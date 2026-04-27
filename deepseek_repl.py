import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
import os
import sys
import platform
import traceback
from pathlib import Path

# Get system info for display
def get_system_info():
    try:
        gpu_info = "No GPU detected"
        if torch.cuda.is_available():
            gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
            gpu_info += f" | Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        
        return {
            "Platform": platform.platform(),
            "Python": platform.python_version(),
            "Torch": torch.__version__,
            "GPU": gpu_info
        }
    except Exception as e:
        return {"Error": str(e)}

# Setup logging
def log(message, level="INFO"):
    print(f"[{level}] {message}")

def log_vram(label=""):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
    tag = f" [{label}]" if label else ""
    log(f"VRAM{tag}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.2f} GB total")

# Configure and load model
def load_model():
    log("Loading model...")
    
    # Get model name from environment variable or find it in models directory
    model_name = os.environ.get('MODEL_NAME')
    
    if not model_name:
        # Try to find a model in LLM_MODEL_HOME or local models/
        model_home = os.environ.get('LLM_MODEL_HOME', os.path.expanduser('~/.models'))
        for search_dir in [model_home, './models']:
            p = Path(search_dir)
            if p.exists():
                model_dirs = [d for d in p.iterdir() if d.is_dir()]
                if model_dirs:
                    model_name = model_dirs[0].name
                    log(f"Found model directory: {model_name} in {search_dir}")
                    break
        if not model_name:
            log("No model directories found", "ERROR")
            return None, None
    
    # Use MODEL_PATH if set, otherwise resolve from MODEL_HOME
    model_path = os.environ.get('MODEL_PATH')
    if not model_path:
        model_home = os.environ.get('LLM_MODEL_HOME', os.path.expanduser('~/.models'))
        model_path = os.path.join(model_home, model_name)
        if not os.path.exists(model_path):
            model_path = f"./models/{model_name}"  # local fallback
    log(f"Using model: {model_name} at {model_path}")
    
    try:
        # Load tokenizer first as it's usually smaller
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        log("Tokenizer loaded successfully")
        
        # Configure model loading based on available hardware
        if torch.cuda.is_available():
            log("Loading model with GPU acceleration (float16)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model.eval()
            log("Model loaded with float16 precision")
            log_vram("after load")
        else:
            log("Loading model on CPU (will be slow)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model.eval()
            log("Model loaded on CPU")
        
        return model, tokenizer
    
    except Exception as e:
        log(f"Error loading model: {str(e)}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        return None, None

# Get the maximum supported length for the model
def get_max_length(tokenizer):
    if hasattr(tokenizer, "model_max_length"):
        return min(tokenizer.model_max_length, 2048)
    return 1024

# Process file content — accepts string paths (multimodal) or legacy file objects
def process_files(files):
    file_context = ""
    if not files:
        return file_context

    image_exts = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}

    for file in files:
        try:
            if isinstance(file, str):
                file_path = file
                file_name = os.path.basename(file_path)
            elif isinstance(file, dict):
                file_path = file.get("path", "")
                file_name = file.get("orig_name", os.path.basename(file_path))
            else:
                file_path = getattr(file, "path", None) or getattr(file, "name", "")
                file_name = getattr(file, "orig_name", None) or os.path.basename(file_path)

            if not file_path or not os.path.exists(file_path):
                continue

            file_size = os.path.getsize(file_path)
            ext = os.path.splitext(file_name)[1].lower()

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                file_context += f"\nFile: {file_name} ({file_size} bytes)\n{content}\n"
            except UnicodeDecodeError:
                kind = "image" if ext in image_exts else "binary"
                file_context += f"\n[{kind} file: {file_name}, {file_size} bytes]\n"

        except Exception as e:
            file_context += f"\n[error reading file: {e}]\n"

    return file_context

# Generator that yields response text chunks — used by gr.ChatInterface (multimodal=True)
def chat_fn(message, history):
    # multimodal=True: message is {"text": "...", "files": ["path", ...]}
    if isinstance(message, dict):
        text = (message.get("text") or "").strip()
        files = message.get("files") or []
    else:
        text = str(message).strip()
        files = []

    if not text and not files:
        return

    file_context = process_files(files)
    if file_context and text:
        input_text = f"Files:\n{file_context}\nQuestion: {text}"
    elif file_context:
        input_text = file_context.strip()
    else:
        input_text = text

    try:
        tokenizer = chat_fn._tokenizer
        model = chat_fn._model

        # Build message list for chat template, including prior turns
        messages = []
        for msg in (history or []):
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c) for c in content
                )
            if role in ("user", "assistant") and str(content).strip():
                messages.append({"role": role, "content": str(content).strip()})
        messages.append({"role": "user", "content": input_text})

        # Apply the model's instruct chat template so generation is non-empty
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = input_text

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60.0
        )

        def generate():
            with torch.no_grad():
                model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    streamer=streamer,
                )

        log_vram("before generate")
        thread = threading.Thread(target=generate)
        thread.start()

        partial = ""
        for token in streamer:
            partial += token
            yield partial

        thread.join()
        log_vram("after generate")

        # Release GPU memory held by this request
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not partial.strip():
            yield "I couldn't generate a response. The model may not understand this type of query."

    except Exception as e:
        log(f"Generation error: {e}", "ERROR")
        yield f"Error: {str(e)}"


# Main function
def main():
    model, tokenizer = load_model()

    if model is None or tokenizer is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)

    chat_fn._model = model
    chat_fn._tokenizer = tokenizer

    system_info = get_system_info()
    info_text = "\n".join([f"**{k}**: {v}" for k, v in system_info.items()])

    interface = gr.ChatInterface(
        fn=chat_fn,
        multimodal=True,
        title=f"DeepSeek Local — {os.environ.get('MODEL_NAME', 'Unknown Model')}",
        description="All processing occurs locally. Files and queries never leave your machine.",
        chatbot=gr.Chatbot(show_label=False),
        textbox=gr.MultimodalTextbox(
            placeholder="Message... (Enter to send, Shift+Enter for new line)",
            file_types=["image", "text", ".pdf", ".md", ".csv"],
            show_label=False,
        ),
        fill_height=True,
    )

    with interface:
        with gr.Accordion("System Information", open=False):
            gr.Markdown(info_text)
            gr.Markdown(f"**Model**: {os.environ.get('MODEL_NAME', 'Unknown')}")

    port = int(os.environ.get('DEEPSEEK_PORT', 7860))
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(),
        js="() => { const app = document.querySelector('gradio-app'); if (app) app.theme_mode = 'dark'; }",
    )

if __name__ == "__main__":
    main()
