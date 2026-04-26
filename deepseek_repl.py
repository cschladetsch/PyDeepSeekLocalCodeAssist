import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
            log("Loading model with GPU acceleration...")
            
            # For 32GB RAM system, we can use 8-bit quantization for better quality
            # while still allowing for larger models
            try:
                # First try 8-bit quantization (better quality than 4-bit)
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_has_fp16_weight=True
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: "24GiB", "cpu": "8GiB"}  # Optimize for 32GB system
                )
                log("Model loaded with 8-bit quantization (optimized for 32GB RAM)")
            except Exception as e:
                log(f"8-bit quantization failed: {str(e)}", "WARNING")
                log("Falling back to 16-bit precision...")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: "24GiB", "cpu": "8GiB"}  # Optimize for 32GB system
                )
                log("Model loaded with 16-bit precision (optimized for 32GB RAM)")
        else:
            log("Loading model on CPU (will be slow)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
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

# Process file content
def process_files(files):
    file_context = ""
    
    for file in files:
        try:
            file_name = os.path.basename(file.name)
            file_size = os.path.getsize(file.name)
            
            # First try to read as text
            try:
                with open(file.name, "r", encoding="utf-8") as f:
                    content = f.read()
                    file_context += f"\nFile: {file_name}\nSize: {file_size} bytes\nContent:\n{content}\n\n"
            except UnicodeDecodeError:
                # Try to handle binary files
                file_context += f"\nBinary File: {file_name}\nSize: {file_size} bytes\n"
                
                # Check if it's a common binary type we can read
                if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.pdf')):
                    file_context += f"File type: {file_name.split('.')[-1]}\n"
                    
        except Exception as e:
            file_context += f"\nError reading file {file_name}: {str(e)}\n"
    
    return file_context

# Function to process user queries
def process_query(message, history, files, model, tokenizer):
    if model is None or tokenizer is None:
        return "Error: Model failed to load. Please check the logs and restart the application."
    
    # Process file content if files are uploaded
    file_context = process_files(files) if files else ""
    
    # Combine file context with user message
    if file_context:
        input_text = f"I've uploaded the following files (processed locally):\n{file_context}\n\nMy question is: {message}"
    else:
        input_text = message
    
    # Generate response
    try:
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        max_length = min(input_ids.shape[1] + 1024, get_max_length(tokenizer))
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                do_sample=True,
                num_beams=1,  # Disable beam search for faster generation
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1  # Slightly penalize repetition
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the response part (not the input)
        if response.startswith(input_text):
            response = response[len(input_text):].strip()
        
        # If response is empty or just whitespace, provide a fallback
        if not response or response.isspace():
            response = "I couldn't generate a meaningful response. Please try rephrasing your question."
        
        return response
    
    except Exception as e:
        return f"An error occurred while generating a response: {str(e)}"

# Main function
def main():
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Create a simple cache mechanism for large texts
    input_cache = {}
    response_cache = {}
    
    # Get system info
    system_info = get_system_info()
    info_text = "\n".join([f"**{k}**: {v}" for k, v in system_info.items()])
    
    # Create Gradio interface
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# DeepSeek Local Interface - {os.environ.get('MODEL_NAME', 'Unknown Model')}")
        gr.Markdown("All processing occurs locally on your machine. Uploaded files are never sent over the internet.")
        
        with gr.Accordion("System Information", open=False):
            gr.Markdown(info_text)
            gr.Markdown(f"**Model**: {os
.environ.get('MODEL_NAME', 'Unknown')}")
            gr.Markdown("**Privacy Notice**: All processing occurs locally. Files and queries never leave your machine.")
        
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(file_count="multiple", label="Upload Files (Processed Locally)")
                gr.Markdown("Files are processed entirely on your local machine.")
            
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(label="Enter your query", placeholder="Type your question here...", lines=3)
        clear = gr.Button("Clear Chat")
        
        # Handle query submission
        msg.submit(
            fn=lambda message, history, files: process_query(message, history, files, model, tokenizer),
            inputs=[msg, chatbot, files],
            outputs=chatbot
        )
        
        # Clear chat history
        clear.click(lambda: None, None, chatbot, queue=False)
    
    # Get port from environment variable or use default
    port = int(os.environ.get('DEEPSEEK_PORT', 7860))
    
    # Launch the interface
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

if __name__ == "__main__":
    main()
