#!/bin/bash

# Enhanced Local DeepSeek Installation Script for Ubuntu
# This script installs an open source language model with a REPL interface and file upload capabilities
# All processing remains local - files never leave your machine

# Exit on error
set -e

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default model
DEFAULT_MODEL="deepseek-ai/deepseek-v2"

# Parse command line arguments
function show_help {
    echo -e "${BLUE}Usage:${NC} $0 [options]"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -h, --help                  Show this help message"
    echo "  -m, --model MODEL_ID        Specify model ID (e.g., deepseek-ai/deepseek-v2)"
    echo "  -l, --list                  List available recommended models"
    echo "  -s, --small                 Use smaller models (for systems with less RAM)"
    echo "  --no-auth                   Skip Hugging Face authentication"
    echo "  --cleanup                   Remove temporary files and fix permissions"
    echo "  --uninstall                 Remove the installation completely"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 --model microsoft/phi-3-mini-4k-instruct    # Install Phi-3 Mini"
    echo "  $0 --small                                    # Install smaller models"
    echo "  $0 --list                                     # Show available models"
    echo "  $0 --cleanup                                  # Fix permissions and clean temp files"
    exit 0
}

function list_models {
    echo -e "${BLUE}Available recommended models:${NC}"
    echo -e "${GREEN}Large models (7B+ parameters):${NC}"
    echo "  deepseek-ai/deepseek-v2                   # DeepSeek V2 (requires auth) - 32GB+ RAM"
    echo "  deepseek-ai/deepseek-llm-67b-chat         # DeepSeek LLM 67B (requires auth) - 64GB+ RAM"
    echo "  deepseek-ai/deepseek-coder-33b-instruct   # DeepSeek Coder 33B (requires auth) - 48GB+ RAM"
    echo "  mistralai/Mistral-7B-Instruct-v0.2        # Mistral 7B - 16GB+ RAM"
    echo "  NousResearch/Nous-Hermes-2-Yi-9B          # Nous Hermes 9B - 24GB+ RAM"
    echo "  google/gemma-7b-it                        # Google Gemma 7B - 16GB+ RAM"
    echo ""
    echo -e "${YELLOW}Medium models (3-7B parameters):${NC}"
    echo "  deepseek-ai/deepseek-coder-6.7b-instruct  # DeepSeek Coder 6.7B - 16GB+ RAM"
    echo "  deepseek-ai/deepseek-math-7b-instruct     # DeepSeek Math 7B - 16GB+ RAM"
    echo "  deepseek-ai/deepseek-llm-7b-chat          # DeepSeek LLM 7B - 16GB+ RAM"
    echo "  microsoft/phi-3-mini-4k-instruct          # Microsoft Phi-3 Mini - 8GB+ RAM"
    echo ""
    echo -e "${GREEN}Small models (1-3B parameters):${NC}"
    echo "  deepseek-ai/deepseek-coder-1.3b-instruct  # DeepSeek Coder 1.3B - 4GB+ RAM"
    echo "  microsoft/phi-2                           # Microsoft Phi-2 - 4GB+ RAM"
    echo "  TinyLlama/TinyLlama-1.1B-Chat-v1.0        # TinyLlama 1.1B - 4GB+ RAM"
    echo ""
    echo -e "${YELLOW}Note:${NC} RAM requirements assume 8-bit quantization. With 4-bit quantization,"
    echo "      you can run models with approximately half the listed RAM requirements."
    exit 0
}

# Parse command line options
USE_AUTH=true
USE_SMALL_MODELS=false
SPECIFIED_MODEL=""
DO_CLEANUP=false
DO_UNINSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -m|--model)
            SPECIFIED_MODEL="$2"
            shift
            shift
            ;;
        -l|--list)
            list_models
            ;;
        -s|--small)
            USE_SMALL_MODELS=true
            shift
            ;;
        --no-auth)
            USE_AUTH=false
            shift
            ;;
        --cleanup)
            DO_CLEANUP=true
            shift
            ;;
        --uninstall)
            DO_UNINSTALL=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            ;;
    esac
done

# Use the directory containing this script as the project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Handle cleanup mode
if [ "$DO_CLEANUP" = true ]; then
    if [ ! -d "$PROJECT_DIR" ]; then
        echo -e "${RED}Installation directory not found. Nothing to clean up.${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Cleaning up temporary files and fixing permissions...${NC}"
    cd "$PROJECT_DIR"
    
    # Remove temporary files
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} +
    find . -name ".git" -type d -exec chmod -R 755 {} +
    
    # Fix permissions
    chmod +x start_deepseek.sh
    chmod +x start_deepseek_network.sh
    
    echo -e "${GREEN}Cleanup complete!${NC}"
    exit 0
fi

# Handle uninstall mode
if [ "$DO_UNINSTALL" = true ]; then
    if [ ! -d "$PROJECT_DIR" ]; then
        echo -e "${RED}Installation directory not found. Nothing to uninstall.${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Warning: This will remove the local venv and models directories.${NC}"
    read -p "Are you sure you want to proceed? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Uninstall cancelled.${NC}"
        exit 0
    fi
    
    echo -e "${YELLOW}Removing venv and models...${NC}"
    rm -rf "$PROJECT_DIR/venv"
    rm -rf "$PROJECT_DIR/models"
    echo -e "${GREEN}Uninstall complete! (repo left intact)${NC}"
    exit 0
fi

# Print header
echo -e "${GREEN}=== DeepSeek Local Installation Script ===${NC}"
echo "This script will install a DeepSeek model (or alternative) with a REPL interface on your Ubuntu system."
echo "All processing remains local - uploaded files never leave your machine."
echo "-------------------------------------------------------------------------"

# Check system requirements
echo -e "${YELLOW}Checking system requirements...${NC}"

# Check if Python 3.8+ is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo "Please install Python 3.8 or newer with: sudo apt install python3 python3-pip"
    exit 1
fi

python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
python_major=$(echo "$python_version" | cut -d. -f1)
python_minor=$(echo "$python_version" | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    echo -e "${RED}Error: Python 3.8 or newer is required. Found Python $python_version${NC}"
    echo "Please upgrade your Python installation"
    exit 1
else
    echo -e "${GREEN}Python version $python_version detected - OK${NC}"
fi

# Check disk space (need at least 15GB free for larger models with 32GB RAM)
free_space=$(df -k $HOME | awk 'NR==2 {print $4}')
free_space_gb=$(echo "scale=1; $free_space / 1024 / 1024" | bc)
if (( $(echo "$free_space_gb < 15.0" | bc -l) )); then
    echo -e "${YELLOW}Warning: You have less than 15GB of free space (${free_space_gb}GB).${NC}"
    echo "For optimal performance with 32GB RAM, we recommend at least 15GB free space."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check memory (need at least 32GB for optimized settings)
total_mem=$(grep MemTotal /proc/meminfo | awk '{print $2}')
total_mem_gb=$(echo "scale=1; $total_mem / 1024 / 1024" | bc)

echo -e "${GREEN}System has ${total_mem_gb}GB of RAM${NC}"
if (( $(echo "$total_mem_gb < 30.0" | bc -l) )); then
    echo -e "${YELLOW}Note: This script is optimized for 32GB RAM systems.${NC}"
    echo -e "${YELLOW}Your system has ${total_mem_gb}GB RAM. Script will still work but may use swap space.${NC}"
fi

# Ask for Hugging Face token if authentication is enabled
if [ "$USE_AUTH" = true ]; then
    # First check if we have the token in environment variable
    if [ -n "$HUGGINGFACE_KEY" ]; then
        echo -e "${GREEN}Using Hugging Face token from HUGGINGFACE_KEY environment variable.${NC}"
        HF_TOKEN="$HUGGINGFACE_KEY"
    elif [ -f ~/.huggingface/token ]; then
        echo -e "${GREEN}Found existing Hugging Face token.${NC}"
        echo -e "If you want to use a different token, edit ~/.huggingface/token manually."
        # Read existing token
        HF_TOKEN=$(grep -o '"token": *"[^"]*"' ~/.huggingface/token | cut -d'"' -f4)
    else
        echo -e "${YELLOW}Hugging Face authentication is enabled${NC}"
        echo -e "You can find your token at https://huggingface.co/settings/tokens"
        echo -e "${GREEN}Note: Your token will only be stored locally on this machine${NC}"
        echo -e "${YELLOW}Security note: The token will be stored in plaintext in ~/.git-credentials${NC}"
        echo -e "This is standard Git behavior but something to be aware of."
        echo -e "To skip this prompt in the future, set the HUGGINGFACE_KEY environment variable."
        read -p "Enter your Hugging Face token (or press Enter to skip and use public models): " HF_TOKEN
    fi
    
    if [ -n "$HF_TOKEN" ]; then
        # Store the token for huggingface-cli
        echo -e "${GREEN}Storing your Hugging Face token locally...${NC}"
        mkdir -p ~/.huggingface
        echo -e "{\n  \"token\": \"$HF_TOKEN\"\n}" > ~/.huggingface/token
        chmod 600 ~/.huggingface/token  # Set proper permissions
        
        # Also store for git credential
        git config --global credential.helper store
        echo "https://USER:$HF_TOKEN@huggingface.co" > ~/.git-credentials
        chmod 600 ~/.git-credentials  # Set proper permissions
        
        echo -e "${GREEN}Hugging Face token configured successfully!${NC}"
    else
        echo -e "${YELLOW}No token provided, will use publicly available models${NC}"
        USE_AUTH=false
    fi
else
    echo -e "${YELLOW}Skipping Hugging Face authentication (--no-auth specified)${NC}"
fi

# Create project directory
echo -e "${GREEN}Creating project directory at $PROJECT_DIR...${NC}"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Update system packages
echo -e "${GREEN}Updating system packages...${NC}"
sudo apt update

# Install system dependencies
echo -e "${GREEN}Installing system dependencies...${NC}"
sudo apt install -y python3 python3-pip python3-venv git build-essential bc curl

# Install Git LFS
echo -e "${GREEN}Installing Git LFS...${NC}"
if ! command -v git-lfs &> /dev/null; then
    sudo apt install -y git-lfs
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Git LFS installation via apt failed, trying alternative method...${NC}"
        curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
        sudo apt install -y git-lfs
    fi
    git lfs install
else
    echo "Git LFS is already installed"
    git lfs install
fi

# Set up Python virtual environment
echo -e "${GREEN}Setting up Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Detect GPU and install appropriate PyTorch version
echo -e "${GREEN}Detecting GPU and installing PyTorch...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected, installing PyTorch with CUDA support...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}No NVIDIA GPU detected, installing CPU-only PyTorch...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    GPU_AVAILABLE=false
fi

# Install required packages
echo -e "${GREEN}Installing required Python packages...${NC}"
pip install transformers accelerate sentencepiece gradio huggingface_hub

# Install bitsandbytes only if GPU is available
if [ "$GPU_AVAILABLE" = true ]; then
    pip install bitsandbytes
fi

# Create models directory
echo -e "${GREEN}Creating models directory...${NC}"
mkdir -p models
cd models

# Define model list based on specified options
if [ -n "$SPECIFIED_MODEL" ]; then
    echo -e "${GREEN}Using specified model: $SPECIFIED_MODEL${NC}"
    MODELS=("$SPECIFIED_MODEL")
elif [ "$USE_SMALL_MODELS" = true ]; then
    echo -e "${YELLOW}Using small models list (optimized for systems with less RAM)...${NC}"
    # Small models (<3B parameters)
    MODELS=(
        "deepseek-ai/deepseek-coder-1.3b-instruct"
        "microsoft/phi-2"
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
elif [ "$USE_AUTH" = true ]; then
    echo -e "${GREEN}Using authenticated models list...${NC}"
    # Default to DeepSeek V2 and other high-quality models that require authentication
    MODELS=(
        "deepseek-ai/deepseek-v2"
        "deepseek-ai/deepseek-llm-67b-chat"
        "deepseek-ai/deepseek-coder-33b-instruct"
        "deepseek-ai/deepseek-coder-6.7b-instruct"
        "deepseek-ai/deepseek-math-7b-instruct"
    )
else
    echo -e "${YELLOW}Using public models list...${NC}"
    # Public models that don't require authentication
    MODELS=(
        "deepseek-ai/deepseek-coder-6.7b-instruct"
        "deepseek-ai/deepseek-math-7b-instruct"
        "deepseek-ai/deepseek-llm-7b-chat"
        "NousResearch/Nous-Hermes-2-Yi-9B"
        "microsoft/phi-3-mini-4k-instruct"
        "mistralai/Mistral-7B-Instruct-v0.2"
    )
fi

MODEL_DOWNLOADED=false

# First, check if any of the models are already downloaded
if [ -d "$PROJECT_DIR/models" ]; then
    for MODEL in "${MODELS[@]}"; do
        MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
        if [ -d "$PROJECT_DIR/models/$MODEL_NAME" ]; then
            echo -e "${GREEN}Found already downloaded model: $MODEL_NAME${NC}"
            
            if [ -n "$SPECIFIED_MODEL" ] && [ "$SPECIFIED_MODEL" != "$MODEL" ]; then
                echo -e "${YELLOW}This is not the model you specified. Continuing search...${NC}"
            else
                echo -e "${GREEN}Using existing model: $MODEL_NAME${NC}"
                MODEL_DOWNLOADED=true
                break
            fi
        fi
    done
fi

# If not found, download a new model
if [ "$MODEL_DOWNLOADED" = false ]; then
    for MODEL in "${MODELS[@]}"; do
        if [ "$MODEL_DOWNLOADED" = false ]; then
            MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
            
            # Check RAM requirements for large models
            total_mem=$(grep MemTotal /proc/meminfo | awk '{print $2}')
            total_mem_gb=$(echo "scale=1; $total_mem / 1024 / 1024" | bc)
            
            # Warn about potential RAM issues
            ram_warning=""
            if [[ "$MODEL_NAME" == *"67b"* || "$MODEL_NAME" == *"70b"* ]] && (( $(echo "$total_mem_gb < 64" | bc -l) )); then
                ram_warning="${RED}Warning: This model requires 64GB+ RAM, you have ${total_mem_gb}GB${NC}"
            elif [[ "$MODEL_NAME" == *"33b"* || "$MODEL_NAME" == *"40b"* ]] && (( $(echo "$total_mem_gb < 48" | bc -l) )); then
                ram_warning="${RED}Warning: This model requires 48GB+ RAM, you have ${total_mem_gb}GB${NC}"
            elif [[ "$MODEL_NAME" == *"13b"* || "$MODEL_NAME" == *"7b"* ]] && (( $(echo "$total_mem_gb < 16" | bc -l) )); then
                ram_warning="${YELLOW}Warning: This model works best with 16GB+ RAM, you have ${total_mem_gb}GB${NC}"
            fi
            
            if [ -n "$ram_warning" ]; then
                echo -e "$ram_warning"
                if [ -n "$SPECIFIED_MODEL" ]; then
                    read -p "Continue anyway? (y/n) " -n 1 -r
                    echo
                    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                        echo -e "${YELLOW}Skipping this model due to RAM constraints${NC}"
                        continue
                    fi
                fi
            fi
            
            echo -e "${YELLOW}Attempting to download $MODEL...${NC}"
            
            # Try to clone the model repository
            if git clone "https://huggingface.co/$MODEL" "$MODEL_NAME"; then
                MODEL_DOWNLOADED=true
                echo -e "${GREEN}Successfully downloaded $MODEL_NAME${NC}"
                # Check size of downloaded model
                model_size=$(du -sh "$MODEL_NAME" | cut -f1)
                echo -e "${GREEN}Model size: $model_size${NC}"
                break
            else
                echo -e "${RED}Failed to download $MODEL_NAME, trying next model...${NC}"
                rm -rf "$MODEL_NAME"
                
                # Special handling if this was the specified model
                if [ -n "$SPECIFIED_MODEL" ] && [ "$SPECIFIED_MODEL" = "$MODEL" ]; then
                    echo -e "${RED}Failed to download the specified model: $SPECIFIED_MODEL${NC}"
                    echo -e "${YELLOW}This could be due to:${NC}"
                    echo "  - The model requires authentication (try providing a Hugging Face token)"
                    echo "  - The model repository doesn't exist"
                    echo "  - Network connectivity issues"
                    
                    read -p "Do you want to try using a different model from the recommended list? (y/n) " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        if [ "$USE_AUTH" = true ]; then
                            MODELS=(
                                "deepseek-ai/deepseek-v2"
                                "deepseek-ai/deepseek-coder-6.7b-instruct"
                                "mistralai/Mistral-7B-Instruct-v0.2"
                                "microsoft/phi-3-mini-4k-instruct"
                            )
                        else
                            MODELS=(
                                "deepseek-ai/deepseek-coder-6.7b-instruct"
                                "microsoft/phi-3-mini-4k-instruct"
                                "microsoft/phi-2"
                            )
                        fi
                        SPECIFIED_MODEL=""
                    else
                        echo -e "${RED}Exiting installation.${NC}"
                        exit 1
                    fi
                fi
            fi
        fi
    done
fi

# Check if a model was successfully downloaded
if [ "$MODEL_DOWNLOADED" = false ]; then
    echo -e "${RED}Error: Failed to download any model. Please check your internet connection or try again.${NC}"
    exit 1
fi

# Navigate back to project directory
cd "$PROJECT_DIR"

# Create REPL interface Python script
echo -e "${GREEN}Creating REPL interface script...${NC}"
cat > deepseek_repl.py << 'EOL'
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
        # Try to find a model directory
        models_dir = Path("./models")
        if models_dir.exists():
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
            if model_dirs:
                model_name = model_dirs[0].name
                log(f"Found model directory: {model_name}")
            else:
                log("No model directories found in ./models", "ERROR")
                return None, None
        else:
            log("Models directory not found", "ERROR")
            return None, None
    
    model_path = f"./models/{model_name}"
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
    demo.launch(server_name="127.0.0.1", port=port, share=False)

if __name__ == "__main__":
    main()
EOL

# Create startup script
echo -e "${GREEN}Creating startup script...${NC}"
cat > start_deepseek.sh << EOL
#!/bin/bash
cd "\$(dirname "\$0")"
source venv/bin/activate

# Default settings
PORT=7860
MODEL_DIR=""

# Parse command line options
while [[ \$# -gt 0 ]]; do
    case \$1 in
        -p|--port)
            PORT="\$2"
            shift
            shift
            ;;
        -m|--model)
            MODEL_DIR="\$2"
            shift
            shift
            ;;
        -h|--help)
            echo "Usage: \$0 [options]"
            echo "Options:"
            echo "  -p, --port PORT    Specify port (default: 7860)"
            echo "  -m, --model DIR    Specify model directory name (default: auto-detect)"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: \$1"
            exit 1
            ;;
    esac
done

# Set environment variable to help with CUDA memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export DEEPSEEK_PORT=\$PORT

# Find the model directory if not specified
if [ -z "\$MODEL_DIR" ]; then
    export MODEL_NAME=\$(ls models | head -1)
else
    if [ -d "models/\$MODEL_DIR" ]; then
        export MODEL_NAME="\$MODEL_DIR"
    else
        echo "Error: Model directory 'models/\$MODEL_DIR' not found"
        echo "Available models:"
        ls -1 models/
        exit 1
    fi
fi

echo "Starting with model: \$MODEL_NAME on port \$PORT"

# Launch the interface
python deepseek_repl.py
EOL

# Create a local-only startup script
cat > start_deepseek_network.sh << EOL
#!/bin/bash
cd "\$(dirname "\$0")"
source venv/bin/activate

# Default settings
PORT=7860
MODEL_DIR=""
IP="0.0.0.0"

# Parse command line options
while [[ \$# -gt 0 ]]; do
    case \$1 in
        -p|--port)
            PORT="\$2"
            shift
            shift
            ;;
        -m|--model)
            MODEL_DIR="\$2"
            shift
            shift
            ;;
        -i|--ip)
            IP="\$2"
            shift
            shift
            ;;
        -h|--help)
            echo "Usage: \$0 [options]"
            echo "Options:"
            echo "  -p, --port PORT    Specify port (default: 7860)"
            echo "  -m, --model DIR    Specify model directory name (default: auto-detect)"
            echo "  -i, --ip IP        Specify IP address to bind to (default: 0.0.0.0)"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: \$1"
            exit 1
            ;;
    esac
done

# Set environment variable to help with CUDA memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export DEEPSEEK_PORT=\$PORT

# Find the model directory if not specified
if [ -z "\$MODEL_DIR" ]; then
    export MODEL_NAME=\$(ls models | head -1)
else
    if [ -d "models/\$MODEL_DIR" ]; then
        export MODEL_NAME="\$MODEL_DIR"
    else
        echo "Error: Model directory 'models/\$MODEL_DIR' not found"
        echo "Available models:"
        ls -1 models/
        exit 1
    fi
fi

echo "Starting with model: \$MODEL_NAME on \$IP:\$PORT (network accessible)"

# Create a temporary modified copy of the script
cp deepseek_repl.py deepseek_repl_network.py

# Modify the launch line in the script to allow network access
sed -i "s/server_name=\"127.0.0.1\", port=port, share=False/server_name=\"\$IP\", port=port, share=False/g" deepseek_repl_network.py

# Launch the interface
python deepseek_repl_network.py

# Clean up
rm deepseek_repl_network.py
EOL

# Make startup scripts executable
chmod +x start_deepseek.sh
chmod +x start_deepseek_network.sh

# Create a model info file for reference
cat > model_info.txt << EOF
Model: $MODEL_NAME
Original repository: $MODEL
Installed on: $(date)
System: $(uname -a)
RAM: ${total_mem_gb}GB
CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)
GPU: $(command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=gpu_name --format=csv,noheader || echo "None detected")
Python: $(python3 --version)
EOF

echo -e "${GREEN}-------------------------------------------------------------------------${NC}"
echo -e "${GREEN}Installation complete!${NC}"
echo -e "Using model: ${YELLOW}$MODEL_NAME${NC}"

# Usage instructions
echo -e "\n${BLUE}Usage instructions:${NC}"
echo -e "1. To start the REPL interface (localhost only), run:"
echo -e "   ${YELLOW}cd $PROJECT_DIR && ./start_deepseek.sh${NC}"
echo -e ""
echo -e "2. To allow access from other computers on your network, run:"
echo -e "   ${YELLOW}cd $PROJECT_DIR && ./start_deepseek_network.sh${NC}"
echo -e ""
echo -e "3. To install a different model in the future, run:"
echo -e "   ${YELLOW}$0 --model MODEL_NAME${NC}"
echo -e ""
echo -e "4. To use a different already-installed model:"
echo -e "   ${YELLOW}cd $PROJECT_DIR && ./start_deepseek.sh --model MODEL_DIRECTORY_NAME${NC}"
echo -e ""
echo -e "5. To see available models for installation:"
echo -e "   ${YELLOW}$0 --list${NC}"
echo -e ""
echo -e "6. To see installed models:"
echo -e "   ${YELLOW}ls -l $PROJECT_DIR/models/${NC}"
echo -e ""
echo -e "${GREEN}PRIVACY NOTICE:${NC}"
echo -e "- All processing is done locally on your machine"
echo -e "- Uploaded files are never sent over the internet"
echo -e "- The interface is set to localhost-only by default for maximum privacy"
echo -e "${GREEN}-------------------------------------------------------------------------${NC}"
echo -e "Notes:"
echo -e "- The first start might take a few minutes as the model is loaded into memory"
echo -e "- CPU-only mode will be very slow - a GPU is recommended"
echo -e "- If you encounter errors, check the terminal output for details"
echo -e "${GREEN}-------------------------------------------------------------------------${NC}"
