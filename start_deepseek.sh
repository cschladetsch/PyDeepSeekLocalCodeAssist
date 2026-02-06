#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

# Default settings (allow env override)
PORT="${DEEPSEEK_PORT:-7860}"
MODEL_DIR=""

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift
            shift
            ;;
        -m|--model)
            MODEL_DIR="$2"
            shift
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -p, --port PORT    Specify port (default: 7860)"
            echo "  -m, --model DIR    Specify model directory name (default: auto-detect)"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set environment variable to help with CUDA memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export DEEPSEEK_PORT=$PORT
export GRADIO_SERVER_PORT=$PORT

# Find the model directory if not specified
if [ -z "$MODEL_DIR" ]; then
    export MODEL_NAME=$(ls models | head -1)
else
    if [ -d "models/$MODEL_DIR" ]; then
        export MODEL_NAME="$MODEL_DIR"
    else
        echo "Error: Model directory 'models/$MODEL_DIR' not found"
        echo "Available models:"
        ls -1 models/
        exit 1
    fi
fi

echo "Starting with model: $MODEL_NAME on port $PORT"

# Launch the interface
python deepseek_repl.py
