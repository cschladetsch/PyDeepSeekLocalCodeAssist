#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

# Canonical model store
MODEL_HOME="${LLM_MODEL_HOME:-$HOME/.models}"

# Default settings
PORT=7860
MODEL_DIR=""

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -p, --port PORT    Specify port (default: 7860)"
            echo "  -m, --model DIR    Specify model directory name (default: auto-detect)"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Model store: ${MODEL_HOME} (override with LLM_MODEL_HOME)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export DEEPSEEK_PORT=$PORT
export GRADIO_SERVER_PORT=$PORT

if [ -z "$MODEL_DIR" ]; then
    if [ -d "$MODEL_HOME" ] && [ -n "$(ls -A "$MODEL_HOME" 2>/dev/null)" ]; then
        export MODEL_NAME=$(ls "$MODEL_HOME" | head -1)
        export MODEL_PATH="$MODEL_HOME/$MODEL_NAME"
    elif [ -d "models" ] && [ -n "$(ls -A models 2>/dev/null)" ]; then
        echo "Warning: using local models/ dir. Set LLM_MODEL_HOME to use ~/.models"
        export MODEL_NAME=$(ls models | head -1)
        export MODEL_PATH="$(pwd)/models/$MODEL_NAME"
    else
        echo "Error: No models found in $MODEL_HOME or local models/"
        exit 1
    fi
else
    if [ -d "$MODEL_HOME/$MODEL_DIR" ]; then
        export MODEL_NAME="$MODEL_DIR"
        export MODEL_PATH="$MODEL_HOME/$MODEL_DIR"
    elif [ -d "models/$MODEL_DIR" ]; then
        echo "Warning: found model in local models/ not in $MODEL_HOME"
        export MODEL_NAME="$MODEL_DIR"
        export MODEL_PATH="$(pwd)/models/$MODEL_DIR"
    else
        echo "Error: Model '$MODEL_DIR' not found in $MODEL_HOME or local models/"
        ls -1 "$MODEL_HOME" 2>/dev/null || true
        ls -1 models/ 2>/dev/null || true
        exit 1
    fi
fi

echo "Starting with model: $MODEL_NAME on port $PORT"
echo "Model path: $MODEL_PATH"

python deepseek_repl.py
