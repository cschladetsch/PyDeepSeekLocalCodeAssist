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

# Kill anything already occupying the target port
existing_pid=$(fuser "${PORT}/tcp" 2>/dev/null)
if [ -n "$existing_pid" ]; then
    echo "Killing existing process(es) on port $PORT: $existing_pid"
    fuser -k "${PORT}/tcp" 2>/dev/null
    sleep 1
fi

# Ensure WSL2 localhost forwarding so 127.0.0.1 works from Windows.
# NAT mode (WSL2 default) + localhostForwarding=true is the correct setup.
# Mirrored networking breaks the port relay mechanism and should not be used here.
if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
    WIN_USER_HOME=$(cmd.exe /c "echo %USERPROFILE%" 2>/dev/null | tr -d '\r\n')
    WIN_USER_HOME_UNIX=$(wslpath "$WIN_USER_HOME" 2>/dev/null)
    WSLCONFIG="${WIN_USER_HOME_UNIX}/.wslconfig"
    if [ -n "$WIN_USER_HOME_UNIX" ]; then
        NEEDS_RESTART=0

        # Remove mirrored networking if present — it breaks localhost forwarding
        if grep -qs "networkingMode\s*=\s*mirrored" "$WSLCONFIG" 2>/dev/null; then
            sed -i 's/^\s*networkingMode\s*=\s*mirrored.*/networkingMode=NAT/' "$WSLCONFIG"
            NEEDS_RESTART=1
        fi

        # Ensure localhostForwarding is enabled
        if ! grep -qs "localhostForwarding\s*=\s*true" "$WSLCONFIG" 2>/dev/null; then
            if ! grep -qs "\[wsl2\]" "$WSLCONFIG" 2>/dev/null; then
                printf '\n[wsl2]\nlocalhostForwarding=true\n' >> "$WSLCONFIG"
            else
                sed -i '/\[wsl2\]/a localhostForwarding=true' "$WSLCONFIG"
            fi
            NEEDS_RESTART=1
        fi

        if [ "$NEEDS_RESTART" -eq 1 ]; then
            echo "---"
            echo "WSL2 config updated in $WSLCONFIG"
            echo "Run 'wsl --shutdown' in Windows PowerShell, then restart WSL2 for changes to take effect."
            echo "---"
        fi
    fi
fi

# Find the first directory containing a HuggingFace config.json, up to 3 levels deep
find_model_path() {
    local base="$1"
    local name="$2"
    local search_root="${name:+$base/$name}"
    search_root="${search_root:-$base}"
    find "$search_root" -maxdepth 3 -name "config.json" 2>/dev/null \
        | head -1 | xargs -I{} dirname {}
}

if [ -z "$MODEL_DIR" ]; then
    MODEL_PATH=$(find_model_path "$MODEL_HOME")
    if [ -z "$MODEL_PATH" ] && [ -d "models" ]; then
        echo "Warning: using local models/ dir. Set LLM_MODEL_HOME to use ~/.models"
        MODEL_PATH=$(find_model_path "$(pwd)/models")
    fi
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: No models found under $MODEL_HOME or local models/"
        exit 1
    fi
    export MODEL_PATH
    export MODEL_NAME=$(basename "$MODEL_PATH")
else
    MODEL_PATH=$(find_model_path "$MODEL_HOME" "$MODEL_DIR")
    if [ -z "$MODEL_PATH" ] && [ -d "models/$MODEL_DIR" ]; then
        echo "Warning: found model in local models/ not in $MODEL_HOME"
        MODEL_PATH=$(find_model_path "$(pwd)/models" "$MODEL_DIR")
    fi
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: Model '$MODEL_DIR' not found under $MODEL_HOME or local models/"
        find "$MODEL_HOME" -name "config.json" 2>/dev/null | xargs -I{} dirname {} || true
        exit 1
    fi
    export MODEL_PATH
    export MODEL_NAME="$MODEL_DIR"
fi

echo "Starting with model: $MODEL_NAME on port $PORT"
echo "Model path: $MODEL_PATH"

python deepseek_repl.py
