#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d venv ]; then
    echo "[ERROR] venv not found in $(pwd). Create it first: python3 -m venv venv"
    exit 1
fi

source venv/bin/activate

# Ensure tokenizer backends are present (fixes the sentencepiece/tiktoken load error)
python -c "import sentencepiece" 2>/dev/null || pip install -q sentencepiece
python -c "import tiktoken" 2>/dev/null || pip install -q tiktoken

exec ./start_deepseek.sh
