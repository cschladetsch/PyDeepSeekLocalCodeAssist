#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d venv ]; then
    echo "[ERROR] venv not found in $(pwd). Create it first: python3 -m venv venv"
    exit 1
fi

source venv/bin/activate

echo "--- Python: $(which python) ---"
echo "--- transformers version: $(python -c 'import transformers; print(transformers.__version__)') ---"

echo "--- Installing/upgrading tokenizer deps ---"
pip install --upgrade sentencepiece tiktoken protobuf

echo "--- Verifying imports ---"
python -c "import sentencepiece; print('sentencepiece OK:', sentencepiece.__file__)"
python -c "import tiktoken; print('tiktoken OK:', tiktoken.__file__)"
python -c "import google.protobuf; print('protobuf OK')"

echo "--- Model directory contents ---"
ls -la /home/christian/.models/deepseek-r1-7b

exec ./start_deepseek.sh
