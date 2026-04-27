# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A local, privacy-first web interface for running DeepSeek and other open-source LLMs entirely on the user's machine via a Gradio UI. No data leaves the host. Supports GPU acceleration with 4-bit quantization, file uploads, and multi-turn chat.

## Commands

```bash
# Install dependencies and download a model
./install_deepseek.sh

# Run the app (localhost only, port 7860)
./start_deepseek.sh

# Run with a specific model and port
./start_deepseek.sh --model deepseek-coder-6.7b-instruct --port 8080

# Run accessible on the local network
./start_deepseek_network.sh

# Run tests
source venv/bin/activate
pytest tests/

# Run a single test
pytest tests/test_deepseek_repl.py::test_get_max_length_uses_tokenizer_cap
```

## Architecture

The entire application lives in `deepseek_repl.py` (~260 lines). The call graph is:

```
main()
  load_model()               → finds model in LLM_MODEL_HOME (~/.models) or ./models/
  process_query()            → Gradio callback: files → text context → model.generate()
    process_files()          → reads uploaded files as UTF-8, falls back to binary description
    get_max_length()         → caps token limit at min(tokenizer.model_max_length, 2048)
  get_system_info()          → GPU/VRAM/platform info displayed in UI
```

Key generation parameters (hardcoded in `process_query`): `max_new_tokens=512, temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.1`.

## Configuration

All configuration is via environment variables, set by the startup scripts:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL_HOME` | `~/.models` | Where models are stored |
| `MODEL_PATH` | auto-detected | Full path to selected model dir |
| `DEEPSEEK_PORT` | `7860` | Gradio server port |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | CUDA memory optimization |

## Dependencies

`requirements.txt` is the single source of truth. Key packages: `torch`, `transformers`, `accelerate`, `bitsandbytes` (quantization), `gradio>=5.0.0`, `huggingface-hub`. No `setup.py` or `pyproject.toml`.

The virtual environment lives at `./venv/` and is excluded from git. Models are stored in `~/.models` (or `./models/` locally) and are also excluded from git.
