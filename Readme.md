# DeepSeek Local

A local, private interface for running DeepSeek language models on your own machine. All processing happens locally - files and queries never leave your computer.

## Demo

![Demo](resources/Untitled.jpg)
## Features

- Run DeepSeek language models entirely on your local machine
- Upload and process files privately
- Optimized for systems with 8GB GPU memory using 4-bit quantization
- Support for various DeepSeek models including DeepSeek-Coder
- Code-specific formatting with syntax highlighting

## Installation

Suggestion: use `pip freeze` to ensure the floor doesn't become lava later.

### Option 1: Using the Installation Script (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/cschladetsch/deepseek-local.git
   cd deepseek-local
   ```

2. Run the installation script:
   ```bash
   chmod +x install_deepseek.sh
   ./install_deepseek.sh
   ```

3. During installation, you'll be prompted for your Hugging Face token if you want to use gated models.

4. Install additional dependencies for code formatting:
   ```bash
   source venv/bin/activate
   pip install markdown pygments
   ```

### Option 2: Manual Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/deepseek-local.git
   cd deepseek-local
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download a model from Hugging Face:
   ```bash
   mkdir -p models
   cd models
   git clone https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct
   cd ..
   ```

## Usage

### Starting the Interface

1. Start the local interface:
   ```bash
   ./start_deepseek.sh
   ```

2. To share with other devices on your network:
   ```bash
   ./start_deepseek_network.sh
   ```

3. Access the interface at http://127.0.0.1:7860 in your web browser

### Command-line Options

The installation script supports various options:

```bash
./install_deepseek.sh [options]

Options:
  -h, --help                  Show help message
  -m, --model MODEL_ID        Specify model ID (e.g., deepseek-ai/deepseek-v2)
  -l, --list                  List available recommended models
  -s, --small                 Use smaller models (for systems with less RAM)
  --no-auth                   Skip Hugging Face authentication
  --cleanup                   Remove temporary files and fix permissions
  --uninstall                 Remove the installation completely
```

The startup script also supports options:

```bash
./start_deepseek.sh [options]

Options:
  -p, --port PORT    Specify port (default: 7860)
  -m, --model DIR    Specify model directory name (default: auto-detect)
  -h, --help         Show help
```

## Supported Models

Models are downloaded from Hugging Face. Some recommended models:

### Large models (7B+ parameters, 32GB+ RAM recommended)
- deepseek-ai/deepseek-v2 (requires auth)
- deepseek-ai/deepseek-coder-33b-instruct (requires auth)
- mistralai/Mistral-7B-Instruct-v0.2

### Medium models (3-7B parameters, 16GB+ RAM recommended)
- deepseek-ai/deepseek-coder-6.7b-instruct
- microsoft/phi-3-mini-4k-instruct

### Small models (1-3B parameters, 4GB+ RAM)
- deepseek-ai/deepseek-coder-1.3b-instruct
- microsoft/phi-2

## System Requirements

- Ubuntu 20.04+ or Windows WSL2
- Python 3.8+
- 8GB RAM minimum (16GB+ recommended for medium models)
- NVIDIA GPU with 8GB VRAM (for GPU acceleration) or CPU-only mode

## Customization

You can customize the interface by editing:
- `style.css` - For UI appearance
- `deepseek_repl.py` - For functionality changes

## Project Structure

```
deepseek-local/
- install_deepseek.sh       # Installation script
- start_deepseek.sh         # Local startup script
- start_deepseek_network.sh # Network-accessible startup script
- deepseek_repl.py          # Main Python interface
- style.css                 # CSS styling
- requirements.txt          # Python dependencies
- models/                   # Downloaded models
�   - deepseek-coder-6.7b-instruct/  # Example model
- venv/                     # Python virtual environment
```

## Dependencies

The project requires several Python packages listed in `requirements.txt`. The main dependencies are:

- torch - For deep learning operations
- transformers - For loading and running the models
- gradio - For the web interface
- bitsandbytes - For model quantization
- markdown and pygments - For code formatting

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

## Privacy

All processing happens locally on your machine. Files uploaded to the interface and all queries are processed entirely on your local hardware and are never sent to external servers.

## Troubleshooting

### Port Already in Use
If port 7860 is already in use, specify a different port:
```bash
./start_deepseek.sh --port 7861
```

### Out of Memory Errors
If you encounter GPU memory errors, try a smaller model:
```bash
./install_deepseek.sh --model deepseek-ai/deepseek-coder-1.3b-instruct
```

### Permission Issues
If you encounter permission issues:
```bash
./install_deepseek.sh --cleanup
```

### Missing Dependencies
If you see errors about missing Python modules:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
