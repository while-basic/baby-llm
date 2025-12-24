# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites Check

1. **Python 3.11** installed
2. **NVIDIA GPU** with CUDA (recommended)
3. **Ollama** installed and running

### Step 1: Install Dependencies

```bash
cd neural-child-unified
pip install -r requirements.txt
```

### Step 2: Set Up Ollama

```bash
# Pull the model
ollama pull gemma3:1b

# Set GPU usage (Linux/Mac)
export OLLAMA_NUM_GPU=1

# On Windows, set environment variable:
# OLLAMA_NUM_GPU=1
```

### Step 3: Start the Web Interface

```bash
python -m neural_child --web
```

### Step 4: Access the Dashboard

Open your browser to: `http://localhost:5000`

## 📋 Common Commands

```bash
# Start web interface
python -m neural_child --web

# Start on custom port
python -m neural_child --web --port 8080

# Run tests
python -m neural_child --test

# Run smoke tests
python -m neural_child --smoke

# Show help
python -m neural_child --help
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- Ollama model and settings
- Neural network parameters
- Development stage
- Memory settings

## 🐛 Troubleshooting

### Python Not Found
- Ensure Python 3.11 is installed
- Use `python` instead of `python3` on Windows

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Check model is available: `ollama list`
- Verify base_url in config.yaml

### Import Errors
- Install all dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.11)

### GPU Not Detected
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Set OLLAMA_NUM_GPU=1 environment variable

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for project overview
- Review [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

---

**Built with 🤍 by [Celaya Solutions](https://celayasolutions.com)**

