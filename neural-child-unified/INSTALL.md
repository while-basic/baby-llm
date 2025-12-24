# 📦 Installation Guide - Unified Neural Child Development System

**Complete installation instructions**

---

## 🚀 Quick Installation

### Option 1: Install in Development Mode (Recommended)

```bash
cd neural-child-unified
pip install -e .
```

This installs the package in "editable" mode, so changes to the code are immediately available.

### Option 2: Install as Regular Package

```bash
cd neural-child-unified
pip install .
```

### Option 3: Use Without Installation (Development)

If you don't want to install, ensure the `src` directory is in your Python path:

```bash
cd neural-child-unified
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# Or on Windows:
set PYTHONPATH=%PYTHONPATH%;%CD%\src
```

---

## 📋 Prerequisites

### Required
- **Python 3.11** (not higher)
- **pip** (Python package manager)

### Optional but Recommended
- **Ollama** - For LLM functionality
- **NVIDIA GPU** - For accelerated inference

---

## 🔧 Step-by-Step Installation

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Package

```bash
# Development mode (recommended)
pip install -e .

# Or regular installation
pip install .
```

### 4. Verify Installation

```bash
# Test import
python -c "import neural_child; print('Installation successful!')"

# Test CLI
python -m neural_child --help
```

---

## 🐛 Troubleshooting

### Module Not Found Error

**Error**: `No module named 'neural_child'`

**Solutions**:

1. **Install in development mode**:
   ```bash
   pip install -e .
   ```

2. **Check Python path**:
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```
   Ensure `src` directory is in the path.

3. **Verify installation**:
   ```bash
   pip list | grep neural-child
   ```

4. **Reinstall**:
   ```bash
   pip uninstall neural-child-unified
   pip install -e .
   ```

### Import Errors After Installation

**Error**: `ImportError` or `ModuleNotFoundError`

**Solutions**:

1. **Check package structure**:
   ```bash
   python -c "import neural_child; print(neural_child.__file__)"
   ```

2. **Verify src directory**:
   ```bash
   ls src/neural_child/__init__.py
   ```

3. **Reinstall dependencies**:
   ```bash
   pip install --force-reinstall -r requirements.txt
   ```

### Python Version Issues

**Error**: `Python version not supported`

**Solution**: Ensure Python 3.11 is installed:
```bash
python --version  # Should show Python 3.11.x
```

---

## ✅ Verification

After installation, verify everything works:

```bash
# Test imports
python -c "from neural_child.web.app import app_factory; print('✓ Web app OK')"
python -c "from neural_child.emotional.regulation import EmotionalRegulation; print('✓ Emotional systems OK')"
python -c "from neural_child.dream.dream_system import DreamSystem; print('✓ Dream system OK')"

# Test CLI
python -m neural_child --help

# Test web interface (if Ollama is running)
python -m neural_child --web
```

---

## 🔄 Updating Installation

### Update Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Reinstall Package

```bash
pip install --force-reinstall -e .
```

---

## 📝 Development Setup

For development, install with dev dependencies:

```bash
pip install -e ".[dev]"
```

This includes:
- pytest (testing)
- black (formatting)
- isort (import sorting)
- mypy (type checking)
- ruff (linting)

---

## 🌐 Ollama Setup

After installing the package, set up Ollama:

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai/)

2. **Start Ollama**:
   ```bash
   ollama serve
   ```

3. **Download Model**:
   ```bash
   ollama pull gemma3:1b
   ```

4. **Verify**:
   ```bash
   ollama list
   ```

---

## 🎯 Next Steps

After installation:

1. **Read**: `GETTING_STARTED.md` for quick start
2. **Configure**: Edit `config/config.yaml` if needed
3. **Run**: `python -m neural_child --web` to start
4. **Test**: `python -m neural_child --test` to verify

---

## 💡 Tips

- **Always use a virtual environment** to avoid conflicts
- **Install in development mode** (`-e`) for active development
- **Check Python version** before installation
- **Verify Ollama** is running before using LLM features

---

**Built with 🤍 by [Celaya Solutions](https://celayasolutions.com)**

