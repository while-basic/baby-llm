# 🔧 Troubleshooting Guide - Unified Neural Child Development System

**Common issues and solutions**

---

## ❌ Module Not Found Errors

### Error: `No module named 'neural_child'`

**Cause**: The package is not installed or Python can't find it.

**Solutions**:

1. **Install the package in development mode** (Recommended):
   ```bash
   cd neural-child-unified
   pip install -e .
   ```

2. **Verify installation**:
   ```bash
   python -c "import neural_child; print('OK')"
   ```

3. **Check if package is installed**:
   ```bash
   pip list | grep neural-child
   ```

4. **Reinstall if needed**:
   ```bash
   pip uninstall neural-child-unified
   pip install -e .
   ```

### Error: `ModuleNotFoundError` when running `python -m neural_child`

**Cause**: Package not installed or wrong Python environment.

**Solutions**:

1. **Ensure you're in the correct directory**:
   ```bash
   cd neural-child-unified
   ```

2. **Install the package**:
   ```bash
   pip install -e .
   ```

3. **Verify Python environment**:
   ```bash
   which python  # Check which Python is being used
   python --version  # Should be 3.11
   ```

4. **Check virtual environment**:
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```

---

## 🌐 Ollama Connection Issues

### Error: `Connection refused` or `Ollama not available`

**Cause**: Ollama server is not running or not accessible.

**Solutions**:

1. **Start Ollama server**:
   ```bash
   ollama serve
   ```

2. **Verify Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

3. **Check model is downloaded**:
   ```bash
   ollama list
   ```

4. **Download model if missing**:
   ```bash
   ollama pull gemma3:1b
   ```

5. **Check configuration**:
   ```yaml
   # config/config.yaml
   ollama:
     base_url: "http://localhost:11434"
     model: "gemma3:1b"
   ```

### Error: `Model not found`

**Cause**: The specified model is not downloaded.

**Solutions**:

1. **List available models**:
   ```bash
   ollama list
   ```

2. **Download the model**:
   ```bash
   ollama pull gemma3:1b
   ```

3. **Update config if using different model**:
   ```yaml
   ollama:
     model: "your-model-name"
   ```

---

## 🐍 Python Version Issues

### Error: `Python version not supported` or syntax errors

**Cause**: Wrong Python version (need 3.11).

**Solutions**:

1. **Check Python version**:
   ```bash
   python --version  # Should be 3.11.x
   ```

2. **Install Python 3.11** if needed:
   - Download from [python.org](https://www.python.org/downloads/)
   - Or use pyenv: `pyenv install 3.11.0`

3. **Use specific Python version**:
   ```bash
   python3.11 -m venv venv
   python3.11 -m pip install -e .
   ```

---

## 📦 Dependency Issues

### Error: `ImportError` for specific packages

**Cause**: Dependencies not installed or version conflicts.

**Solutions**:

1. **Install all dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

3. **Reinstall dependencies**:
   ```bash
   pip install --force-reinstall -r requirements.txt
   ```

4. **Check for conflicts**:
   ```bash
   pip check
   ```

### Error: `torch` or `pydantic` import errors

**Cause**: Core dependencies not installed.

**Solutions**:

1. **Install PyTorch** (if needed):
   ```bash
   pip install torch
   ```

2. **Install Pydantic**:
   ```bash
   pip install pydantic>=2.0.0
   ```

3. **Install all at once**:
   ```bash
   pip install -e .
   ```

---

## 🌐 Web Interface Issues

### Error: `Address already in use`

**Cause**: Port 5000 is already in use.

**Solutions**:

1. **Use a different port**:
   ```bash
   python -m neural_child --web --port 8080
   ```

2. **Find and kill process using port**:
   ```bash
   # Linux/Mac
   lsof -ti:5000 | xargs kill
   
   # Windows
   netstat -ano | findstr :5000
   taskkill /PID <PID> /F
   ```

### Error: `Flask app not found` or `app_factory not found`

**Cause**: Web module not properly imported.

**Solutions**:

1. **Verify package installation**:
   ```bash
   pip install -e .
   ```

2. **Check imports**:
   ```bash
   python -c "from neural_child.web.app import app_factory; print('OK')"
   ```

3. **Reinstall package**:
   ```bash
   pip install --force-reinstall -e .
   ```

---

## 🧪 Test Issues

### Error: Tests fail with import errors

**Cause**: Package not installed or path issues.

**Solutions**:

1. **Install package first**:
   ```bash
   pip install -e .
   ```

2. **Run tests from project root**:
   ```bash
   cd neural-child-unified
   python -m neural_child --test
   ```

3. **Run tests directly**:
   ```bash
   python tests/test_integration.py
   ```

---

## 💾 Memory/Storage Issues

### Error: ChromaDB errors or memory issues

**Cause**: ChromaDB not properly initialized or storage issues.

**Solutions**:

1. **Check ChromaDB installation**:
   ```bash
   pip install chromadb>=0.4.0
   ```

2. **Clear ChromaDB data** (if corrupted):
   ```bash
   rm -rf .chroma  # Linux/Mac
   rmdir /s .chroma  # Windows
   ```

3. **Check disk space**:
   ```bash
   df -h  # Linux/Mac
   ```

---

## 🔍 General Debugging

### Enable Debug Mode

```bash
# Web interface with debug
python -m neural_child --web --debug

# Python with verbose errors
python -v -m neural_child --web
```

### Check System Information

```bash
# Python version
python --version

# Installed packages
pip list

# Package location
python -c "import neural_child; print(neural_child.__file__)"

# Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Verify Installation

```bash
# Test imports
python -c "from neural_child.web.app import app_factory; print('✓ Web OK')"
python -c "from neural_child.emotional.regulation import EmotionalRegulation; print('✓ Emotional OK')"
python -c "from neural_child.dream.dream_system import DreamSystem; print('✓ Dream OK')"

# Test CLI
python -m neural_child --help
```

---

## 📞 Getting More Help

1. **Check Documentation**:
   - `README.md` - Main documentation
   - `INSTALL.md` - Installation guide
   - `GETTING_STARTED.md` - Quick start
   - `DOCUMENTATION_INDEX.md` - All documentation

2. **Review Error Messages**: Check the full error traceback for clues

3. **Check Logs**: Look for warning messages in console output

4. **Verify Prerequisites**: Ensure all prerequisites are installed

---

## ✅ Quick Verification Checklist

Before reporting issues, verify:

- [ ] Python 3.11 is installed and in PATH
- [ ] Package is installed: `pip list | grep neural-child`
- [ ] Virtual environment is activated (if using)
- [ ] Dependencies are installed: `pip install -r requirements.txt`
- [ ] Ollama is running: `ollama serve`
- [ ] Model is downloaded: `ollama list`
- [ ] Configuration file exists: `config/config.yaml`
- [ ] Can import module: `python -c "import neural_child"`

---

**Built with 🤍 by [Celaya Solutions](https://celayasolutions.com)**

