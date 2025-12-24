# 🚀 Getting Started - Unified Neural Child Development System

**Quick guide to start using the system immediately**

---

## ⚡ Quick Start (5 Minutes)

### 1. Prerequisites

Ensure you have:
- **Python 3.11** installed
- **Ollama** installed and running
- **NVIDIA GPU** (optional but recommended)

### 2. Install Ollama Model

```bash
ollama pull gemma3:1b
```

### 3. Install Package and Dependencies

```bash
cd neural-child-unified

# Install in development mode (recommended)
pip install -e .

# Or install dependencies separately
pip install -r requirements.txt
```

### 4. Verify Installation (Optional)

```bash
python verify_installation.py
```

This will check that everything is set up correctly.

### 5. Start the System

```bash
python -m neural_child --web
```

Open your browser to: `http://127.0.0.1:5000`

---

## 📖 Common Tasks

### Start Web Interface

```bash
python -m neural_child --web
```

**Options:**
- `--port 8080` - Use different port
- `--host 0.0.0.0` - Allow external access
- `--debug` - Enable debug mode

### Run Tests

```bash
# Integration tests
python -m neural_child --test

# Or directly
python tests/test_integration.py
```

### Run Smoke Tests

```bash
python -m neural_child --smoke
```

---

## 🎯 What You Can Do

### 1. Monitor Development State

Visit the dashboard at `http://127.0.0.1:5000` to see:
- Current developmental stage
- Emotional state
- Neural activity
- System warnings

### 2. Chat with the Neural Child

Use the chat interface on the dashboard to interact with the AI system.

### 3. Use the API

```python
import requests

# Get current state
response = requests.get('http://127.0.0.1:5000/api/state')
state = response.json()

# Send a chat message
response = requests.post('http://127.0.0.1:5000/api/chat', 
    json={'message': 'Hello!'})
result = response.json()
```

### 4. Programmatic Access

```python
from neural_child.emotional.regulation import EmotionalRegulation
from neural_child.interaction.llm.llm_module import chat_completion

# Use emotional regulation
emotion_system = EmotionalRegulation()

# Use LLM integration
response = chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gemma3:1b"
)
```

---

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
ollama:
  model: "gemma3:1b"
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 500
```

---

## 🐛 Troubleshooting

### Ollama Not Found

**Error**: `Connection refused` or `Ollama not available`

**Solution**:
1. Ensure Ollama is running: `ollama serve`
2. Verify model is downloaded: `ollama list`
3. Check base URL in `config/config.yaml`

### Import Errors

**Error**: `ModuleNotFoundError`

**Solution**:
1. Ensure you're in the `neural-child-unified` directory
2. Install dependencies: `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.11)

### Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Use a different port
python -m neural_child --web --port 8080
```

---

## 📚 Next Steps

1. **Read the README**: `README.md` for comprehensive documentation
2. **Check Examples**: See `QUICK_START.md` for more examples
3. **Explore API**: See API endpoints in `README.md`
4. **Review Architecture**: See `ARCHITECTURE_PROPOSAL.md` (if available)

---

## 🎓 Learning Resources

- **Project Overview**: `README.md`
- **Quick Start**: `QUICK_START.md`
- **Contributing**: `CONTRIBUTING.md`
- **Changelog**: `CHANGELOG.md`
- **Project Summary**: `PROJECT_SUMMARY.md`

---

## 💡 Tips

1. **Start Simple**: Begin with the web interface to see the system in action
2. **Check Logs**: Monitor console output for warnings and errors
3. **Use Debug Mode**: Add `--debug` flag for detailed error messages
4. **Test First**: Run `python -m neural_child --test` to verify setup

---

## 🆘 Need Help?

- Check the documentation files in the project root
- Review error messages in the console
- Verify all prerequisites are installed
- Ensure Ollama is running and model is downloaded

---

**Built with 🤍 by [Celaya Solutions](https://celayasolutions.com)**

