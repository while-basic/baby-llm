# 🧠 Baby LLM - Unified Neural Child Development System

<div align="center">

**A sophisticated AI system that simulates child cognitive development through interconnected neural networks**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

*An open-source research project by Celaya Solutions AI Research Lab*

</div>

---

## 🌟 Overview

The Unified Neural Child Development System is a comprehensive AI framework that simulates child cognitive development from newborn to mature stages. It combines neural networks, large language models (via Ollama), and developmental psychology to create a unique simulation of cognitive growth, emotional development, and learning.

This unified system consolidates five distinct iterations into a single, cohesive codebase featuring:

- **Developmental Stages**: Progressive cognitive development from newborn to mature adult
- **Neural Networks**: Specialized networks for consciousness, emotions, perception, and thoughts
- **Mother LLM**: Nurturing AI caregiver providing stage-appropriate guidance
- **Quantum Emotional Processing**: Quantum-inspired emotional state modeling
- **Memory Systems**: Multiple memory types with RAG integration
- **Physiological Systems**: Heartbeat simulation and dream generation
- **Obsidian Integration**: Knowledge graph and memory storage
- **Web Interface**: Flask-based dashboard for monitoring and interaction

## ✨ Key Features

### Core Systems

- **🧒 Developmental Stages**: 8 stages from NEWBORN to MATURE_ADULT
- **🧠 Neural Architecture**: Brain region simulation with specialized networks
- **💭 Memory Systems**: Episodic, semantic, emotional, and working memory
- **🎯 Decision Making**: Q-Learning and decision networks
- **📚 Language Development**: Vocabulary acquisition and language learning
- **👁️ Vision Systems**: Visual perception and processing
- **🔍 Metacognition**: Self-awareness and hypothesis processing
- **⚖️ Moral Network**: Ethical reasoning and decision-making

### Emotional & Psychological

- **😊 Emotional Regulation**: Real-time emotional state management
- **🌊 Quantum Emotional Processing**: Quantum-inspired emotional superposition
- **💓 Heartbeat System**: Physiological responses based on emotional state
- **💤 Dream System**: Dream generation and analysis
- **🔗 Attachment System**: Caregiver-child relationship modeling
- **🧠 Theory of Mind**: Understanding others' mental states
- **🛡️ Defense Mechanisms**: Psychological coping strategies

### Unique Features

- **📝 Obsidian Integration**: Automatic memory storage and knowledge graphs
- **🚌 Message Bus**: Publish-subscribe inter-network communication
- **🎓 Autonomous Learning**: Curiosity-driven self-directed learning
- **🛡️ Safety Monitor**: Harm detection and ethical constraints
- **📊 Visualization Tools**: Emotional state and network visualization

### Web Interface

- **🌐 Flask Web Application**: Lightweight web interface
- **📡 RESTful API**: 11+ endpoints for state, emotions, chat, visualization
- **📈 Real-time Monitoring**: Live state updates and metrics
- **💬 Chat Interface**: Interactive conversation with the neural child

## 🚀 Quick Start

### Prerequisites

- Python 3.11 (required)
- NVIDIA GPU with CUDA support (recommended)
- Ollama installed and running with `gemma3:1b` model

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/while-basic/baby-llm.git
cd baby-llm/neural-child-unified
```

2. **Create virtual environment:**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the package (in development mode):**
```bash
pip install -e .
```

This installs the package and all dependencies. Alternatively, you can install dependencies separately:
```bash
pip install -r requirements.txt
```

**Note**: The package must be installed for `python -m neural_child` to work. See `INSTALL.md` for detailed installation instructions.

4. **Configure Ollama:**
```bash
# Pull the model
ollama pull gemma3:1b

# Set GPU usage (Linux/Mac)
export OLLAMA_NUM_GPU=1

# Verify GPU usage
nvidia-smi  # Should show Ollama process
```

5. **Configure the system:**
```bash
# Edit config/config.yaml
# Ensure Ollama settings are correct:
# ollama:
#   model: "gemma3:1b"
#   base_url: "http://localhost:11434"
```

### Running the System

**Verify installation (optional):**
```bash
python verify_installation.py
```

**Start the web interface:**
```bash
python -m neural_child --web
```

Access the dashboard at: `http://localhost:5000`

**Note**: If you get `No module named 'neural_child'`, run `pip install -e .` first.

**Run with custom port:**
```bash
python -m neural_child --web --port 8080
```

**Run integration tests:**
```bash
python -m neural_child --test
```

**Run smoke tests:**
```bash
python -m neural_child --smoke
```

## 📖 Usage

### Command-Line Interface

The system provides a unified CLI through `python -m neural_child`:

```bash
# Start web interface
python -m neural_child --web [--port PORT] [--host HOST] [--debug]

# Run tests
python -m neural_child --test
python -m neural_child --smoke

# Show help
python -m neural_child --help
```

### Web API Endpoints

**State & Monitoring:**
- `GET /api/state` - Get current development state
- `GET /api/emotions` - Get emotional state
- `GET /api/memory` - Get memory status
- `GET /api/development/warnings` - Get warnings
- `GET /api/health` - Health check

**Interaction:**
- `POST /api/chat` - Send chat message
- `POST /api/emotions` - Update emotional state
- `POST /api/development/speed` - Update development speed

**Visualization:**
- `GET /api/visualization/data?type=emotional` - Get emotional visualization data
- `GET /api/neural/activity` - Get neural activity data
- `GET /api/neural/topology` - Get network topology

### Python API

```python
from neural_child.web.app import create_app
from neural_child.emotional.regulation import EmotionalRegulation
from neural_child.interaction.llm.llm_module import chat_completion

# Create Flask app
app = create_app()

# Use emotional regulation
emotion_system = EmotionalRegulation()

# Use LLM integration
response = chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    model="gemma3:1b"
)
```

## 🏗️ Architecture

### Directory Structure

```
neural-child-unified/
├── src/
│   └── neural_child/
│       ├── core/              # Core systems (brain, decision, development, training)
│       ├── cognitive/         # Cognitive systems (memory, language, vision, metacognition)
│       ├── emotional/          # Emotional systems (regulation, development, memory, embedding)
│       ├── interaction/       # Interaction systems (chat, LLM integration)
│       ├── psychological/     # Psychological components (attachment, theory of mind, defense)
│       ├── physiological/     # Physiological systems (heartbeat)
│       ├── dream/             # Dream system
│       ├── communication/     # Message bus
│       ├── learning/          # Autonomous learning
│       ├── safety/             # Safety monitor
│       ├── integration/       # External integrations (Obsidian)
│       ├── visualization/     # Visualization tools
│       ├── web/                # Flask web application
│       ├── models/             # Data models and schemas
│       └── utils/              # Utilities (logger, config, helpers)
├── config/
│   └── config.yaml            # Configuration file
├── tests/                      # Integration tests
├── scripts/                    # Utility scripts
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

### System Components

**Core Systems:**
- `core/brain/` - Integrated brain architecture
- `core/decision/` - Decision making and Q-Learning
- `core/development/` - Developmental stages and milestones
- `core/training/` - Training systems and meta-learning

**Cognitive Systems:**
- `cognitive/memory/` - Memory systems (RAG, episodic, semantic)
- `cognitive/language/` - Language development
- `cognitive/vision/` - Vision systems
- `cognitive/metacognition/` - Self-awareness networks
- `cognitive/moral/` - Moral reasoning network

**Emotional Systems:**
- `emotional/regulation.py` - Emotional regulation
- `emotional/development.py` - Emotional development
- `emotional/memory.py` - Emotional memory
- `emotional/embedding.py` - Emotional embedding with quantum processing

**Interaction Systems:**
- `interaction/chat/` - Chat systems (integrated, emotional, self-awareness)
- `interaction/llm/` - LLM integration (Ollama, Mother LLM)

## ⚙️ Configuration

Configuration is managed through `config/config.yaml`:

```yaml
# Ollama Configuration
ollama:
  model: "gemma3:1b"
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 512
  timeout: 30

# Neural Network Configuration
neural_network:
  device: "cuda"  # or "cpu"
  embedding_dim: 128
  learning_rate: 0.0001

# Development Configuration
development:
  starting_stage: "NEWBORN"
  development_speed: 1.0

# Memory Configuration
memory:
  persist_directory: "memories"
  collection_name: "neural_child_memories"
```

## 🧪 Testing

**Run all integration tests:**
```bash
python -m neural_child --test
```

**Run specific test file:**
```bash
pytest tests/test_integration.py -v
```

**Run smoke tests:**
```bash
python -m neural_child --smoke
```

## 📚 Documentation

**🚀 Start Here**: `START_HERE.md` - Your entry point to all documentation

**Essential Guides**:
- **`GETTING_STARTED.md`** - Quick 5-minute setup guide
- **`INSTALL.md`** - Detailed installation instructions
- **`TROUBLESHOOTING.md`** - Common issues and solutions
- **`README.md`** - Complete project documentation (this file)

**Status & Reports**:
- **`PROJECT_COMPLETE.md`** - Final completion summary
- **`STATUS.md`** - Current project status
- **`COMPLETION_REPORT.md`** - Detailed completion report

**Reference**:
- **`DOCUMENTATION_INDEX.md`** - Complete guide to all documentation files
- **API Documentation**: See `src/neural_child/web/app.py` for API endpoints
- **Configuration**: See `config/config.yaml` for configuration options
- **Architecture**: See Architecture section above and `neural-child-jan-2026/init.md`

## 🔧 Development

### Project Structure

The project follows a modular architecture with clear separation of concerns:

- **Core Systems**: Foundation (brain, decision, development, training)
- **Cognitive Systems**: Higher-level cognition (memory, language, vision, metacognition)
- **Emotional Systems**: Emotional processing and regulation
- **Interaction Systems**: Chat and LLM integration
- **Unique Features**: Specialized systems (heartbeat, dreams, Obsidian, etc.)
- **Web Interface**: Flask application and API

### Code Style

- **File Headers**: All Python files include Celaya Solutions header
- **Line Limit**: Files should be under 400 lines
- **Type Hints**: Use type hints throughout
- **Documentation**: Docstrings for all classes and functions
- **Imports**: Use optional imports for dependencies not yet extracted

### Adding New Features

1. Create feature in appropriate module directory
2. Add proper file header
3. Use optional imports for dependencies
4. Create `__init__.py` exports
5. Add integration tests
6. Update documentation

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Celaya Solutions AI Research Lab** - Original development
- **Ollama** - LLM integration
- **PyTorch** - Neural network framework
- **Flask** - Web framework

## 📞 Contact

- **Author**: Christopher Celaya
- **Email**: chris@chriscelaya.com
- **Website**: [Celaya Solutions](https://celayasolutions.com)
- **GitHub**: [while-basic](https://github.com/while-basic)

---

**Built with 🤍 by [Celaya Solutions](https://celayasolutions.com)**

