# 🧠 NeuralChild

<div align="center">

**A psychological brain simulation framework modeling child cognitive development**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*An open-source research project by Celaya Solutions AI Research Lab*

[Documentation](https://neuralchild.readthedocs.io) | [Examples](./examples) | [Contributing](./CONTRIBUTING.md) | [Changelog](./CHANGELOG.md)

</div>

---

## 🌟 Overview

NeuralChild simulates an artificial mind that learns and grows through developmental stages (Infant → Toddler → Child → Adolescent → Mature), guided by a "Mother" LLM that provides nurturing interactions and developmental guidance. This framework combines neural networks, large language models, and developmental psychology to create a unique simulation of cognitive growth.

### Key Features

- **🧒 Developmental Stages**: Progressive cognitive development from infant to mature stages
- **🧠 Neural Networks**: Specialized networks for consciousness, emotions, perception, and thoughts
- **👩 Mother LLM**: Nurturing AI caregiver providing stage-appropriate guidance
- **💭 Memory System**: Short-term and long-term memory with consolidation and clustering
- **🔄 Message Bus**: Inter-network communication using pub-sub pattern
- **📊 Interactive Dashboard**: Real-time visualization of development and neural states
- **🎯 Belief Formation**: Dynamic belief network with evidence tracking
- **📈 Metrics & Monitoring**: Comprehensive tracking of development milestones

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/celayasolutions/neuralchild.git
cd neuralchild

# Install dependencies
pip install -e .

# For development with all extras
pip install -e ".[dev,viz]"
```

### Basic Usage

```python
from neuralchild import Mind, MotherLLM, Config
from neuralchild.core import DevelopmentalStage

# Load configuration
config = Config.from_yaml("config.yaml")

# Create the mind
mind = Mind(config=config)

# Create the mother LLM
mother = MotherLLM()

# Run simulation
for step in range(100):
    # Process a simulation step
    observable_state = mind.step()

    # Mother observes and responds
    response = mother.observe_and_respond(observable_state)

    # Feed response back to mind
    mind.receive_maternal_input(response)

    print(f"Stage: {mind.current_stage.name}, Step: {step}")
```

### Run Interactive Dashboard

```bash
# Start the dashboard
neuralchild dashboard

# Or use Python directly
python -m neuralchild.dashboard
```

Then open your browser to `http://localhost:8050`

## 📖 Core Concepts

### Developmental Stages

The system progresses through five developmental stages:

1. **Infant** (0-12 months): Basic sensory processing, simple emotions, reflexive responses
2. **Toddler** (1-3 years): Language acquisition begins, basic self-awareness, exploration
3. **Child** (3-12 years): Complex reasoning, social understanding, belief formation
4. **Adolescent** (12-18 years): Abstract thinking, identity formation, moral reasoning
5. **Mature** (18+ years): Advanced cognition, wisdom, emotional regulation

### Neural Networks

- **ConsciousnessNetwork**: Integrates information into unified awareness and self-model
- **EmotionsNetwork**: Processes and generates emotional states with quantum-inspired dynamics
- **PerceptionNetwork**: Handles visual and auditory input processing
- **ThoughtsNetwork**: Generates cognitive processes and decision-making

### Mother LLM

The Mother LLM provides:
- Stage-appropriate responses and guidance
- Emotional attunement and support
- Developmental scaffolding
- Language model and social interaction

## 🏗️ Architecture

```
NeuralChild/
├── neuralchild/              # Main package
│   ├── core/                 # Core schemas and base classes
│   │   ├── schemas.py        # Data models (Memory, Belief, NetworkMessage)
│   │   └── neural_network.py # Base neural network class
│   ├── mind/                 # Mind simulation
│   │   ├── mind_core.py      # Main Mind orchestrator
│   │   ├── schemas.py        # Mind-specific schemas
│   │   └── networks/         # Neural network implementations
│   │       ├── consciousness.py
│   │       ├── emotions.py
│   │       ├── perception.py
│   │       └── thoughts.py
│   ├── mother/               # Mother LLM component
│   │   └── mother_llm.py     # Nurturing AI caregiver
│   ├── communication/        # Message bus system
│   │   └── message_bus.py    # Pub-sub communication
│   ├── visualization/        # Dashboard components
│   │   └── dashboard.py      # Dash-based UI
│   ├── utils/                # Utility functions
│   │   └── llm_module.py     # LLM integration utilities
│   ├── config.py             # Configuration management
│   └── cli.py                # Command-line interface
├── docs/                     # Documentation
├── examples/                 # Usage examples
├── tests/                    # Test suite
└── models/                   # Saved model checkpoints
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=neuralchild --cov-report=html

# Run specific test file
pytest neuralchild/tests/test_mind.py
```

### Code Quality

```bash
# Format code
black neuralchild/
isort neuralchild/

# Lint
flake8 neuralchild/
mypy neuralchild/
```

## 📚 Documentation

Comprehensive documentation is available at [neuralchild.readthedocs.io](https://neuralchild.readthedocs.io)

Topics covered:
- **Getting Started**: Installation and basic usage
- **Core Concepts**: Developmental stages, neural networks, memory systems
- **API Reference**: Detailed API documentation
- **Examples**: Tutorials and use cases
- **Contributing**: Development guidelines

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Ways to Contribute

- 🐛 Report bugs and issues
- 💡 Suggest new features or improvements
- 📝 Improve documentation
- 🧪 Add tests and increase coverage
- 🔧 Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

NeuralChild is inspired by research in:
- Developmental psychology and cognitive science
- Neural network architectures and deep learning
- Large language models and natural language processing
- Computational models of consciousness

## 📧 Contact

**Celaya Solutions AI Research Lab**
- Website: [celayasolutions.com](https://celayasolutions.com)
- Email: research@celayasolutions.com
- GitHub: [@celayasolutions](https://github.com/celayasolutions)

## 🗺️ Roadmap

- [ ] Enhanced memory consolidation with sleep cycles
- [ ] Multi-modal sensory processing (visual, auditory, tactile)
- [ ] Social interaction with multiple agents
- [ ] Transfer learning from pre-trained models
- [ ] Mobile device support and edge deployment
- [ ] Integration with robotics platforms

---

<div align="center">

**Made with ❤️ by Celaya Solutions AI Research Lab**

*First open-source contribution from our AI research team*

</div>
