# NeuralChild Examples

This directory contains example scripts demonstrating various features of the NeuralChild framework.

## Prerequisites

```bash
# Install NeuralChild
pip install -e ..

# Or with all dependencies
pip install -e "..[all]"
```

## Examples

### 1. Basic Simulation (`basic_simulation.py`)

**What it demonstrates:**
- Creating a Mind instance
- Creating a Mother LLM
- Running a simple simulation loop
- Observing developmental progress
- Saving simulation state

**Run it:**
```bash
python basic_simulation.py
```

**Expected output:**
- 100 simulation steps
- Periodic status updates
- Developmental milestones (if reached)
- Final statistics and saved state

**Key concepts:**
- Mind initialization and configuration
- Simulation step loop
- Observable state monitoring
- Mother-child interaction basics

---

### 2. Mother-Child Interaction (`mother_child_interaction.py`)

**What it demonstrates:**
- Detailed interaction patterns
- Mother's adaptive responses to child's development
- Stage-appropriate communication
- Developmental milestone celebration
- Rich state visualization

**Run it:**
```bash
python mother_child_interaction.py
```

**Expected output:**
- Detailed interaction logs every 20 steps
- Child's internal state display
- Mother's understanding and responses
- Celebration of developmental advances
- Comprehensive final summary

**Key concepts:**
- Observable state interpretation
- Mother's response generation
- Developmental stage progression
- Belief and memory formation
- Emotional state tracking

---

## Configuration

All examples use `config.yaml` from the parent directory. You can customize:

```yaml
mind:
  development_acceleration: 2.0  # Speed up development for demos
  starting_stage: "INFANT"       # Starting point

development:
  simulate_llm: true              # Use simulated mode (no API key needed)
  debug_mode: false               # Enable for detailed logs
```

## Common Patterns

### Creating a Mind

```python
from neuralchild import Mind, load_config

config = load_config("config.yaml")
mind = Mind(config=config)
```

### Running Simulation Steps

```python
for step in range(100):
    observable_state = mind.step()
    # Use observable_state for monitoring
```

### Mother-Child Interaction

```python
from neuralchild import MotherLLM

mother = MotherLLM()
response = mother.observe_and_respond(observable_state)
print(response.response)  # Mother's nurturing message
```

### Saving and Loading State

```python
# Save
mind.save_state("models/my_simulation.pt")

# Load (future simulation)
mind = Mind(config=config)
mind.load_state("models/my_simulation.pt")
```

## Advanced Examples (Coming Soon)

- **`custom_network.py`** - Adding custom neural networks
- **`developmental_tracking.py`** - Detailed milestone analysis
- **`memory_exploration.py`** - Memory clustering and consolidation
- **`belief_reasoning.py`** - Belief formation and evidence tracking
- **`multi_agent.py`** - Multiple minds interacting (planned for v1.1)

## Tips

### Simulated vs. Real LLM

**Simulated Mode** (no API key needed):
```python
config.development.simulate_llm = True
```

**Real LLM** (requires OpenAI API key):
```python
config.development.simulate_llm = False
config.model.llm_model = "gpt-3.5-turbo"
# Set OPENAI_API_KEY environment variable
```

### Accelerating Development

For demos, speed up development:
```python
config.mind.development_acceleration = 5.0  # 5x faster
```

### Debug Mode

Enable detailed logging:
```python
config.development.debug_mode = True
config.logging.level = "DEBUG"
```

## Troubleshooting

### Import Errors

```bash
# Make sure NeuralChild is installed
pip install -e ..

# Verify
python -c "from neuralchild import Mind; print('✓ Imports work')"
```

### Missing Dependencies

```bash
# Install all optional dependencies
pip install -e "..[all]"
```

### Config File Not Found

```bash
# Create default config
cd ..
python -m neuralchild init
```

## Learn More

- **Documentation**: https://neuralchild.readthedocs.io
- **API Reference**: See docstrings in source code
- **Contributing**: See `../CONTRIBUTING.md`
- **Issues**: https://github.com/celayasolutions/neuralchild/issues

---

*Happy experimenting! 🧠✨*

*Celaya Solutions AI Research Lab*
