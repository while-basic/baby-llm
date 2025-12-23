# Contributing to NeuralChild

Thank you for your interest in contributing to NeuralChild! This document provides guidelines and instructions for contributing to the project.

## 🌟 How to Contribute

We welcome contributions in many forms:
- 🐛 Bug reports and fixes
- 💡 Feature requests and implementations
- 📝 Documentation improvements
- 🧪 Test coverage expansion
- 🎨 UI/UX enhancements
- 🔬 Research and experimentation

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/neuralchild.git
cd neuralchild

# Add upstream remote
git remote add upstream https://github.com/celayasolutions/neuralchild.git
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
python -c "from neuralchild import Mind, MotherLLM; print('✓ Setup complete')"
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bugfix branch
git checkout -b fix/issue-description
```

## 📋 Development Workflow

### Code Style

We use the following tools to maintain code quality:

#### 1. Black (Code Formatting)
```bash
# Format all code
black neuralchild/

# Check without modifying
black --check neuralchild/
```

**Configuration**: Line length 100, Python 3.8+ target (see `pyproject.toml`)

#### 2. isort (Import Sorting)
```bash
# Sort imports
isort neuralchild/

# Check without modifying
isort --check neuralchild/
```

#### 3. flake8 (Linting)
```bash
# Lint code
flake8 neuralchild/
```

#### 4. mypy (Type Checking)
```bash
# Check types
mypy neuralchild/
```

### Running Tests

```bash
# Run all tests
pytest neuralchild/tests/ -v

# Run specific test file
pytest neuralchild/tests/test_mind.py -v

# Run with coverage
pytest neuralchild/tests/ --cov=neuralchild --cov-report=html

# View coverage report
open htmlcov/index.html
```

**Coverage requirement**: Maintain >70% coverage for all new code

### Pre-commit Checklist

Before committing, ensure:
- [ ] Code is formatted with Black
- [ ] Imports are sorted with isort
- [ ] No linting errors from flake8
- [ ] Type hints pass mypy checks (if applicable)
- [ ] All tests pass
- [ ] New code has tests
- [ ] Documentation is updated

### Commit Message Format

Follow conventional commits:

```
type(scope): brief description

Longer description if needed, explaining:
- What changed
- Why it changed
- Any breaking changes or migrations needed

Closes #123  # If fixing an issue
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build process, dependencies, etc.

**Examples**:
```
feat(networks): add attention mechanism to perception network
fix(mind): correct memory consolidation timing
docs(readme): add installation troubleshooting section
test(mother): add tests for stage transition responses
```

## 🧪 Testing Guidelines

### Writing Tests

1. **Use pytest fixtures** from `conftest.py`
2. **Test both success and error cases**
3. **Mock external dependencies** (LLM APIs, file I/O)
4. **Use descriptive test names**: `test_<what>_<condition>_<expected_outcome>`

Example:
```python
def test_memory_access_increases_strength(sample_memory):
    """Test that accessing a memory increases its strength."""
    initial_strength = sample_memory.strength
    sample_memory.access()
    assert sample_memory.strength > initial_strength
```

### Test Organization

- Unit tests: Test individual components in isolation
- Integration tests: Test component interactions
- End-to-end tests: Test complete workflows

## 📝 Documentation Guidelines

### Code Documentation

#### Docstrings
Use Google-style docstrings:

```python
def process_input(self, stimulus: torch.Tensor, source: str) -> Dict[str, Any]:
    """Process sensory input through the perception network.

    Args:
        stimulus: Input tensor representing sensory data
        source: Source of the stimulus ('visual', 'auditory', etc.)

    Returns:
        Dictionary containing processed perception data with keys:
        - 'attention_level': Current attention focus (0-1)
        - 'recognized_patterns': List of recognized patterns
        - 'novelty_score': How novel the input is (0-1)

    Raises:
        ValueError: If stimulus has invalid dimensions

    Example:
        >>> network = PerceptionNetwork()
        >>> stimulus = torch.randn(1, 128)
        >>> result = network.process_input(stimulus, 'visual')
    """
```

#### Type Hints
Always include type hints:

```python
from typing import List, Dict, Optional

def consolidate_memories(
    self,
    memories: List[Memory],
    threshold: float = 0.5
) -> Optional[Dict[str, Any]]:
    ...
```

### README and Documentation

- Update `README.md` for user-facing changes
- Add docstrings to all public APIs
- Include examples for new features
- Update `CHANGELOG.md` for version releases

## 🔬 Adding New Features

### New Neural Network

1. Create file in `neuralchild/mind/networks/`
2. Inherit from `NeuralNetwork` base class
3. Implement required methods: `forward()`, `process_message()`, `generate_text_output()`
4. Add developmental stage logic
5. Register in `Mind` class
6. Add tests
7. Document in README

### New Developmental Stage

1. Add to `DevelopmentalStage` enum in `core/schemas.py`
2. Update advancement thresholds in `Mind.__init__()`
3. Add stage-specific behaviors to networks
4. Update Mother LLM techniques
5. Add tests for new stage
6. Document progression requirements

## 🐛 Reporting Bugs

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Create Mind with config '...'
2. Run step() with input '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment:**
 - OS: [e.g. macOS 13.0]
 - Python version: [e.g. 3.10.5]
 - NeuralChild version: [e.g. 1.0.0]

**Additional context**
Stack trace, logs, screenshots, etc.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature description**
Clear description of the proposed feature.

**Motivation**
Why is this feature valuable? What problem does it solve?

**Proposed implementation**
How might this feature be implemented? (optional)

**Alternatives considered**
What other approaches did you consider?

**Additional context**
Examples, mockups, related research, etc.
```

## 🔄 Pull Request Process

### Before Submitting

1. **Rebase on latest upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run full test suite**: `pytest`

3. **Check code quality**: `black`, `isort`, `flake8`

4. **Update documentation** if needed

### PR Template

Your PR will be reviewed based on:
- [ ] Code quality and style
- [ ] Test coverage
- [ ] Documentation
- [ ] No breaking changes (or migration guide provided)
- [ ] Passes CI checks

### Review Process

1. **Automated checks** run (tests, linting, coverage)
2. **Maintainer review** provides feedback
3. **Address feedback** with additional commits
4. **Approval** once all checks pass and feedback addressed
5. **Merge** by maintainer

## 🎯 Areas Needing Contribution

Current priority areas:

### High Priority
- [ ] Multi-modal sensory processing (tactile, proprioceptive)
- [ ] Advanced memory consolidation (sleep cycles, dreaming)
- [ ] Social interaction between multiple agents
- [ ] Performance optimization (GPU acceleration, batching)

### Medium Priority
- [ ] Additional neural networks (language, motor control)
- [ ] Enhanced dashboard visualizations
- [ ] More developmental milestones and behaviors
- [ ] Integration with robotics platforms

### Good First Issues
- [ ] Add more example scripts
- [ ] Improve error messages
- [ ] Add docstrings to undocumented functions
- [ ] Write tutorials for specific features

## 📞 Getting Help

- **Documentation**: https://neuralchild.readthedocs.io
- **Issues**: https://github.com/celayasolutions/neuralchild/issues
- **Discussions**: https://github.com/celayasolutions/neuralchild/discussions
- **Email**: research@celayasolutions.com

## 📜 Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior**:
- Using welcoming and inclusive language
- Respecting differing viewpoints
- Accepting constructive criticism gracefully
- Focusing on what's best for the community
- Showing empathy towards others

**Unacceptable behavior**:
- Harassment, trolling, or discriminatory comments
- Personal or political attacks
- Publishing others' private information
- Other conduct inappropriate for a professional setting

### Enforcement

Violations may result in temporary or permanent ban from the project. Report incidents to research@celayasolutions.com.

## 🏆 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Credited in academic papers if applicable

Thank you for contributing to NeuralChild! 🧠✨

---

*Celaya Solutions AI Research Lab*
*Licensed under the MIT License*
