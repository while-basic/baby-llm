# Contributing to Baby LLM - Unified Neural Child Development System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. **Fork and clone the repository:**
```bash
git clone https://github.com/while-basic/baby-llm.git
cd baby-llm/neural-child-unified
```

2. **Create virtual environment:**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install development dependencies:**
```bash
pip install pytest pytest-cov black isort mypy ruff
```

## Code Style

### File Headers

All Python files must include the standard Celaya Solutions header:

```python
#----------------------------------------------------------------------------
#File:       filename.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Brief description of what the file does
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------
```

### Code Standards

- **Line Limit**: Keep files under 400 lines
- **Type Hints**: Use type hints for all function parameters and return values
- **Docstrings**: Include docstrings for all classes and functions
- **Imports**: Use optional imports for dependencies that may not be available
- **Error Handling**: Use try/except blocks with appropriate error messages

### Formatting

```bash
# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
ruff check src/
```

## Project Structure

Follow the existing directory structure:

```
src/neural_child/
├── core/              # Core systems
├── cognitive/         # Cognitive systems
├── emotional/         # Emotional systems
├── interaction/       # Interaction systems
├── psychological/     # Psychological components
├── physiological/     # Physiological systems
├── dream/             # Dream system
├── communication/     # Message bus
├── learning/          # Autonomous learning
├── safety/            # Safety monitor
├── integration/       # External integrations
├── visualization/     # Visualization tools
├── web/               # Flask web application
├── models/            # Data models
└── utils/             # Utilities
```

## Adding New Features

1. **Create feature in appropriate directory**
2. **Add file header**
3. **Use optional imports for dependencies**
4. **Create `__init__.py` exports**
5. **Add integration tests**
6. **Update documentation**

## Testing

### Running Tests

```bash
# Run all tests
python -m neural_child --test

# Run specific test file
pytest tests/test_integration.py -v

# Run with coverage
pytest tests/ --cov=src/neural_child --cov-report=html
```

### Writing Tests

- Create test files in `tests/` directory
- Use descriptive test function names
- Test both success and failure cases
- Include docstrings for test functions

## Pull Request Process

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes:**
   - Follow code style guidelines
   - Add tests for new features
   - Update documentation

3. **Run tests:**
```bash
python -m neural_child --test
```

4. **Commit changes:**
```bash
git add .
git commit -m "Add: Description of changes"
```

5. **Push and create pull request:**
```bash
git push origin feature/your-feature-name
```

## Commit Messages

Use clear, descriptive commit messages:

- `Add: Feature description`
- `Fix: Bug description`
- `Update: Change description`
- `Remove: Removal description`
- `Docs: Documentation update`

## Questions?

- **Email**: chris@chriscelaya.com
- **GitHub Issues**: Create an issue for questions or bugs

---

**Built with 🤍 by [Celaya Solutions](https://celayasolutions.com)**

