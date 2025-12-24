.. File:       docs/development.rst
.. Project:    Baby LLM - Unified Neural Child Development System
.. Created by: Celaya Solutions, 2025
.. Author:     Christopher Celaya <chris@chriscelaya.com>
.. Description: Development documentation
.. Version:    1.0.0
.. License:    MIT
.. Last Update: January 2025

Development
==========

Project Structure
----------------

The project follows a modular architecture with clear separation of concerns:

* **Core Systems**: Foundation (brain, decision, development, training)
* **Cognitive Systems**: Higher-level cognition (memory, language, vision, metacognition)
* **Emotional Systems**: Emotional processing and regulation
* **Interaction Systems**: Chat and LLM integration
* **Unique Features**: Specialized systems (heartbeat, dreams, Obsidian, etc.)
* **Web Interface**: Flask application and API

Code Style
----------

* **File Headers**: All Python files include Celaya Solutions header
* **Line Limit**: Files should be under 400 lines
* **Type Hints**: Use type hints throughout
* **Documentation**: Docstrings for all classes and functions
* **Imports**: Use optional imports for dependencies not yet extracted

Adding New Features
-------------------

1. Create feature in appropriate module directory
2. Add proper file header
3. Use optional imports for dependencies
4. Create ``__init__.py`` exports
5. Add integration tests
6. Update documentation

Testing
-------

Run all integration tests:

.. code-block:: bash

   python -m neural_child --test

Run specific test file:

.. code-block:: bash

   pytest tests/test_integration.py -v

Run smoke tests:

.. code-block:: bash

   python -m neural_child --smoke
