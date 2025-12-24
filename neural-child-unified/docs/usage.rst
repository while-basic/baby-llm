.. File:       docs/usage.rst
.. Project:    Baby LLM - Unified Neural Child Development System
.. Created by: Celaya Solutions, 2025
.. Author:     Christopher Celaya <chris@chriscelaya.com>
.. Description: Usage documentation
.. Version:    1.0.0
.. License:    MIT
.. Last Update: January 2025

Usage Guide
===========

Command-Line Interface
-----------------------

The system provides a unified CLI through ``python -m neural_child``:

.. code-block:: bash

   # Start web interface
   python -m neural_child --web [--port PORT] [--host HOST] [--debug]

   # Run tests
   python -m neural_child --test
   python -m neural_child --smoke

   # Show help
   python -m neural_child --help

Web API Endpoints
-----------------

State & Monitoring
~~~~~~~~~~~~~~~~~~

* ``GET /api/state`` - Get current development state
* ``GET /api/emotions`` - Get emotional state
* ``GET /api/memory`` - Get memory status
* ``GET /api/development/warnings`` - Get warnings
* ``GET /api/health`` - Health check

Interaction
~~~~~~~~~~~

* ``POST /api/chat`` - Send chat message
* ``POST /api/emotions`` - Update emotional state
* ``POST /api/development/speed`` - Update development speed

Visualization
~~~~~~~~~~~~~

* ``GET /api/visualization/data?type=emotional`` - Get emotional visualization data
* ``GET /api/neural/activity`` - Get neural activity data
* ``GET /api/neural/topology`` - Get network topology

Python API
----------

.. code-block:: python

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
