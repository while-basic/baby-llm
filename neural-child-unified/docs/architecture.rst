.. File:       docs/architecture.rst
.. Project:    Baby LLM - Unified Neural Child Development System
.. Created by: Celaya Solutions, 2025
.. Author:     Christopher Celaya <chris@chriscelaya.com>
.. Description: Architecture documentation
.. Version:    1.0.0
.. License:    MIT
.. Last Update: January 2025

Architecture
============

Directory Structure
-------------------

.. code-block:: text

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
   └── README.md                  # Project documentation

System Components
-----------------

Core Systems
~~~~~~~~~~~~

* ``core/brain/`` - Integrated brain architecture
* ``core/decision/`` - Decision making and Q-Learning
* ``core/development/`` - Developmental stages and milestones
* ``core/training/`` - Training systems and meta-learning

Cognitive Systems
~~~~~~~~~~~~~~~~~

* ``cognitive/memory/`` - Memory systems (RAG, episodic, semantic)
* ``cognitive/language/`` - Language development
* ``cognitive/vision/`` - Vision systems
* ``cognitive/metacognition/`` - Self-awareness networks
* ``cognitive/moral/`` - Moral reasoning network

Emotional Systems
~~~~~~~~~~~~~~~~~

* ``emotional/regulation.py`` - Emotional regulation
* ``emotional/development.py`` - Emotional development
* ``emotional/memory.py`` - Emotional memory
* ``emotional/embedding.py`` - Emotional embedding with quantum processing

Interaction Systems
~~~~~~~~~~~~~~~~~~~

* ``interaction/chat/`` - Chat systems (integrated, emotional, self-awareness)
* ``interaction/llm/`` - LLM integration (Ollama, Mother LLM)
