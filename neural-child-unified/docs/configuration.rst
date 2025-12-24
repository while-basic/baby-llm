.. File:       docs/configuration.rst
.. Project:    Baby LLM - Unified Neural Child Development System
.. Created by: Celaya Solutions, 2025
.. Author:     Christopher Celaya <chris@chriscelaya.com>
.. Description: Configuration documentation
.. Version:    1.0.0
.. License:    MIT
.. Last Update: January 2025

Configuration
=============

Configuration is managed through ``config/config.yaml``:

Ollama Configuration
--------------------

.. code-block:: yaml

   ollama:
     model: "gemma3:1b"
     base_url: "http://localhost:11434"
     temperature: 0.7
     max_tokens: 512
     timeout: 30

Neural Network Configuration
----------------------------

.. code-block:: yaml

   neural_network:
     device: "cuda"  # or "cpu"
     embedding_dim: 128
     learning_rate: 0.0001

Development Configuration
--------------------------

.. code-block:: yaml

   development:
     starting_stage: "NEWBORN"
     development_speed: 1.0

Memory Configuration
--------------------

.. code-block:: yaml

   memory:
     persist_directory: "memories"
     collection_name: "neural_child_memories"
