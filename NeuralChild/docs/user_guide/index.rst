User Guide
==========

Welcome to the NeuralChild User Guide. This comprehensive guide will help you understand,
configure, and use the NeuralChild framework to simulate artificial cognitive development.

What is NeuralChild?
--------------------

NeuralChild is a psychological brain simulation framework that models child cognitive development
through five distinct developmental stages: Infant, Toddler, Child, Adolescent, and Mature.
The system combines neural networks with a nurturing "Mother" LLM to create a unique simulation
of cognitive growth and learning.

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Key Components
--------------

The framework consists of several interconnected components:

**Mind System**
   The central orchestrator managing all neural networks, memory systems, and developmental
   progression.

**Neural Networks**
   Specialized networks for consciousness, emotions, perception, and thoughts that grow and
   adapt based on developmental stage.

**Mother LLM**
   An AI caregiver that provides stage-appropriate guidance, emotional attunement, and
   developmental scaffolding.

**Memory System**
   Short-term and long-term memory with automatic consolidation, clustering, and retrieval
   mechanisms.

**Communication Bus**
   Inter-network message passing using publish-subscribe pattern for coordinated behavior.

**Belief Network**
   Dynamic belief formation and updating based on experiences and evidence.

Who Should Use This Guide?
---------------------------

This guide is designed for:

- **Researchers** exploring computational models of cognitive development
- **AI Developers** building systems that learn and grow over time
- **Students** studying developmental psychology or artificial intelligence
- **Enthusiasts** interested in brain simulation and emergent behavior

Guide Structure
---------------

This user guide is organized into the following sections:

.. toctree::
   :maxdepth: 2

   concepts
   developmental_stages
   neural_networks
   mother_llm
   configuration
   cli
   dashboard

:doc:`concepts`
   Core concepts and architectural overview of the NeuralChild framework.

:doc:`developmental_stages`
   Detailed information about the five developmental stages and progression criteria.

:doc:`neural_networks`
   In-depth documentation of the specialized neural networks and their roles.

:doc:`mother_llm`
   Understanding the Mother LLM component and its nurturing interactions.

:doc:`configuration`
   Complete configuration guide for customizing your simulation.

:doc:`cli`
   Command-line interface reference and usage examples.

:doc:`dashboard`
   Interactive dashboard for visualizing and monitoring development.

Quick Start Example
-------------------

Here's a simple example to get you started:

.. code-block:: python

   from neuralchild import Mind, MotherLLM, Config
   from neuralchild.core import DevelopmentalStage

   # Load configuration
   config = Config.from_yaml("config.yaml")

   # Create the mind and mother
   mind = Mind(config=config)
   mother = MotherLLM()

   # Run simulation loop
   for step in range(100):
       # Mind processes one step
       observable_state = mind.step()

       # Mother observes and responds
       response = mother.observe_and_respond(observable_state)

       # Display progress
       print(f"Stage: {mind.current_stage.name}")
       print(f"Mood: {observable_state.apparent_mood}")
       print(f"Vocalization: {observable_state.vocalization}")

For more detailed examples, see the :doc:`../examples/index` section.

Getting Help
------------

If you need assistance:

- Check the :doc:`../api/index` for detailed API documentation
- Browse :doc:`../examples/index` for practical usage examples
- Visit the GitHub repository: https://github.com/celayasolutions/neuralchild
- Join discussions: https://github.com/celayasolutions/neuralchild/discussions
- Report issues: https://github.com/celayasolutions/neuralchild/issues

Next Steps
----------

- Read :doc:`concepts` to understand the architecture
- Explore :doc:`developmental_stages` to learn about cognitive progression
- Review :doc:`configuration` to customize your simulation
- Try the :doc:`cli` commands to run your first simulation

Contributing
------------

We welcome contributions! See the :doc:`../contributing` guide for details on how to:

- Report bugs and request features
- Submit pull requests
- Improve documentation
- Add new examples

License
-------

NeuralChild is licensed under the MIT License. See :doc:`../license` for details.
