NeuralChild Documentation
========================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg
   :target: https://pytorch.org/
   :alt: PyTorch 2.0+

**A psychological brain simulation framework modeling child cognitive development**

NeuralChild simulates an artificial mind that learns and grows through developmental stages
(Infant → Toddler → Child → Adolescent → Mature), guided by a "Mother" LLM that provides
nurturing interactions and developmental guidance.

Quick Links
-----------

* :doc:`getting_started` - Installation and first steps
* :doc:`user_guide/index` - Comprehensive user guide
* :doc:`api/index` - API reference
* :doc:`examples/index` - Code examples
* :doc:`contributing` - Contributing guidelines

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/concepts
   user_guide/developmental_stages
   user_guide/neural_networks
   user_guide/mother_llm
   user_guide/configuration
   user_guide/cli
   user_guide/dashboard

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/core
   api/mind
   api/mother
   api/communication
   api/utils
   api/config

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index
   examples/basic_simulation
   examples/mother_child_interaction
   examples/custom_network
   examples/developmental_tracking

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Features
--------

🧒 **Developmental Stages**
   Progressive cognitive development from infant to mature stages

🧠 **Neural Networks**
   Specialized networks for consciousness, emotions, perception, and thoughts

👩 **Mother LLM**
   Nurturing AI caregiver providing stage-appropriate guidance

💭 **Memory System**
   Short-term and long-term memory with consolidation and clustering

🔄 **Message Bus**
   Inter-network communication using pub-sub pattern

📊 **Interactive Dashboard**
   Real-time visualization of development and neural states

🎯 **Belief Formation**
   Dynamic belief network with evidence tracking

📈 **Metrics & Monitoring**
   Comprehensive tracking of development milestones

Installation
------------

.. code-block:: bash

   pip install neuralchild

Or from source:

.. code-block:: bash

   git clone https://github.com/celayasolutions/neuralchild.git
   cd neuralchild
   pip install -e .

Quick Example
-------------

.. code-block:: python

   from neuralchild import Mind, MotherLLM, Config

   # Load configuration
   config = Config.from_yaml("config.yaml")

   # Create the mind
   mind = Mind(config=config)

   # Create the mother LLM
   mother = MotherLLM()

   # Run simulation
   for step in range(100):
       observable_state = mind.step()
       response = mother.observe_and_respond(observable_state)
       print(f"Stage: {mind.current_stage.name}, Step: {step}")

Community
---------

* **GitHub**: https://github.com/celayasolutions/neuralchild
* **Issues**: https://github.com/celayasolutions/neuralchild/issues
* **Discussions**: https://github.com/celayasolutions/neuralchild/discussions
* **Email**: research@celayasolutions.com

License
-------

NeuralChild is licensed under the MIT License. See :doc:`license` for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
