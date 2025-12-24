Getting Started
===============

Welcome to NeuralChild! This guide will help you get up and running quickly.

Prerequisites
-------------

Before installing NeuralChild, ensure you have:

* Python 3.8 or higher
* pip (Python package manager)
* Virtual environment (recommended)

System Requirements
~~~~~~~~~~~~~~~~~~~

**Minimum**:

* CPU: Dual-core processor
* RAM: 4 GB
* Storage: 500 MB free space

**Recommended**:

* CPU: Quad-core processor or better
* RAM: 8 GB or more
* Storage: 1 GB free space
* GPU: CUDA-compatible GPU (optional, for faster processing)

Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~

Install NeuralChild using pip:

.. code-block:: bash

   pip install neuralchild

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/celayasolutions/neuralchild.git
   cd neuralchild
   pip install -e ".[dev]"

With Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install with all optional features:

.. code-block:: bash

   pip install neuralchild[all]

Or install specific features:

.. code-block:: bash

   # Development tools
   pip install neuralchild[dev]

   # Visualization tools
   pip install neuralchild[viz]

Verify Installation
-------------------

Test that NeuralChild is installed correctly:

.. code-block:: bash

   neuralchild info

Or in Python:

.. code-block:: python

   import neuralchild
   print(neuralchild.__version__)

You should see version 1.0.0 (or higher).

Configuration
-------------

Create a default configuration file:

.. code-block:: bash

   neuralchild init

This creates ``config.yaml`` in your current directory. Edit this file to customize:

* Model parameters
* Learning rates
* Developmental settings
* Logging preferences
* LLM API settings

Quick Start
-----------

Run Your First Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   neuralchild run --steps 100

This runs a 100-step simulation using the default configuration.

Launch the Dashboard
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   neuralchild dashboard

Open your browser to http://localhost:8050 to see the interactive dashboard.

Try an Example
~~~~~~~~~~~~~~

.. code-block:: bash

   cd examples
   python basic_simulation.py

Next Steps
----------

* Read the :doc:`quickstart` for a detailed tutorial
* Explore :doc:`user_guide/index` for in-depth documentation
* Check out :doc:`examples/index` for more examples
* Learn about :doc:`user_guide/concepts` and architecture

Getting Help
------------

If you encounter issues:

1. Check the :doc:`troubleshooting` guide
2. Search `existing issues <https://github.com/celayasolutions/neuralchild/issues>`_
3. Ask in `Discussions <https://github.com/celayasolutions/neuralchild/discussions>`_
4. Email us at research@celayasolutions.com

Common Issues
~~~~~~~~~~~~~

**Import Errors**

If you see ``ModuleNotFoundError``, ensure NeuralChild is installed:

.. code-block:: bash

   pip install neuralchild

**Dependency Conflicts**

Use a virtual environment to avoid conflicts:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install neuralchild

**CUDA/GPU Issues**

NeuralChild works on CPU by default. For GPU support, install PyTorch with CUDA:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu118

See `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ for details.
