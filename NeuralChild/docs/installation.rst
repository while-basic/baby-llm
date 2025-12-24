Installation Guide
==================

This guide covers various installation methods for NeuralChild.

Standard Installation
---------------------

Using pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install NeuralChild:

.. code-block:: bash

   pip install neuralchild

This installs the core package with all required dependencies.

From Source
~~~~~~~~~~~

For the latest development version:

.. code-block:: bash

   git clone https://github.com/celayasolutions/neuralchild.git
   cd neuralchild
   pip install -e .

The ``-e`` flag installs in editable mode, useful for development.

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

NeuralChild requires:

* **Python**: 3.8, 3.9, 3.10, or 3.11
* **PyTorch**: 2.0.0 or higher
* **Pydantic**: 2.0.0 or higher
* **PyYAML**: 6.0 or higher
* **NumPy**: 1.20.0 or higher

These are installed automatically with the package.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For additional features:

**Dashboard and Visualization**:

.. code-block:: bash

   pip install dash>=2.9.0 dash-bootstrap-components>=1.4.0 plotly>=5.14.0

Or:

.. code-block:: bash

   pip install neuralchild[viz]

**Development Tools**:

.. code-block:: bash

   pip install pytest black isort mypy flake8

Or:

.. code-block:: bash

   pip install neuralchild[dev]

**All Features**:

.. code-block:: bash

   pip install neuralchild[all]

Virtual Environment Setup
--------------------------

Using venv (Python Built-in)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create virtual environment
   python -m venv neuralchild-env

   # Activate (Linux/macOS)
   source neuralchild-env/bin/activate

   # Activate (Windows)
   neuralchild-env\\Scripts\\activate

   # Install NeuralChild
   pip install neuralchild

Using conda
~~~~~~~~~~~

.. code-block:: bash

   # Create conda environment
   conda create -n neuralchild python=3.10

   # Activate
   conda activate neuralchild

   # Install NeuralChild
   pip install neuralchild

GPU Support
-----------

NeuralChild works on CPU by default. For GPU acceleration:

CUDA (NVIDIA GPUs)
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install PyTorch with CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118

   # Or CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121

Verify GPU support:

.. code-block:: python

   import torch
   print(torch.cuda.is_available())  # Should print True

Platform-Specific Instructions
-------------------------------

Linux
~~~~~

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install python3-pip python3-venv

   # Install NeuralChild
   pip3 install neuralchild

macOS
~~~~~

.. code-block:: bash

   # Using Homebrew
   brew install python3

   # Install NeuralChild
   pip3 install neuralchild

Windows
~~~~~~~

.. code-block:: powershell

   # Download Python from python.org
   # Then in PowerShell or Command Prompt:
   pip install neuralchild

Docker Installation
-------------------

Using the official Docker image (coming soon):

.. code-block:: bash

   docker pull celayasolutions/neuralchild:latest
   docker run -it celayasolutions/neuralchild:latest

Build from Dockerfile:

.. code-block:: bash

   git clone https://github.com/celayasolutions/neuralchild.git
   cd neuralchild
   docker build -t neuralchild .
   docker run -it neuralchild

Verification
------------

Verify Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   neuralchild info

Expected output:

.. code-block:: text

   ============================================================
   🧠 NeuralChild - AI Brain Simulation Framework
   ============================================================

   Version: 1.0.0
   Organization: Celaya Solutions AI Research Lab
   License: MIT

   📦 Package Information:
      Name: neuralchild
      Version: 1.0.0

   🔧 Dependencies:
      ✅ PyTorch: 2.0.0+
      ✅ Pydantic: 2.0.0+
      ...

Run Test Suite
~~~~~~~~~~~~~~

.. code-block:: bash

   pytest neuralchild/tests/ -v

All tests should pass.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ModuleNotFoundError: No module named 'neuralchild'**

Solution: Ensure NeuralChild is installed in your current environment:

.. code-block:: bash

   pip install neuralchild

**ImportError: cannot import name 'X'**

Solution: Update to the latest version:

.. code-block:: bash

   pip install --upgrade neuralchild

**CUDA errors**

Solution: Ensure compatible PyTorch CUDA version is installed. See :ref:`GPU Support`.

**Permission denied**

Solution: Use ``--user`` flag or virtual environment:

.. code-block:: bash

   pip install --user neuralchild

Getting Help
------------

If you continue experiencing issues:

* Check `GitHub Issues <https://github.com/celayasolutions/neuralchild/issues>`_
* Ask in `Discussions <https://github.com/celayasolutions/neuralchild/discussions>`_
* Email: research@celayasolutions.com

Uninstallation
--------------

To remove NeuralChild:

.. code-block:: bash

   pip uninstall neuralchild

To completely remove including cache:

.. code-block:: bash

   pip uninstall neuralchild
   pip cache purge
