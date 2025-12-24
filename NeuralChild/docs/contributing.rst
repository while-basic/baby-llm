Contributing
============

Thank you for your interest in contributing to NeuralChild!

Please see our comprehensive contributing guide:

`CONTRIBUTING.md <https://github.com/celayasolutions/neuralchild/blob/main/CONTRIBUTING.md>`_

Quick Links
-----------

* `Report a Bug <https://github.com/celayasolutions/neuralchild/issues/new?template=bug_report.md>`_
* `Request a Feature <https://github.com/celayasolutions/neuralchild/issues/new?template=feature_request.md>`_
* `Ask a Question <https://github.com/celayasolutions/neuralchild/discussions>`_

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/celayasolutions/neuralchild.git
   cd neuralchild
   pip install -e ".[dev]"

Running Tests
-------------

.. code-block:: bash

   pytest neuralchild/tests/ -v

Code Style
----------

.. code-block:: bash

   black neuralchild/
   isort neuralchild/
   flake8 neuralchild/

Pull Request Process
--------------------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

For detailed guidelines, see `CONTRIBUTING.md <https://github.com/celayasolutions/neuralchild/blob/main/CONTRIBUTING.md>`_.
