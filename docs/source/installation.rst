.. _installation:

Installation 
**********************

Installing Python API
=========================================

Basic installation
--------------------------

Run the below command to run the stable version of the Python API

.. code-block:: shell

   pip install git+https://github.com/hungpham2511/toppra

The development version can be installed with this instead.

.. code-block:: shell

   pip install git+https://github.com/hungpham2511/toppra@develop

You can now try a basic example:

.. code-block:: shell

   python examples/plot_kinematics.py


Installing other solver backends
--------------------------------

The default installation comes with an implementation of the seidel LP
solver, specialized for parametrization problem. Other backends are
also available.

To install `qpoases` run following command after installing `pyinvoke`

.. code-block:: shell

   inv install-solvers

To install other backends (cvxpy, cvxopt, ecos), install the `dev`
extra requirements:

.. code-block:: shell

   pip install .[dev]

Installing C++ API
=========================================

See `cpp/README.md
<https://github.com/hungpham2511/toppra/blob/develop/cpp/README.md>`_
for more details.

Building docs
==============================

The latest documentation is available at
`<https://hungpham2511.github.io/toppra>`_.

To build and view the documentation locally, run the following command
in the terminal.

.. code-block:: shell

   cd toppra/docs
   pip install -r requirements.txt
   make livehtml

The C++ API has its own doxygen-based API.

Testing
===============================

The Python API test suites use :code:`pytest` for running unittests,
you will need to install the `dev` extra (See above) and run `pytest`
to run the full suite.

.. code-block:: sh

   cd <toppra-dir>/
   pytest -v

The C++ API has a set of unit tests as well, from the build folder run:

.. code-block:: sh

   ./tests/all_tests
