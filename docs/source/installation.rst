.. _installation:

Installation 
**********************

Installation instructions for Python API
=========================================

Basic installation
--------------------------

To install `toppra`, ensure that `numpy` and `cython` are installed, then run

.. code-block:: shell

   pip install numpy cython
   git clone https://github.com/hungpham2511/toppra && cd toppra/
   pip install .

You can now try a basic example:

.. code-block:: shell

   python examples/kinematics.py


Advanced: Other solver backends
--------------------------------

The default installation comes with an implementation of the seidel LP
solver, specialized for parametrization problem. Other backends are
also available.

To install `qpoases` run following command after installing `pyinvoke`

.. code-block:: shell

   invoke install-solvers


To install other backends (cvxpy, cvxopt, ecos), install the `dev`
extra requirements:

.. code-block:: shell

   pip install .[dev]


Advanced: OpenRAVE and Pymanoid
--------------------------------------

In order to run some of the examples, it is necessary to install
`openRAVE <https://github.com/rdiankov/openrave>`_. A good instruction
for installing this library on Ubuntu 16.04 can be found `here
<https://scaron.info/teaching/installing-openrave-on-ubuntu-16.04.html>`_.

.. note:: The humanoid and redundantly-actuated torque examples are not
          yet included in the current library. See tag ``v0.1`` if you
          want to run these examples.

Multi-contact and torque bounds.  To use these functionality, the
following libraries are needed:

1. [openRAVE](https://github.com/rdiankov/openrave)
2. [pymanoid](https://github.com/stephane-caron/pymanoid)

`openRAVE` can be tricky to install, a good instruction for installing
`openRAVE` on Ubuntu 16.04 can be
found
[here](https://scaron.info/teaching/installing-openrave-on-ubuntu-16.04.html).

To install `pymanoid` locally, do the following
``` sh
mkdir git && cd git
git clone <pymanoid-git-url>
git checkout 54299cf
export PYTHONPATH=$PYTHONPATH:$HOME/git/pymanoid
```

Advanced: toppra C++ and bindings
-----------------------------------

A C++ toppra API with python bindings is being developed. For
instruction see `toppra/cpp/README.md`.

Installation instructions for C++ API
=========================================

See `cpp/README.md` for more details.

Building docs
==============================

The latest documentation is available at
`<https://toppra.readthedocs.io/en/latest/>`_.

To build and view the documentation locally, install `sphinx
<http://www.sphinx-doc.org/en/stable/index.html>`_ then run the
following command in the terminal

.. code-block:: shell

   invoke build-docs

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
