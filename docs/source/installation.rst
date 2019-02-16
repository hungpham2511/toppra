.. _installation:

Installation instructions
==========================

Basic functionality
--------------------------

To install the core functionality of TOPP-RA, simply do:

.. code-block:: shell

   git clone https://github.com/hungpham2511/toppra && cd toppra/
   pip install -r requirements.txt --user
   python setup.py install --user

You can now use TOPP-RA to parametrize trajectories/paths subject to
the robot kinematic limits as well as tool tip Cartesian limits. Try a
basic example here: :doc:`tutorials`.

.. note:: Generally, Python packages should be installed in a `virtual
	  environment
	  <https://docs.python-guide.org/dev/virtualenvs/>`_. See the
	  hyperlink for more details. To install TOPP-RA in a virtual
	  environment, simply activate the environment and do the
	  above steps omitting the :code:`--user` flag.


(Optional) Install qpOASES
--------------------------------

It is possible to use `qpOASES
<https://projects.coin-or.org/qpOASES/wiki/QpoasesInstallation>`_
instead of the default LP solver ``seidel`` as a back-end solver
wrapper running in background. To install run following commands in
a terminal:

.. code-block:: shell

   git clone https://github.com/hungpham2511/qpOASES
   cd qpOASES/ && mkdir bin && make
   cd interfaces/python/
   pip install cython
   python setup.py install --user
   
Advanced functionality
--------------------------------------

In order to run some of the examples, it is necessary to install
`openRAVE <https://github.com/rdiankov/openrave>`_. A good instruction
for installing this library on Ubuntu 16.04 can be found `here
<https://scaron.info/teaching/installing-openrave-on-ubuntu-16.04.html>`_.

.. note:: The humanoid and redundantly-actuated torque examples are not
          yet included in the current library. See tag ``v0.1`` if you
          want to run these examples.

..
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

Building docs
------------------------------

The latest documentation is available at
`<https://hungpham2511.github.io/toppra/>`_.

To build and view the documentation, install `sphinx
<http://www.sphinx-doc.org/en/stable/index.html>`_ then run the
following commands in the terminal

.. code-block:: shell

   cd <toppra-dir>/docs/
   make clean && make html
   <browser> build/index.html

Testing
-------------------------------

TOPP-RA makes use of :code:`pytest` and ``cvxpy`` for testing. Both
can be installed from :code:`pip`.  To run all the tests, do:

.. code-block:: sh

   cd <toppra-dir>/tests/
   pytest -v


