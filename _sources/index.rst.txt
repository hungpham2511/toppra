.. TOPP-RA documentation master file, created by
   sphinx-quickstart on Sat Nov 18 00:03:54 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TOPP-RA's documentation!
===================================

TOPP-RA is a library for time-parameterizing robot trajectories
subject to kinematic and dynamic constraints. It is able to compute
the *time-optimal* time-parametrization of a given path, as well as a
time-parametrization with a *specified duration*.  See the examples
for more details.

The current implementation supports the following constraints:

1. joint velocity and acceleration bounds;
2. *robust* acceleration bounds;
3. torque bounds (including redundantly-actuated manipulators);
4. *robust* torque bounds;
5. contact stability for legged robots.

If you use this library for your research, please reference the accompanying paper:
      `«A new approach to Time-Optimal Path Parameterization based on Reachability Analysis» <https://arxiv.org/abs/1707.07239>`_,
      *IEEE Transactions on Robotics*, vol. 34(3), pp. 645–659, 2018.

See below for installation instructions and some tutorials.

Quick Installation
~~~~~~~~~~~~~~~~~~~~~~
TOPP-RA can be installed from source by the following steps:

.. code-block:: shell

   git clone https://github.com/hungpham2511/toppra && cd toppra/
   pip install -r requirements.txt --user
   python setup.py install --user

These lines implement the most basic functionalities, such as
parametrize trajectories subject to velocity and acceleration
constraints. Some advanced functionalities require `openrave`, `cvxpy`
and so on. See :doc:`installation` for the full instruction.

.. toctree::
   :maxdepth: 2

   installation

Tutorials
~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   tutorials/0_kinematic_example
   tutorials/1_geometric_path
   tutorials/2_can_linear_constraints
   tutorials/3_catersian_constraints

Package References
~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   hello

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

