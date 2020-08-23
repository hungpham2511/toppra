"""
toppra.solverwrapper
---------------------------

All computations in TOPP-RA algorithms are done by the linear and
quadratic solvers, wrapped in solver wrappers.


SolverWrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.solverwrapper.SolverWrapper
   :members: 


hotqpOASESSolverWrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.solverwrapper.hotqpOASESSolverWrapper
   :members: close_solver, setup_solver, solve_stagewise_optim

qpOASESSolverWrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.solverwrapper.qpOASESSolverWrapper
   :members: close_solver, setup_solver, solve_stagewise_optim


seidelWrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.solverwrapper.seidelWrapper	       
   :members: solve_stagewise_optim


ecosWrapper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.solverwrapper.ecosWrapper
   :members: solve_stagewise_optim

"""

from .hot_qpoases_solverwrapper import hotqpOASESSolverWrapper
from .cy_seidel_solverwrapper import seidelWrapper
from .ecos_solverwrapper import ecosWrapper
from .qpoases_solverwrapper import qpOASESSolverWrapper
from .solverwrapper import available_solvers, SolverWrapper
