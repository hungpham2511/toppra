"""Parametrization algorithms
------------------------

TOPPRA
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.algorithm.TOPPRA
   :show-inheritance:
   :members: compute_trajectory, compute_parameterization

TOPPRAsd
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.algorithm.TOPPRAsd
   :members: compute_trajectory, set_desired_duration, compute_parameterization
   :show-inheritance:

ParameterizationReturnCode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: toppra.algorithm.ParameterizationReturnCode
   :members:

   .. autoattribute:: Ok
   .. autoattribute:: ErrUnknown
   .. autoattribute:: ErrShortPath
   .. autoattribute:: FailUncontrollable
   .. autoattribute:: ErrForwardPassFail

ParameterizationData
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: toppra.algorithm.ParameterizationData
   :members:

[abstract]ParameterizationAlgorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: toppra.algorithm.ParameterizationAlgorithm
   :members:

"""
from .algorithm import ParameterizationAlgorithm, ParameterizationData, ParameterizationReturnCode
from .reachabilitybased import TOPPRA, TOPPRAsd

__all__ = ["ParameterizationData", "ParameterizationAlgorithm", "TOPPRA", "TOPPRAsd"]
