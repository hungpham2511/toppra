"""toppra.algorithm
------------------------


ParameterizationAlgorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: toppra.algorithm.ParameterizationAlgorithm
   :members:

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


TOPPRA
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.algorithm.TOPPRA
   :show-inheritance:
   :members: compute_trajectory

TOPPRAsd
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: toppra.algorithm.TOPPRAsd
   :members: compute_trajectory
   :show-inheritance:


"""
from .algorithm import ParameterizationAlgorithm, ParameterizationData, ParameterizationReturnCode
from .reachabilitybased import TOPPRA, TOPPRAsd

__all__ = ["ParameterizationData", "ParameterizationAlgorithm", "TOPPRA", "TOPPRAsd"]
