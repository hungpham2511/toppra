"""Exceptions used in the toppra library."""


class ToppraError(Exception):
    """A generic error class used in the toppra library."""


class BadInputVelocities(ToppraError):
    """Raise when given input velocity is invalid."""


class SolverNotFound(ToppraError):
    """Unable to find a solver."""
