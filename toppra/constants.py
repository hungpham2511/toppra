"""
Some constants used by TOPPRA solvers.
"""
import logging

try:
    import openravepy as orpy
    FOUND_OPENRAVE = True
except (ImportError, SyntaxError) as err:
    FOUND_OPENRAVE = False
    logging.getLogger("toppra").debug("Unable to import openrave.")


# Constants
SUPERTINY = 1e-10
TINY = 1e-8
SMALL = 1e-5
LARGE = 1000.0
VERYLARGE = 1e8
INFTY = 1e16

# Number of times xs[i] is lowered during the forward pass to
# accomodate for numerical error from solvers.
MAX_TRIES = 10

# TODO: What are these constant used for?
MAXU = 10000  # Max limit for `u`
MAXX = 10000  # Max limit for `x`
MAXSD = 100  # square root of maxx

# constraint creation
JVEL_MAXSD = 1e8  # max sd when creating joint velocity constraints
JACC_MAXU = 1e16  # max u when creating joint acceleration constraint

# solver wrapper related constants.
# NOTE: Read the wrapper's documentation for more details.

# qpoases
QPOASES_INFTY = 1e16

# cvxpy
CVXPY_MAXX = 10000
CVXPY_MAXU = 10000

# ecos
ECOS_MAXX = 10000
ECOS_INFTY = 1000
