# Constants
SUPERTINY = 1e-10
TINY = 1e-8
SMALL = 1e-5
LARGE = 1000.0
INFTY = 1e8

# TODO: What are these constant used for?
MAXU = 10000  # Max limit for `u`
MAXX = 10000  # Max limit for `x`
MAXSD = 100   # square root of maxx

# constraint creation
JVEL_MAXSD = 10000   # max sd when creating joint velocity constraints
JACC_MAXU = 1000000  # max u when creating joint acceleration constraint


# solver wrapper related constants.
# NOTE: Read the wrapper's documentation for more details.

# cvxpy
CVXPY_MAXX = 10000
CVXPY_MAXU = 10000

# ecos
ECOS_MAXX = 10000
ECOS_INFTY = 1000
