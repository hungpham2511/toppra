"""toppra Testing flags.

This file defines several flags used in testing. Other tests should
import these flags instead of defining their own.

"""


try:
    import openravepy as orpy
    FOUND_OPENRAVEPY = True
except:
    # Unable to find openrave
    FOUND_OPENRAVEPY = False

# try:
#     import mosek
#     FOUND_MOSEK = True
# except ImportError:
#     FOUND_MOSEK = False
FOUND_MOSEK = False  # permanently disable MOSEK

try:
    import cvxpy
    FOUND_CXPY = True
except ImportError:
    FOUND_CXPY = False
