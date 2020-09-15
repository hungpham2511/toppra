# testing flags
try:
    import openravepy as orpy
    IMPORT_OPENRAVEPY = True
    IMPORT_OPENRAVEPY_MSG = ""
except Exception as err:
    # Unable to find openrave
    IMPORT_OPENRAVEPY = False
    IMPORT_OPENRAVEPY_MSG = err.args[0]

FOUND_MOSEK = False  # permanently disable MOSEK during testing

try:
    import cvxpy
    FOUND_CXPY = True
except ImportError:
    FOUND_CXPY = False
