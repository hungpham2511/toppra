import logging
import toppra
import pytest
try:
    import openravepy as orpy
    IMPORT_OPENRAVE = True
except ImportError as err:
    IMPORT_OPENRAVE = False
except SyntaxError as err:
    IMPORT_OPENRAVE = False
logger = logging.getLogger('toppra.bug')

import sys
print(sys.path)

@pytest.fixture(autouse=True, scope="session")
def rave_env():
    if IMPORT_OPENRAVE:
        logger.warn("Starting openrave")
        orpy.RaveInitialize(load_all_plugins=True)
        logger.warn("Starting a new environment")
        env = orpy.Environment()
        yield env
        logger.warn("Destroying a new environment")
        env.Destroy()
        logger.warn("Destroying Rave runtime")
        orpy.RaveDestroy()
    else:
        yield None


def pytest_addoption(parser):
    parser.addoption(
        "--loglevel", action="store", default="WARNING",
        help="Set toppra loglevel during testing."
    )

    parser.addoption(
        "--robust_regex", action="store", default=".*oa.*",
        help="Regex to choose problems to test when running test_robustness_main.py. "
             "Select '.*oa.*' to run only tests for hotqpoases."
    )

    parser.addoption(
        "--visualize", action="store_true", default=False,
        help="If True visualize test instance."
    )


def pytest_collection_modifyitems(config, items):
    toppra.setup_logging(config.getoption("--loglevel"))
