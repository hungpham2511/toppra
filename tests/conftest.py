import toppra
import pytest
try:
    import openravepy as orpy
    IMPORT_OPENRAVE = True
except ImportError as err:
    IMPORT_OPENRAVE = False
except SyntaxError as err:
    IMPORT_OPENRAVE = False


@pytest.fixture(scope="session")
def rave_env():
    env = orpy.Environment()
    yield env
    env.Destroy()
    orpy.RaveDestroy()


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
