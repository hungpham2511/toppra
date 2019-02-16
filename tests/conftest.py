try:
    import openravepy as orpy
except:
    # Unable to find openrave
    FOUND_OPENRAVEPY = False
import pytest


@pytest.fixture(scope="session")
def rave_env():
    env = orpy.Environment()
    yield env
    env.Destroy()
    orpy.RaveDestroy()

