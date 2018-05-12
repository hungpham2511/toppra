import openravepy as orpy
import pytest
    

@pytest.fixture(scope="session")
def rave_env():
    env = orpy.Environment()
    yield env
    env.Destroy()
    orpy.RaveDestroy()

