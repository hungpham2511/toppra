from invoke import task
import os
try:
    import pathlib2 as pathlib
except ImportError:
    import pathlib


@task
def install_solvers(c, user=False):
    install_dir = '/tmp/tox-qpoases'
    path = pathlib.Path(install_dir)
    if path.exists():
        print("Path already exist")
    else:
        c.run("git clone https://github.com/hungpham2511/qpOASES {}".format(install_dir))
        c.run("cd /tmp/tox-qpoases/ && mkdir bin && make")
    if user:
        flag = "--user"
    else:
        flag = ""
    c.run("cd /tmp/tox-qpoases/interfaces/python/ && python setup.py install {}".format(flag))

@task
def make_venvs(c, python3=False, run_tests=False):
    """Convenient command to create different environments for testing."""
    if not python3:
        venv_path = "/tmp/venv"
        flag = ""
        test_flag = "export PYTHONPATH=$PYTHONPATH:`openrave-config --python-dir` &&"
    else:
        venv_path = "/tmp/venv3"
        flag = "--python python3"
        test_flag = ""

    c.run("python -m virtualenv {flag} {venv_path} && \
             {venv_path}/bin/pip install invoke pathlib numpy cython pytest".format(
                 venv_path=venv_path, flag=flag
             ))
    c.run(". {venv_path}/bin/activate && \
             invoke install-solvers && \
             pip install -e .[dev]".format(venv_path=venv_path))
    if run_tests:
        c.run("{test_flag} {venv_path}/bin/pytest -x".format(test_flag=test_flag, venv_path=venv_path))


@task
def docker_build(c):
    c.run("docker build -f dockerfiles/Dockerfile . -t toppra-dev")


@task
def docker_start(c):
    c.run("docker run --rm --name toppra-dep -d \
                  -v /home/hung/git/toppra:$HOME/toppra \
                  -e DISPLAY=unix$DISPLAY \
                  --net=host \
                  hungpham2511/toppra-dep:0.0.2 sleep infinity")


@task
def docker_exec(c):
    c.run("docker exec -it toppra-dep bash", pty=True)

@task
def docker_stop(c):
    c.run("docker stop toppra-dep")
