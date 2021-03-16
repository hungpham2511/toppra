"""Collection of different operational tasts."""
from invoke import task

try:
    import pathlib2 as pathlib
except ImportError:
    import pathlib

TOCHECK = [
    "toppra",
]


@task
def type_check(c):
    """Check static typing."""
    for path in TOCHECK:
        if not c.run("mypy {path}".format(path=path)):
            return 1
    return 0


@task
def strip_types(c):
    """Strip type hints from source code."""
    from strip_hints import strip_file_to_string
    import glob
    def process_file(f):
        print(f)
        out = strip_file_to_string(f)
        with open(f, 'w') as fh:
            fh.write(out)
    for f in glob.glob("toppra/*/*.py"):
        process_file(f)
    for f in glob.glob("toppra/*.py"):
        process_file(f)
        
@task
def build_docs(c):
    """Build documentation"""
    c.run("cp HISTORY.md docs/source/HISTORY.md")
    c.run("cd docs && make html")


@task
def install_solvers(c, user=False):
    """Install backend solvers, e.g, qpoases."""
    install_dir = "/tmp/tox-qpoases"
    path = pathlib.Path(install_dir)
    if path.exists():
        print("Path already exist")
    else:
        c.run(
            "git clone https://github.com/hungpham2511/qpOASES {}".format(install_dir)
        )
        c.run("cd /tmp/tox-qpoases/ && mkdir bin && make")
    if user:
        flag = "--user"
    else:
        flag = ""
    c.run(
        "cd /tmp/tox-qpoases/interfaces/python/ && python setup.py install {}".format(
            flag
        )
    )


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

    c.run(
        "python -m virtualenv {flag} {venv_path} && \
             {venv_path}/bin/pip install invoke pathlib numpy cython pytest".format(
            venv_path=venv_path, flag=flag
        )
    )
    c.run(
        ". {venv_path}/bin/activate && \
             invoke install-solvers && \
             pip install -e .[dev]".format(
            venv_path=venv_path
        )
    )
    if run_tests:
        c.run(
            "{test_flag} {venv_path}/bin/pytest -x".format(
                test_flag=test_flag, venv_path=venv_path
            )
        )


@task
def lint(c, pycodestyle=False, pydocstyle=False):
    """Run linting on selected source files."""
    c.run(
        "python -m pylint --rcfile=.pylintrc \
                toppra/__init__.py \
                toppra/utils.py \
                toppra/interpolator.py \
                toppra/exceptions.py \
                toppra/parametrizer.py \
           "
    )
    # toppra/solverwrapper/solverwrapper.py
    if pycodestyle:
        c.run("pycodestyle toppra --max-line-length=120 --ignore=E731,W503,W605")
    if pydocstyle:
        c.run("pydocstyle toppra")


@task
def docker_build(c):
    """Build docker image to run toppra development."""
    c.run("docker build -f dockerfiles/Dockerfile . -t toppra-dev")


@task
def docker_start(c):
    """Start the development docker container."""
    c.run(
        "docker run --rm --name toppra-dep -d \
                  -v /home/hung/git/toppra:$HOME/toppra \
                  -e DISPLAY=unix$DISPLAY \
                  --net=host \
                  hungpham2511/toppra-dep:0.0.3 sleep infinity"
    )
    print("Started new docker instance with name toppra-dep")
