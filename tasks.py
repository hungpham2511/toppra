"""Collection of different operational tasks."""
from invoke import task

try:
    import pathlib2 as pathlib
except ImportError:
    import pathlib

TOCHECK = [
    "python/toppra",
]


@task
def type_check(c):
    """Check static typing."""
    for path in TOCHECK:
        if not c.run(f"mypy {path}"):
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
        print("qpOASES install path already exists")
    else:
        branch = "master"
        c.run(f"git clone -b {branch} https://github.com/hungpham2511/qpOASES.git {install_dir}")
        c.run(f"cd {install_dir} && mkdir bin && make")
    if user:
        flag = "--user"
    else:
        flag = ""
    c.run(f"cd {install_dir}/interfaces/python/ && \
        pip install --no-build-isolation {flag} .")


@task
def make_venvs(c, run_tests=False):
    """Convenient command to create different environments for testing."""
    venv_path = "/tmp/venv"
    test_flag = ""

    c.run(
        f"python3 -m virtualenv {venv_path} && \
            {venv_path}/bin/pip install invoke pathlib numpy cython pytest"
    )
    c.run(
        f". {venv_path}/bin/activate && \
            invoke install-solvers && \
            pip install -e ./python[dev]"
    )
    if run_tests:
        c.run(f"{test_flag} {venv_path}/bin/pytest -x")


@task
def lint(c, pycodestyle=False, pydocstyle=False):
    """Run linting on selected source files."""
    c.run(
        "python -m pylint --rcfile=.pylintrc \
                python/toppra/__init__.py \
                python/toppra/utils.py \
                python/toppra/interpolator.py \
                python/toppra/exceptions.py \
                python/toppra/parametrizer.py \
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
