from invoke import task
import os
try:
    import pathlib2 as pathlib
except ImportError:
    import pathlib


@task
def install_solvers(c):
    install_dir = '/tmp/tox-qpoases'
    path = pathlib.Path(install_dir)
    if path.exists():
        print("Path already exist")
    else:
        c.run("git clone https://github.com/hungpham2511/qpOASES {}".format(install_dir))
        c.run("cd /tmp/tox-qpoases/ && mkdir bin && make")
    c.run("cd /tmp/tox-qpoases/interfaces/python/ && python setup.py install")
