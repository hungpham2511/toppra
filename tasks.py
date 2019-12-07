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


@task
def docker_build(c):
    c.run("docker build -f dockerfiles/Dockerfile . -t toppra-dev")


@task
def docker_start(c):
    c.run("docker run --rm --name toppra-dep -d \
    -v /home/hung/git/toppra:$HOME/toppra \
    hungpham2511/toppra-dep:0.0.2 sleep infinity")


@task
def docker_exec(c):
    c.run("docker exec -it toppra-dep bash", pty=True)

@task
def docker_stop(c):
    c.run("docker stop toppra-dep")
