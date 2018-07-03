from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

NAME = "toppra"
VERSION = "0.2"
DESCR = "An implementation of TOPP-RA (TOPP via Reachability Analysis) for time-parametrizing" \
        "trajectories for robots subject to kinematic (velocity and acceleration) and dynamic" \
        "(torque) constraints. Some other kinds of constraints are also supported."
URL = "https://github.com/hungpham2511/toppra"
REQUIRES = ['numpy', 'cython', "scipy>0.18", "coloredlogs", "enum", "cvxpy>=0.4.11, <1.0.0", "pytest"]

AUTHOR = "Hung Pham"
EMAIL = "hungpham2511@gmail.com"

LICENSE = "MIT"

SRC_DIR = "toppra"
PACKAGES = ["toppra",
            "toppra.constraint",
            "toppra.algorithm",
            "toppra.algorithm.reachabilitybased",
            "toppra.solverwrapper"]

ext_1 = Extension(SRC_DIR + "._CythonUtils",
                  [SRC_DIR + "/_CythonUtils.pyx"],
                  libraries=[],
                  include_dirs=[np.get_include()])

ext_2 = Extension(SRC_DIR + ".solverwrapper.cy_seidel_solverwrapper",
                  [SRC_DIR + "/solverwrapper/cy_seidel_solverwrapper.pyx"],
                  extra_compile_args=['-O1'],
                  include_dirs=[np.get_include()])


EXTENSIONS = [ext_1, ext_2]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=EXTENSIONS
          )
