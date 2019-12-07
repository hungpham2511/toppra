from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import sys

NAME = "toppra"
with open("VERSION", "r") as file_:
    VERSION = file_.read()
DESCR = "toppra: time-optimal parametrization of trajectories for robots subject to constraints."
LONG_DESCRIPTION = "An implementation of TOPP-RA (TOPP via Reachability Analysis) for time-parametrizing" \
    "trajectories for robots subject to kinematic (velocity and acceleration) and dynamic" \
    "(torque) constraints. Some other kinds of constraints are also supported."

URL = "https://github.com/hungpham2511/toppra"

# requirements
if sys.version[0] == '2':
    with open("requirements.txt", "r") as f:
        REQUIRES = ["scipy==0.18.0", "numpy", "enum34", "coloredlogs"]
        DEV_REQUIRES = [line.strip() for line in f if line.strip()]
else:
    with open("requirements3.txt", "r") as f:
        REQUIRES = ["scipy>0.18", "numpy", "enum34", "coloredlogs"]
        DEV_REQUIRES = [line.strip() for line in f if line.strip()]

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
          setup_requires=["numpy", "cython"],
          extras_require={
              'dev': DEV_REQUIRES
          },
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          long_description=LONG_DESCRIPTION,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=cythonize(EXTENSIONS)
          )
