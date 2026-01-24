from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import sys

def get_numpy_include():
    import numpy
    return numpy.get_include()

NAME = "toppra"
with open("VERSION", "r", encoding='UTF-8') as file_:
    VERSION = file_.read()
DESCR = "toppra: time-optimal parametrization of trajectories for robots subject to constraints."
with open("README.md", "r", encoding='UTF-8') as file_:
    LONG_DESCRIPTION = file_.read()

URL = "https://github.com/hungpham2511/toppra"

# setup requirements
with open("requirements.txt", "r") as f:
    REQUIRES = ["scipy>0.18", "numpy", "matplotlib"]
    DEV_REQUIRES = [line.strip() for line in f if line.strip()]

AUTHOR = "Hung Pham"
EMAIL = "hungpham2511@gmail.com"

LICENSE = "MIT"

SRC_DIR = "toppra"
PACKAGES = [
    "toppra", "toppra.constraint", "toppra.algorithm",
    "toppra.algorithm.reachabilitybased", "toppra.solverwrapper", "toppra.cpp"
]

ext_1 = Extension(SRC_DIR + "._CythonUtils", [SRC_DIR + "/_CythonUtils.pyx"],
                  extra_compile_args=['-O2'],
                  libraries=[],
                  include_dirs=[get_numpy_include()])

ext_2 = Extension(SRC_DIR + ".solverwrapper.cy_seidel_solverwrapper",
                  [SRC_DIR + "/solverwrapper/cy_seidel_solverwrapper.pyx"],
                  extra_compile_args=['-O1'],
                  include_dirs=[get_numpy_include()])

EXTENSIONS = [ext_1, ext_2]
SETUP_REQUIRES = ["numpy", "cython"]


if __name__ == "__main__":
    setup(
        # Dependencies installed when running `pip install .`
        install_requires=REQUIRES,
        setup_requires=["numpy", "cython"],
        extras_require={
            # Dependencies installed when running `pip install -e .[dev]`

            # NOTE: This is deprecated in favour of the simpler workflow
            # of installing from requirements3.txt before installing
            # this pkg.
            'dev': DEV_REQUIRES
        },
        packages=PACKAGES,
        zip_safe=False,
        name=NAME,
        version=VERSION,
        description=DESCR,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        license=LICENSE,

        # This is used to build the Cython modules. Will be run
        # automatically if not found by pip. Otherwise run
        #
        #      python setup.py build
        #
        # to trigger manually.
        cmdclass={"build_ext": build_ext},
        ext_modules=cythonize(EXTENSIONS))
