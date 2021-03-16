from setuptools import setup, Extension
from distutils.command.install import install
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import sys

NAME = "toppra"
with open("VERSION", "r") as file_:
    VERSION = file_.read()
DESCR = "toppra: time-optimal parametrization of trajectories for robots subject to constraints."
with open("README.md", "r") as file_:
    LONG_DESCRIPTION = file_.read()

URL = "https://github.com/hungpham2511/toppra"


# setup requirements
if sys.version[0] == '2':
    with open("requirements.txt", "r") as f:
        REQUIRES = ["scipy==0.18.0", "numpy", "matplotlib",
                    # only required on python2.7
                    "pathlib2", "enum34", "strip_hints", "typing"]
        DEV_REQUIRES = [line.strip() for line in f if line.strip()]
else:
    with open("requirements3.txt", "r") as f:
        REQUIRES = ["scipy>0.18", "numpy", "matplotlib"]
        DEV_REQUIRES = [line.strip() for line in f if line.strip()]

AUTHOR = "Hung Pham"
EMAIL = "hungpham2511@gmail.com"

LICENSE = "MIT"

SRC_DIR = "toppra"
PACKAGES = ["toppra",
            "toppra.constraint",
            "toppra.algorithm",
            "toppra.algorithm.reachabilitybased",
            "toppra.solverwrapper",
            "toppra.cpp"]

ext_1 = Extension(SRC_DIR + "._CythonUtils",
                  [SRC_DIR + "/_CythonUtils.pyx"],
                  libraries=[],
                  include_dirs=[np.get_include()])

ext_2 = Extension(SRC_DIR + ".solverwrapper.cy_seidel_solverwrapper",
                  [SRC_DIR + "/solverwrapper/cy_seidel_solverwrapper.pyx"],
                  extra_compile_args=['-O1'],
                  include_dirs=[np.get_include()])

EXTENSIONS = [ext_1, ext_2]
SETUP_REQUIRES = ["numpy", "cython"]
if sys.version[0] == '2' or sys.version[:3] == '3.5':
    SETUP_REQUIRES = ["numpy", "cython", "strip_hints"]


# custom install command: strip type-hints before installing toppra
# for python2.7 and pthon3.5
class install2(install):
    def run(self, *args, **kwargs):
        # stripping
        if sys.version[0] == '2' or sys.version[:3] == '3.5':
            from strip_hints import strip_file_to_string
            import glob
            import os.path
            def process_file(f):
                print(os.path.abspath(f))
                out = strip_file_to_string(f)
                with open(f, 'w') as fh:
                    fh.write(out)
            for f in glob.glob("%s/*/toppra/*/*.py" % self.build_base):
                process_file(f)
            for f in glob.glob("%s/*/toppra/*.py" % self.build_base):
                process_file(f)

            print(os.path.abspath("."))
            print(os.path.abspath(self.build_base))
        # install new files
        install.run(self, *args, **kwargs)


if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          # Dependencies installed when running `pip install .`
          setup_requires=["numpy", "cython"],

          # Dependencies installed when running `pip install -e .[dev]`
          extras_require={
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
          cmdclass={"build_ext": build_ext, "install": install2},
          ext_modules=cythonize(EXTENSIONS)
          )
