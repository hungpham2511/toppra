from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

NAME = "toppra"
VERSION = "0.1"
DESCR = "An implementation of TOPP via Reachability Analysis (TOPP-RA)"
URL = "nil"
REQUIRES = ['numpy', 'cython']

AUTHOR = "Hung Pham"
EMAIL = "hungpham2511@gmail.com"

LICENSE = "Apache 2.0"

SRC_DIR = "toppra"
PACKAGES = [SRC_DIR]

ext_1 = Extension(SRC_DIR + "._CythonUtils",
                  [SRC_DIR + "/_CythonUtils.pyx"],
                  libraries=[],
                  include_dirs=[np.get_include()])


EXTENSIONS = [ext_1]

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
