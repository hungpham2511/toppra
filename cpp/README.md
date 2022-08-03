# Description
This is a C++ implementation of the TOPP-RA algorithm.

# Building

```sh
# clone
git clone -b develop https://github.com/hungpham2511/toppra

# build
mkdir -p cpp/build && cd cpp/build
cmake ..
make -j4

# run test
./tests/all_tests
```

## Running test with pinocchio
`pinocchio` is used in toppra to calculate kinematical quantities such
as end-effector pose jacobians, required for task-space velocity
consraint. 

Install `pinocchio` bindings (see below) and example from robotpkg,
then setup environment variables before building

``` sh
# Building
export LD_LIBRARY_PATH=/opt/openrobots/lib:${LD_LIBRARY_PATH}
export CMAKE_PREFIX_PATH=/opt/openrobots
```

Add `-DBUILD_WITH_PINOCCHIO=ON` to the cmake command.

To run the tests

``` sh
# Running
export ROS_PACKAGE_PATH=/opt/openrobots/share
export PYTHONPATH=/opt/openrobots/lib/python3.6/site-packages:$PYTHONPATH
# or any python version that you install
```

## How to install optional dependencies:

- GLPK: `sudo apt install libglpk-dev`
- qpOASES: `sudo apt install robotpkg-qpoases` (follow http://robotpkg.openrobots.org/debian.html for robotpkg)
- pinocchio: `sudo apt install robotpkg-pinocchio` (follow http://robotpkg.openrobots.org/debian.html for robotpkg)


## Building Python bindings

We use `pybind11` to provide the bindings. To build the bindings you
need to first install pybind11 version 2.5.0.
``` sh
cd ~/ && git clone https://github.com/pybind/pybind11
cd pybind11 && git checkout v2.5.0
mkdir build && cd build && cmake -DPYBIND11_TEST=false .. && sudo make install
```

Build `toppra` with Python bindings and all optional dependencies:
``` sh
cmake -DBUILD_WITH_PINOCCHIO=ON -DBUILD_WITH_qpOASES=ON -DBUILD_WITH_GLPK=ON -DPYTHON_BINDINGS=ON -DPYBIND11_PYTHON_VERSION=3.7 ..
make
```
Running this step shall build the shared library `libtoppra.so` and create a CPython
module at `toppra/cpp/` that linked to this shared library.

## Issues during build

``` sh
/usr/local/include/pybind11/detail/common.h:112:10: fatal error: Python.h: No such file or directory
 #include <Python.h>
          ^~~~~~~~~~
compilation terminated.
```
This is because during compilation, `pybind11` can not find the Python header files. 
Check that you have `libpython-dev` or `libpython3-dev` installed and whether the 
correct Python executable  is found during cmake step. You should see something similar to
below:

``` sh
-- Found PythonInterp: /usr/bin/python3.7 (found suitable version "3.7.4", minimum required is "3.7")
-- Found PythonLibs: /usr/lib/x86_64-linux-gnu/libpython3.7m.so
-- Found /usr/include/python3.7m /usr/bin/python3.7
```

If this error persists, try give more hint to CMake `FindPython`
module by adding these flags to the cmake command:
``` sh
-DPYTHON_BINDINGS=ON -DPYBIND11_PYTHON_VERSION=3.7 -DPython_EXECUTABLE=/usr/bin/python3.7 
```

## Building doxygen doc

Make sure that `doxygen` is installed and available. Running cmake
normally will create a rule for building doc with doxygen.

``` sh
# Run cmake normally
make doc
# The documentation is available at doc/doc/html/index.html
```

# Using TOPPRA in CMake-based project

In your CMakeLists.txt,
```cmake
# The following line defines cmake target toppra::toppra
find_package(toppra)
...
target_link_library(foo PUBLIC toppra::toppra)
```



