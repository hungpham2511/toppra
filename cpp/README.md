# Description

This is a header-only implementation of the TOPP-RA algorithm.

# Building Tests

```sh
# clone
git clone --recursive -b feat-cpp https://github.com/hungpham2511/toppra

# build deps
cd toppra/cpp/extern/googletest
mkdir build && cd build && cmake .. 
make -j4

# build toppra
cd ../../.. && mkdir build && cd build && cmake ..
make -j4
```

