# Description
This is a implementation of the TOPP-RA algorithm.

# Building Tests

```sh
# clone
git clone -b develop https://github.com/hungpham2511/toppra

# build
export LD_LIBRARY_PATH=/opt/openrobots/lib:${LD_LIBRARY_PATH}
export CMAKE_PREFIX_PATH=/opt/openrobots
mkdir build && cd build && cmake -DBUILD_WITH_PINOCCHIO=ON -DBUILD_WITH_qpOASES=ON ..
make -j4

# run test
./tests/all_tests
```

