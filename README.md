## 


# Installation
## Basic

From pip

``` sh
pip install Cython numpy coloredlogs enum scipy
```

to run the examples, you need in addition

``` sh
pip install matplotlib
```


Install qpOASES, the numerical solver

``` sh
git clone https://github.com/stephane-caron/qpOASES
cd qpOASES
```


Finall, install `toppra` with

``` sh
sudo python setup.py install
```

## Multi-contact and torque bounds examples
To use these functionalities, the following libraries are needed:

1. [openRAVE](https://github.com/rdiankov/openrave)
2. [pymanoid](https://github.com/stephane-caron/pymanoid)

`openRAVE` can be tricky to install, a good instruction for installing
`openRAVE` on Ubuntu 16.06 can be
found
[here](https://scaron.info/teaching/installing-openrave-on-ubuntu-16.04.html).


