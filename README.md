# QMKL


**QMKL** is Math Kernel Library for VideoCore IV QPU.
QMKL is **fully compatible with Intel MKL**,
except that you need to call initialization function `qmkl_init` at first.


## Installation

```
$ git clone https://github.com/Terminus-IMRC/qmkl.git
$ mkdir -p qmkl/build/
$ cd qmkl/build/
$ cmake ..
$ make
$ sudo make install
```
