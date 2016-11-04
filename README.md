# QMKL


**QMKL** is Math Kernel Library for VideoCore IV QPU.
QMKL is **fully compatible with Intel MKL**,
except that you need to call initialization function `qmkl_init` at first.


## Requirements

You need to install [qasm2](https://github.com/Terminus-IMRC/qpu-assembler2)
and [qbin2hex](https://github.com/Terminus-IMRC/qpu-bin-to-hex) to compile
this library. Just clone them and do `make && sudo make install`.


## Installation

```
$ git clone https://github.com/Terminus-IMRC/qmkl.git
$ cd qmkl/
$ cmake .
$ make
$ sudo make install
```


## Running tests

```
$ sudo test/sgemm
```
