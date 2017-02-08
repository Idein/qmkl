# QMKL - Idein's internal version


QMKL is **Math Kernel Library for VideoCore IV QPU**.
QMKL is **compatible with Intel MKL**
except for double precision etc.


## Requirements

You need to install [qasm2](https://github.com/Terminus-IMRC/qpu-assembler2)
and [qbin2hex](https://github.com/Terminus-IMRC/qpu-bin-to-hex) to compile
this library. Just clone them and do `make && sudo make install`.


## Installation

```
$ git clone https://github.com/Idein/qmkl.git
$ cd qmkl/
$ cmake .
$ make
$ sudo make install
```


## Running tests

```
$ sudo test/sgemm
$ sudo test/scopy
$ sudo test/vsAbs
```

The results are [here](https://gist.github.com/Terminus-IMRC/1ec399a64edcacfc3040baf3c97f0895).
