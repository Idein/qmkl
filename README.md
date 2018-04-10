# QMKL


QMKL is a **Math Kernel Library for VideoCore IV QPU**. QMKL is **compatible
with Intel MKL** except for double precision etc.

We, Idein Inc., did some object recognition demos (GoogLeNet etc.) on Raspberry
Pi. The demos run on QPU with QMKL and our private libraries which are highly
optimized for neural networks. There are movies of that:
[Raspberry Pi 3](https://twitter.com/9_ties/status/858291781133148160) and
[Raspberry Pi Zero](https://twitter.com/9_ties/status/858300756092375040).


## Requirements

You need to install:

- [qasm2](https://github.com/Terminus-IMRC/qpu-assembler2)
- [qbin2hex](https://github.com/Terminus-IMRC/qpu-bin-to-hex)
- [mailbox](https://github.com/Terminus-IMRC/mailbox)
- [librpimemmgr](https://github.com/Idein/librpimemmgr)

In addition, make sure Linux kernel 4.9.79 or above is running on your Pi. e.g.:

```
$ uname -r
4.9.80-v7+
```


## Installation

```
$ git clone https://github.com/Idein/qmkl.git
$ cd qmkl/
$ cmake .
$ make
$ sudo make install
```

Or you can create Debian package and install it:

```
$ make package
$ sudo dpkg -i qmkl-x.y.x-system.deb
```


## Running tests

```
$ test/sgemm
$ test/scopy
$ test/vsAbs
$ test/sgemm_spec
```
