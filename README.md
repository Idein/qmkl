# QMKL


QMKL is a **Math Kernel Library for VideoCore IV QPU**. QMKL is **compatible
with Intel MKL** except for double precision etc.

We, Idein Inc., built object recognition demos (GoogLeNet etc.) on Raspberry
Pi. The demos run on QPU using both QMKL and our private libraries, which are highly
optimized for neural networks. Please check out our video on [YouTube](https://www.youtube.com/watch?v=R5niixLtf2Q).



## Requirements

You need to install:

- [py-videocore](https://github.com/nineties/py-videocore)
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

If you want to use a specific version of Python, do:

```
$ cmake -DPYTHON_EXECUTABLE=/path/to/python
```

You can create Debian package and install it:

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
