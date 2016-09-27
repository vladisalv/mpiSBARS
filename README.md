**mpiSBARS** - parallel version of program [sbars](http://sbars.psn.ru/gccount.html).

## Overview

mpiSBARS is parallel program for recognition of extended inexact repeats in genome.
It based on spectral-analytical method recognition of repeats in character sequences.
The theoretical justification of the algorithm is based
on the theorem on equivalent representation of the character sequence
by the vector of continuous characteristic functions. More information
you can find [here](http://sbars.psn.ru/papers/pyatkov14.pdf).

Method implemented using C++ and MPI parallel technology.
Also, it has GPU support (CUDA-C) for performance on heterogeneous computer system.

### Compile

To build mpiSBARS, do:
```bash
$ ./autogen.sh
$ ./configure
$ make
$ make install
```

For enable GPU support config it with --enable-gpu option:
```bash
$ ./autogen.sh
$ ./configure --enable-gpu
```

Sometimes, you need specify path to your cuda libs:
```bash
$ ./autogen.sh
$ CPPFLAGS="-I/opt/cuda/include/" LDFLAGS="-L/opt/cuda/lib64/" ./configure --enable-gpu
```

Also, you can disable MPI support:
```bash
$ ./autogen.sh
$ ./configure --without-mpi
```

So, you can have 4 version of program:

* sequence program (like original sbars)
* sequence program using GPU
* parallel program
* parallel program using GPU

### Launch

Try `mpisbars --help` for more detail.

Also, you can use `test/run` script for simple launch,
which use MPIRUN wrap script (`test/MPIRUN -h`).

You can verify your build launch `test/verify` script:
```bash
$ SYSTEM=slurm EXTRA="ompi" _PARTITION=test test/verify
You use slurm: sbatch --ntasks=4 --partition=test
--output=test/result//_out_file50K--4--2016-09-27_05-42-36 ompi
test/../src/mpisbars --profiling-window 250 --step-profiling 1
--decompose-window 250 --step-decompose 100 --number-coefficient 75 --eps 0.00001
-f test/input//file50K
--image-save test/output//_pic_file50K--4--2016-09-27_05-42-36
--repeats-analysis-save test/output//_ana_file50K--4--2016-09-27_05-42-36
--limit-memory 100000000
Submitted batch job 1325736
Press any key when your task will completed
Test is OK.
$
```



### Web Page

[https://github.com/vladisalv/mpiSBARS](https://github.com/vladisalv/mpiSBARS)

### Bug Reports

For bug report send a description of the problem to [Vladislav Toigildin](mailto:Vladislav.Toigildin@cs.msu.su)
