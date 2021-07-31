# Parallel: USM vs Buffer with Accessor (DPC++)

Using Intel's oneAPI DPC++.  Here we compare sharing data between host and device by USM to sharing by buffer with accessor.

Linux distro "Pop!_OS 21.04"

cmake version 3.18.4

Intel(R) oneAPI DPC++/C++ Compiler 2021.3.0 (2021.3.0.20210619)

## Steps

*I placed this in my .bashrc file so the dpcpp compiler can be found.*

> *export ONEAPI_DIR="/opt/intel/oneapi"*

> *[ -s "$ONEAPI_DIR/setvars.sh" ] && \. "$ONEAPI_DIR/setvars.sh"  # initialize oneAPI environment*

./build.sh

./build/usm-vs-buffer-with-accessor

## History

dates: July 30 - Aug 1, 2021

duration: 8 hours

### Code from Another Developer

This code inspired by Intel sample code.

[oneAPI Base Training Module 3: Unified Shared Memory](https://devcloud.intel.com/oneapi/get_started/baseTrainingModules/)

## Output

[Output](https://github.com/TallDave67/parallel-usm-vs-buffer-with-accessor/blob/main/oneapi/output.txt)
