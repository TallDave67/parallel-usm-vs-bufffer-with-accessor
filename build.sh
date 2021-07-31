#!/bin/sh
export CC=/usr/bin/clang-11
export CXX=/opt/intel/oneapi/compiler/2021.3.0/linux/bin/dpcpp

[ ! -d "build" ] && echo "making build directory" && mkdir build
cd build
cmake ../
make VERBOSE=1
