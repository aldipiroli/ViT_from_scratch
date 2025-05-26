#!/bin/bash
mkdir build 
cd build 
cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch/libtorch/ ..
cmake --build . --config Release
export LD_LIBRARY_PATH=/usr/local/libtorch/libtorch/lib/:$LD_LIBRARY_PATH