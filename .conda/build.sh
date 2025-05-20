#!/bin/bash

mkdir build
cd build

# Install the C API library for DeePMD-kit: libdeepmd_c library
# The library is stored in local machine.
wget https://github.com/deepmodeling/deepmd-kit/releases/latest/download/libdeepmd_c.tar.gz
tar -xf libdeepmd_c.tar.gz -C ${PREFIX}

cmake -DOPENMM_DIR=${PREFIX} -DDEEPMD_DIR=${PREFIX}/libdeepmd_c ..

make #-j${NUM_CPUS}
make install
make test
make PythonInstall

rm -r ${PREFIX}/libdeepmd_c
