#!/bin/bash

mkdir build
cd build

# Install the C API library for DeePMD-kit: libdeepmd_c library
# The library is stored in local machine.
wget https://github.com/deepmodeling/deepmd-kit/releases/latest/download/libdeepmd_c.tar.gz
tar -xf libdeepmd_c.tar.gz -C ${PREFIX}

mkdir -p ${PREFIX}/lib/libdeepmd_c
mkdir -p ${PREFIX}/include/libdeepmd_c
cp -r ${PREFIX}/libdeepmd_c/include/* ${PREFIX}/include/
cp -r ${PREFIX}/libdeepmd_c/lib/* ${PREFIX}/lib/

cmake -DOPENMM_DIR=${PREFIX} -DDEEPMD_DIR=${PREFIX}/libdeepmd_c ..

make #-j${NUM_CPUS}
make install
make test
make PythonInstall

rm -r ${PREFIX}/libdeepmd_c
