# OpenMM Plugin for Deepmd-kit


This is a plugin for [OpenMM](http://openmm.org) that allows DeepPotential model
to be used for defining forces.  It is implemented with [Deepmd-kit](https://github.com/deepmodeling/deepmd-kit) and [TensorFlow](https://www.tensorflow.org/).
To use it, you create a TensorFlow graph with Deepmd-kit that takes particle positions as input
and produces forces and energy as output. This plugin uses the graph to apply
forces to particles during a simulation.

## Installation

This plugin requires the c++ library of **OpenMM, v7.5**, **Tensorflow, v1.14**, **Deepmd-kit, v1.2.0**. It uses CMake as its build tool. CUDA is also needed for this plugin. This plugin support **CUDA, Reference** platform for now.
And then compile this plugin with steps below.

1. Clone this repository and create a directory in which to build the plugin.
   ```
   git clone https://github.com/dingye18/openmm_deepmd_plugin.git
   cd openmm_deepmd_plugin && mkdir build && cd build
   ```
2. Run `cmake` command with required parameters.
   ```
   cmake .. -DOPENMM_DIR={OPENMM_INSTALLED_DIR} -DDEEPMD_DIR={DEEPMD_INSTALLED_DIR} -DTENSORFLOW_DIR=${TENSORFLOW_DIR}
   ```
   You can also specify the CUDA platform with `-DCUDA_TOOLKIT_ROOT_DIR={CUDA_DIR}`.
   The default value for `OPENMM_DIR`, `DEEPMD_DIR`, `TENSORFLOW_DIR` are `/usr/local/openmm/`, `/usr/local/deepmd`, `/usr/local/tensorflow` respectively. 
3. Compile the shared library with command `make` running in `build` directory.
   ```
   make && sudo make install
   ```
   It will install the plugin to the subdirectory of `OPENMM_DIR` automatically.
4. Compile the Python interface of this plugin with
   ```
   make PythonInstall
   ```
   Attention that running of this plugin with python need OpenMM python library to be installed first.

## Usage

In the [tests](./tests) directory, you can find [test_deepmd_simulation.py](./tests/test_deepmd_simulation.py) and [test_deepmd_alchemical.py](./tests/test_deepmd_alchemical.py) two files for reference.
That's used for running of this plugin with on trained [water model](./tests/frozen_model/lw_pimd.v1.pb).
Alchemical simulation feature for Deepmd-kit is also implement in this plugin. More details about the alchemical simulation can be refered to [AlchemicalProtocol.pdf](./tests/refer/AlchemicalProtocol.pdf).

## Problem to Be Solved

**Energy not conserved with NVE simulation......**