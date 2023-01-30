# OpenMM Plugin for Deepmd-kit


This is a plugin for [OpenMM](http://openmm.org) that allows DeepPotential model
to be used for defining forces. 
It is implemented with [Deepmd-kit](https://github.com/deepmodeling/deepmd-kit).
To use it, you create a TensorFlow graph with Deepmd-kit that takes particle positions as input
and produces forces and energy as output. This plugin uses the graph to apply
forces to particles during a simulation.

## Installation

### Install from source
This plugin requires the library of **OpenMM, v7.6**, **Deepmd-kit C API package, v2.2.0.beta.0**. 
Compile plugin from source with following steps.

1. Prepare the conda environment.
   ```
   conda create -n dp_openmm
   conda activate dp_openmm
   conda install -c conda-forge openmm cudatoolkit=11.6
   ```

2. Download and install the Deepmd-kit C API library.
   ```shell
   wget https://github.com/deepmodeling/deepmd-kit/releases/download/v2.2.0.b0/libdeepmd_c.tar.gz
   # Extract the C API library of Deepmd-kit to the directory of your choice.
   tar -xf libdeepmd_c.tar.gz -C /usr/local/libdeepmd_c 
   ```

3. Clone this repository and create a directory in which to build the plugin.
   ```shell
   git clone https://github.com/JingHuangLab/openmm_deepmd_plugin.git
   cd openmm_deepmd_plugin && mkdir build && cd build
   ```
4. Run `cmake` command with required parameters.
   ```shell
   cmake .. -DOPENMM_DIR=${OPENMM_INSTALLED_DIR} -DDEEPMD_DIR=${LIBDEEPMD_C_INSTALLED_DIR}
   ```
   `OPENMM_INSTALLED_DIR` is the directory where OpenMM is installed.
   If you installed OpenMM from conda, it is the directory of the location of environment `dp_openmm`.
   `LIBDEEPMD_C_INSTALLED_DIR` is the directory where Deepmd-kit C API library is installed.
   For example, if you installed Deepmd-kit C API library to `/usr/local/libdeepmd_c`, 
   then `LIBDEEPMD_C_INSTALLED_DIR` is `/usr/local/libdeepmd_c`.

5. Compile the shared library with command `make` running in `build` directory.
   ```shell
   make && make install
   ```
   It will install the plugin to the subdirectory of `OPENMM_DIR` automatically.

6. Test the plugin C++ interface and compile the Python interface of this plugin with
   ```shell
   make test
   make PythonInstall
   ```
## Usage

In the [tests](./tests) directory, you can find [test_deepmd_simulation.py](./tests/test_deepmd_simulation.py) and [test_deepmd_alchemical.py](./tests/test_deepmd_alchemical.py) two files for reference.
That's used for running of this plugin with on trained [water model](./tests/frozen_model/water.pb).
Alchemical simulation feature for Deepmd-kit is also implemented in this plugin. 
More details about the alchemical simulation can be refered to [AlchemicalProtocol.pdf](./tests/refer/AlchemicalProtocol.pdf).
