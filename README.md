# OpenMM Plugin for Deep Potential Model

This plugin is specifically designed for [OpenMM](http://openmm.org), enabling the integration of the Deep Potential model to define forces. The implementation has been made possible with the [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) C API interface.

To use this plugin, you must create a TensorFlow graph using DeePMD-kit that takes particle positions as input and produces energy and forces as output. During simulation, this plugin utilizes the graph to apply forces to the particles.


## Installation

### Installing from Source
To install this plugin, you will need the **OpenMM** and **DeePMD-kit C API package** libraries. You can compile the plugin from source by following these steps:

1. Prepare a conda environment by running the following commands:
   ```
   conda create -n dp_openmm
   conda activate dp_openmm
   conda install -c conda-forge openmm cudatoolkit=11.6
   ```

2. Download and install the DeePMD-kit C API library by running the following command:
   ```shell
   wget https://github.com/deepmodeling/deepmd-kit/releases/download/v2.2.2/libdeepmd_c.tar.gz
   # Extract the C API library of Deepmd-kit to the directory of your choice.
   tar -xf libdeepmd_c.tar.gz -C ${LIBDEEPMD_C_INSTALLED_DIR}
   ```

3. Clone this repository and create a directory in which to build the plugin.
   ```shell
   git clone https://github.com/JingHuangLab/openmm_deepmd_plugin.git
   cd openmm_deepmd_plugin && mkdir build && cd build
   ```

4. Run `cmake` command with the required parameters.
   ```shell
   cmake .. -DOPENMM_DIR=${OPENMM_INSTALLED_DIR} -DDEEPMD_DIR=${LIBDEEPMD_C_INSTALLED_DIR}
   ```
   `OPENMM_INSTALLED_DIR` is the directory where OpenMM is installed.
   If you installed OpenMM from conda, it is the directory of the location of environment `dp_openmm`.
   `LIBDEEPMD_C_INSTALLED_DIR` is the directory where the DeePMD-kit C API library is installed.

5. Compile the shared library.
   ```shell
   make && make install
   ```
   It will install the plugin to the subdirectory of `OPENMM_DIR` automatically.

6. Test the plugin C++ interface and install the Python module of this plugin into conda environment by running the following commands:
   ```shell
   make test
   make PythonInstall
   python -m OpenMMDeepmdPlugin.tests.test_dp_plugin_nve
   python -m OpenMMDeepmdPlugin.tests.test_dp_plugin_nve --platform CUDA
   ```




## Usage of `DeepPotentialModel` class

To make the plugin more user-friendly, we have created a Python class called `DeepPotentialModel`. 
This class wraps the C++ interface and raw force object `DeepmdForce`, and provides several methods to facilitate the use of the plugin.

### Creating a `DeepPotentialModel` object
To create a `DeepPotentialModel` object, use the following code:

```python
dp_model = DeepPotentialModel(dp_model_file, Lambda=1.0)
```

Here, `dp_model_file` refers to the path of the Deep Potential model file. 
The parameter `Lambda` is used for interpolating the Deep Potential model. 
The output forces and energy values from the Deep Potential model are multiplied by `Lambda` before being added into the OpenMM context. 
By default, the value of `Lambda` is set to 1.0.


### Setting the unit transformation coefficients
To set the unit transformation coefficients, use the following code:

```python
dp_model.setUnitTransformCoefficients(coord_coefficient, force_coefficient, energy_coefficient)
```

In OpenMM, the units for coordinates, forces, and energy are *nanometers*, *kJ/(mol\*nm)*, and *kJ/mol*, respectively. However, the Deep Potential models have their own units, which are determined by the training data. To make the Deep Potential model compatible with OpenMM, three coefficients are needed for transforming the units.

- `coord_coefficient`: This coefficient transforms the input coordinates from the OpenMM context to the Deep Potential model. The values of the coordinates in OpenMM (in nanometers) will be multiplied by `coord_coefficient` as the input values to the Deep Potential model.
- `force_coefficient`: This coefficient transforms the output forces from the Deep Potential model to units that are compatible with OpenMM (i.e., kJ/(mol\*nm)). The output force values will be multiplied by `force_coefficient` and added into the OpenMM context.
- `energy_coefficient`: This coefficient transforms the output energy values from the Deep Potential model to units that are compatible with OpenMM (i.e., kJ/mol). The output energy values will be multiplied by `energy_coefficient` and added into the OpenMM context.


### Creating an OpenMM System object with the Deep Potential model

To create an OpenMM System object with the Deep Potential model, use the following code:

```python
dp_system = dp_model.createSystem(topology)
```

- `topology` is an OpenMM Topology object of the system. 

The returned `dp_system` is an OpenMM System object with the Deep Potential model.

### Passing part of the system to the Deep Potential model

To pass part of the system to the Deep Potential model, use the following code:

```python
dp_force = dp_model.addParticlesToDPRegion(dp_particles, topology, particleNameLabeler="element")
```

- `dp_particles` is a list of particles to be passed to the Deep Potential model (e.g., ligand particles in a protein-ligand system). 
- `topology` is the OpenMM Topology object of the whole system. 
- `particleNameLabeler` is the labeler used to identify the atom type of the input particles in the `topology`. It is optional and can be set to either "element" (default) or "atom_name".

The returned `dp_force` is the `Force` object that can be added to the simulation system.

### Adapting the selection of particles to be passed to the Deep Potential model

To adaptively select the particles to be passed to the Deep Potential model, use the following code:

```python
dp_force = dp_model.addCenterParticlesToAdaptiveDPRegion(center_particles, topology, sel_num4each_type=None, radius=0.35, atom_names_to_add_forces=None, extend_residues=True)
```

Here, `addCenterParticlesToAdaptiveDPRegion` dynamically selects the `center_particles` and their surrounding particles (within a distance less than `radius`) to the Deep Potential models.

- `center_particles` are the center particles of the adaptively selected DP region. 
- `topology` is the OpenMM Topology object of the whole system. 
- `sel_num4each_type` is a list of the maximum number for each particle type. 
- `radius` is the distance (in nanometers) used to select other particles in the adaptive DP region based on their proximity to `center_particles`. 

- `atom_names_to_add_forces` is a list of the atom names that will add DP forces. If it is None or an empty list, all selected particles in the adaptive DP region will have forces added from the DP model (default).

- `extend_residues` is a boolean value indicating whether or not to extend the selected particles to their belonged residues. It is set to True by default.

The returned `dp_force` is the `Force` object that can be added to the simulation system. 


## Application Scenarios

### Conventional Simulations with Deep Potential

To construct a DP simulation with OpenMM, use the following three lines of code:

```python
dp_model = DeepPotentialModel(dp_model_file)
dp_model.setUnitTransformCoefficients(coord_coeffi, force_coeffi, energy_coeffi)
dp_system = dp_model.createSystem(topology)
```

Here, `dp_model_file` is the path to the Deep Potential model file. `coord_coeffi`, `force_coeffi`, and `energy_coeffi` are the coefficients for unit transformation between the DP model and OpenMM. `topology` is the OpenMM Topology object of the whole system.

For more practical examples, refer to the provided simulation scripts:
- [test_deepmd_nvt.py](./python/tests/test_deepmd_nvt.py)
- [test_deepmd_nve.py](./python/tests/test_deepmd_nve.py)
- [test_deepmd_npt.py](./python/tests/test_deepmd_npt.py)

### DP/MM or DP+MM Simulations

To perform DP/MM simulations with fixed or adaptive DP regions, use the methods `addParticlesToDPRegion` and `addCenterParticlesToAdaptiveDPRegion`.

For example, to include ligand intramolecular interactions governed by the DP models, use `addParticlesToDPRegion`. 
If we select the ligand particles as the center particles in `addCenterParticlesToAdaptiveDPRegion`, 
the adaptively selected DP region could include protein residues and water molecules that directly interacted with the ligand.

For more practical usage of these two methods, refer to the provided simulation scripts:
- [test_deepmd_dp_region.py](./python/tests/test_deepmd_dp_region.py)
- [test_deepmd_dp_adaptive_region.py](./python/tests/test_deepmd_dp_adaptive_region.py)
- [test_deepmd_multi_dp_adaptive_region.py](./python/tests/test_deepmd_multi_dp_adaptive_region.py)

### Alchemical Simulations with Deep Potential

To perform alchemical simulations with the DP models based, combine `lambda` and `addParticlesToDPRegion`. 
The alchemical simulations protocol with the DP models is described in `AlchemicalProtocol.pdf`.

For an example script about alchemical simulation for water's hydration-free energy calculation, refer to [test_deepmd_alchemical.py](./python/tests/test_deepmd_alchemical.py).