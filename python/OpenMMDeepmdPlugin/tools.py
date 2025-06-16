from __future__ import absolute_import
try:
    from openmm import app, KcalPerKJ
    import openmm as mm
    from openmm import unit as u
    from openmm.app import *
    import openmm.unit as unit
except:
    from simtk import unit as u
    import simtk.openmm as mm
    from simtk.openmm.app import *
    import simtk.openmm as mm
    import simtk.unit as unit
    
import sys
from datetime import datetime, timedelta
try:
    import matplotlib.pyplot as plt
except:
    print("matplotlib is not installed.")
import numpy as np

try:
    string_types = (unicode, str)
except NameError:
    string_types = (str,)

from .OpenMMDeepmdPlugin import DeepmdForce

class ForceReporter(object):
    def __init__(self, file, group_num, reportInterval):
        self.group_num = group_num
        if self.group_num is None:
            self._out = open(file, 'w')
            #self._out.write("Get the forces of all components"+"\n")
        else:
            self._out = open(file, 'w')
            #self._out.write("Get the forces of group "+str(self.group_num) + "\n") 
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        # return (steps, positions, velocities, forces, energies)
        return (steps, False, False, True, False)

    def report(self, simulation, state):
        if self.group_num is not None:
            state = simulation.context.getState(getForces=True, groups={self.group_num})
        else:
            state = simulation.context.getState(getForces=True)
        forces = state.getForces().value_in_unit(u.kilojoules_per_mole/u.nanometers)
        self._out.write(str(forces)+"\n")



def DrawScatter(x, y, name, xlabel="Time", ylabel="Force, unit is KJ/(mol*nm)", withLine = True, fitting = False):
    plt.clf()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(name)

    color_list = ['r', 'g', 'b']

    if len(y.shape) > 1 and y.shape[1] != 0:
        for ii, y_row in enumerate(y):
            plt.scatter(x, y_row, c=color_list[ii], alpha=0.5)
            if withLine:
                plt.plot(x, y_row)    
    else:
        plt.scatter(x, y, c='b', alpha=0.5)
        if withLine:
            plt.plot(x, y)
    if fitting:
        coef, bias = np.polyfit(x, y, 1)
        min_x = min(x)
        max_x = max(x)
        fitting_x = np.linspace(min_x, max_x, 500)
        fitting_y = coef * fitting_x + bias
        plt.plot(fitting_x, fitting_y, '-r')

    plt.savefig("./output/"+name+'.png')
    return



class DeepPotentialModel():
    def __init__(self, model_file, LambdaName = "dp_alchem_lambda", Lambda = 1.0) -> None:
        """Initialize the Deep Potential model.

        Args:
            model_file (string): Path to the Deep Potential model file.
            Lambda (float, optional): Weight of the Deep Potential models in use. Output forces and energy from the Deep Potential model will be multiplied by Lambda and return to the OpenMM context. Defaults to 1.0.
        """
        self.model_file = model_file
        self.dp_force = DeepmdForce(model_file)
        self.cutoff = self.dp_force.getCutoff()
        self.numb_types = self.dp_force.getNumberTypes()
        self.type_map_raw = self.dp_force.getTypesMap()
        self.type_map_dict, self.dp_model_types = self.__decode_type_map(self.type_map_raw)
    
        self.dp_force.addLambdaParameter(LambdaName, Lambda)
        self.lambda_name = LambdaName
        
        # Set up the atom type
        for atom_type in self.type_map_dict.keys():
            self.dp_force.addType(self.type_map_dict[atom_type], atom_type)
        
        return
    
    def __decode_type_map(self, type_map_string):
        type_map_dict = dict()
        type_list = type_map_string.split()
        for ii, atom_type in enumerate(type_list):
            type_map_dict[atom_type] = ii
        dp_model_types = list(type_map_dict.keys())
        
        assert len(dp_model_types) == self.numb_types, "Number of types is not consistent with numb_types from dp model"
        
        return type_map_dict, dp_model_types
    
    def setGPURank(self, rank = 0):
        """Specify the GPU device to run DP model. Default is 0.

        Args:
            rank (int, optional): Rank id of the GPU device to run DP model. Defaults to 0.
        """
        self.dp_force.setGPURank(rank)
        return
    
    def setUnitTransformCoefficients(self, coordinatesCoefficient, forceCoefficient, energyCoefficient):
        """ Set the coefficients for transforming the units of the DP-predicted forces and energies to the units used by OpenMM.

        Args:
            coordinatesCoefficient (float): Coefficient for input coordinates that transforms the units of the coordinates from nanometers to the units used by the DP model.
            forceCoefficient (float): Coefficient for forces that transforms the units of the DP-predicted forces from the units used by the DP model to kJ/(mol * nm).
            energyCoefficient (float): Coefficient for energies that transforms the units of the DP-predicted energy from the units used by the DP model to kJ/mol.
        """
        self.dp_force.setUnitTransformCoefficients(coordinatesCoefficient, forceCoefficient, energyCoefficient)
            
        return
    
    def createSystem(self, topology, particleNameLabeler = "element"):
        """Create a OpenMM system with Deep Potential force.
            Used for conventional MD simulation. (NVT, NPT, NVE simulations with the DP model only etc.)

        Args:
            topology (Topology): OpenMM Topology object
            particleNameLabeler (str, optional): Labeler of the atom type in topology, element or atom_name. Defaults to "element".
        
        """
        dp_system = mm.System()
        
        # Add particles into force.
        for atom in topology.atoms():
            if particleNameLabeler == "element":
                atom_type = atom.element.symbol
            elif particleNameLabeler == "atom_name":
                atom_type = atom.name
            if atom_type not in self.dp_model_types:
                raise Exception(f"Atom type {atom_type} is not found in {self.dp_model_types}.")
            
            dp_system.addParticle(atom.element.mass)
            self.dp_force.addParticle(atom.index, atom_type)
            
        
        # Add bond information into DeepmdForce for the PBC issue 
        # during the trajectory saving. For visualization, the bond information is not necessary.
        for bond in topology.bonds():
            self.dp_force.addBond(bond[0].index, bond[1].index)
        
        dp_system.addForce(self.dp_force)
        return dp_system
    
    def addParticlesToDPRegion(self, dp_particles, topology, particleNameLabeler = "element"):
        """Add particles into DP region.
            Only the particles in the DP region will be used to calculate the DP force and energy.

        Args:
            particles (List): list of particle index
            topology (Topology): OpenMM Topology object
            particleNameLabeler (str, optional): labeler of atom type in topology, element or atom_name. Defaults to "element".
        """
        for atom in topology.atoms():
            if atom.index in dp_particles:
                if particleNameLabeler == "element":
                    atom_type = atom.element.symbol
                elif particleNameLabeler == "atom_name":
                    atom_type = atom.name
                self.dp_force.addParticle(atom.index, atom_type)

        return self.dp_force
    
    def addCenterParticlesToAdaptiveDPRegion(
        self, 
        center_particles, 
        topology, 
        sel_num4each_type = None,
        radius = 0.35, 
        atom_names_to_add_forces = None,
        extend_residues = True
        ):
        """Add center particles into adaptive DP region.
            Put the required information for adaptive DP region into the DeepmdForce object.

        Args:
            center_particles (List): list of particle index
            topology (Topology): OpenMM Topology object
            sel_num4each_type (List, optional): list of number of particles for each type. Defaults to None.
            radius (float, optional): radius of the adaptive region. Defaults to 0.35 nanometers.
            atom_names_to_add_forces (List): list of atom names to add forces.
            
        """
        self.dp_force.setAdaptiveRegion(True)
        # Feed the topology information to the DeepmdForce object.
        for chain in topology.chains():
            chainIndex = chain.index
            chainId = chain.id
            
            if chainId.isnumeric():
                self.dp_force.addChain(chainIndex, chainId)
            else:
                chainId = chainIndex
                self.dp_force.addChain(chainIndex, chainId)
            
            residues = chain._residues
            for res in residues:
                resId = int(res.id)
                resIndex = res.index
                resName = str(res.name)
                
                self.dp_force.addResidue(chainIndex, resName, resIndex, resId)
                atoms = res._atoms
                for at in atoms:
                    atomIndex = at.index
                    atomId = int(at.id)
                    atomName = str(at.name)
                    atomElement = at.element._symbol
                    
                    # Small patch for Zn
                    if atomElement == "Zn":
                        atomElement = "ZN"
                    # Small patch for oxygen name in water molecule.
                    if res.name == "HOH" and atomName == "O":
                        atomName = "OH2"
                    
                    self.dp_force.addAtom(resIndex, atomName, atomElement, atomIndex, atomId)
                        
        # add Center atoms into the DeepmdForce object.
        self.dp_force.setCenterAtoms(center_particles)
        self.dp_force.setRegionRadius(radius)
        if atom_names_to_add_forces is not None:
            self.dp_force.setAtomNames4DPForces(atom_names_to_add_forces)
        
        if sel_num4each_type is not None:
            num4type = []
            for type_name in self.dp_model_types:
                num4type.append(sel_num4each_type[type_name])
            self.dp_force.setSelNum4EachType(self.dp_model_types, num4type)        

        return self.dp_force
    
    def setLambdaName(lambda_name):
        """Set the name of the lambda parameter for the DP model.

        Args:
            lambda_name (str): Name of the lambda parameter.
        """
        self.lambda_name = lambda_name
        self.dp_force.setLambdaName(lambda_name)
        assert self.dp_force.getLambdaName() == lambda_name, "Failed to set the lambda name."
        return