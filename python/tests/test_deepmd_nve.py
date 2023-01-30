from sys import stdout, exit
import sys
import os, platform
import numpy as np
from math import sqrt
import glob, random
try:
    import openmm as mm
    from openmm import unit as u
    from openmm.app import PDBFile, StateDataReporter, DCDReporter, element
except:
    import simtk.openmm as mm
    from simtk import unit as u
    from simtk.openmm.app import PDBFile, StateDataReporter, DCDReporter, element

import time

from OpenMMDeepmdPlugin import DeepmdForce
from OpenMMDeepmdPlugin import ForceReporter, Simulation4Deepmd


output_temp_dir = "/tmp/openmm_deepmd_plugin_test_output"
energy_std_tol = 0.0005 # unit is kJ/mol


def test_deepmd_nve_reference():
    pdb_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "lw_256_test.pdb")
    output_dcd = os.path.join(output_temp_dir, "lw_256_test.reference.nve.dcd")
    output_log = os.path.join(output_temp_dir, "lw_256_test.reference.nve.log")
    dp_model = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "water.pb")
    
    time_step = 0.2 # unit is femtosecond.
    report_frequency = 10
    nsteps = 100    
    platform_name = "Reference"
    box = [19.807884, 0, 0, 0, 19.807884, 0, 0, 0, 19.807884]
    box = [mm.Vec3(box[0], box[1], box[2]), mm.Vec3(box[3], box[4], box[5]), mm.Vec3(box[6], box[7], box[8])] * u.angstroms
    
    liquid_water = PDBFile(pdb_file)
    topology = liquid_water.topology
    positions = liquid_water.getPositions()
    num_atoms = topology.getNumAtoms()
    
    dp_system = mm.System()
    integrator = mm.VerletIntegrator(time_step*u.femtoseconds)
    platform = mm.Platform.getPlatformByName(platform_name)
    
    dp_force = DeepmdForce(dp_model)
    dp_force.addType(0, element.oxygen.symbol)
    dp_force.addType(0, element.hydrogen.symbol)
    
    nHydrogen = 0
    nOxygen = 0
    # Add particles into force.
    for atom in topology.atoms():
        if atom.element == element.oxygen:
            dp_system.addParticle(element.oxygen.mass)
            dp_force.addParticle(atom.index, element.oxygen.symbol)
            nOxygen += 1
            for at in atom.residue.atoms():
                topology.addBond(at, atom)
                if at.index != atom.index:
                    dp_force.addBond(atom.index, at.index)
        elif atom.element == element.hydrogen:
            dp_system.addParticle(element.hydrogen.mass)
            dp_force.addParticle(atom.index, element.hydrogen.symbol)
            nHydrogen += 1
    
    dp_force.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
    dp_system.addForce(dp_force)
    
    sim = Simulation4Deepmd(topology, dp_system, integrator, platform)
    sim.context.setPeriodicBoxVectors(box[0], box[1], box[2])
    sim.context.setPositions(positions)
    
    sim.reporters.append(DCDReporter(output_dcd, report_frequency, enforcePeriodicBox=False))
    sim.reporters.append(
        StateDataReporter(output_log, report_frequency, step=True, time=True, totalEnergy=True, kineticEnergy=True, potentialEnergy=True, temperature=True, progress=True,
                          remainingTime=True, speed=True,  density=True,totalSteps=nsteps, separator='\t')
    )
    
    # Run dynamics
    print("Running dynamics")
    start_time = time.time()
    sim.step(nsteps)
    end_time = time.time()
    cost_time = end_time - start_time
    print("Running on %s platform, time cost: %.4f s"%(platform_name, cost_time))
    
    total_energy = []
    tot_energy_index = -5
    with open(output_log, "r") as f:
        log_content = f.readlines()
    for ii , line in enumerate(log_content):
        if ii == 0:
            continue
        temp = line.split()
        total_energy.append(float(temp[tot_energy_index]))
    total_energy = np.array(total_energy)
    
    # Check the total energy fluctuations over # of atoms is smaller than energy_std_tol. 
    assert(np.std(total_energy) / num_atoms < energy_std_tol)
        

def test_deepmd_nve_cuda():
    pdb_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "lw_256_test.pdb")
    output_dcd = os.path.join(output_temp_dir, "lw_256_test.reference.nve.dcd")
    output_log = os.path.join(output_temp_dir, "lw_256_test.reference.nve.log")
    dp_model = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "water.pb")
    
    time_step = 0.2 # unit is femtosecond.
    report_frequency = 10
    nsteps = 100    
    platform_name = "CUDA"
    box = [19.807884, 0, 0, 0, 19.807884, 0, 0, 0, 19.807884]
    box = [mm.Vec3(box[0], box[1], box[2]), mm.Vec3(box[3], box[4], box[5]), mm.Vec3(box[6], box[7], box[8])] * u.angstroms
    
    liquid_water = PDBFile(pdb_file)
    topology = liquid_water.topology
    positions = liquid_water.getPositions()
    num_atoms = topology.getNumAtoms()
    
    dp_system = mm.System()
    integrator = mm.VerletIntegrator(time_step*u.femtoseconds)
    platform = mm.Platform.getPlatformByName(platform_name)
    
    dp_force = DeepmdForce(dp_model)
    dp_force.addType(0, element.oxygen.symbol)
    dp_force.addType(0, element.hydrogen.symbol)
    
    nHydrogen = 0
    nOxygen = 0
    # Add particles into force.
    for atom in topology.atoms():
        if atom.element == element.oxygen:
            dp_system.addParticle(element.oxygen.mass)
            dp_force.addParticle(atom.index, element.oxygen.symbol)
            nOxygen += 1
            for at in atom.residue.atoms():
                topology.addBond(at, atom)
                if at.index != atom.index:
                    dp_force.addBond(atom.index, at.index)
        elif atom.element == element.hydrogen:
            dp_system.addParticle(element.hydrogen.mass)
            dp_force.addParticle(atom.index, element.hydrogen.symbol)
            nHydrogen += 1
    
    dp_force.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
    dp_system.addForce(dp_force)
    
    sim = Simulation4Deepmd(topology, dp_system, integrator, platform)
    sim.context.setPeriodicBoxVectors(box[0], box[1], box[2])
    sim.context.setPositions(positions)
    
    sim.reporters.append(DCDReporter(output_dcd, report_frequency, enforcePeriodicBox=False))
    sim.reporters.append(
        StateDataReporter(output_log, report_frequency, step=True, time=True, totalEnergy=True, kineticEnergy=True, potentialEnergy=True, temperature=True, progress=True,
                          remainingTime=True, speed=True,  density=True,totalSteps=nsteps, separator='\t')
    )
    
    # Run dynamics
    print("Running dynamics")
    start_time = time.time()
    sim.step(nsteps)
    end_time = time.time()
    cost_time = end_time - start_time
    print("Running on %s platform, time cost: %.4f s"%(platform_name, cost_time))
    
    total_energy = []
    tot_energy_index = -5
    with open(output_log, "r") as f:
        log_content = f.readlines()
    for ii , line in enumerate(log_content):
        if ii == 0:
            continue
        temp = line.split()
        total_energy.append(float(temp[tot_energy_index]))
    total_energy = np.array(total_energy)
    
    # Check the total energy fluctuations over # of atoms is smaller than energy_std_tol.
    assert(np.std(total_energy) / num_atoms < energy_std_tol)