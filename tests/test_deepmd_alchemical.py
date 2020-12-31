from sys import stdout, exit
import sys
import os, platform
import numpy as np
from math import sqrt
import glob, random
from simtk.openmm import app, KcalPerKJ
import simtk.openmm as mm
from simtk.openmm import CustomNonbondedForce
from simtk import unit as u
# OpenMM Imports
import simtk.openmm as mm
import time
import simtk.openmm.app as app
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

from OpenMMDeepmdPlugin import *
from utils import ForceReporter, Simulation4Deepmd



nsteps = 1000000
nstout = 100
alchemical_resid = 1
Lambda = 0.3
box = [20, 0, 0, 0, 20, 0, 0, 0, 20]
mini_Tol = 10 #KJ/mol
mini_nstep = 1000

pdb_file = "./input/lw_pimd_64.pdb"
output_dcd = "./output/lw_pimd_64_alchem_lambda_"+str(Lambda)+".dcd"
output_force_txt = "./output/lw_pimd_64.alchem."+str(Lambda)+".force.txt"
show_force = False
used4Alchemical = True

# This model is trained by deepmd-kit 1.2.0
model_file = "./frozen_model/lw_pimd.v1.pb"

print("nsteps:", nsteps, " Resid: ", alchemical_resid, " Lambda: ", Lambda)
print(pdb_file)
print(output_dcd)
print(show_force, output_force_txt)
print(model_file)
print("---------------------------------------")
print("\n\n")


# Load the system initial information from pdb.
lw_pdb = PDBFile(pdb_file)
topology = lw_pdb.topology
positions = lw_pdb.getPositions()
natoms = topology.getNumAtoms()

randomSeed = random.randint(0, natoms)
# Transform the units from angstrom to nanometers
box = [mm.Vec3(box[0], box[1], box[2]), mm.Vec3(box[3], box[4], box[5]), mm.Vec3(box[6], box[7], box[8])] * u.angstroms
box = [x.value_in_unit(u.nanometers) for x in box]

# Create the system.
dp_system = mm.System()
integrator = mm.LangevinIntegrator(
                        300*u.kelvin,       # Temperature of heat bath
                        5.0/u.picoseconds,  # Friction coefficient
                        1*u.femtoseconds, # Time step
)
platform = mm.Platform.getPlatformByName('CUDA')

# Set the dp force for alchemical simulation.
dp_force = DeepmdForce(model_file, model_file, model_file, used4Alchemical)
dp_force.addType(0, element.oxygen.symbol)
dp_force.addType(1, element.hydrogen.symbol)

nHydrogen = 0
nOxygen = 0

graph1_particles = []
graph2_particles = []

for atom in topology.atoms():
    if int(atom.residue.id) == alchemical_resid:
        graph2_particles.append(atom.index)
    else:
        graph1_particles.append(atom.index)

    if atom.element == element.oxygen:
        dp_system.addParticle(element.oxygen.mass)
        dp_force.addParticle(atom.index, element.oxygen.symbol)
        nOxygen += 1
        '''
        for at in atom.residue.atoms():
            if at.element == element.oxygen:
                continue
            lw_pdb.topology.addBond(atom, at)
        '''
    elif atom.element == element.hydrogen:
        dp_system.addParticle(element.hydrogen.mass)
        dp_force.addParticle(atom.index, element.hydrogen.symbol)
        nHydrogen += 1

dp_force.setAtomsIndex4Graph1(graph1_particles)
dp_force.setAtomsIndex4Graph2(graph2_particles)
dp_force.setLambda(Lambda)


print(len(graph1_particles), len(graph2_particles))

dp_force.setForceGroup(1)
# Add force in dp_system
dp_system.addForce(dp_force)
# Add Barostat for NPT ensemble simulation.
dp_system.addForce(MonteCarloBarostat(1*u.atmospheres, 300*u.kelvin))


num4forces = dp_system.getNumForces()
print(num4forces, " forces totally.")
for ii in range(num4forces):
    print(dp_system.getForce(ii))

force_reporter_1 = ForceReporter(output_force_txt, 1, nstout)

simulation = Simulation4Deepmd(topology, dp_system, integrator, platform)

simulation.context.setPeriodicBoxVectors(box[0], box[1], box[2])
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(300*u.kelvin, randomSeed)

# Add reporter.
simulation.reporters.append(app.DCDReporter(output_dcd, nstout))
if show_force:
    simulation.reporters.append(force_reporter_1)
simulation.reporters.append(
        StateDataReporter(sys.stdout, nstout, step=True, time=True, potentialEnergy=True, temperature=True, progress=True,
                          remainingTime=True, speed=True, totalSteps=nsteps, separator='\t')
    )

# Run dynamics
print('Running dynamics')
start_time = time.time()
simulation.step(nsteps)
end_time = time.time()
cost_time = end_time - start_time
print(platform.getName(),"%.4f s" % cost_time)

