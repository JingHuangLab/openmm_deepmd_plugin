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
import argparse
from OpenMMDeepmdPlugin import *
from utils import ForceReporter, Simulation4Deepmd, DrawScatter


parser = argparse.ArgumentParser()
parser.add_argument('-m','--mole', dest='mole', help='Molecule name, used for pdb select.', required=True)
parser.add_argument('-n', '--nsteps', type = int, dest='nsteps', help='Number of steps', default=100000)
parser.add_argument('--dt', type = float, dest='timestep', help='Time step for simulation, unit is femtosecond', default=0.2)
parser.add_argument('--nstout', type = int, dest='nstout', help='Frame steps for saved log.', default=100)
parser.add_argument('--box', type = float, dest='box', help='Box dimension size for simulation, unit is angstrom', default=19.807884)
parser.add_argument('--state', type = str, dest='state', help='Initial state file for simulation', default="")
parser.add_argument('--graph', type = str, dest='graph', help='Trained TF graph, used for simulation.', default="./frozen_model/graph_from_han_dp2.0_compress.pb")

args = parser.parse_args()

mole_name = args.mole
nsteps = args.nsteps
time_step = args.timestep
nstout = args.nstout

#nsteps = 150000
#nstout = 100
#time_step = 0.2  # unit is femtosecond.
temp = 300 # system temperature

# Units is angstrom here.
#box = [20, 0, 0, 0, 20, 0, 0, 0, 20]
#box = [19.807884, 0, 0, 0, 19.807884, 0, 0, 0, 19.807884]
box = [args.box, 0, 0, 0, args.box, 0, 0, 0, args.box]

if not os.path.exists("output"):
    os.mkdir("./output")
pdb_file = "./input/"+mole_name+".pdb"
output_dcd = "./output/"+mole_name+".nve.dcd"
output_force_txt = "./output/"+mole_name+".force.txt"
output_log = "./output/"+mole_name+".nve.log"
tot_index = -5
used4Alchemical = False
show_force = False
loadFromState = False 
NPT = False
NVE = True
NVT = False
# Thermostat setting:
Thermostat = "Langevin"
#Thermostat = "Anderson"
#Thermostat = "NoseHoover"

# Integrator setting:
Integrator = "VerletIntegrator"
#Integrator = "VariableVerletIntegrator"
#Integrator = "VelocityVerletIntegrator"

model_file = args.graph
state_file = args.state

print("nsteps:", nsteps, ". NPT:", NPT, ". NVT:", NVT, ". NVE:", NVE, ". Thermostat:", Thermostat)
print(pdb_file)
print("System Temperature: %.2f kelvin", temp)
print("Time Step: %.2f fs"%time_step)
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

print("Box setting (unit is nanometers):", box)
# Create the system.
dp_system = mm.System()
# Set the thermostat or Integrator here.
integrator = None
if NPT or NVT:
    if Thermostat == "Langevin":
        integrator = mm.LangevinIntegrator(
                                temp*u.kelvin,       # Temperature of heat bath
                                1.0/u.picoseconds,  # Friction coefficient
                                time_step*u.femtoseconds, # Time step
        )
    elif Thermostat == "NoseHoover":
        integrator = mm.NoseHooverIntegrator(
                                temp*u.kelvin,       # Temperature of heat bath
                                1.0/u.picoseconds,  # Collision Frequency
                                time_step*u.femtoseconds, # Time step
        )
    elif Thermostat == "Anderson":
        integrator = mm.VerletIntegrator(time_step*u.femtoseconds)
elif NVE:
    if Integrator == "VerletIntegrator":
        print("VerletIntegrator is used")
        integrator = mm.VerletIntegrator(time_step*u.femtoseconds)
    elif Integrator == "VariableVerletIntegrator":
        print("VariableVerletIntegrator is used")
        integrator = mm.VariableVerletIntegrator(0.000001)

# Get platform
platform = mm.Platform.getPlatformByName('CUDA')

# Initialize the deepmd force and add type here.
dp_force = DeepmdForce(model_file, "", "", used4Alchemical)
dp_force.addType(0, element.oxygen.symbol)
dp_force.addType(1, element.hydrogen.symbol)

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

# Set the units transformation coefficients from openmm to graph input tensors.
# First is the coordinates coefficient, which used for transformation from nanometers to graph needed coordinate unit.
# Second number is force coefficient, which used for transformation graph output force unit to openmm used unit (kJ/(mol * nm))
# Third number is energy coefficient, which used for transformation graph output energy unit to openmm used unit (kJ/mol)
dp_force.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)

#dp_force.setForceGroup(1)
# Add force in dp_system
dp_system.addForce(dp_force)
# Runnning the NPT ensemble
if NPT:
    dp_system.addForce(MonteCarloBarostat(1*u.atmospheres, temp*u.kelvin))
if NVT:
    # Setttings for Anderson thermostat.
    if Thermostat == "Anderson":
        dp_system.addForce(AndersenThermostat(temp*u.kelvin, 1.0/u.picoseconds))

num4forces = dp_system.getNumForces()
print(num4forces, " forces totally.")
for ii in range(num4forces):
    print(dp_system.getForce(ii))

print(dp_system.usesPeriodicBoundaryConditions())

sim = Simulation4Deepmd(topology, dp_system, integrator, platform)

sim.context.setPeriodicBoxVectors(box[0], box[1], box[2])

sim.context.setPositions(lw_pdb.getPositions())
if not NVE:
    sim.context.setVelocitiesToTemperature(temp*u.kelvin, randomSeed)

# Add reporter.
sim.reporters.append(app.DCDReporter(output_dcd, nstout, enforcePeriodicBox=False))
sim.reporters.append(
        StateDataReporter(output_log, nstout, step=True, time=True, totalEnergy=True, kineticEnergy=True, potentialEnergy=True, temperature=True, progress=True,
                          remainingTime=True, speed=True,  density=True,totalSteps=nsteps, separator='\t')
    )

if show_force:
    force_reporter_1 = ForceReporter(output_force_txt, None, nstout)
    sim.reporters.append(force_reporter_1)

if loadFromState:
    # Read the state from MPID simulation.
    with open(state_file, "r") as f:
        load_state = mm.XmlSerializer.deserialize(f.read())
    sim.context.setPositions(load_state.getPositions())
    sim.context.setVelocities(load_state.getVelocities())
else:
    sim.context.setPositions(lw_pdb.getPositions())
    #if not NVE:
    sim.context.setVelocitiesToTemperature(temp*u.kelvin, randomSeed)

# Run dynamics
print('Running dynamics')
start_time = time.time()
sim.step(nsteps)
end_time = time.time()
cost_time = end_time - start_time
print(platform.getName(),"%.4f s" % cost_time)

with open(output_log, "r")  as f:
    log_content = f.readlines()

total_energy = []

for ii, line in enumerate(log_content):    
    if ii == 0:
        continue
    temp = line.split()
    total_energy.append(float(temp[tot_index]))

total_energy = np.array(total_energy)
print(output_log, total_energy.shape[0], np.average(total_energy), np.std(total_energy), np.std(total_energy)/natoms)

