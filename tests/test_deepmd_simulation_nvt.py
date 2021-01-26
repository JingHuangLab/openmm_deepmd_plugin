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
from simtk.openmm.openmm import AndersenThermostat
from simtk.unit import *
#from openmmtools.integrators import VelocityVerletIntegrator
import argparse
from math import floor, ceil

from OpenMMDeepmdPlugin import *
from utils import ForceReporter, Simulation4Deepmd, DrawScatter


parser = argparse.ArgumentParser()
parser.add_argument('-m','--mole', dest='mole', help='Molecule name, used for pdb select.', required=True)
parser.add_argument('-n', '--nsteps', type = int, dest='nsteps', help='Number of steps', default=100000)
parser.add_argument('--dt', type = float, dest='timestep', help='Time step for simulation, unit is femtosecond', default=0.2)
parser.add_argument('--nstout', type = int, dest='nstout', help='Frame steps for saved log.', default=100)
parser.add_argument('--dcd-dt', type = float, dest='dcd_dt', help='dcd file save time gap. Unit is ns. Default to be 0.1ns ', default=0.1)
parser.add_argument('--box', type = float, dest='box', help='Box dimension size for simulation, unit is angstrom', default=19.807884)
parser.add_argument('--output', type = str, dest='output', help='Output directory when write logs and dcd.', default="./output/")
parser.add_argument('--restart', type = bool, dest='restart', help='Restart or not', default=False)
parser.add_argument('--chk', type = str, dest='chk', help='Path to .rst file. Used when restart set to be true.', default="")
parser.add_argument('--temp', type = int, dest='temp', help='Temperature of this simulation.', default=300)

args = parser.parse_args()

mole_name = args.mole
nsteps = args.nsteps
time_step = args.timestep
nstout = args.nstout
output_dir = os.path.join(args.output, mole_name)
restart = args.restart
checkpoint = args.chk
dcd_dt = args.dcd_dt

#nsteps = 150000
#nstout = 100
#time_step = 0.2  # unit is femtosecond.
temp = args.temp # system temperature

# Units is angstrom here.
#box = [20, 0, 0, 0, 20, 0, 0, 0, 20]
#box = [19.807884, 0, 0, 0, 19.807884, 0, 0, 0, 19.807884]
box = [args.box, 0, 0, 0, args.box, 0, 0, 0, args.box]


if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

#mole_name = "lw_packmol_256_10ns_npt_tip3p.npt"
pdb_file = "./input/"+mole_name+".pdb"

output_dcd =  os.path.join(output_dir, mole_name+".nvt." + str(temp))
output_force_txt = os.path.join(output_dir, mole_name+".force.txt")
output_log = os.path.join(output_dir, mole_name+".nvt." + str(temp))
output_state = os.path.join(output_dir, mole_name+".nvt." + str(temp))
output_chk = os.path.join(output_dir, mole_name+".nvt." + str(temp))
output_pdb = os.path.join(output_dir, mole_name+".nvt." + str(temp))


num_dcd = ceil(nsteps / ((dcd_dt / time_step)* 1000000))
dcd_steps = int(nsteps/num_dcd)


tot_index = -5
used4Alchemical = False
show_force = False
loadFromState = False
NPT = False
NVE = False
NVT = True
# Thermostat setting:
Thermostat = "Langevin"
#Thermostat = "Anderson"
#Thermostat = "NoseHoover"

# Integrator setting:
Integrator = "VerletIntegrator"
#Integrator = "VariableVerletIntegrator"
#Integrator = "VelocityVerletIntegrator"

# This model is trained by deepmd-kit 1.2.0
model_file = "./frozen_model/lw_pimd.se_a.pb"

print("nsteps:", nsteps, ". NPT:", NPT, ". NVT:", NVT, ". NVE:", NVE, ". Thermostat:", Thermostat)
print(pdb_file)
print("System Temperature: ", temp)
print("Time Step: ", time_step)
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
    elif Integrator == "VelocityVerletIntegrator":
        print("VelocityVerletIntegrator is used")
        integrator = VelocityVerletIntegrator(time_step*u.femtoseconds)

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

# Set the deepmd compiled op library file path so that we can load it.
dp_force.setDeepmdOpFile("/home/dingye/local/deepmd1.2.0_tf1.14/lib/libdeepmd_op.so")
# Set the units transformation coefficients from openmm to graph input tensors.
# First is the coordinates coefficient, which used for transformation from nanometers to graph needed coordinate unit.
# Second number is force coefficient, which used for transformation graph output force unit to openmm used unit (kJ/(mol * nm))
# Third number is energy coefficient, which used for transformation graph output energy unit to openmm used unit (kJ/mol)
dp_force.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)

print(nHydrogen, nOxygen)

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


for ii in range(num_dcd):
    dcd = output_dcd+ "."+str(ii)+".dcd"
    chk = output_chk + "."+str(ii)+".chk"
    state = output_state + '.'+str(ii)+".state"
    log = output_log + "."+str(ii)+".log"
    pdb = output_pdb + "."+str(ii)+".pdb"

    # Set initial velocities and positions.
    if not restart:
        sim.context.setPositions(lw_pdb.getPositions())
        if not NVE:
            sim.context.setVelocitiesToTemperature(temp*u.kelvin, randomSeed)
    elif restart:
        sim.loadCheckpoint(checkpoint)

    # Add reporter.
    sim.reporters.append(app.DCDReporter(dcd, nstout))
    sim.reporters.append(
            StateDataReporter(log, nstout, step=True, time=True, totalEnergy=True, kineticEnergy=True, potentialEnergy=True, temperature=True, progress=True,
                            remainingTime=True, speed=True,  density=True,totalSteps=nsteps, separator='\t')
        )

    # Run dynamics
    print('Running dynamics of %dth dcd, %d steps, %s, %s, %s, %s'%(ii+1, dcd_steps, dcd, chk, log, state))
    start_time = time.time()
    sim.step(dcd_steps)
    sim.reporters = [] # Clear reporters.
    end_time = time.time()
    cost_time = end_time - start_time
    print(platform.getName(),"%.4f s" % cost_time)
    # Save state and checkpoint.
    sim.saveCheckpoint(chk)
    save_state=sim.context.getState(getPositions=True, getVelocities=True)    
    with open(state, 'w') as f:
        f.write(mm.XmlSerializer.serialize(save_state))
    # Set restart = True here. And update the checkpoint file.
    restart = True
    checkpoint = chk

    position=save_state.getPositions()
    velocity=save_state.getVelocities()
    app.PDBFile.writeFile(sim.topology, position, open(pdb, 'w'))

    with open(log, "r")  as f:
        log_content = f.readlines()

    total_energy = []

    for ii, line in enumerate(log_content):    
        if ii == 0:
            continue
        temp = line.split()
        total_energy.append(float(temp[tot_index]))

    total_energy = np.array(total_energy)
    print("%d th dcd saved at %s: "%(ii+1, dcd))
    print(log, total_energy.shape[0], np.average(total_energy), np.std(total_energy), np.std(total_energy)/natoms)

