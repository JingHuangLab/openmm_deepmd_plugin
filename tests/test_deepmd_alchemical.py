from sys import stdout, exit
import sys
import os, platform
import numpy as np
from math import sqrt, ceil, floor
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
from utils import ForceReporter, Simulation4Deepmd




parser = argparse.ArgumentParser()
parser.add_argument('-m','--mole', dest='mole', help='Molecule name, used for pdb select.', required=True)
parser.add_argument('-n', '--nsteps', type = int, dest='nsteps', help='Number of steps', default=100000)
parser.add_argument('--dt', type = float, dest='timestep', help='Time step for simulation, unit is femtosecond', default=0.2)
parser.add_argument('--nstout', type = int, dest='nstout', help='Frame steps for saved log.', default=100)
parser.add_argument('--temp', type = int, dest='temp', help='Temperature of this simulation.', default=300)
parser.add_argument('--dcd-dt', type = float, dest='dcd_dt', help='dcd file save time gap. Unit is ns. Default to be 0.1ns ', default=0.1)
parser.add_argument('--box', type = float, dest='box', help='Box dimension size for simulation, unit is angstrom', default=19.807884)
parser.add_argument('--output', type = str, dest='output', help='Output directory when write logs and dcd.', default="./output/")
parser.add_argument('--restart', type = bool, dest='restart', help='Restart or not', default=False)
parser.add_argument('--chk', type = str, dest='chk', help='Path to .chk file. Used when restart set to be true.', default="")
parser.add_argument('--dcd-bias', type = int, dest='dcd_bias', help='Bias index for output dcd files when using restart.', default=0)

# These two parameters are related with alchemical simulation.
parser.add_argument('--lambda', type = float, dest='Lambda', help='Lambda state for this simulation.', default=0.1)
parser.add_argument('--resid', type = int, dest='alchemical_resid', help='Residue id for alchemical simulation.', default=1)

args = parser.parse_args()

mole_name = args.mole
# Set the alchemical simulation output directory.
output_dir = os.path.join(args.output, mole_name+".alchem")
nsteps = args.nsteps
nstout = args.nstout
temp = args.temp
timestep = args.timestep
restart = args.restart
checkpoint = args.chk
dcd_dt = args.dcd_dt
dcd_bias = args.dcd_bias

alchemical_resid = args.alchemical_resid
Lambda = args.Lambda
box = [args.box, 0, 0, 0, args.box, 0, 0, 0, args.box]

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exists_ok=True)

num_dcd = ceil(nsteps / ((dcd_dt / timestep)* 1000000))
dcd_steps = int(nsteps/num_dcd)

# Set the input .pdb file
pdb_file = "./input/"+mole+".pdb"

output_dcd = os.path.join(output_dir, mole_name+".lambda."+str(Lambda)+".nvt")
output_force_txt = os.path.join(output_dir, mole_name+".alchem."+str(Lambda)+".nvt")
output_log = os.path.join(output_dir, mole_name+".lambda."+str(Lambda)+".nvt")
output_state = os.path.join(output_dir, mole_name+".lambda."+str(Lambda)+".nvt")
output_chk = os.path.join(output_dir, mole_name+".lambda."+str(Lambda)+".nvt")
output_pdb = os.path.join(output_dir, mole_name+".lambda."+str(Lambda)+".nvt")

show_force = False
used4Alchemical = True
tot_index = -5
used4Alchemical = True
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

# This model is trained by deepmd-kit 1.2.0
model_file = "./frozen_model/lw_pimd.se_a.pb"

print("nsteps:", nsteps, " Resid: ", alchemical_resid, " Lambda: ", Lambda)
print(pdb_file)
print("System Temperature: ", temp)
print("Time Step: ", timestep)
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
# Set the thermostat or Integrator here.
integrator = None
if NPT or NVT:
    if Thermostat == "Langevin":
        integrator = mm.LangevinIntegrator(
                                temp*u.kelvin,       # Temperature of heat bath
                                1.0/u.picoseconds,  # Friction coefficient
                                timestep*u.femtoseconds, # Time step
        )
    elif Thermostat == "NoseHoover":
        integrator = mm.NoseHooverIntegrator(
                                temp*u.kelvin,       # Temperature of heat bath
                                1.0/u.picoseconds,  # Collision Frequency
                                timestep*u.femtoseconds, # Time step
        )
    elif Thermostat == "Anderson":
        integrator = mm.VerletIntegrator(timestep*u.femtoseconds)
elif NVE:
    if Integrator == "VerletIntegrator":
        print("VerletIntegrator is used")
        integrator = mm.VerletIntegrator(timestep*u.femtoseconds)
    elif Integrator == "VariableVerletIntegrator":
        print("VariableVerletIntegrator is used")
        integrator = mm.VariableVerletIntegrator(0.000001)

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
        for at in atom.residue.atoms():
            topology.addBond(at, atom)
            if at.index != atom.index:
                dp_force.addBond(atom.index, at.index)
    elif atom.element == element.hydrogen:
        dp_system.addParticle(element.hydrogen.mass)
        dp_force.addParticle(atom.index, element.hydrogen.symbol)
        nHydrogen += 1

dp_force.setAtomsIndex4Graph1(graph1_particles)
dp_force.setAtomsIndex4Graph2(graph2_particles)
dp_force.setLambda(Lambda)

# Set the deepmd compiled op library file path so that we can load it.
dp_force.setDeepmdOpFile("/home/dingye/local/deepmd1.2.0_tf1.14/lib/libdeepmd_op.so")
# Set the units transformation coefficients from openmm to graph input tensors.
# First is the coordinates coefficient, which used for transformation from nanometers to graph needed coordinate unit.
# Second number is force coefficient, which used for transformation graph output force unit to openmm used unit (kJ/(mol * nm))
# Third number is energy coefficient, which used for transformation graph output energy unit to openmm used unit (kJ/mol)
dp_force.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)

print(len(graph1_particles), len(graph2_particles))

dp_force.setForceGroup(1)
# Add force in dp_system
dp_system.addForce(dp_force)

# Check if running the NPT ensemble
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



simulation = Simulation4Deepmd(topology, dp_system, integrator, platform)
simulation.context.setPeriodicBoxVectors(box[0], box[1], box[2])

for ii in range(num_dcd):
    index = ii + dcd_bias
    dcd = output_dcd+ "."+str(index)+".dcd"
    chk = output_chk + "."+str(index)+".chk"
    state = output_state + '.'+str(index)+".state"
    log = output_log + "."+str(index)+".log"
    pdb = output_pdb + "."+str(index)+".pdb"
    force = output_force_txt + "."+str(index)+".force.txt"

    # Set initial velocities and positions.
    if not restart:
        simulation.context.setPositions(lw_pdb.getPositions())
        if not NVE:
            simulation.context.setVelocitiesToTemperature(temp*u.kelvin, randomSeed)
    elif restart:
        simulation.loadCheckpoint(checkpoint)
    
    # Add reporter.
    simulation.reporters.append(app.DCDReporter(output_dcd, nstout))
    simulation.reporters.append(
            StateDataReporter(log, nstout, step=True, time=True, totalEnergy=True, kineticEnergy=True, potentialEnergy=True, temperature=True, progress=True, remainingTime=True, speed=True,  density=True,totalSteps=nsteps, separator='\t')
        )
    if show_force:
        force_reporter_1 = ForceReporter(force, 1, nstout)
        simulation.reporters.append(force_reporter_1)
    
    # Run dynamics
    print('Running dynamics of %dth dcd, %d steps, %s, %s, %s, %s'%(ii+1, dcd_steps, dcd, chk, log, state))
    start_time = time.time()
    simulation.step(dcd_steps)
    simulation.reporters = [] # Clear reporters.
    end_time = time.time()
    cost_time = end_time - start_time
    print(platform.getName(),"%.4f s" % cost_time)
    # Save state and checkpoint.
    simulation.saveCheckpoint(chk)
    save_state=simulation.context.getState(getPositions=True, getVelocities=True)    
    with open(state, 'w') as f:
        f.write(mm.XmlSerializer.serialize(save_state))
    # Set restart = True here. And update the checkpoint file.
    restart = True
    checkpoint = chk

    position=save_state.getPositions()
    velocity=save_state.getVelocities()
    app.PDBFile.writeFile(simulation.topology, position, open(pdb, 'w'))

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

