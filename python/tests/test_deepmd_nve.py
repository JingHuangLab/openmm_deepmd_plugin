import os
import numpy as np
import time
try:
    import openmm as mm
    from openmm import unit as u
    from openmm.app import PDBFile, StateDataReporter, DCDReporter, Simulation
except:
    import simtk.openmm as mm
    from simtk import unit as u
    from simtk.openmm.app import PDBFile, StateDataReporter, DCDReporter, Simulation
    
from OpenMMDeepmdPlugin import DeepPotentialModel


def test_deepmd_nve_reference(nsteps = 1000, time_step = 0.2, platform_name = "Reference", output_temp_dir = "/tmp/openmm_deepmd_plugin_test_nve_output", energy_std_tol = 0.0005 ):
    if not os.path.exists(output_temp_dir):
        os.mkdir(output_temp_dir)
    
    pdb_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "lw_256_test.pdb")
    dp_model = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "water.pb")
    output_dcd = os.path.join(output_temp_dir, "lw_256_test.nve.reference.dcd")
    output_log = os.path.join(output_temp_dir, "lw_256_test.nve.reference.log")
    
    # Set up the simulation parameters.
    time_step = time_step # unit is femtosecond.
    report_frequency = 100
    nsteps = nsteps
    box = [19.807884, 0, 0, 0, 19.807884, 0, 0, 0, 19.807884]
    box = [mm.Vec3(box[0], box[1], box[2]), mm.Vec3(box[3], box[4], box[5]), mm.Vec3(box[6], box[7], box[8])] * u.angstroms
    
    liquid_water = PDBFile(pdb_file)
    topology = liquid_water.topology
    positions = liquid_water.getPositions()
    num_atoms = topology.getNumAtoms()
    
    # Set up the dp_system with the dp_model.    
    dp_model = DeepPotentialModel(dp_model)
    dp_model.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
    dp_system = dp_model.createSystem(topology)
    
    integrator = mm.VerletIntegrator(time_step*u.femtoseconds)
    platform = mm.Platform.getPlatformByName(platform_name)
    
    # Build up the simulation object.
    sim = Simulation(topology, dp_system, integrator, platform)
    sim.context.setPeriodicBoxVectors(box[0], box[1], box[2])
    sim.context.setPositions(positions)

    # Add state reporters
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
    
    # Fetch the total energy from the log file.
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
    
    # Check the total energy fluctuations over # of atoms is smaller than energy_std_tol, unit in kJ/mol.
    assert(np.std(total_energy) / num_atoms < energy_std_tol)    
    
    
def test_deepmd_nve_cuda(nsteps = 1000, time_step = 0.2, platform_name = "CUDA", output_temp_dir = "/tmp/openmm_deepmd_plugin_test_nve_output", energy_std_tol = 0.0005 ):
    if not os.path.exists(output_temp_dir):
        os.mkdir(output_temp_dir)
    
    pdb_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "lw_256_test.pdb")
    dp_model = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "water.pb")
    output_dcd = os.path.join(output_temp_dir, "lw_256_test.nve.cuda.dcd")
    output_log = os.path.join(output_temp_dir, "lw_256_test.nve.cuda.log")
    
    # Set up the simulation parameters.
    time_step = time_step # unit is femtosecond.
    report_frequency = 100
    nsteps = nsteps
    box = [19.807884, 0, 0, 0, 19.807884, 0, 0, 0, 19.807884]
    box = [mm.Vec3(box[0], box[1], box[2]), mm.Vec3(box[3], box[4], box[5]), mm.Vec3(box[6], box[7], box[8])] * u.angstroms
    
    liquid_water = PDBFile(pdb_file)
    topology = liquid_water.topology
    positions = liquid_water.getPositions()
    num_atoms = topology.getNumAtoms()
    
    # Set up the dp_system with the dp_model.    
    dp_model = DeepPotentialModel(dp_model)
    dp_model.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
    dp_system = dp_model.createSystem(topology)
    
    integrator = mm.VerletIntegrator(time_step*u.femtoseconds)
    platform = mm.Platform.getPlatformByName(platform_name)
    
    # Build up the simulation object.
    sim = Simulation(topology, dp_system, integrator, platform)
    sim.context.setPeriodicBoxVectors(box[0], box[1], box[2])
    sim.context.setPositions(positions)

    # Add state reporters
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
    
    # Fetch the total energy from the log file.
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
    
    # Check the total energy fluctuations over # of atoms is smaller than energy_std_tol, unit in kJ/mol.
    assert(np.std(total_energy) / num_atoms < energy_std_tol)    
