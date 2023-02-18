import os
import time
import numpy as np
try:
    import openmm as mm
    from openmm import unit as u
    from openmm.app import PDBFile, StateDataReporter, DCDReporter, Simulation
except:
    import simtk.openmm as mm
    from simtk import unit as u
    from simtk.openmm.app import PDBFile, StateDataReporter, DCDReporter, Simulation
    
from OpenMMDeepmdPlugin import DeepPotentialModel



def test_deepmd_alchemical_reference(nsteps = 1000, time_step = 0.2, Lambda = 0.5, platform_name = "Reference", output_temp_dir = "/tmp/openmm_deepmd_plugin_test_alchemical_output", temperature_std_tol = 25):
    if not os.path.exists(output_temp_dir):
        os.mkdir(output_temp_dir)
    
    pdb_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "lw_256_test.pdb")
    dp_model_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "water.pb")
    output_dcd = os.path.join(output_temp_dir, f"lw_256_test.alchemical.reference.{Lambda}.dcd")
    output_log = os.path.join(output_temp_dir, f"lw_256_test.alchemical.reference.{Lambda}.log")
    
    # Set up the simulation parameters.
    nsteps = nsteps
    time_step = time_step # unit is femtosecond.
    temperature = 300 # Kelvin
    report_frequency = 100 # Save trajectory every report_frequency steps.
    box = [19.807884, 0, 0, 0, 19.807884, 0, 0, 0, 19.807884]
    box = [mm.Vec3(box[0], box[1], box[2]), mm.Vec3(box[3], box[4], box[5]), mm.Vec3(box[6], box[7], box[8])] * u.angstroms
    
    liquid_water = PDBFile(pdb_file)
    topology = liquid_water.topology
    positions = liquid_water.getPositions()
    
    # Set up the dp_system with the dp_model.    
    dp_model = DeepPotentialModel(dp_model_file, Lambda = Lambda)
    dp_model.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
    # By default, createSystem from dp_model will put all atoms in topology into the DP particles for dp_model.
    dp_system = dp_model.createSystem(topology)
    
    # Initial the other two dp_models for alchemical simulation.
    dp_model_1 = DeepPotentialModel(dp_model_file, Lambda = 1 - Lambda)
    dp_model_2 = DeepPotentialModel(dp_model_file, Lambda = 1 - Lambda)
    dp_model_1.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
    dp_model_2.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
    
    # Split the system particles into two groups for alchemical simulation.
    graph1_particles = []
    graph2_particles = []
    for atom in topology.atoms():
        if int(atom.residue.id) == 1:
            graph2_particles.append(atom.index)
        else:
            graph1_particles.append(atom.index)
    dp_force_1 =  dp_model_1.addParticlesToDPRegion(graph1_particles, topology)
    dp_force_2 = dp_model_2.addParticlesToDPRegion(graph2_particles, topology)
    
    # Add the two dp_forces to the dp_system.
    dp_system.addForce(dp_force_1)
    dp_system.addForce(dp_force_2)    
    
    integrator = mm.LangevinIntegrator(
                                temperature*u.kelvin, # Temperature of heat bath
                                1.0/u.picoseconds, # Friction coefficient
                                time_step*u.femtoseconds, # Time step
        )
    platform = mm.Platform.getPlatformByName(platform_name)
    
    # Build up the simulation object.
    sim = Simulation(topology, dp_system, integrator, platform)
    sim.context.setPeriodicBoxVectors(box[0], box[1], box[2])
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(temperature * u.kelvin)

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
    
    # Fetch the temperature info from the log file.
    temperature_trajectory = []
    temperature_index = -4
    with open(output_log, "r") as f:
        log_content = f.readlines()
    for ii , line in enumerate(log_content):
        if ii == 0:
            continue
        tmp = line.split()
        temperature_trajectory.append(float(tmp[temperature_index]))
    temperature_trajectory = np.array(temperature_trajectory)
    
    # Check the temperature fluctuations is smaller than temperature_std_tol, unit in kelvin.
    assert(np.std(temperature_trajectory) < temperature_std_tol)
    
    
def test_deepmd_alchemical_cuda(nsteps = 1000, time_step = 0.2, Lambda = 0.5, platform_name = "CUDA", output_temp_dir = "/tmp/openmm_deepmd_plugin_test_alchemical_output", temperature_std_tol = 25):
    if not os.path.exists(output_temp_dir):
        os.mkdir(output_temp_dir)
    
    pdb_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "lw_256_test.pdb")
    dp_model_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "water.pb")
    output_dcd = os.path.join(output_temp_dir, f"lw_256_test.alchemical.cuda.{Lambda}.dcd")
    output_log = os.path.join(output_temp_dir, f"lw_256_test.alchemical.cuda.{Lambda}.log")
    
    # Set up the simulation parameters.
    nsteps = nsteps
    time_step = time_step # unit is femtosecond.
    temperature = 300 # Kelvin
    report_frequency = 100 # Save trajectory every report_frequency steps.
    box = [19.807884, 0, 0, 0, 19.807884, 0, 0, 0, 19.807884]
    box = [mm.Vec3(box[0], box[1], box[2]), mm.Vec3(box[3], box[4], box[5]), mm.Vec3(box[6], box[7], box[8])] * u.angstroms
    
    liquid_water = PDBFile(pdb_file)
    topology = liquid_water.topology
    positions = liquid_water.getPositions()
    
    # Set up the dp_system with the dp_model.    
    dp_model = DeepPotentialModel(dp_model_file, Lambda = Lambda)
    dp_model.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
    # By default, createSystem from dp_model will put all atoms in topology into the DP particles for dp_model.
    dp_system = dp_model.createSystem(topology)
    
    # Initial the other two dp_models for alchemical simulation.
    dp_model_1 = DeepPotentialModel(dp_model_file, Lambda = 1 - Lambda)
    dp_model_2 = DeepPotentialModel(dp_model_file, Lambda = 1 - Lambda)
    dp_model_1.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
    dp_model_2.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
    
    # Split the system particles into two groups for alchemical simulation.
    graph1_particles = []
    graph2_particles = []
    for atom in topology.atoms():
        if int(atom.residue.id) == 1:
            graph2_particles.append(atom.index)
        else:
            graph1_particles.append(atom.index)
    dp_force_1 =  dp_model_1.addParticlesToDPRegion(graph1_particles, topology)
    dp_force_2 = dp_model_2.addParticlesToDPRegion(graph2_particles, topology)
    
    # Add the two dp_forces to the dp_system.
    dp_system.addForce(dp_force_1)
    dp_system.addForce(dp_force_2)
    
    integrator = mm.LangevinIntegrator(
                                temperature*u.kelvin,       # Temperature of heat bath
                                1.0/u.picoseconds,  # Friction coefficient
                                time_step*u.femtoseconds, # Time step
        )
    platform = mm.Platform.getPlatformByName(platform_name)
    
    # Build up the simulation object.
    sim = Simulation(topology, dp_system, integrator, platform)
    sim.context.setPeriodicBoxVectors(box[0], box[1], box[2])
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(temperature * u.kelvin)

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
    
    # Fetch the temperature info from the log file.
    temperature_trajectory = []
    temperature_index = -4
    with open(output_log, "r") as f:
        log_content = f.readlines()
    for ii , line in enumerate(log_content):
        if ii == 0:
            continue
        tmp = line.split()
        temperature_trajectory.append(float(tmp[temperature_index]))
    temperature_trajectory = np.array(temperature_trajectory)
    
    # Check the temperature fluctuations is smaller than temperature_std_tol, unit in kelvin.
    assert(np.std(temperature_trajectory) < temperature_std_tol)
    
    
if __name__ == "__main__":
    test_deepmd_alchemical_reference()
    test_deepmd_alchemical_cuda()