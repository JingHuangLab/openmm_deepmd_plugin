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



def run_deepmd_alchemical_cuda(nsteps = 1000, time_step = 0.2, Lambda = 1.0, platform_name = "CUDA", output_temp_dir = "/tmp/openmm_deepmd_plugin_test_alchemical_output"):
    if not os.path.exists(output_temp_dir):
        os.mkdir(output_temp_dir)
    
    pdb_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "lw_256_test.pdb")
    dp_model = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "water.pb")
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
    dp_model = DeepPotentialModel(dp_model, dp_model, dp_model)
    dp_model.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
    
    # Set up alchemical simulation for DP
    graph1_particles = []
    graph2_particles = []
    for atom in topology.atoms():
        if int(atom.residue.id) == 1:
            graph2_particles.append(atom.index)
        else:
            graph1_particles.append(atom.index)
    dp_system = dp_model.createSystem(topology, particles_group_1 = graph1_particles, particles_group_2 = graph2_particles, Lambda = Lambda)
    
    
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
    

if __name__=="__main__":
    nsteps = 4 * 1000 * 5000 * 0.2 # 4 ns production simulation.
    time_step = 0.2 # fs
    Lambda_list = [0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.2, 0.4, 0.7, 1.0]
    
    for ith_lambda in Lambda_list:
        run_deepmd_alchemical_cuda(nsteps=nsteps, time_step=time_step, Lambda=ith_lambda)