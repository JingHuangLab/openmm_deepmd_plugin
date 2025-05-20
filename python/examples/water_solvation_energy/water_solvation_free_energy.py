import os
import time
import numpy as np
import pickle
    
try:
    import openmm as mm
    from openmm import unit as u
    from openmm.app import PDBFile, StateDataReporter, DCDReporter, Simulation
except:
    import simtk.openmm as mm
    from simtk import unit as u
    from simtk.openmm.app import PDBFile, StateDataReporter, DCDReporter, Simulation

    
from OpenMMDeepmdPlugin import DeepPotentialModel

from bar import bar


def create_alchemical_builder(pdb_file, dp_model_file, dp_model_file1, dp_model_file2,):
    liquid_water = PDBFile(pdb_file)
    topology = liquid_water.topology
    positions = liquid_water.getPositions()
    
    def get_alchemical_system(lambda_value):
        # Set up the dp_system with the dp_model.    
        dp_model = DeepPotentialModel(dp_model_file, Lambda = lambda_value)
        dp_model.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
        # By default, createSystem from dp_model will put all atoms in topology into the DP particles for dp_model.
        dp_system = dp_model.createSystem(topology)
        
        # Initial the other two dp_models for alchemical simulation.
        dp_model_1 = DeepPotentialModel(dp_model_file1, Lambda = 1 - lambda_value)
        dp_model_2 = DeepPotentialModel(dp_model_file2, Lambda = 1 - lambda_value)
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

        return dp_system, topology, positions
        
    return get_alchemical_system


def build_alchemical_simulation(alchemical_system, topology, box_vectors, init_positions, nsteps, time_step, temperature, report_frequency, output_dcd, output_log):
    platform = mm.Platform.getPlatformByName("CUDA")
    integrator = mm.LangevinIntegrator(temperature * u.kelvin, 1.0 / u.picosecond, time_step * u.femtoseconds)
    alchemical_simulation = Simulation(topology, alchemical_system, integrator, platform)
    
    alchemical_simulation.context.setPositions(init_positions)
    alchemical_simulation.context.setPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])
    
    if output_log is not None:
        alchemical_simulation.reporters.append(StateDataReporter(output_log, 
                                                             report_frequency, 
                                                             step=True, 
                                                             time=True, 
                                                             totalEnergy=True,          
                                                             kineticEnergy=True, 
                                                             potentialEnergy=True, 
                                                             temperature=True, 
                                                             progress=True, 
                                                             remainingTime=True, 
                                                             speed=True, 
                                                             density=True, 
                                                             totalSteps=nsteps, 
                                                             separator='\t'))
    if output_dcd is not None:
        alchemical_simulation.reporters.append(DCDReporter(output_dcd, report_frequency))

    return alchemical_simulation
    

def run_simulation(alchemical_simulation, nsteps, minimized=False):
    if not minimized:
        print("Minimizing energy")
        alchemical_simulation.minimizeEnergy()
        state = alchemical_simulation.context.getState(getEnergy=True, getPositions=True)
        print("Minimized energy: ", state.getPotentialEnergy())
        
        
    print("Running dynamics")
    start_time = time.time()
    alchemical_simulation.step(nsteps)
    end_time = time.time()
    cost_time = end_time - start_time
    print("Running on %s platform, time cost: %.4f s" % (alchemical_simulation.context.getPlatform().getName(), cost_time))
    return cost_time

def reevaluate_energy(simulation, dcd_files, pdb_file):
    try:
        import mdtraj
    except ImportError:
        raise ImportError("mdtraj is not installed. Please install it to use this function.")
    
    energy_list = []
    for dcd_file in dcd_files:
        traj = mdtraj.load(dcd_file, top = pdb_file)
        energy = []
        for frame in traj:
            simulation.context.setPositions(frame.xyz[0])
            state = simulation.context.getState(getEnergy=True)
            potential_energy = state.getPotentialEnergy()
            energy.append(potential_energy.value_in_unit(u.kilojoule_per_mole))
        energy_list.append(energy)
    energy_list = np.array(energy_list)
    
    return energy_list





if __name__ == "__main__":
    
    pdb_file = os.path.join(os.path.dirname(__file__), "openmm_deepmd_plugin/python/OpenMMDeepmdPlugin/data", "lw_256_test.pdb")
    
    dp_model_file = os.path.join(os.path.dirname(__file__), "openmm_deepmd_plugin/python/OpenMMDeepmdPlugin/data", "water.pb")
    dp_model_file1 = os.path.join(os.path.dirname(__file__), "openmm_deepmd_plugin/python/OpenMMDeepmdPlugin/data", "water.pb")
    dp_model_file2 = os.path.join(os.path.dirname(__file__), "openmm_deepmd_plugin/python/OpenMMDeepmdPlugin/data", "water.pb")
    
    output_temp_dir = "/tmp/omm_dp_water_solvation_free_energy"
    if not os.path.exists(output_temp_dir):
        os.mkdir(output_temp_dir)
    box = [19.807884, 0, 0, 0, 19.807884, 0, 0, 0, 19.807884]
    box = [mm.Vec3(box[0], box[1], box[2]), mm.Vec3(box[3], box[4], box[5]), mm.Vec3(box[6], box[7], box[8])] * u.angstroms
    
    
    lambda_list = [1.0, 0.7, 0.4, 0.2, 0.12, 0.08, 0.05, 0.03, 0.01, 0.0]
    
    time_step = 1.0 # unit is femtosecond.
    temperature = 300 # Kelvin
    report_frequency = 1000 # Save trajectory every 1 ps.
    nsteps = int(0.1 * 1000 * 1000 / time_step) # 1 ns
    
    alchemical_builder = create_alchemical_builder(pdb_file, dp_model_file, dp_model_file1, dp_model_file2)
    
    all_alchem_simulations = []

    ## Run the alchemical simulation for each lambda value.
    ## This is a time consuming step, and it is recommended to run it in parallel.
    for lambda_value in lambda_list:
        output_dcd = os.path.join(output_temp_dir, f"lw_256.alchemical.{lambda_value}.dcd")
        output_log = os.path.join(output_temp_dir, f"lw_256.alchemical.{lambda_value}.log")
        
        alchemical_system, topology, positions = alchemical_builder(lambda_value)
        
        print("Lambda: ", lambda_value)
        # Build up the simulation object.
        alchem_simulation = build_alchemical_simulation(alchemical_system, topology, box, positions, nsteps, time_step, temperature, report_frequency, output_dcd, output_log)
        
        # Run dynamics
        run_simulation(alchem_simulation, nsteps)
    
    
    ## Re-evaluate the energy of the trajectory for each BAR calculation.
    dcd_files = [os.path.join(output_temp_dir, f"lw_256.alchemical.{lambda_value}.dcd") for lambda_value in lambda_list]
    
    num_lambdas = len(lambda_list)
    energy_matrix = {}
    
    
    for ii, lambda_value in enumerate(lambda_list):
        energy_matrix[ii] = {"00": None, "01": None, "10": None, "11": None}
        
        print("Re-evaluating energy for lambda: ", lambda_value)
        alchemical_system, topology, positions = alchemical_builder(lambda_value)
        alchem_simulation = build_alchemical_simulation(alchemical_system, topology, box, positions, nsteps, time_step, temperature, report_frequency, None, None)
        
        if ii == 0:
            input_dcd_files = [dcd_files[ii], dcd_files[ii+1]]
        elif ii == len(lambda_list) - 1:
            input_dcd_files = [dcd_files[ii-1], dcd_files[ii]]
        else:
            input_dcd_files = [dcd_files[ii - 1], dcd_files[ii], dcd_files[ii + 1]]
        
        energy_list = reevaluate_energy(alchem_simulation, input_dcd_files, pdb_file)
        
        if ii == 0:
            energy_matrix[ii]["00"] = energy_list[0]
            energy_matrix[ii]["01"] = energy_list[1]
        elif ii == len(lambda_list) - 1:
            energy_matrix[ii]["00"] = energy_list[1]
            energy_matrix[ii]["01"] = energy_list[0]
            energy_matrix[ii - 1]["10"] = energy_list[0]
            energy_matrix[ii - 1]["11"] = energy_list[1]
        else:
            energy_matrix[ii]["00"] = energy_list[1]
            energy_matrix[ii]["01"] = energy_list[2]
            energy_matrix[ii - 1]["10"] = energy_list[0]
            energy_matrix[ii - 1]["11"] = energy_list[1]
            
    with open(os.path.join(output_temp_dir, "energy_matrix.pickle"), "wb") as f:
        pickle.dump(energy_matrix, f)
    
    if os.path.exists(os.path.join(output_temp_dir, "energy_matrix.pickle")):
        with open(os.path.join(output_temp_dir, "energy_matrix.pickle"), "rb") as f:
            energy_matrix = pickle.load(f)
        
    dG_all = 0.0
    for ii in range(num_lambdas - 1):
        w_F = np.concatenate((energy_matrix[ii]["00"], energy_matrix[ii]["01"]))
        w_R = np.concatenate((energy_matrix[ii]["10"], energy_matrix[ii]["11"]))
        results = bar(w_F, w_R)
        dG = results["Delta_f"]
        dG_std = results["dDelta_f"]
        
        print("Lambda: ", lambda_list[ii], " dG: ", dG, " dG_std: ", dG_std)
        
        dG_all += dG
    
    print("Free energy change: ", dG_all)
    
    
    