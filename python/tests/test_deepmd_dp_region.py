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


def test_deepmd_fixed_dp_particles_reference(nsteps = 1000, time_step = 0.2, Lambda = 1.0, platform_name = "Reference", output_temp_dir = "/tmp/openmm_deepmd_plugin_test_fixed_dp_particles_output", energy_std_tol = 25):
    if not os.path.exists(output_temp_dir):
        os.mkdir(output_temp_dir)
    
    pdb_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "lw_256_test.pdb")
    dp_model = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "water.pb")
    output_dcd = os.path.join(output_temp_dir, f"lw_256_test.fixed.dp.region.reference.{Lambda}.dcd")
    output_log = os.path.join(output_temp_dir, f"lw_256_test.fixed.dp.region.reference.{Lambda}.log")
    
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
    
    # Create a mm system from scratch.
    # In practice for DP/MM simulation with fixed DP region, the mm system can be created from a force field. Such as:
    # forcefield = ForceField('amber99sbildn.xml', 'tip3p.xml')
    # mm_system = forcefield.createSystem(topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometers, constraints=HBonds)
    mm_system = mm.System()
    for atom in topology.atoms():
        mm_system.addParticle(atom.element.mass)
    
    # Set up the dp_system with the dp_model.
    dp_model = DeepPotentialModel(dp_model, Lambda = Lambda)
    dp_model.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
    
    # Append the selected DP particles to DP region
    dp_particles = []
    for atom in topology.atoms():
        # Assume the water molecules with residue.id = 1 will be selected as DP particles.
        if int(atom.residue.id) == 1 or int(atom.residue.id) == 47 or int(atom.residue.id) == 17 or int(atom.residue.id) == 184:
            dp_particles.append(atom.index)
    dp_force = dp_model.addParticlesToDPRegion(dp_particles=dp_particles, topology = topology)
    mm_system.addForce(dp_force)
    
    integrator = mm.VerletIntegrator(time_step*u.femtoseconds)
    platform = mm.Platform.getPlatformByName(platform_name)
    
    # Build up the simulation object.
    sim = Simulation(topology, mm_system, integrator, platform)
    
    # In practice for DP/MM simulation with fixed DP region.
    # The bonded forces within the DP region should be removed from the mm system.
    # num_forces = mm_system.getNumForces()
    # for force in mm_system.getForces():
    #     if isinstance(force, mm.HarmonicBondForce):
    #         num_bonds = force.getNumBonds()
    #         for bond_index in range(num_bonds):
    #             atom1, atom2, length, k = force.getBondParameters(bond_index)
    #             if atom1 in dp_particles and atom2 in dp_particles:
    #                 # Remove the bond forces within the DP region.
    #                 force.setBondParameters(bond_index, atom1, atom2, length, 0.0)
    #         force.updateParametersInContext(sim.context)


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
    
    # Fetch the temperature info from the log file.
    energy_trajectory = []
    energy_index = -5
    with open(output_log, "r") as f:
        log_content = f.readlines()
    for ii , line in enumerate(log_content):
        if ii == 0:
            continue
        tmp = line.split()
        energy_trajectory.append(float(tmp[energy_index]))
    energy_trajectory = np.array(energy_trajectory)
    
    assert (np.std(energy_trajectory) < energy_std_tol)
    assert (np.std(energy_trajectory) > 0.0001)
    
        
def test_deepmd_fixed_dp_particles_cuda(nsteps = 1000, time_step = 0.2, Lambda = 1.0, platform_name = "CUDA", output_temp_dir = "/tmp/openmm_deepmd_plugin_test_fixed_dp_particles_output", energy_std_tol = 1):
    if not os.path.exists(output_temp_dir):
        os.mkdir(output_temp_dir)
    
    pdb_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "lw_256_test.pdb")
    dp_model = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "water.pb")
    output_dcd = os.path.join(output_temp_dir, f"lw_256_test.fixed.dp.region.cuda.{Lambda}.dcd")
    output_log = os.path.join(output_temp_dir, f"lw_256_test.fixed.dp.region.cuda.{Lambda}.log")
    
    # Set up the simulation parameters.
    nsteps = nsteps
    time_step = time_step # unit is femtosecond.
    report_frequency = 100 # Save trajectory every report_frequency steps.
    box = [19.807884, 0, 0, 0, 19.807884, 0, 0, 0, 19.807884]
    box = [mm.Vec3(box[0], box[1], box[2]), mm.Vec3(box[3], box[4], box[5]), mm.Vec3(box[6], box[7], box[8])] * u.angstroms
    
    liquid_water = PDBFile(pdb_file)
    topology = liquid_water.topology
    positions = liquid_water.getPositions()
    
    # Create a mm system from scratch.
    # In practice for DP/MM simulation with fixed DP region, the mm system can be created from a force field. Such as:
    # forcefield = ForceField('amber99sbildn.xml', 'tip3p.xml')
    # mm_system = forcefield.createSystem(topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometers, constraints=HBonds)
    mm_system = mm.System()
    for atom in topology.atoms():
        mm_system.addParticle(atom.element.mass)
    
    # Set up the dp_system with the dp_model.
    dp_model = DeepPotentialModel(dp_model, Lambda = Lambda)
    dp_model.setUnitTransformCoefficients(10.0, 964.8792534459, 96.48792534459)
    
    # Append the selected DP particles to DP region
    dp_particles = []
    for atom in topology.atoms():
        # Assume the water molecules with residue.id = 1 will be selected as DP particles.
        if int(atom.residue.id) == 1 or int(atom.residue.id) == 47 or int(atom.residue.id) == 17 or int(atom.residue.id) == 184:
            dp_particles.append(atom.index)
    dp_force = dp_model.addParticlesToDPRegion(dp_particles=dp_particles, topology = topology)
    mm_system.addForce(dp_force)
    
    integrator = mm.VerletIntegrator(time_step*u.femtoseconds)
    platform = mm.Platform.getPlatformByName(platform_name)
    
    # Build up the simulation object.
    sim = Simulation(topology, mm_system, integrator, platform)
    
    # In practice for DP/MM simulation with fixed DP region.
    # The bonded forces within the DP region should be removed from the mm system.
    # num_forces = mm_system.getNumForces()
    # for force in mm_system.getForces():
    #     if isinstance(force, mm.HarmonicBondForce):
    #         num_bonds = force.getNumBonds()
    #         for bond_index in range(num_bonds):
    #             atom1, atom2, length, k = force.getBondParameters(bond_index)
    #             if atom1 in dp_particles and atom2 in dp_particles:
    #                 # Remove the bond forces within the DP region.
    #                 force.setBondParameters(bond_index, atom1, atom2, length, 0.0)
    #         force.updateParametersInContext(sim.context)


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
    
    # Fetch the temperature info from the log file.
    energy_trajectory = []
    energy_index = -5
    with open(output_log, "r") as f:
        log_content = f.readlines()
    for ii , line in enumerate(log_content):
        if ii == 0:
            continue
        tmp = line.split()
        energy_trajectory.append(float(tmp[energy_index]))
    energy_trajectory = np.array(energy_trajectory)
    
    assert (np.std(energy_trajectory) < energy_std_tol)
    assert (np.std(energy_trajectory) > 0.0001)
    
if __name__ == "__main__":
    test_deepmd_fixed_dp_particles_reference(nsteps = 5000, Lambda= 0.0)
    test_deepmd_fixed_dp_particles_cuda(nsteps = 5000, Lambda= 0.0)
    