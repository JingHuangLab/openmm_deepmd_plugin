import os
import numpy as np
import time
import argparse

try:
    import openmm as mm
    from openmm import unit as u
    from openmm.app import PDBFile, StateDataReporter, DCDReporter, element
except:
    import simtk.openmm as mm
    from simtk import unit as u
    from simtk.openmm.app import PDBFile, StateDataReporter, DCDReporter, element

from OpenMMDeepmdPlugin import DeepmdForce, Simulation4Deepmd


def test_deepmd_nve(nsteps, time_step, platform_name = "Reference", output_temp_dir = "/tmp/openmm_deepmd_plugin_test_nve_output", energy_std_tol = 0.0005 ):
    if not os.path.exists(output_temp_dir):
        os.mkdir(output_temp_dir)
    
    pdb_file = os.path.join(os.path.dirname(__file__), "../data", "lw_256_test.pdb")
    dp_model = os.path.join(os.path.dirname(__file__), "../data", "water.pb")
    output_dcd = os.path.join(output_temp_dir, "lw_256_test.nve.dcd")
    output_log = os.path.join(output_temp_dir, "lw_256_test.nve.log")
    
    
    time_step = 0.2 # unit is femtosecond.
    report_frequency = 100
    nsteps = 200
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
    dp_force.addType(1, element.hydrogen.symbol)
    
    # Add particles into force.
    for atom in topology.atoms():
        if atom.element == element.oxygen:
            dp_system.addParticle(element.oxygen.mass)
            dp_force.addParticle(atom.index, element.oxygen.symbol)
            for at in atom.residue.atoms():
                topology.addBond(at, atom)
                if at.index != atom.index:
                    dp_force.addBond(atom.index, at.index)
        elif atom.element == element.hydrogen:
            dp_system.addParticle(element.hydrogen.mass)
            dp_force.addParticle(atom.index, element.hydrogen.symbol)
    
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
    
    # Check the total energy fluctuations over # of atoms is smaller than energy_std_tol, unit in kJ/mol.
    assert(np.std(total_energy) / num_atoms < energy_std_tol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nsteps', type = int, dest='nsteps', help='Number of steps', default=100)
    parser.add_argument('--dt', type = float, dest='timestep', help='Time step for simulation, unit is femtosecond', default=0.2)
    parser.add_argument('--platform', type = str, dest='platform', help='Platform for simulation.', default="Reference")
    
    args = parser.parse_args()

    nsteps = args.nsteps
    time_step = args.timestep
    platform_name = args.platform
    
    test_deepmd_nve(nsteps=nsteps, time_step=time_step, platform_name=platform_name)
    