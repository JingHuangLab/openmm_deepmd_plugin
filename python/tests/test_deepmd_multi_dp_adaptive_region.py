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
from utils import read_top, read_crd, read_params, read_box, vfswitch, restraints

def test_deepmd_adaptive_dp_particles_reference(nsteps = 100, time_step = 1, Lambda = 1.0, platform_name = "Reference", output_temp_dir = "/tmp/openmm_deepmd_plugin_test_multi_adaptive_dp_particles_output"):
    if not os.path.exists(output_temp_dir):
        os.mkdir(output_temp_dir)
    
    crd_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "1aay.crd")
    psf_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "1aay.psf")
    restrain_txt = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "1aay.restraints_prot_pos")
    sysinfo_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "1aay.sysinfo")
    toppar_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "toppar.str")
    
    output_dcd = os.path.join(output_temp_dir, f"1aay_test.multi.adaptive.dp.region.reference.dp.mm.dcd")
    output_log = os.path.join(output_temp_dir, f"1aay_test.multi.adaptive.dp.region.reference.dp.mm.log")
    
    # Set up the simulation parameters.
    nsteps = nsteps
    time_step = time_step # unit is femtosecond.
    temp = 300 # Kelvin
    report_frequency = 100 # Save trajectory every report_frequency steps.
    fric_coeff = 1.0     # friction coefficient
    mini_nstep = 100      # minimization step
    mini_Tol = 100      # minimization tolerance
    r_on = 1.0           # unit is nanometer
    r_off = 1.2          # unit is nanometer
    
    psf = read_top(psf_file)
    crd = read_crd(crd_file)
    params = read_params(toppar_file)
    psf = read_box(psf, sysinfo_file)    
    
    nonbond_options = dict( nonbondedMethod=mm.app.PME, 
                            nonbondedCutoff=1.2*u.nanometers,
                            constraints=mm.app.HBonds,
                            ewaldErrorTolerance=0.0005,
                            switchDistance=1.0*u.nanometers,)
    mm_sys = psf.createSystem(params, **nonbond_options)
    mm_sys = vfswitch(mm_sys, psf, r_on, r_off)
    mm_sys = restraints(mm_sys, crd, 400.0, 40.0, restrain_txt)
    
    # Search for zinc atoms in the topology.
    zinc_atoms = []
    for atom in psf.topology.atoms():
        if atom.element.symbol == "Zn":
            zinc_atoms.append(atom.index)
    
    dp_model_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "dp_mask_test_plugin.pb")
    for zinc in zinc_atoms:
        print("Add deep potential force for zinc atom %d" % zinc)
        # Set up the dp force and add into system.
        dp_model = DeepPotentialModel(dp_model_file, Lambda = Lambda)
        dp_model.setUnitTransformCoefficients(10., 41.84, 4.184)
        dp_force = dp_model.addCenterParticlesToAdaptiveDPRegion(
            center_particles = [zinc], 
            topology = psf.topology, 
            sel_num4each_type={"C": 36, "O": 16, "N":24, "H":64, "S":6, "ZN":1}
            )
        # Add force into the system.
        mm_sys.addForce(dp_force)
        

    integrator = mm.LangevinIntegrator(temp*u.kelvin, fric_coeff/u.picosecond, time_step*u.femtosecond)
    platform = mm.Platform.getPlatformByName(platform_name)
    
    # Build up the simulation object.
    sim = Simulation(psf.topology, mm_sys, integrator, platform)
    sim.context.setPositions(crd.positions)
    

    print("\nInitial system energy")
    print(sim.context.getState(getEnergy=True).getPotentialEnergy())
    
    if mini_nstep > 0:
        print("\nEnergy minimization: %s steps" % mini_nstep)
        sim.minimizeEnergy(tolerance=mini_Tol*u.kilojoule/u.mole, maxIterations=mini_nstep)
        print(sim.context.getState(getEnergy=True).getPotentialEnergy()) 

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
    
def test_deepmd_adaptive_dp_particles_cuda(nsteps = 100, time_step = 1, Lambda = 1.0, platform_name = "CUDA", output_temp_dir = "/tmp/openmm_deepmd_plugin_test_multi_adaptive_dp_particles_output"):
    if not os.path.exists(output_temp_dir):
        os.mkdir(output_temp_dir)
    
    crd_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "1aay.crd")
    psf_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "1aay.psf")
    restrain_txt = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "1aay.restraints_prot_pos")
    sysinfo_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "1aay.sysinfo")
    toppar_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "toppar.str")
    
    output_dcd = os.path.join(output_temp_dir, f"1aay_test.multi.adaptive.dp.region.cuda.dp.mm.dcd")
    output_log = os.path.join(output_temp_dir, f"1aay_test.multi.adaptive.dp.region.cuda.dp.mm.log")
    
    # Set up the simulation parameters.
    nsteps = nsteps
    time_step = time_step # unit is femtosecond.
    temp = 300 # Kelvin
    report_frequency = 100 # Save trajectory every report_frequency steps.
    fric_coeff = 1.0     # friction coefficient
    mini_nstep = 100     # minimization step
    mini_Tol = 100       # minimization tolerance
    r_on = 1.0           # unit is nanometer
    r_off = 1.2          # unit is nanometer
    
    psf = read_top(psf_file)
    crd = read_crd(crd_file)
    params = read_params(toppar_file)
    psf = read_box(psf, sysinfo_file)    
    
    nonbond_options = dict( nonbondedMethod=mm.app.PME, 
                            nonbondedCutoff=1.2*u.nanometers,
                            constraints=mm.app.HBonds,
                            ewaldErrorTolerance=0.0005,
                            switchDistance=1.0*u.nanometers,)
    mm_sys = psf.createSystem(params, **nonbond_options)
    mm_sys = vfswitch(mm_sys, psf, r_on, r_off)
    mm_sys = restraints(mm_sys, crd, 400.0, 40.0, restrain_txt)
    
    # Search for zinc atoms in the topology.
    zinc_atoms = []
    for atom in psf.topology.atoms():
        if atom.element.symbol == "Zn":
            zinc_atoms.append(atom.index)
    
    dp_model_file = os.path.join(os.path.dirname(__file__), "../OpenMMDeepmdPlugin/data", "dp_mask_test_plugin.pb")
    for zinc in zinc_atoms:
        print("Add deep potential force for zinc atom %d" % zinc)
        # Set up the dp force and add into system.
        dp_model = DeepPotentialModel(dp_model_file, Lambda = Lambda)
        dp_model.setUnitTransformCoefficients(10., 41.84, 4.184)
        dp_force = dp_model.addCenterParticlesToAdaptiveDPRegion(
            center_particles = [zinc], 
            topology = psf.topology, 
            sel_num4each_type={"C": 36, "O": 16, "N":24, "H":64, "S":6, "ZN":1}
            )
        # Add force into the system.
        mm_sys.addForce(dp_force)
        

    integrator = mm.LangevinIntegrator(temp*u.kelvin, fric_coeff/u.picosecond, time_step*u.femtosecond)
    platform = mm.Platform.getPlatformByName(platform_name)
    
    # Build up the simulation object.
    sim = Simulation(psf.topology, mm_sys, integrator, platform)
    sim.context.setPositions(crd.positions)
    

    print("\nInitial system energy")
    print(sim.context.getState(getEnergy=True).getPotentialEnergy())
    
    if mini_nstep > 0:
        print("\nEnergy minimization: %s steps" % mini_nstep)
        sim.minimizeEnergy(tolerance=mini_Tol*u.kilojoule/u.mole, maxIterations=mini_nstep)
        print(sim.context.getState(getEnergy=True).getPotentialEnergy()) 

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
        
if __name__ == "__main__":
    test_deepmd_adaptive_dp_particles_reference(nsteps = 100, Lambda= 1.0)
    test_deepmd_adaptive_dp_particles_cuda(nsteps = 1000, Lambda= 1.0)
    