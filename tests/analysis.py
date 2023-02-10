import MDAnalysis
import numpy as np
import argparse
import numpy as np
import math
from OpenMMDeepmdPlugin import DrawScatter, AlchemicalContext
from scipy import stats
import simtk.unit as unit
import csv

def readEnergyAndDraw(mole_name, log_file, totalOnly=False, natoms = 256 * 3):
    output_log = log_file
    tot_index = -5
    kinetic_index = 4
    potential_index = 3 

    init_total = None
    init_kinetic = None
    init_potential = None

    with open(output_log, "r")  as f:
        log_content = f.readlines()

    total_energy = []
    kinetic_energy = []
    potential_energy = []

    for ii, line in enumerate(log_content):    
        if ii == 0:
            continue
        temp = line.split()
        if ii == 1:
            init_potential = float(temp[potential_index])
            init_kinetic = float(temp[kinetic_index])
            init_total = float(temp[tot_index])

        if totalOnly:
            total_energy.append(float(temp[tot_index]))
        else:
            total_energy.append(float(temp[tot_index]) - init_total)
            potential_energy.append(float(temp[potential_index]) - init_potential)
            kinetic_energy.append(float(temp[kinetic_index]) - init_kinetic)
    
    selected_frames_num = len(total_energy)

    total_energy = np.array(total_energy[:selected_frames_num])
    kinetic_energy = np.array(kinetic_energy[:selected_frames_num])
    potential_energy = np.array(potential_energy[:selected_frames_num])

    print(output_log, total_energy.shape[0], np.average(total_energy), np.std(total_energy), np.std(total_energy)/natoms)
    if not totalOnly:
        y = [total_energy, potential_energy, kinetic_energy]
    else:
        y = total_energy
    y = np.array(y)

    x = np.array(range(selected_frames_num))
    xlabel = "Frame"
    ylabel = "Energy, kJ/mol"
    name = mole_name
    DrawScatter(x, y, name, xlabel, ylabel)
    return

def draw_potential_distribution(logs_1, logs_2, temp1, temp2, bins_num=30):
    potential = dict()
    potential[1] = []
    potential[2] = []
    kB = 1.380649 * 6.0221409 * 1e-3

    for log in logs_1:
        with open(log, 'r') as f:
            log_content = f.readlines()
        for ii, line in enumerate(log_content):
            if ii == 0:
                continue
            temp = line.split()
            ene = float(temp[3])
            if (ii - 1) % 10 == 0: 
                potential[1].append(ene)
    
    for log in logs_2:
        with open(log, 'r') as f:
            log_content = f.readlines()
        for ii, line in enumerate(log_content):
            if ii == 0:
                continue
            temp = line.split()
            ene = float(temp[3])
            if ( ii - 1 ) % 10 == 0:
                potential[2].append(ene)
    
    potential[1] = np.array(potential[1])
    potential[2] = np.array(potential[2])
    max_ene = np.max(potential[1]) if np.max(potential[1]) > np.max(potential[2]) else np.max(potential[2])
    min_ene = np.min(potential[1]) if np.min(potential[1]) < np.min(potential[2]) else np.min(potential[2])

    hist1, bins_edge_1 = np.histogram(potential[1], bins = bins_num, range= [min_ene, max_ene], density=True)
    hist2, bins_edge_2 = np.histogram(potential[2], bins = bins_num, range=[min_ene, max_ene], density=True)

    assert hist1.shape[0] == hist2.shape[0], "Number of two histograms are not equal."
    assert hist1.shape[0] == bins_num, "Number of histogram is not equal to number of bins."

    hist1 = hist1 * (bins_edge_1[1] - bins_edge_1[0])
    hist2 = hist2 * (bins_edge_2[1] - bins_edge_2[0])

    log_p_ene = []
    ene = []
    for ii in range(bins_num):
        assert (bins_edge_1[ii] + bins_edge_1[ii + 1]) * 0.5 == (bins_edge_2[ii] + bins_edge_2[ii + 1]) * 0.5, " %d bins edge are not equal."%ii
        try:
            if hist1[ii] == 0 or hist2[ii] == 0:
                continue
            if hist1[ii] < 1e-3 or hist2[ii] < 1e-3:
                continue
            log_p = math.log(hist2[ii]/hist1[ii])
            ene.append((bins_edge_1[ii] + bins_edge_1[ii + 1]) * 0.5)
            log_p_ene.append(log_p)
        except:
            print(hist1[ii], hist2[ii], ii)


    #fitted_coef, b = np.polyfit(ene, log_p_ene, 1)
    fitted_coef, intercept, r_value, p_value, std_err = stats.linregress(ene, log_p_ene)
    theory_coef = 1.0/(kB * temp1) - 1.0/(kB * temp2)
    
    ene = np.array(ene)
    log_p_ene = np.array(log_p_ene)

    print("Number of potential 1: %d, Number of potential 2: %d, Number of bins: %d"%(potential[1].shape[0], potential[2].shape[0], len(ene)))
    print(fitted_coef, std_err)
    print(theory_coef)

    name = "T1_%d.T2_%d"%(temp1, temp2)
    xlabel = "E, kJ/mol"
    ylabel = "ln(P(E|T_2)/P(E|T_1))"
    DrawScatter(ene, log_p_ene, name, xlabel, ylabel, fitting = True)
    return

def alchemEnergy(alchem_context, pdb, dcd_files, save_file):
    try:
        import MDAnalysis as mda
        import simtk.openmm as mm
        import simtk.unit as unit
        from tqdm import tqdm
    except ImportError:
        print("MDAnalysis can not be imported.")
        exit(1)
    potential_e = []
    u = MDAnalysis.Universe(pdb, dcd_files)
    natoms = len(u.atoms)
    
    for ts in tqdm(u.trajectory, desc="Iterate for each frame in trajectory: "):
        positions = ts._pos
        posi = []
        for ii in range(natoms):
            posi.append(mm.Vec3(positions[ii][0], positions[ii][1], positions[ii][2])* unit.angstroms)
        temp_p = alchem_context.getPotentialEnergy(posi)
        potential_e.append(temp_p)
    
    potential_e_np = np.array(potential_e)
    np.save(save_file+".npy", potential_e_np)
    potential_e_str = [str(x) for x in potential_e]
    with open(save_file+".log", 'w') as f:
        f.write("\n".join(potential_e_str))
    return

def alchemMBAR(npy_prefix, N_max, lambda_list, RT = unit.AVOGADRO_CONSTANT_NA._value * unit.BOLTZMANN_CONSTANT_kB._value * 300/1000): # Default value for RT is 2.479 * 300/298 kJ/mol in 300K.
    try:
        from pymbar import MBAR
    except ImportError:
        print("pymbar is not installed.")
        exit(1)
    u_kn = []
    N_k = []

    # Load energy from system.
    for hamilton_lambda in lambda_list:
        u = []
        for dcd_lambda in lambda_list:
            npy_file = npy_prefix + "_dcd.lambda." + str(dcd_lambda) + "_hamilton.lambda." + str(hamilton_lambda) + ".npy"
            ene = np.load(npy_file, allow_pickle=True)
            if N_max > ene.shape[0]:
                raise ValueError("%s don't have enough energy points. %d points needed"%(npy_file, N_max))

            ene = [(x._value)/RT for x in ene]
            ene = ene[:N_max]
            u = u + ene
        N_k.append(N_max)
        u_kn.append(u)
    u_kn = np.array(u_kn)
    N_k = np.array(N_k)
    #print(len(u_kn), len(u_kn[0]), len(N_k), N_k[0])
    
    mbar = MBAR(u_kn, N_k, verbose=False)
    hfe = mbar.getFreeEnergyDifferences()

    print("Result for %s,  number of select frames: %d * 0.2ps "%(npy_prefix, N_max))
    print(hfe[0][0][-1] * RT,'unit is kJ/mol')

    return hfe[0][0][-1] * RT

def getRDF_Oxygen_Oxygen(pdb_file, dcd_files, fig_name):
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.rdf import InterRDF
        import matplotlib.pyplot as plt
    except ImportError:
        print("MDAnalysis can not be imported.")
        exit(1)

    u = mda.Universe(pdb_file, dcd_files)
    all_atom = u.select_atoms("all")
    with MDAnalysis.Writer("./output/lw_256_nvt_300K_4ns.xtc", all_atom.n_atoms) as W:
        for ts in u.trajectory:
            if ts.frame % 10 == 0:
                W.write(all_atom)

    oxygen = u.select_atoms("name O")

    print("Running RDF")
    rdf = InterRDF(oxygen, oxygen, nbins=400, range=[1.2, 8.0], verbose=True)
    rdf.run()

    num = len(rdf.bins)

    with open("output/rdf_dp_omm_o_o.csv", "w+") as f:
        csv_writer = csv.writer(f, delimiter=',')
        for ii in range(num):
            csv_writer.writerow([rdf.bins[ii], rdf.rdf[ii]])

    fig_name = fig_name.split("/")[-1]
    plt.clf()
    plt.ylabel("g(r)")
    plt.xlabel("r($\AA$)")
    plt.title(fig_name)
    plt.plot(rdf.bins, rdf.rdf)
    plt.savefig("./output/"+fig_name+'.png')
    print("Figure saved at %s"%("./output/"+fig_name+'.png'))

    return

def getRDF_Oxygen_Hydrogen(pdb_file, dcd_files, fig_name):
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.rdf import InterRDF
        import matplotlib.pyplot as plt
    except ImportError:
        print("MDAnalysis can not be imported.")
        exit(1)

    u = mda.Universe(pdb_file, dcd_files)
    all_atom = u.select_atoms("all")
    oxygen = u.select_atoms("name O")
    hydrogen = u.select_atoms("type H")

    print("Running RDF")
    rdf = InterRDF(oxygen, hydrogen, nbins=400, range=[1.2, 8.0], verbose=True)
    rdf.run()

    num = len(rdf.bins)

    with open("output/rdf_dp_omm_o_h.csv", "w+") as f:
        csv_writer = csv.writer(f, delimiter=',')
        for ii in range(num):
            csv_writer.writerow([rdf.bins[ii], rdf.rdf[ii]])

    fig_name = fig_name.split("/")[-1]
    plt.clf()
    plt.ylabel("g(r)")
    plt.xlabel("r($\AA$)")
    plt.title(fig_name)
    plt.plot(rdf.bins, rdf.rdf)
    plt.savefig("./output/"+fig_name+'.png')
    print("Figure saved at %s"%("./output/"+fig_name+'.png'))

    return

def getRDF_Hydrogen_Hydrogen(pdb_file, dcd_files, fig_name):
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.rdf import InterRDF
        import matplotlib.pyplot as plt
    except ImportError:
        print("MDAnalysis can not be imported.")
        exit(1)

    u = mda.Universe(pdb_file, dcd_files)
    all_atom = u.select_atoms("all")
    hydrogen_1 = u.select_atoms("type H")
    hydrogen_2 = u.select_atoms("type H")

    print("Running RDF")
    rdf = InterRDF(hydrogen_1, hydrogen_2, nbins=400, range=[1.2, 8.0], verbose=True)
    rdf.run()

    num = len(rdf.bins)

    with open("output/rdf_dp_omm_h_h.csv", "w+") as f:
        csv_writer = csv.writer(f, delimiter=',')
        for ii in range(num):
            csv_writer.writerow([rdf.bins[ii], rdf.rdf[ii]])

    fig_name = fig_name.split("/")[-1]
    plt.clf()
    plt.ylabel("g(r)")
    plt.xlabel("r($\AA$)")
    plt.title(fig_name)
    plt.plot(rdf.bins, rdf.rdf)
    plt.savefig("./output/"+fig_name+'.png')
    print("Figure saved at %s"%("./output/"+fig_name+'.png'))

    return



def draw_nve_figure4presentation(nve_log, save_fig):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib can not be imported.")
    with open(nve_log, "r") as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = content[1:]
    step_num = []
    step_time = []
    total_energy = []

    for line in content:
        temp = line.split()
        step_num.append(int(temp[1]))
        step_time.append(float(temp[2]))
        total_energy.append(float(temp[5]))

    total_energy = np.array(total_energy)
    total_energy = total_energy / (3 * 256 * 3 - 6)
    total_energy = total_energy - total_energy[0]
    # Convert unit from kJ/mol to kcal/mol.
    total_energy = total_energy * 0.239006
    step_time = np.array(step_time)

    plt.clf()
    plt.figure(figsize=(12,8))
    font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 13}
    plt.rc('font', **font)
    plt.tick_params(direction='in')

    plt.ylim(-0.001, +0.001)
    plt.yticks(np.arange(-0.001, 0.002, step=0.001))
    plt.xlim(0, 1000)
    plt.plot(step_time, total_energy, '-')
    plt.xlabel("Time (ps)")
    plt.ylabel("Total Energy per DOF (kcal/mol)")

    plt.savefig(save_fig)

    return

def draw_MSD(pdb_file, dcd_files, fig_name):
    try:
        import MDAnalysis as mda
        import MDAnalysis.analysis.msd as MSD
        import matplotlib.pyplot as plt
    except ImportError:
        print("MDAnalysis can not be imported.")
        exit(1)

    u = mda.Universe(pdb_file, dcd_files)

    MSD4Water = MSD.EinsteinMSD(u, "all", msd_type='xyz', fft=True)
    MSD4Water.run()
    MSD4Hydrogen = MSD.EinsteinMSD(u, "name O", msd_type='xyz', fft=True)
    MSD4Hydrogen.run()
    MSD4Oxygen = MSD.EinsteinMSD(u, "type H", msd_type='xyz', fft=True)
    MSD4Oxygen.run()

    time = 0
    nframes = MSD4Water.n_frames
    timestep = 0.1
    lagtimes = np.arange(nframes) * timestep
    
    with open("output/msd_dp_omm_water_oxygen.csv", "w+") as f:
        csv_writer = csv.writer(f, delimiter=',')
        for ii in range(nframes):
            csv_writer.writerow([lagtimes[ii], MSD4Water.results.timeseries[ii], MSD4Oxygen.results.timeseries[ii]])

    plt.clf()
    plt.figure(figsize=(12,8))
    font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 13}
    plt.rc('font', **font)
    plt.tick_params(direction='in')

    plt.xlabel('Time (ps)')
    plt.ylabel(r'$MSD~(\AA^2)$')
    plt.plot(lagtimes,MSD4Water.results.timeseries, label = "Water")
    plt.plot(lagtimes,MSD4Oxygen.results.timeseries, label = "Oxygen")
    plt.plot(lagtimes,MSD4Hydrogen.results.timeseries, label = "Hydrogen")
    plt.legend()
    fig_name = fig_name.split("/")[-1]
    plt.savefig("./output/"+fig_name+'.png')
    print("Figure saved at %s"%("./output/"+fig_name+'.png'))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='type', type = str, help='Analysis type for processing. nveEnergy, logEnergy, oxygenRDF, alchemEnergy, alchemMBAR, compareRDF', required=True)
    
    # Parameters for nveEnergy analysis.
    parser.add_argument('--mole', dest='mole', type = str, help='Molecule name for nveEnergy, alchemEnergy analysis.', default=None, required=False)
    parser.add_argument('--log', dest='log', type = str, help='Log file for nveEnergy analysis.', default=None, required=False)
    parser.add_argument('--totalOnly', dest='totalOnly', type = bool, help='Whether to plot total energy only.', default=False, required=False)
    
    # Parameters for logEnergy analysis.
    parser.add_argument('--num-log', dest='nlog', type = int, help='Number of log files for logarithm energy (logEnergy) analysis.', default=-1, required=False)
    parser.add_argument('--log-prefix', dest='log_prefix', type = str, help='Log files prefix, including the path to log file.', default=None, required=False)
    parser.add_argument('--temp1', dest='temp1', type = int, help='Temperature 1 for log files analysis.', default=None, required=False)
    parser.add_argument('--temp2', dest='temp2', type = int, help='Temperature 2 for log files analysis.', default=None, required=False)

    # Parameters for oxygenRDF analysis.
    parser.add_argument('--pdb', dest='pdb', type =str, help='Input .pdb file for oxygenRDF analysis.', default=None, required=False)
    parser.add_argument('--num-dcd', dest='ndcd', type = int, help='Number of dcd for oxygenRDF, alchemEnergy analysis.', default=-1, required=False)
    parser.add_argument('--dcd-prefix', dest='dcd_prefix', type = str, help='.dcd files prefix, including the path to dcd file.', default=None, required=False)

    # Parameters for alchemEnergy analysis.
    parser.add_argument('--lambda', dest='Lambda', type = float, help='lambda value for potential energy.', default=None, required=False)
    parser.add_argument('--box', type = float, dest='box', help='Box dimension size for simulation, unit is angstrom', default=19.807884)
    parser.add_argument('--dcd-lambda', type = float, dest='dcd_lambda', help='The lambda setting value for dcd files.', default=None)

    # Parameters for alchemMBAR analysis.
    parser.add_argument('--npy-prefix', dest='npy_prefix', type = str, help='.npy files prefix, including the path to npy file.', default=None, required=False)
    parser.add_argument('--maxframe', dest='max_frame', type = int, help='Max number of frames which should be read.', default=None, required=False)
    parser.add_argument('--lambda-list', dest='lambda_list', type = str, help='List of lambdas which used for .npy files searching. Each lambda is separated by commas.', default=None, required=False)

    # Parameters for RDF comparision.
    parser.add_argument('--rdf-gmx', dest='rdf_gmx', type = str, help='Path to csv file which contains the value of RDF by DP Gromacs simulation.', default=None, required=False)
    parser.add_argument('--rdf-lmp', dest='rdf_lmp', type = str, help='Path to .csv file which contains the value of RDF by DP Lammps simulation.', default=None, required=False)
    parser.add_argument('--rdf-omm', dest='rdf_omm', type = str, help='Path to .csv file which contains the value of RDF by DP OpenMM simulation.', default=None, required=False)


    args = parser.parse_args()
    
    analysis = args.type
    mole = args.mole
    output_log = args.log
    pdb_file = args.pdb

    num_log = args.nlog
    log_prefix = args.log_prefix
    temp1 = args.temp1
    temp2 = args.temp2

    num_dcd = args.ndcd
    dcd_prefix = args.dcd_prefix

    Lambda = args.Lambda
    dcd_lambda = args.dcd_lambda
    box = args.box

    npy_prefix = args.npy_prefix
    N_max = args.max_frame
    lambda_list = args.lambda_list
    if lambda_list is not None:
        lambda_list = lambda_list.split(',')
        lambda_list = [float(x) for x in lambda_list]

    if analysis == "nveEnergy":
        readEnergyAndDraw(mole, output_log, totalOnly=args.totalOnly)
    if analysis == "logEnergy":
        logs_1 = []
        logs_2 = []
        for ii in range(num_log):
            if ii == 0:
                continue
            logs_1.append(log_prefix + "."+str(temp1)+"."+str(ii)+".log")
            logs_2.append(log_prefix + "."+str(temp2)+"."+str(ii)+".log")

        draw_potential_distribution(logs_1, logs_2, temp1, temp2) 
    if analysis == "ooRDF":
        dcd_files = []
        for ii in range(num_dcd):
            if ii == 0:
                continue
            temp_dcd = dcd_prefix+"."+str(ii)+".dcd"
            dcd_files.append(temp_dcd)
        getRDF_Oxygen_Oxygen(pdb_file, dcd_files, dcd_prefix+"_oxygen_oxygen")
    if analysis == "ohRDF":
        dcd_files = []
        for ii in range(num_dcd):
            if ii == 0:
                continue
            temp_dcd = dcd_prefix+"."+str(ii)+".dcd"
            dcd_files.append(temp_dcd)
        getRDF_Oxygen_Hydrogen(pdb_file, dcd_files, dcd_prefix+"_oxygen_hydrogen")

    if analysis == "hhRDF":
        dcd_files = []
        for ii in range(num_dcd):
            if ii == 0:
                continue
            temp_dcd = dcd_prefix+"."+str(ii)+".dcd"
            dcd_files.append(temp_dcd)
        getRDF_Hydrogen_Hydrogen(pdb_file, dcd_files, dcd_prefix+"_hydrogen_hydrogen")

    if analysis == "alchemEnergy":
        model_file = "./frozen_model/graph_from_han_dp2.0_compress.pb"
        # Set the alchemical residue id.
        resid = 1
        box = [args.box, 0, 0, 0, args.box, 0, 0, 0, args.box]
        # Construct the alchemical context for potential energy calculation.
        alchem_context = AlchemicalContext(resid, Lambda, model_file, model_file, model_file, pdb_file, box)        
        
        dcd_files = []
        for ii in range(num_dcd):
            temp_dcd = dcd_prefix+"."+str(ii)+".dcd"
            dcd_files.append(temp_dcd)
        save_file = "./output/" + mole + "_dcd.lambda."+str(dcd_lambda)+"_hamilton.lambda."+str(Lambda)
        alchemEnergy(alchem_context, pdb_file, dcd_files, save_file)

    if analysis == "alchemMBAR":
        hfe = []
        x = []
        points_gap = 100
        n = int(N_max /points_gap)
        
        for ii in range(n):
            temp = alchemMBAR(npy_prefix, int((ii+1) * points_gap), lambda_list)
            hfe.append(temp)
            x.append((ii+1)*points_gap*1)
        hfe = np.array(hfe)
        x = np.array(x)
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib can not be imported.")

        plt.clf()
        plt.figure(figsize=(10,8))
        font = {'family' : 'normal',
            #'weight' : 'bold',
            'size'   : 13}
        plt.rc('font', **font)
        plt.tick_params(direction='in')

        plt.ylim(-32, -29)
        plt.yticks(np.arange(-32, -28.5, step=0.5))
        plt.xlim(0, 3100)
        plt.scatter(x, hfe)
        plt.plot(x, hfe, '-')
        plt.xlabel("Time (ps)")
        plt.ylabel("Hydration Free Energy (kJ/mol)")

        plt.savefig("./output/alchem_timeseries.png")        
        
        #DrawScatter(x, hfe, "alchem_timeseries", xlabel="Time (ps)", ylabel="HFE (kJ/mol)")
    
    if analysis == "compareRDF":
        rdf_gmx_file = args.rdf_gmx
        rdf_lmp_file = args.rdf_lmp
        rdf_omm_file = args.rdf_omm
        fig_name = "rdf_omm_gmx_lmp"
        import matplotlib.pyplot as plt
        plt.clf()
        font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 13}
        plt.rc('font', **font)
        plt.tick_params(direction='in')
        
        plt.xlim(2, 10)
        plt.ylim(0.0, 3.7)
        plt.ylabel("g(r)")
        plt.xlabel("r($\AA$)")
        #plt.title("RDF of Oxygen-Oxygen", fontweight='bold')

        if rdf_gmx_file is not None:
            temp_bins = []
            temp_rdf = []
            with open(rdf_gmx_file, 'r') as f:
                content = f.readlines()
                for line in content:
                    temp = line.split(',')
                    bin_index = float(temp[0])
                    rdf = float(temp[1])
                    if bin_index * 10 >= 2:
                        temp_bins.append(bin_index * 10)
                        temp_rdf.append(rdf)
            plt.plot(temp_bins, temp_rdf, 'r-', label = "Gromacs")

        if rdf_lmp_file is not None:
            temp_bins = []
            temp_rdf = []
            with open(rdf_lmp_file, 'r') as f:
                content = f.readlines()
                for line in content:
                    temp = line.split(',')
                    bin_index = float(temp[0])
                    rdf = float(temp[1])
                    if bin_index * 10 >= 2:
                        temp_bins.append(bin_index * 10)
                        temp_rdf.append(rdf)

            plt.plot(temp_bins, temp_rdf, 'b-', label = "LAMMPS")

        if rdf_omm_file is not None:
            temp_bins = []
            temp_rdf = []
            with open(rdf_omm_file, 'r') as f:
                content = f.readlines()
                for line in content:
                    temp = line.split(',')
                    bin_index = float(temp[0])
                    rdf = float(temp[1])
                    if bin_index >= 2:
                        temp_bins.append(bin_index)
                        temp_rdf.append(rdf)

            plt.plot(temp_bins, temp_rdf, 'g-', label = "OpenMM")
        
        plt.legend()
        plt.savefig("./output/"+fig_name+'.png')
        
    if analysis == "drawNVE":
        nve_log = args.log
        save_fig = "output/nve.png"
        draw_nve_figure4presentation(nve_log, save_fig)
    
    if analysis == "drawMSD":
        dcd_files = []
        for ii in range(num_dcd):
            temp_dcd = dcd_prefix+"."+str(ii)+".dcd"
            dcd_files.append(temp_dcd)
        draw_MSD(pdb_file, dcd_files, dcd_prefix+".msd")
