#%%
import os, sys, time
import numpy as np
from tqdm import tqdm
from utils import DeepPotentialContext
import simtk.openmm as mm
from simtk import unit as u
import argparse



def check_energy_force_set(set_dir, dp_context, natoms):
    force_npy = os.path.join(set_dir, "force.npy")
    energy_npy = os.path.join(set_dir, "energy.npy")
    coord_npy = os.path.join(set_dir, "coord.npy")
    box_npy = os.path.join(set_dir, "box.npy")

    force = np.load(force_npy)
    energy = np.load(energy_npy)
    coord = np.load(coord_npy)
    box = np.load(box_npy)
    frame = 0
    omm_dp_energy = []
    omm_dp_forces = []
    diff = {"force":[], "energy":[], "posi":[]}

    natoms = len(type_list)
    for ii in tqdm(range(coord.shape[0]), desc="Iterate for each frame in coord"):
        posi = []
        box4omm = [mm.Vec3(box[ii][0], box[ii][1], box[ii][2]), mm.Vec3(box[ii][3], box[ii][4], box[ii][5]), mm.Vec3(box[ii][6], box[ii][7], box[ii][8])] * u.angstroms
        #print("---------------")
        #print("Input coord from npy:")
        for jj in range(natoms):
            atom_posi = mm.Vec3( coord[ii][jj * 3 + 0], coord[ii][jj * 3 + 1], coord[ii][jj * 3 + 2]) * u.angstroms
            posi.append(atom_posi)

        ene, f, omm_posi = dp_context.getEnergyForcesPositions(posi, box4omm)
        omm_posi = omm_posi.reshape((-1))
        omm_posi = omm_posi.value_in_unit(u.angstroms)

        #print(omm_posi, coord[ii])

        f = f.reshape((-1))
        ene = ene/ 96.48792534459
        f = f/964.8792534459
        f = np.array(f)
        ene = ene._value

        diff_posi = np.sqrt(np.mean(np.power(coord[ii] - omm_posi, 2)))
        #print(ene, f[:3], diff_posi, energy[ii], force[ii][:3])

        temp_force_diff = np.sqrt(np.mean((force[ii] - f)**2))
        temp_ene_diff = np.sqrt((energy[ii] - ene)*(energy[ii] - ene))
        diff['force'].append(temp_force_diff)
        diff['energy'].append(temp_ene_diff)
        diff['posi'].append(diff_posi)

        omm_dp_forces.append(f)
        omm_dp_energy.append(ene)

    omm_dp_energy = np.array(omm_dp_energy)
    omm_dp_forces = np.array(omm_dp_forces)
    diff['force'] = np.array(diff['force'])
    diff['energy'] = np.array(diff['energy'])

    #print(np.mean(diff['force']), np.mean(diff['energy']))

    #print(omm_dp_energy.shape, energy.shape, omm_dp_forces.shape, force.shape)

    diff_energy = np.sqrt(np.mean(np.power(np.subtract(omm_dp_energy, energy), 2))) 
    diff_forces = np.sqrt(np.mean(np.power(np.subtract(omm_dp_forces, force), 2)))

    print("Diff on Energy: ", diff_energy, coord.shape[0])
    print("Diff on Forces: ", diff_forces, coord.shape[0])

    print(omm_dp_energy.shape, energy.shape)

    print(omm_dp_energy[:10], energy[:10])
    print(omm_dp_forces[0][:10], force[0][:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dat-dir', dest = "dat_dir", help='Data directory.')
    parser.add_argument('--model', dest = "graph", help='Path to graph model.')
    args = parser.parse_args()

    dat_dir = args.dat_dir
    graph = args.graph
    type_raw = os.path.join(dat_dir, "type.raw")
    with open(type_raw, 'r') as f:
        content = f.readlines()
    type_list = [int(x) for x in content]
    natoms = len(type_list)

    dp_context = DeepPotentialContext(graph, type_list)

    directory_list = os.listdir(dat_dir)
    set_dir_list = [x for x in directory_list if x.startswith("set")]

    for set_dir in set_dir_list:
        print(set_dir, dat_dir)
        check_energy_force_set(os.path.join(dat_dir, set_dir), dp_context, natoms)
    