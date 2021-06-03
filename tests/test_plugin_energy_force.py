#%%
import os, sys, time
import numpy as np
from tqdm import tqdm
from utils import DeepPotentialContext
import simtk.openmm as mm
from simtk import unit as u

set_dir = "/home/dingye/Documents/Data/Water/lw_pimd/set.000"
force_npy = os.path.join(set_dir, "force.npy")
energy_npy = os.path.join(set_dir, "energy.npy")
coord_npy = os.path.join(set_dir, "coord.npy")
box_npy = os.path.join(set_dir, "box.npy")
type_npy = os.path.join(set_dir, "type.npy")
graph = "./graph/lw_pimd.se_a_v1.2.0.float_step.1M_batchSize.1.pb"

#%%
force = np.load(force_npy)
energy = np.load(energy_npy)
coord = np.load(coord_npy)
types = np.load(type_npy)
box = np.load(box_npy)


frame = 0
type_list = types[0]

dp_context = DeepPotentialContext(graph, type_list)
#%%

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



