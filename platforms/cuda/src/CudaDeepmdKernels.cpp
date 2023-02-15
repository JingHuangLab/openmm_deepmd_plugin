/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaDeepmdKernels.h"
#include "CudaDeepmdKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <algorithm>

using namespace DeepmdPlugin;
using namespace OpenMM;
using namespace std;


CudaCalcDeepmdForceKernel::~CudaCalcDeepmdForceKernel(){
   return;
}

void CudaCalcDeepmdForceKernel::initialize(const System& system, const DeepmdForce& force){
    graph_file = force.getDeepmdGraphFile();
    type4EachParticle = force.getType4EachParticle();
    typesIndexMap = force.getTypesIndexMap();
    used4Alchemical = force.alchemical();
    forceUnitCoeff = force.getForceUnitCoefficient();
    energyUnitCoeff = force.getEnergyUnitCoefficient();
    coordUnitCoeff = force.getCoordUnitCoefficient();
    
    //natoms = system.getNumParticles();
    natoms = type4EachParticle.size();
    tot_atoms = system.getNumParticles();

    // Load the ordinary graph firstly.
    dp = DeepPot(graph_file);
    if(used4Alchemical){
        cout<<"Deep Potential Alchemical simulation. Load the other two graphs here."<<endl;
        graph_file_1 = force.getGraph1_4Alchemical();
        graph_file_2 = force.getGraph2_4Alchemical();
        dp_1 = DeepPot(graph_file_1);
        dp_2 = DeepPot(graph_file_2);
        lambda = force.getLambda();
        atomsIndex4Graph1 = force.getAtomsIndex4Graph1();
        atomsIndex4Graph2 = force.getAtomsIndex4Graph2();
        natoms4alchemical[1] = atomsIndex4Graph1.size();
        natoms4alchemical[2] = atomsIndex4Graph2.size();
        
        // pair<int, int> stores the atoms index in U_B. This might be useful for force assign.
        atomsIndexMap4U_B = vector<pair<int,int>>(natoms4alchemical[1] + natoms4alchemical[2]);

        // Initialize the input and output array for alchemical simulation.
        dener4alchemical[1] = 0.0;
        dforce4alchemical[1] = vector<VALUETYPE>(natoms4alchemical[1] * 3, 0.);
        dvirial4alchemical[1] = vector<VALUETYPE>(9, 0.);
        dcoord4alchemical[1] = vector<VALUETYPE>(natoms4alchemical[1] * 3, 0.);
        dbox4alchemical[1] = vector<VALUETYPE>(9, 0.);
        dtype4alchemical[1] = vector<int>(natoms4alchemical[1], 0);
        
        for(int ii = 0; ii < natoms4alchemical[1]; ++ii){
            int index = atomsIndex4Graph1[ii];
            atomsIndexMap4U_B[index] = make_pair(1, ii);
            dtype4alchemical[1][ii] = typesIndexMap[type4EachParticle[index]];
        }
        
        dener4alchemical[2] = 0.0;
        dforce4alchemical[2] = vector<VALUETYPE>(natoms4alchemical[2] * 3, 0.);
        dvirial4alchemical[2] = vector<VALUETYPE>(9, 0.);
        dcoord4alchemical[2] = vector<VALUETYPE>(natoms4alchemical[2] * 3, 0.);
        dbox4alchemical[2] = vector<VALUETYPE>(9, 0.);
        dtype4alchemical[2] = vector<int>(natoms4alchemical[2], 0);
        
        for(int ii = 0; ii < natoms4alchemical[2]; ++ii){
            int index = atomsIndex4Graph2[ii];
            atomsIndexMap4U_B[index] = make_pair(2, ii);
            dtype4alchemical[2][ii] = typesIndexMap[type4EachParticle[index]];
        }

        if ((natoms4alchemical[1] + natoms4alchemical[2]) != natoms){
        throw OpenMMException("Wrong atoms number for graph1 and graph2. Summation of atoms number in graph 1 and 2 is not equal to total atoms number.");
        }
    }

    // Initialize the ordinary input and output array.
    // Initialize the input tensor.
    dener = 0.;
    dforce = vector<VALUETYPE>(natoms * 3, 0.);
    dvirial = vector<VALUETYPE>(9, 0.);
    dcoord = vector<VALUETYPE>(natoms * 3, 0.);
    dbox = vector<VALUETYPE>(9, 0.);
    //dtype = vector<int>(natoms, 0);    
    // Set atom type;
    //for(int ii = 0; ii < natoms; ii++){
        // ii is the atom index of each particle.
    //    dtype[ii] = typesIndexMap[type4EachParticle[ii]];
    //}

    for(std::map<int, string>::iterator it = type4EachParticle.begin(); it != type4EachParticle.end(); ++it){
        dp_particles.push_back(it->first);
        dtype.push_back(typesIndexMap[it->second]);
    }

    AddedForces = vector<double>(tot_atoms * 3, 0.0);
    // Set for CUDA context.
    cu.setAsCurrent();
    map<string, string> defines;
    defines["FORCES_TYPE"] = "double";
    networkForces.initialize(cu, 3*natoms, sizeof(double), "networkForces");
    CUmodule module = cu.createModule(CudaDeepmdKernelSources::DeepmdForce, defines);
    addForcesKernel = cu.getKernel(module, "addForces");
}


double CudaCalcDeepmdForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3> pos;
    context.getPositions(pos);
    Vec3 box[3];

    // Set box size.
    if (context.getSystem().usesPeriodicBoundaryConditions()){
        cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
        // Transform unit from nanometers to angstrom.
        dbox[0] = box[0][0] * coordUnitCoeff;
        dbox[1] = box[0][1] * coordUnitCoeff;
        dbox[2] = box[0][2] * coordUnitCoeff;
        dbox[3] = box[1][0] * coordUnitCoeff;
        dbox[4] = box[1][1] * coordUnitCoeff;
        dbox[5] = box[1][2] * coordUnitCoeff;
        dbox[6] = box[2][0] * coordUnitCoeff;
        dbox[7] = box[2][1] * coordUnitCoeff;
        dbox[8] = box[2][2] * coordUnitCoeff;
    }else{
        dbox = {}; // No PBC.
    }
    // Set input coord.
    for(int ii = 0; ii < natoms; ++ii){
        // Multiply by coordUnitCoeff means the transformation of the unit from nanometers to required input unit for positions in trained DP model.
        int atom_index = dp_particles[ii];
        dcoord[ii * 3 + 0] = pos[atom_index][0] * coordUnitCoeff;
        dcoord[ii * 3 + 1] = pos[atom_index][1] * coordUnitCoeff;
        dcoord[ii * 3 + 2] = pos[atom_index][2] * coordUnitCoeff;
    }
    // Assign the input coord for alchemical simulation.
    if(used4Alchemical){
        // Set the input coord and box array for graph 1 first.
        for(int ii = 0; ii < natoms4alchemical[1]; ii ++){
            int index = atomsIndex4Graph1[ii];
            dcoord4alchemical[1][ii * 3 + 0] = pos[index][0] * coordUnitCoeff;
            dcoord4alchemical[1][ii * 3 + 1] = pos[index][1] * coordUnitCoeff;
            dcoord4alchemical[1][ii * 3 + 2] = pos[index][2] * coordUnitCoeff;
        }
        dbox4alchemical[1] = dbox;

        // Set the input coord and box array for graph 2.
        for(int ii = 0; ii < natoms4alchemical[2]; ii ++){
            int index = atomsIndex4Graph2[ii];
            dcoord4alchemical[2][ii * 3 + 0] = pos[index][0] * coordUnitCoeff;
            dcoord4alchemical[2][ii * 3 + 1] = pos[index][1] * coordUnitCoeff;
            dcoord4alchemical[2][ii * 3 + 2] = pos[index][2] * coordUnitCoeff;
        }
        dbox4alchemical[2] = dbox;
    }

    dp.compute (dener, dforce, dvirial, dcoord, dtype, dbox);
    
    if (used4Alchemical){
        // Compute the first graph.
        dp_1.compute (dener4alchemical[1], dforce4alchemical[1], dvirial4alchemical[1], dcoord4alchemical[1], dtype4alchemical[1], dbox4alchemical[1]);
        // Compute the second graph.
        dp_2.compute (dener4alchemical[2], dforce4alchemical[2], dvirial4alchemical[2], dcoord4alchemical[2], dtype4alchemical[2], dbox4alchemical[2]);
    }

    if(used4Alchemical){
        for(int ii = 0; ii < natoms; ii++){
            // ii is the index of the atom.
            // Interpolate the alchemical forces.
            if(atomsIndexMap4U_B[ii].first == 1){
                // Get the force from ordinary graph and graph_1.
                int index4U_B = atomsIndexMap4U_B[ii].second;
                // F = \lambda * (F_A) + (1 - \lambda) * F_1
                AddedForces[ii * 3 + 0] = (lambda * dforce[ii * 3 + 0] + (1 - lambda) * (dforce4alchemical[1][index4U_B * 3 + 0])) * forceUnitCoeff;
                AddedForces[ii * 3 + 1] = (lambda * dforce[ii * 3 + 1] + (1 - lambda) * (dforce4alchemical[1][index4U_B * 3 + 1])) * forceUnitCoeff;
                AddedForces[ii * 3 + 2] = (lambda * dforce[ii * 3 + 2] + (1 - lambda) * (dforce4alchemical[1][index4U_B * 3 + 2])) * forceUnitCoeff;
            } else if (atomsIndexMap4U_B[ii].first == 2){
                // Get the force from ordinary graph and graph_2.
                int index4U_B = atomsIndexMap4U_B[ii].second;
                // F = \lambda * (F_A) + (1 - \lambda) * F_1
                AddedForces[ii * 3 + 0] = (lambda * dforce[ii * 3 + 0] + (1 - lambda) * (dforce4alchemical[2][index4U_B * 3 + 0])) * forceUnitCoeff;
                AddedForces[ii * 3 + 1] = (lambda * dforce[ii * 3 + 1] + (1 - lambda) * (dforce4alchemical[2][index4U_B * 3 + 1])) * forceUnitCoeff;
                AddedForces[ii * 3 + 2] = (lambda * dforce[ii * 3 + 2] + (1 - lambda) * (dforce4alchemical[2][index4U_B * 3 + 2])) * forceUnitCoeff;
            }
        }
        dener = lambda * dener + (1 - lambda) * (dener4alchemical[1] + dener4alchemical[2]);
        // Transform the unit from output energy unit to KJ/mol
        dener = dener * energyUnitCoeff;
    } else{
        // Transform the unit from output forces unit to KJ/(mol*nm)
        for(int ii = 0; ii < natoms; ii ++){
            int atom_index = dp_particles[ii];

            AddedForces[atom_index * 3 + 0] = dforce[ii * 3 + 0] * forceUnitCoeff;
            AddedForces[atom_index * 3 + 1] = dforce[ii * 3 + 1] * forceUnitCoeff;
            AddedForces[atom_index * 3 + 2] = dforce[ii * 3 + 2] * forceUnitCoeff;
        }
        dener = dener * energyUnitCoeff;
    }

    if (includeForces) {
        // Change to OpenMM CUDA context.
        cu.setAsCurrent();
        networkForces.upload(AddedForces);
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&networkForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &natoms, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, natoms);
    }
    return dener;
}



