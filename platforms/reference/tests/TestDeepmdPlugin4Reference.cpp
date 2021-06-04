/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
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

#include "DeepmdForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/Platform.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include "neighborList.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>


using namespace OpenMM;
using namespace DeepmdPlugin;
using namespace std;

extern "C" OPENMM_EXPORT void registerDeepmdReferenceKernelFactories();

const double TOL = 1e-5;
const string graph = "../../../../frozen_model/lw_pimd.se_a_v1.2.0.float_step.1M_batchSize.1.pb";
const string op_path = "/home/dingye/.local/deepmd-kit-1.2.0/lib/libdeepmd_op.so";
const double coordUnitCoeff = 10;
const double forceUnitCoeff = 964.8792534459;
const double energyUnitCoeff = 96.48792534459;



void referenceDeepmdForce(vector<Vec3> positions, vector<Vec3> box, vector<int> types, vector<Vec3>& force, double& energy, NNPInter nnp_inter){
    int natoms = positions.size();
    vector<VALUETYPE> nnp_coords(natoms*3);
    vector<VALUETYPE> nnp_box(9);
    vector<VALUETYPE> nnp_force(natoms*3);
    vector<VALUETYPE> nnp_virial(9);
    double nnp_energy;
    
    // Set box and coordinates input for NNPInter.
    for (int ii = 0; ii < natoms; ++ii){
        nnp_coords[ii * 3 + 0] = positions[ii][0] * coordUnitCoeff;
        nnp_coords[ii * 3 + 1] = positions[ii][1] * coordUnitCoeff;
        nnp_coords[ii * 3 + 2] = positions[ii][2] * coordUnitCoeff;
    }
    nnp_box[0] = box[0][0] * coordUnitCoeff;
    nnp_box[1] = box[0][1] * coordUnitCoeff;
    nnp_box[2] = box[0][2] * coordUnitCoeff;
    nnp_box[3] = box[1][0] * coordUnitCoeff;
    nnp_box[4] = box[1][1] * coordUnitCoeff;
    nnp_box[5] = box[1][2] * coordUnitCoeff;
    nnp_box[6] = box[2][0] * coordUnitCoeff;
    nnp_box[7] = box[2][1] * coordUnitCoeff;
    nnp_box[8] = box[2][2] * coordUnitCoeff;
    int nghost = 0;
    // Compute the nnp forces and energy.
    nnp_inter.compute (nnp_energy, nnp_force, nnp_virial, nnp_coords, types, nnp_box, nghost);
    // Assign the energy and forces for return.
    energy = nnp_energy * energyUnitCoeff;
    for(int ii = 0; ii < natoms; ++ii){
        force[ii][0] = nnp_force[ii * 3 + 0] * forceUnitCoeff;
        force[ii][1] = nnp_force[ii * 3 + 1] * forceUnitCoeff;
        force[ii][2] = nnp_force[ii * 3 + 2] * forceUnitCoeff;
    }
}

void testDeepmdDynamics(vector<VALUETYPE> init_positions, vector<string> particleTypeName, map<string, int> typeDict, vector<double> mass, int nsteps=1000){
    System system;
    VerletIntegrator integrator(0.0001); // Time step is 0.0001 ps here.
    DeepmdForce dp_force = DeepmdForce(graph, " ", " ", false);
    dp_force.setDeepmdOpFile(op_path);
    
    int natoms;
    natoms = int(init_positions.size()/3);
    vector<Vec3> positions;
    // units is nanometers for box size setting.
    vector<Vec3> box = {Vec3(1.3,0.0,0.0),Vec3(0.0,1.3,0.0),Vec3(0.0,0.0,1.3)};
    vector<int> types;

    for(auto it = typeDict.begin(); it != typeDict.end(); it++){
        dp_force.addType(it->second, it->first);
    }
    for (int ii = 0; ii < natoms; ++ii){
        system.addParticle(mass[ii]);
        dp_force.addParticle(ii, particleTypeName[ii]);    
        types.push_back(typeDict[particleTypeName[ii]]);
        positions.push_back(Vec3(init_positions[ii * 3 + 0], init_positions[ii * 3 + 1], init_positions[ii * 3 + 2]));
    }
    dp_force.setUnitTransformCoefficients(coordUnitCoeff, forceUnitCoeff, energyUnitCoeff); 
    system.addForce(&dp_force);
    
    Platform& platform = Platform::getPlatformByName("Reference");

    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setPeriodicBoxVectors(box[0], box[1], box[2]);

    // Initialize the nnp_inter.
    NNPInter nnp_inter = NNPInter(graph, 0);
    // Record the difference of forces and energy on each step.
    vector<double> errorForce;
    vector<double> errorEnergy;
    double err = 0.;

    vector<Vec3> omm_forces(natoms, Vec3(0,0,0));
    vector<Vec3> forces(natoms, Vec3(0,0,0));
    
    double omm_energy, omm_kinetic_energy, energy;

    for (int ii = 0; ii < nsteps; ++ii){
        // Running dynamics 1 step.
        integrator.step(1);
        
        // Get the force from openmm context state.
        State state = context.getState(State::Forces | State::Energy | State::Positions);
        omm_forces = state.getForces();
        omm_energy = state.getPotentialEnergy();
        omm_kinetic_energy = state.getKineticEnergy();
        // Calculate the force from NNPInter directly.
        referenceDeepmdForce(state.getPositions(), box, types, forces, energy, nnp_inter);
        // Compare the difference of these two result.
        err = 0.;
        for (int jj = 0; jj < natoms; ++jj){
            // TODO: NeighborList in reference platform have an problem to be solved in future. 
            cout<<"check force on atom "<<jj<<endl;
            cout<<"Coords of atom "<<jj<<endl;
            for(int kk = 0; kk < 3; ++kk){
                cout<<state.getPositions()[jj][kk]<<" ";
            }
            cout<<endl;
            ASSERT_EQUAL_VEC(omm_forces[jj], forces[jj], TOL);
            Vec3 diff_f = omm_forces[jj] - forces[jj];
            double diff = diff_f.dot(diff_f);
            err += diff;
        }
        ASSERT_EQUAL_TOL(energy, omm_energy, TOL);
        err = err/natoms;
        err = sqrt(err);
        errorForce.push_back(err);
        err = abs(omm_energy - energy);
        cout<<omm_energy + omm_kinetic_energy<< " "<< omm_energy<< " "<< energy << endl;
        errorEnergy.push_back(err);
    }

    double meanErrorEnergy = 0.;
    double meanErrorForce = 0.;
    for(int ii = 0; ii < natoms; ++ii){
        meanErrorForce += errorForce[ii];
        meanErrorEnergy += errorEnergy[ii];
    }
    meanErrorForce = meanErrorForce / natoms;
    meanErrorEnergy = meanErrorEnergy / natoms;

    std::cout.precision(10);

    std::cout<<"Mean Force Difference: "<<meanErrorForce<<"; ";
    std::cout<<"Mean Energy Difference: "<<meanErrorEnergy<<"; ";
    std::cout<<"in "<<nsteps<<" steps."<<std::endl;
}

// TODO: Test Deepmd Plugin with the training data.
void testDeepmdWithNPY(string coord_npy, string force_npy, string type_npy, string box_npy){

}

void testDeepmdNeighbors(vector<VALUETYPE> init_positions, vector<string> particleTypeName, map<string, int> typeDict, vector<double> mass, int nsteps=1000){
    System system;
    VerletIntegrator integrator(0.0001); // Time step is 0.0005 ps here.
    DeepmdForce dp_force = DeepmdForce(graph, " ", " ", false);
    dp_force.setDeepmdOpFile(op_path);
    
    int natoms;
    natoms = int(init_positions.size()/3);
    vector<Vec3> positions;
    // units is nanometers for box size setting.
    vector<Vec3> box = {Vec3(2.0,0.0,0.0),Vec3(0.0,2.0,0.0),Vec3(0.0,0.0,2.0)};
    vector<int> types;

    for(auto it = typeDict.begin(); it != typeDict.end(); it++){
        dp_force.addType(it->second, it->first);
    }
    for (int ii = 0; ii < natoms; ++ii){
        system.addParticle(mass[ii]);
        dp_force.addParticle(ii, particleTypeName[ii]);    
        types.push_back(typeDict[particleTypeName[ii]]);
        positions.push_back(Vec3(init_positions[ii * 3 + 0], init_positions[ii * 3 + 1], init_positions[ii * 3 + 2]));
    }
    dp_force.setUnitTransformCoefficients(coordUnitCoeff, forceUnitCoeff, energyUnitCoeff); 
    system.addForce(&dp_force);
    
    Platform& platform = Platform::getPlatformByName("Reference");

    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setPeriodicBoxVectors(box[0], box[1], box[2]);

    // Set the input for deepmd neighborlist calculation.
    vector<compute_t> box4nei(9, 0.);
    for(int jj = 0; jj < box.size(); ++jj){
        box4nei[jj * 3 + 0] = box[jj][0];
        box4nei[jj * 3 + 1] = box[jj][1];
        box4nei[jj * 3 + 2] = box[jj][2];
    }
    SimulationRegion<compute_t> region;
    region.reinitBox(box4nei.data());

    // Initialize the nnp_inter.
    NNPInter nnp_inter = NNPInter(graph, 0);
    // Record the difference of forces and energy on each step.
    vector<double> errorForce;
    vector<double> errorEnergy;
    double err = 0.;

    vector<Vec3> omm_forces(natoms, Vec3(0,0,0));
    vector<Vec3> forces(natoms, Vec3(0,0,0));
    vector<Vec3> posi;
    vector<set<int> > empty(natoms);
    NeighborList* omm_neighborList = new NeighborList();
    vector<vector<int>> neighbors(natoms, vector<int>());

    double omm_energy, omm_kinetic_energy, energy;
    double cutoff = 0.6+0.2;

    for (int ii = 0; ii < nsteps; ++ii){
        // Running dynamics 1 step.
        integrator.step(1);
        
        cout<<"********************************"<<endl;
        // Get the force from openmm context state.
        State state = context.getState(State::Forces | State::Energy | State::Positions);
        omm_forces = state.getForces();
        omm_energy = state.getPotentialEnergy();
        omm_kinetic_energy = state.getKineticEnergy();
        
        // Calculate the atoms neighbors list for positions.
        posi = state.getPositions();
        computeNeighborListVoxelHash(*omm_neighborList, natoms, posi, empty, box.data(),  true, cutoff, 0.0);

        for(auto it = omm_neighborList->begin(); it != omm_neighborList->end(); ++it){
            neighbors[it->first].push_back(it->second);
            neighbors[it->second].push_back(it->first);
        }

        // Construct the input coordinates for deepmd neighborlist calculation function.
        vector<compute_t> positions4nei(natoms * 3, 0.);
        for (int jj = 0; jj < natoms; ++jj){
            positions4nei[jj * 3 + 0] = posi[jj][0];
            positions4nei[jj * 3 + 1] = posi[jj][1];
            positions4nei[jj * 3 + 2] = posi[jj][2];
        }
        vector<vector<int>> dp_neighbor_a;
        vector<vector<int>> dp_neighbor_r;
        // Calculate the neighbor list with dp's function;
        getNeighborsFromDescriptor(dp_neighbor_a, dp_neighbor_r, box4nei, positions4nei, types, cutoff, cutoff, natoms, region);

        for(int jj = 0; jj < natoms; ++jj){
            cout<<"++++++++++"<<endl;
            cout<<"Neighbors for atom "<<jj<< " with omm";
            cout<<" ("<<posi[jj][0];
            cout<<" "<<posi[jj][1];
            cout<<" "<<posi[jj][2]<<") "<<endl;

            for(int kk = 0 ; kk < neighbors[jj].size(); ++kk){
                cout<<neighbors[jj][kk]<< " ";
                if (find(dp_neighbor_a[jj].begin(), dp_neighbor_a[jj].end(),neighbors[jj][kk]) == dp_neighbor_a[jj].end()){
                    cout<<"("<<posi[neighbors[jj][kk]][0];
                    cout<<" "<<posi[neighbors[jj][kk]][1];
                    cout<<" "<<posi[neighbors[jj][kk]][2]<<") ";
                    vector<compute_t> diff(3);
                    region.diffNearestNeighbor(posi[jj][0], posi[jj][1], posi[jj][2], posi[neighbors[jj][kk]][0], posi[neighbors[jj][kk]][1], posi[neighbors[jj][kk]][2], diff[0], diff[1], diff[2]);
                    cout<<"(diff "<<diff[0];
                    cout<<" "<<diff[1];
                    cout<<" "<<diff[2]<<") ";
                }
            }
            cout<<endl;
            
            cout<<"-----------------"<<endl;

            cout<<"Neighbors for atom "<<jj<< " with dp";
            cout<<" ("<<positions4nei[jj* 3 + 0];
            cout<<" "<<positions4nei[jj* 3 + 1];
            cout<<" "<<positions4nei[jj* 3 + 2]<<") "<<endl;
            for(int kk = 0 ; kk < dp_neighbor_a[jj].size(); ++kk){
                cout<<dp_neighbor_a[jj][kk]<< " ";
                if (find(neighbors[jj].begin(), neighbors[jj].end(),dp_neighbor_a[jj][kk]) == neighbors[jj].end()){
                    cout<<"("<<positions4nei[dp_neighbor_a[jj][kk] * 3 + 0];
                    cout<<" "<<positions4nei[dp_neighbor_a[jj][kk] * 3 + 1];
                    cout<<" "<<positions4nei[dp_neighbor_a[jj][kk] * 3 + 2]<<") ";
                    vector<compute_t> diff(3);
                    region.diffNearestNeighbor(positions4nei[jj * 3 + 0], positions4nei[jj * 3 + 1], positions4nei[jj * 3 + 2],  positions4nei[dp_neighbor_a[jj][kk] * 3 + 0], positions4nei[dp_neighbor_a[jj][kk] * 3 + 1], positions4nei[dp_neighbor_a[jj][kk] * 3 + 2], diff[0], diff[1], diff[2]);
                    cout<<"(diff "<<diff[0];
                    cout<<" "<<diff[1];
                    cout<<" "<<diff[2]<<") ";
                }
            }
            cout<<endl;

            if (dp_neighbor_a[jj].size() != neighbors[jj].size()){
                cout<< "Attention that dp neighbor list length is not the same as the omm neighbor list! "<<dp_neighbor_a[jj].size()<< " "<<neighbors[jj].size()<<endl;
            }

            neighbors[jj].clear();
            dp_neighbor_a[jj].clear();
            dp_neighbor_r[jj].clear();
        }

        // Show potetntial energy and total energy.
        cout<<"-----Total-------Potential-----------"<<endl;
        cout<< omm_energy + omm_kinetic_energy << " " << omm_energy << endl;
        cout<<"-------------------------------------"<<endl;
    }

}



int main() {
    // Initialize positions, unit is angstrom.
    //vector<VALUETYPE> init_positions = {10.543000221252441, 14.571999549865723,7.9380002021789551,10.170000076293945,15.211000442504883,7.3270001411437988,11.420999526977539,14.894000053405762,8.116999626159668,3.4600000381469727,6.3179998397827148,1.784000039100647,2.7950000762939453,6.4429998397827148,1.1039999723434448,3.5339999198913574,5.3730001449584961,1.878000020980835,1.3240000009536743,14.984000205993652,5.8090000152587891,1.6160000562667847,15.803000450134277,6.2129998207092285,0.47200000286102295,15.189999580383301,5.434999942779541,3.9010000228881836,18.275999069213867,0.55199998617172241,3.684999942779541,18.003000259399414,1.4450000524520874,3.5929999351501465,17.562999725341797,0.0010000000474974513};
    //vector<string> particleTypeName = {"O", "H", "H", "O", "H", "H", "O", "H", "H", "O", "H", "H"};

    
    vector<VALUETYPE> init_positions = positions_64_waters;
    vector<string> particleTypeName = types_64_waters;

    map<string, int> typeDict = {{"O", 0}, {"H", 1}};
    vector<double> mass;

    int nsteps = 10;

    int natoms = init_positions.size() / 3;
    for(int ii = 0; ii < natoms; ++ii){
        if (particleTypeName[ii].compare("O") == 0)
            mass.push_back(15.99943);
        else if (particleTypeName[ii].compare("H") == 0)
            mass.push_back(1.007947);
        // Convert the input coordinates unit from angstrom to nanometers.
        init_positions[ii * 3 + 0] = init_positions[ii * 3 + 0] * 0.1;
        init_positions[ii * 3 + 1] = init_positions[ii * 3 + 1] * 0.1;
        init_positions[ii * 3 + 2] = init_positions[ii * 3 + 2] * 0.1;
    }
    
    registerDeepmdReferenceKernelFactories();
    testDeepmdDynamics(init_positions, particleTypeName, typeDict, mass, nsteps);
    //testDeepmdNeighbors(init_positions, particleTypeName, typeDict, mass, nsteps);
    //testDeepmdWithNPY("", "", "", "");

    std::cout<<"Done"<<std::endl;
    return 0;
}