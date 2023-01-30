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
#include <iostream>
#include <vector>
#include <cmath>


using namespace OpenMM;
using namespace DeepmdPlugin;
using namespace std;

extern "C" OPENMM_EXPORT void registerDeepmdReferenceKernelFactories();

const double TOL = 1e-5;
const string graph = "../tests/frozen_model/water.pb";
const double coordUnitCoeff = 10;
const double forceUnitCoeff = 964.8792534459;
const double energyUnitCoeff = 96.48792534459;
const double temperature = 300;
const int randomSeed = 123456;

void referenceDeepmdForce(vector<Vec3> positions, vector<Vec3> box, vector<int> types, vector<Vec3>& force, double& energy, DeepPot nnp_inter){
    
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
    nnp_inter.compute (nnp_energy, nnp_force, nnp_virial, nnp_coords, types, nnp_box);
    // Assign the energy and forces for return.
    energy = nnp_energy * energyUnitCoeff;
    for(int ii = 0; ii < natoms; ++ii){
        force[ii][0] = nnp_force[ii * 3 + 0] * forceUnitCoeff;
        force[ii][1] = nnp_force[ii * 3 + 1] * forceUnitCoeff;
        force[ii][2] = nnp_force[ii * 3 + 2] * forceUnitCoeff;
    }
}

void testDeepmdDynamics(int natoms, vector<string> names, vector<double> coord, vector<int> atype, vector<double> box, vector<double> mass, map<int, string> typeDict, int nsteps=100){


    ASSERT_EQUAL(names.size(), atype.size());

    System system;
    VerletIntegrator integrator(0.0002); // Time step is 0.2 fs here.
    DeepmdForce* dp_force = new DeepmdForce(graph);

    // Convert the units of coordinates and box from angstrom to nanometers.
    vector<Vec3> omm_coord;
    vector<Vec3> omm_box;
    for(auto it = typeDict.begin(); it != typeDict.end(); it++){
        dp_force->addType(it->first, it->second);
    }
    for(int ii = 0; ii < 3; ii++){
        omm_box.push_back(Vec3(box[ii * 3 + 0] / coordUnitCoeff, box[ii * 3 + 1] / coordUnitCoeff, box[ii * 3 + 2] / coordUnitCoeff));
    }
    for (int ii = 0; ii < natoms; ++ii){
        system.addParticle(mass[ii]);
        dp_force->addParticle(ii, names[ii]);    
        omm_coord.push_back(Vec3(coord[ii * 3 + 0] * 0.1, coord[ii * 3 + 1] * 0.1, coord[ii * 3 + 2] * 0.1));
    }
    dp_force->setUnitTransformCoefficients(coordUnitCoeff, forceUnitCoeff, energyUnitCoeff); 
    system.addForce(dp_force);

    Platform& platform = Platform::getPlatformByName("Reference");
    Context context(system, integrator, platform);
    context.setPositions(omm_coord);
    context.setPeriodicBoxVectors(omm_box[0], omm_box[1], omm_box[2]);
    context.setVelocitiesToTemperature(temperature, randomSeed);

    // Initialize the nnp_inter for comparision.
    DeepPot nnp_inter = DeepPot(graph);
    for (int ii = 0; ii < nsteps; ++ii){
        // Running dynamics 1 step.
        integrator.step(1);
        // Get the forces and energy from openmm context state.
        State state = context.getState(State::Forces | State::Energy | State::Positions);
        const vector<Vec3>& omm_forces = state.getForces();
        const double& omm_energy = state.getPotentialEnergy();

        // Calculate the force from NNPInter directly.
        std::vector<Vec3> forces(natoms, Vec3(0,0,0));
        double energy;
        referenceDeepmdForce(state.getPositions(), omm_box, atype, forces, energy, nnp_inter);

        for (int jj = 0; jj < natoms; ++jj){
            ASSERT_EQUAL_VEC(omm_forces[jj], forces[jj], TOL);
        }
        ASSERT_EQUAL_TOL(energy, omm_energy, TOL);
    }
}


int main() {
    // Initialize positions, unit is angstrom.
    std::vector<double> coord = {12.83, 2.56, 2.18,
    12.09, 2.87, 2.74,
    00.25, 3.32, 1.68,
    3.36, 3.00, 1.81,
    3.51, 2.51, 2.60,
    4.27, 3.22, 1.56};
    std::vector<int> atype = {0, 1, 1, 0, 1, 1};
    std::vector<double> box = {
    13., 0., 0., 0., 13., 0., 0., 0., 13.
    };
    
    std::map<int, string> typeDict = {{0, "O"}, {1, "H"}};
    std::vector<double> mass;
    int nsteps = 100;
    int natoms = coord.size() / 3;
    for(int ii = 0; ii < natoms; ++ii){
        if (atype[ii] == 0)
            mass.push_back(15.99943);
        else if (atype[ii] == 1)
            mass.push_back(1.007947);
    }
    std::vector<string> atomNames;
    for(int ii = 0; ii < natoms; ++ii){
        atomNames.push_back(typeDict[atype[ii]]);        
    }


    try{
        registerDeepmdReferenceKernelFactories();
        testDeepmdDynamics(natoms, atomNames, coord, atype, box, mass, typeDict, nsteps);
    }
    catch(const std::exception& e) {
        std::cout << "exception: "<<e.what() << std::endl;
        return 1;
    }
    std::cout<<"Done"<<std::endl;
    return 0;
}
