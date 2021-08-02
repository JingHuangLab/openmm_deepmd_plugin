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
#include <stdio.h>
#include <vector>
#include <math.h>
#include <numeric>


using namespace OpenMM;
using namespace DeepmdPlugin;
using namespace deepmd;
using namespace std;

extern "C" OPENMM_EXPORT void registerDeepmdCudaKernelFactories();

const double TOL = 1e-5;
const string graph = "../tests/frozen_model/graph_from_han_dp2.0_compress.pb";
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
    System system;
    VerletIntegrator integrator(0.0002); // Time step is 0.2 fs here.
    DeepmdForce dp_force = DeepmdForce(graph, " ", " ", false);
    
    // Convert the units of coordinates and box from angstrom to nanometers.
    vector<Vec3> omm_coord;
    vector<Vec3> omm_box;

    ASSERT_EQUAL_TOL(names.size(), atype.size(), TOL);
    
    for(auto it = typeDict.begin(); it != typeDict.end(); it++){
        dp_force.addType(it->first, it->second);
    }
    for(int ii = 0; ii < 3; ii++){
        omm_box.push_back(Vec3(box[ii * 3 + 0] / coordUnitCoeff, box[ii * 3 + 1] / coordUnitCoeff, box[ii * 3 + 2] / coordUnitCoeff));
    }
    for (int ii = 0; ii < natoms; ++ii){
        system.addParticle(mass[ii]);
        dp_force.addParticle(ii, names[ii]);    
        omm_coord.push_back(Vec3(coord[ii * 3 + 0] * 0.1, coord[ii * 3 + 1] * 0.1, coord[ii * 3 + 2] * 0.1));
    }
    dp_force.setUnitTransformCoefficients(coordUnitCoeff, forceUnitCoeff, energyUnitCoeff); 
    system.addForce(&dp_force);

    Platform& platform = Platform::getPlatformByName("CUDA");

    Context context(system, integrator, platform);
    context.setPositions(omm_coord);
    context.setPeriodicBoxVectors(omm_box[0], omm_box[1], omm_box[2]);
    context.setVelocitiesToTemperature(temperature, randomSeed);

    // Initialize the nnp_inter.
    DeepPot nnp_inter = DeepPot(graph);
    // Record the difference of forces and energy on each step.
    vector<double> errorForce;
    vector<double> errorEnergy;
    double err = 0.;

    for (int ii = 0; ii < nsteps; ++ii){
        // Running dynamics 1 step.
        integrator.step(1);
        
        // Get the force from openmm context state.
        State state = context.getState(State::Forces | State::Energy | State::Positions);
        const vector<Vec3>& omm_forces = state.getForces();
        const double& omm_energy = state.getPotentialEnergy();
        const double& omm_kinetic_energy = state.getKineticEnergy();
        // Calculate the force from NNPInter directly.
        vector<Vec3> forces(natoms, Vec3(0,0,0));
        double energy;
        referenceDeepmdForce(state.getPositions(), omm_box, atype, forces, energy, nnp_inter);
        // Compare the difference of these two result.
        err = 0.;
        for (int jj = 0; jj < natoms; ++jj){
            ASSERT_EQUAL_VEC(omm_forces[jj], forces[jj], TOL);
            Vec3 diff_f = omm_forces[jj] - forces[jj];
            double diff = diff_f.dot(diff_f);
            err += diff;
        }
        ASSERT_EQUAL_TOL(energy, omm_energy, TOL);
    }
    
}


void testDeepmdEnergyAndForces(int natoms, vector<string> atomNames, vector<double> coord, vector<int> atype, vector<double> box, vector<double> expected_e, vector<double> expected_f, vector<double> expected_v, vector<double> mass, map<int, string> typeDict){
    System system;
    VerletIntegrator integrator(0.0002); // Time step is 0.2 fs here.
    DeepmdForce dp_force = DeepmdForce(graph, " ", " ", false);
    vector<Vec3> omm_coord;
    vector<Vec3> omm_box;
    ASSERT_EQUAL_TOL(atomNames.size(), atype.size(), TOL);
    
    // Calculate the expected energy force for OpenMM comparision.
    double expected_tot_e = 0;
    vector<Vec3> expected_omm_f;
    for (int ii  = 0; ii < natoms; ii++){
        expected_tot_e += expected_e[ii] * energyUnitCoeff;
        expected_omm_f.push_back(Vec3(expected_f[ii * 3 + 0] * forceUnitCoeff, expected_f[ii * 3 + 1] * forceUnitCoeff, expected_f[ii * 3 + 2] * forceUnitCoeff));
    }

    for(auto it = typeDict.begin(); it != typeDict.end(); it++){
        dp_force.addType(it->first, it->second);
    }
    for(int ii = 0; ii < 3; ii++){
        omm_box.push_back(Vec3(box[ii * 3 + 0] / coordUnitCoeff, box[ii * 3 + 1] / coordUnitCoeff, box[ii * 3 + 2] / coordUnitCoeff));
    }
    for (int ii = 0; ii < natoms; ++ii){
        system.addParticle(mass[ii]);
        dp_force.addParticle(ii, atomNames[ii]);    
        omm_coord.push_back(Vec3(coord[ii * 3 + 0] * 0.1, coord[ii * 3 + 1] * 0.1, coord[ii * 3 + 2] * 0.1));
    }
    dp_force.setUnitTransformCoefficients(coordUnitCoeff, forceUnitCoeff, energyUnitCoeff); 
    system.addForce(&dp_force);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integrator, platform);
    context.setPositions(omm_coord);
    context.setPeriodicBoxVectors(omm_box[0], omm_box[1], omm_box[2]);
    context.setVelocitiesToTemperature(temperature, randomSeed);

    // Get the force from openmm context state.
    State state = context.getState(State::Forces | State::Energy | State::Positions);
    const vector<Vec3>& omm_forces = state.getForces();
    const double& omm_energy = state.getPotentialEnergy();
    const double& omm_kinetic_energy = state.getKineticEnergy();
    
    ASSERT_EQUAL_TOL(omm_energy, expected_tot_e, TOL);
    for (int ii = 0; ii < natoms; ++ii){
        ASSERT_EQUAL_VEC(omm_forces[ii], expected_omm_f[ii], TOL);
    }
}


int main(int argc, char* argv[]) {
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
    std::vector<double> expected_e = {
    -9.275780747115504710e+01,-1.863501786584258468e+02,-1.863392472863538103e+02,-9.279281325486221021e+01,-1.863671545232153903e+02,-1.863619822847602165e+02
    };
    std::vector<double> expected_f = {
    -3.034045420701179663e-01,8.405844663871177014e-01,7.696947487118485642e-02,7.662001266663505117e-01,-1.880601391333554251e-01,-6.183333871091722944e-01,-5.036172391059643427e-01,-6.529525836149027151e-01,5.432962643022043459e-01,6.382357912332115024e-01,-1.748518296794561167e-01,3.457363524891907125e-01,1.286482986991941552e-03,3.757251165286925043e-01,-5.972588700887541124e-01,-5.987006197104716154e-01,-2.004450304880958100e-01,2.495901655353461868e-01
    };
    std::vector<double> expected_v = {
    -2.912234126853306959e-01,-3.800610846612756388e-02,2.776624987489437202e-01,-5.053761003913598976e-02,-3.152373041953385746e-01,1.060894290092162379e-01,2.826389131596073745e-01,1.039129970665329250e-01,-2.584378792325942586e-01,-3.121722367954994914e-01,8.483275876786681990e-02,2.524662342344257682e-01,4.142176771106586414e-02,-3.820285230785245428e-02,-2.727311173065460545e-02,2.668859789777112135e-01,-6.448243569420382404e-02,-2.121731470426218846e-01,-8.624335220278558922e-02,-1.809695356746038597e-01,1.529875294531883312e-01,-1.283658185172031341e-01,-1.992682279795223999e-01,1.409924999632362341e-01,1.398322735274434292e-01,1.804318474574856390e-01,-1.470309318999652726e-01,-2.593983661598450730e-01,-4.236536279233147489e-02,3.386387920184946720e-02,-4.174017537818433543e-02,-1.003500282164128260e-01,1.525690815194478966e-01,3.398976109910181037e-02,1.522253908435125536e-01,-2.349125581341701963e-01,9.515545977581392825e-04,-1.643218849228543846e-02,1.993234765412972564e-02,6.027265332209678569e-04,-9.563256398907417355e-02,1.510815124001868293e-01,-7.738094816888557714e-03,1.502832772532304295e-01,-2.380965783745832010e-01,-2.309456719810296654e-01,-6.666961081213038098e-02,7.955566551234216632e-02,-8.099093777937517447e-02,-3.386641099800401927e-02,4.447884755740908608e-02,1.008593228579038742e-01,4.556718179228393811e-02,-6.078081273849572641e-02
    };    
    map<int, string> typeDict = {{0, "O"}, {1, "H"}};
    vector<double> mass;
    int nsteps = 100;
    int natoms = expected_e.size();
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
    // Test the single point energy and dynamics of Deepmd Plugin.
    try{
        registerDeepmdCudaKernelFactories();
        if (argc > 1)
            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        testDeepmdEnergyAndForces(natoms, atomNames, coord, atype, box, expected_e, expected_f, expected_v, mass, typeDict);
        //testDeepmdDynamics(natoms, atomNames, coord, atype, box, mass, typeDict, nsteps);
    }
    catch(const OpenMM::OpenMMException& e) {
        cout << "OpenMMException: "<<e.what() << endl;
        return 1;
    }
    cout<<"Done"<<endl;
    return 0;
}