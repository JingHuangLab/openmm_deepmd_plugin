#ifndef REFERENCE_DEEPMD_KERNELS_H_
#define REFERENCE_DEEPMD_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
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

#include "DeepmdKernels.h"
#include "openmm/Platform.h"
#include <vector>
using namespace deepmd;

namespace DeepmdPlugin {

/**
 * This kernel is invoked by DeepmdForceImpl to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcDeepmdForceKernel : public CalcDeepmdForceKernel {
public:
    ReferenceCalcDeepmdForceKernel(std::string name, const OpenMM::Platform& platform) : CalcDeepmdForceKernel(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the ExampleForce this kernel will be used for
     */
    ~ReferenceCalcDeepmdForceKernel();
    void initialize(const OpenMM::System& system, const DeepmdForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the ExampleForce to copy the parameters from
     */
private:
    // graph_file 1 and 2 are used for alchemical simulation.
    std::string graph_file, graph_file_1, graph_file_2;
    // dp_1 and dp_2 are used for alchemical simulation.
    DeepPot dp, dp_1, dp_2;

    int natoms;
    ENERGYTYPE dener;
    vector<VALUETYPE> dforce;
    vector<VALUETYPE> dvirial;
    vector<VALUETYPE> dcoord;
    vector<VALUETYPE> dbox;
    vector<int> dtype;

    map<int, string> type4EachParticle;
    map<string, vector<int>> particleGroup4EachType;
    map<string, int> typesIndexMap;
    double forceUnitCoeff, energyUnitCoeff, coordUnitCoeff;
    #ifdef HIGH_PREC
    vector<double> AddedForces;
    #else
    vector<float> AddedForces;
    #endif

    // Parameters for alchemical simulation.
    bool used4Alchemical = false;
    double lambda; // U = lambda * U_A + (1 - lambda) * (U_1 + U_2). Where U_A comes from the original graph, U_1 and U_2 come from two alchemical graph.
    vector<int> atomsIndex4Graph1;
    vector<int> atomsIndex4Graph2;
    map<int, vector<VALUETYPE>> dcoord4alchemical;
    map<int, vector<VALUETYPE>> dbox4alchemical;
    map<int, vector<int>> dtype4alchemical;

    map<int, ENERGYTYPE> dener4alchemical;
    map<int, vector<VALUETYPE>> dforce4alchemical;
    map<int, vector<VALUETYPE>> dvirial4alchemical;

    map<int, int> natoms4alchemical;
    vector<pair<int, int>> atomsIndexMap4U_B;
};

} // namespace DeepmdPlugin

#endif /*REFERENCE_DEEPMD_KERNELS_H_*/
