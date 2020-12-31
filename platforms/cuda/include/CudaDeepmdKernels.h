#ifndef CUDA_DEEPMD_KERNELS_H_
#define CUDA_DEEPMD_KERNELS_H_

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

#include "DeepmdKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"

//#include <cuda_runtime.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>


namespace DeepmdPlugin {

/**
 * This kernel is invoked by Deepmd to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcDeepmdForceKernel : public CalcDeepmdForceKernel{
public:
    CudaCalcDeepmdForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu):CalcDeepmdForceKernel(name, platform), cu(cu){};
    ~CudaCalcDeepmdForceKernel();
    void initialize(const OpenMM::System& system, const DeepmdForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    // Used for CUDA Platform.
    bool hasInitialized;
    OpenMM::CudaContext& cu;
    OpenMM::CudaArray networkForces;
    CUfunction addForcesKernel;

    // graph_file 1 and 2 are used for alchemical simulation.
    std::string graph_file, graph_file_1, graph_file_2;
    // nnp_inter_1 and nnp_inter_2 are used for alchemical simulation.
    NNPInter nnp_inter, nnp_inter_1, nnp_inter_2;
    //NNPInterModelDevi nnp_inter_model_devi;
    unsigned numb_models;
    double cutoff;
    int numb_types;
    vector<vector<double > > all_force;
    ofstream fp;
    int out_freq;
    string out_file;
    int dim_fparam;
    int dim_aparam;
    int out_each;
    int out_rel;
    bool single_model;

    #ifdef HIGH_PREC
    vector<double > fparam;
    vector<double > aparam;
    double eps;
    #else
    vector<float > fparam;
    vector<float > aparam;
    float eps;
    #endif
    void make_ttm_aparam(
    #ifdef HIGH_PREC
        vector<double > & dparam
    #else
        vector<float > & dparam
    #endif
        );
    bool do_ttm;
    string ttm_fix_id;

    int natoms;
    int nghost = 0;
    ENERGYTYPE dener;
    vector<VALUETYPE> dforce;
    vector<VALUETYPE> dvirial;
    vector<VALUETYPE> dcoord;
    vector<VALUETYPE> dbox;
    vector<int> dtype;

    #ifdef HIGH_PREC
    vector<double > daparam;
    #else 
    vector<float > daparam;
    #endif

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
    double lambda; // U = lambda * U_A + (1 - lambda) * (U_1 + U_2). Where U_A comes from the original graph, U_1 and U_2 comes from alchemical graph.
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


#endif /*CUDA_DEEPMD_KERNELS_H_*/