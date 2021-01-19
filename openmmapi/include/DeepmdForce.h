#ifndef OPENMM_DEEPMDFORCE_H_
#define OPENMM_DEEPMDFORCE_H_

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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <vector>
// Include NNPInter.h for TF Inference.
#include "NNPInter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"
#include <tensorflow/core/public/session.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/c/c_api.h"
#include <fstream>
#include "internal/windowsExportDeepmd.h"

#ifdef HIGH_PREC
typedef double VALUETYPE;
typedef double ENERGYTYPE;
#else 
typedef float  VALUETYPE;
typedef double ENERGYTYPE;
#endif


namespace DeepmdPlugin {


class OPENMM_EXPORT_DEEPMD DeepmdForce : public OpenMM::Force {
public:
    /**
     * Create an DeepmdForce.
     */
    DeepmdForce(const string& GraphFile, const string& GraphFile_1, const string& GraphFile_2, const bool used4Alchemical);
    ~DeepmdForce();
    
    // For ordinary simulation.
    void setDeepmdOpFile(const string op_file);
    void setUnitTransformCoefficients(const double coordCoefficient, const double forceCoefficient, const double energyCoefficient);
    void addParticle(const int particleIndex, const string particleType);
    void addType(const int typeIndex, const string Type);
    void addBond(const int particle1, const int particle2);
    const std::string& getDeepmdGraphFile() const;
    const map<int, string>& getType4EachParticle() const;
    const map<string, vector<int>>& getParticles4EachType() const;
    const vector<pair<int, int>> getBondsList() const;
    const map<string, int>& getTypesIndexMap() const;
    const string& getDeepmdOpFile() const;
    double getCoordUnitCoefficient() const;
    double getForceUnitCoefficient() const;
    double getEnergyUnitCoefficient() const;
    
    
    // For alchemical simulation.
    void setAlchemical(const bool used4Alchemical);
    void setAtomsIndex4Graph1(const vector<int> atomsIndex);
    void setAtomsIndex4Graph2(const vector<int> atomsIndex);
    void setLambda(const double lambda);
    // Below are interface for kernel function calls.
    bool alchemical() const {
        return used4Alchemical;
    }
    double getLambda() const;
    const string getGraph1_4Alchemical() const;
    const string getGraph2_4Alchemical() const;
    vector<int> getAtomsIndex4Graph1() const;
    vector<int> getAtomsIndex4Graph2() const;

    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const {
        //return false;
        //Deepmd-kit simulation must needs PBC for now.
        return true;
    }

protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    // graph_1 and 2 are used for alchemical simulation.
    string graph_file, graph_file_1, graph_file_2;
    bool used4Alchemical = false;
    string op_file = "/home/dingye/.local/deepmd-kit-1.2.0/lib/libdeepmd_op.so";
    map<int, string> type4EachParticle;
    map<string, vector<int>> particleGroup4EachType;
    map<string, int> typesIndexMap;
    vector<pair<int, int>> bondsList;
    double coordCoeff, forceCoeff, energyCoeff;

    // Used for alchemical simulation.
    vector<int> atomsIndex4Graph1;
    vector<int> atomsIndex4Graph2;
    double lambda;    
};

} // namespace DeepmdPlugin

#endif /*OPENMM_DEEPMDFORCE_H_*/
