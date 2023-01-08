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
// Include DeepPot.h for DeepPotential model inference.
#include <deepmd/deepmd.hpp>
#include "internal/windowsExportDeepmd.h"

#ifdef HIGH_PREC
typedef double VALUETYPE;
typedef double ENERGYTYPE;
#else 
typedef float  VALUETYPE;
typedef double ENERGYTYPE;
#endif

using namespace std;
using deepmd::hpp::DeepPot;

namespace DeepmdPlugin {


class OPENMM_EXPORT_DEEPMD DeepmdForce : public OpenMM::Force {
public:
    /**
     * @brief Construct a new Deepmd Force object. Used for NVT/NPT/NVE NNP simulation.
     * 
     * @param GraphFile 
     */
    DeepmdForce(const string& GraphFile);
    /**
     * @brief Construct a new Deepmd Force object. Used when running alchemical simulation.
     * 
     * @param GraphFile 
     * @param GraphFile_1 
     * @param GraphFile_2 
     */
    DeepmdForce(const string& GraphFile, const string& GraphFile_1, const string& GraphFile_2);
    ~DeepmdForce();
    
    /**
     * @brief Set the Unit Transform Coefficients.
     * 
     * @param coordCoefficient :  the coordinate transform coefficient.
     * @param forceCoefficient : the force transform coefficient.
     * @param energyCoefficient : the energy transform coefficient.
     */
    void setUnitTransformCoefficients(const double coordCoefficient, const double forceCoefficient, const double energyCoefficient);
    /**
     * @brief Set the gpu id for running Deep Potential model.
     * 
     * @param gpu_id 
     */
    void setGPUNode(const int gpu_id);
    /**
     * @brief Set the NNP whether to use PBC.
     * 
     * @param use_pbc : bool value.
     */
    void setPBC(const bool use_pbc);
    /**
     * @brief Add particle into the Deepmd Force.
     * 
     * @param particleIndex 
     * @param particleType 
     */
    void addParticle(const int particleIndex, const string particleType);
    /**
     * @brief Add the particle types into the Deepmd Force.
     * 
     * @param typeIndex 
     * @param Type 
     */
    void addType(const int typeIndex, const string Type);
    /**
     * @brief Add the bond information into the Deepmd Force. Used for visualization, this bond information is not used in the forces and energy calculation.
     * 
     * @param particle1 
     * @param particle2 
     */
    void addBond(const int particle1, const int particle2);
    /**
     * @brief Get the path to Deepmd Graph (Deep Potential Model) File.
     * 
     * @return const std::string& 
     */
    const std::string& getDeepmdGraphFile() const;
    /**
     * @brief Get the types information for each particle.
     * 
     * @return const map<int, string>& 
     */
    const map<int, string>& getType4EachParticle() const;
    /**
     * @brief Get the particles index vector for each type.
     * 
     * @return const map<string, vector<int>>& 
     */
    const map<string, vector<int>>& getParticles4EachType() const;
    /**
     * @brief Get the bonds list.
     * 
     * @return const vector<pair<int, int>> 
     */
    const vector<pair<int, int>> getBondsList() const;
    /**
     * @brief Get the types map.
     * 
     * @return const map<string, int>& 
     */
    const map<string, int>& getTypesIndexMap() const;
    /**
     * @brief Get the gpu id for running Deep Potential model.
     * 
     * @return const int 
     */
    const int getGPUNode() const;
    /**
     * @brief Get the Coord Unit Coefficient.
     * 
     * @return double 
     */
    double getCoordUnitCoefficient() const;
    /**
     * @brief Get the Force Unit Coefficient.
     * 
     * @return double 
     */
    double getForceUnitCoefficient() const;
    /**
     * @brief Get the Energy Unit Coefficient.
     * 
     * @return double 
     */
    double getEnergyUnitCoefficient() const;
    
    // For alchemical simulation.
    /**
     * @brief Set the Deepmd Force is used for alchemical simulation or not.
     * 
     * @param used4Alchemical 
     */
    void setAlchemical(const bool used4Alchemical);
    /**
     * @brief Set the atoms index list for graph 1 in alchemical simulation.
     * 
     * @param atomsIndex 
     */
    void setAtomsIndex4Graph1(const vector<int> atomsIndex);
    /**
     * @brief Set the atoms index list for graph 2 in alchemical simulation.
     * 
     * @param atomsIndex 
     */
    void setAtomsIndex4Graph2(const vector<int> atomsIndex);
    /**
     * @brief Set the lambda value for this alchemical simulation.
     * 
     * @param lambda 
     */
    void setLambda(const double lambda);
    /**
     * @brief Check whether the Deepmd Force is used for alchemical simulation or not.
     * 
     * @return true : used for alchemical simulation. 
     * @return false : not used for alchemical simulation.
     */
    bool alchemical() const {
        return used4Alchemical;
    }
    /**
     * @brief Get the lambda value for this alchemical simulation.
     * 
     * @return double 
     */
    double getLambda() const;
    /**
     * @brief Get the path to first graph for alchemical simulation.
     * 
     * @return const string 
     */
    const string getGraph1_4Alchemical() const;
    /**
     * @brief Get the path to second graph for alchemical simulation.
     * 
     * @return const string 
     */
    const string getGraph2_4Alchemical() const;
    /**
     * @brief Get the atoms index vector for graph 1 in alchemical simulation.
     * 
     * @return vector<int> 
     */
    vector<int> getAtomsIndex4Graph1() const;
    /**
     * @brief Get the atoms index vector for graph 2 in alchemical simulation.
     * 
     * @return vector<int> 
     */
    vector<int> getAtomsIndex4Graph2() const;

    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const {
        return use_pbc;
    }
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    // graph_1 and 2 are used for alchemical simulation.
    string graph_file, graph_file_1, graph_file_2;
    bool used4Alchemical = false;
    bool use_pbc = true;
    int gpu_node = 0;
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
