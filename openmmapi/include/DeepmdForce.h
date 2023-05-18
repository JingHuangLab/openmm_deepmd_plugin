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
#include "internal/windowsExportDeepmd.h"
#include "Topology.h"
#include <vector>
#include <sstream>
// Include DeepPot.h for DeepPotential model inference.
#include "deepmd/deepmd.hpp"
//#include "deepmd/DeepPot.h"



using namespace std;
//using deepmd::DeepPot;
using deepmd::hpp::DeepPot;

namespace DeepmdPlugin {

typedef double VALUETYPE;
typedef double ENERGYTYPE;

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
     * @param lambda
     */
    DeepmdForce(const string& GraphFile, const double& lambda);
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
    /**
     * @brief Get the Cutoff radius of the model used.
     * 
     * @return double
     */
    double getCutoff() const;
    /**
     * @brief Get the number of types in the model.
     * 
     * @return int 
     */
    int getNumberTypes() const;
    /**
     * @brief Get the string that stores the types information.
     * 
     * @return string
     */
    string getTypesMap() const;
    /**
     * @brief Is the DP region is fixed or adaptively selected.
     * 
     * @return bool
     */
    bool isFixedRegion() const;
    void setAdaptiveRegion(const bool& adaptive_region_sign);
    void setCenterAtoms(const vector<int>& center_atoms);
    void setRegionRadius(const double& region_radius);
    void setAtomNames4DPForces(const vector<string>& atom_names);
    void setSelNum4EachType(const vector<string>& type_names, const vector<int>& sel_num);
    vector<int> getCenterAtoms() const;
    double getRegionRadius() const;
    vector<string> getAtomNames4DPForces() const;
    map<string, int> getSelNum4EachType() const;

    /**
    * Get the topology structure from the python generated topology with OpenMM.
    */
    void addChain(int chainIndex, int Id);
    void addResidue(int chainIndex, string ResName, int ResIndex, int ResId);
    void addAtom(int resIndex, string AtomName, string AtomElement, int atomIndex, int atomId);

    Topology* getTopology() const;
    /**
     * @brief Set the lambda value for this alchemical simulation.
     * 
     * @param lambda 
     */
    void setLambda(const double lambda);
    /**
     * @brief Get the lambda value for DP force scale weights in simulation.
     * 
     * @return double 
     */
    double getLambda() const;

    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const {
        return use_pbc;
    }
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    string graph_file = "";
    double lambda = 1.0;
    bool use_pbc = true;

    int numb_types = 0;
    string type_map = "";
    double cutoff = 0.;
    
    map<int, string> type4EachParticle;
    map<string, vector<int>> particleGroup4EachType;
    map<string, int> typesIndexMap;
    vector<pair<int, int>> bondsList;
    double coordCoeff, forceCoeff, energyCoeff;

    /* The following parameters are prepared for adaptive dp region.
     Especially for the support of zinc-protein simulations with dp-mask.
     Adaptive dp region is constructed by following steps:
     1. Select the **center_atoms** that would be appended into the dp region.
     2. Select the atoms within the **radius4adaptive_dp_region** from the center atoms.
     3. Extend the selected atoms to the whole residues.
     4. Put the selected residues atoms and center atoms into the dp region and dp model, get energy and forces.
     5. Assign the dp forces to the selected atoms that atom names are in the **atom_names4dp_forces**.
    */

    // By default, it is true. If false, the dp region will be selected adaptively with the selected_atoms and radius4adaptive_dp_region parameters.
    bool fixed_dp_region = true;
    double radius4adaptive_dp_region = 0.35; // unit in nanometers.
    // Only the atoms within **atom_names4dp_forces ** would be added in dp forces.
    vector<int> center_atoms;
    vector<string> atom_names4dp_forces;
    map<string, int> sel_num4each_type;
    Topology* topology = NULL;

};
} // namespace DeepmdPlugin

#endif /*OPENMM_DEEPMDFORCE_H_*/
