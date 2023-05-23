%module OpenMMDeepmdPlugin

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_vector.i>
%include <std_map.i>

%inline %{
using namespace std;
%}

namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(StringVector) vector<string>;
   %template(ConstCharVector) vector<const char*>;
}

%{
#include "DeepmdForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <vector>
%}


/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

/*
%feature("shadow") DeepmdPlugin::DeepmdForce::DeepmdForce %{
    def __init__(self, *args):
        this = _OpenMMDeepmdPlugin.new_DeepmdForce(args[0], args[1], args[2], args[3])
        try:
            self.this.append(this)
        except Exception:
            self.this = this
%}
*/
namespace DeepmdPlugin {

class DeepmdForce : public OpenMM::Force {
public:
    //DeepmdForce::DeepmdForce(const string& GraphFile, const string& GraphFile_1, const string& GraphFile_2, const bool used4Alchemical);
    DeepmdForce(const string& GraphFile);
    DeepmdForce(const string& GraphFile, const double& lambda);

    void addParticle(const int particleIndex, const string particleType);
    void addType(const int typeIndex, const string Type);
    void addBond(const int particle1, const int particle2);
    void setPBC(const bool use_pbc);
    void setUnitTransformCoefficients(const double coordCoefficient, const double forceCoefficient, const double energyCoefficient);
    void setLambda(const double lambda);
    void setGPURank(const int gpu_rank);

    // Extract the model info from dp model.    
    double getCutoff() const;
    int getNumberTypes() const;
    string getTypesMap() const;

    // Add methods for adaptive DP region selection.
    bool isFixedRegion() const;
    void setAdaptiveRegion(const bool& adaptive_region_sign);
    void setCenterAtoms(const vector<int>& center_atoms);
    void setRegionRadius(const double& region_radius);
    void setAtomNames4DPForces(const vector<string>& atom_names);
    void setSelNum4EachType(const vector<string>& type_names, const vector<int>& sel_num);

    void addChain(int chainIndex, int Id);
    void addResidue(int chainIndex, string ResName, int ResIndex, int ResId);
    void addAtom(int resIndex, string AtomName, string AtomElement, int atomIndex, int atomId);

    /*
     * Add methods for casting a Force to a DeepmdForce.
    */
    %extend {
        static DeepmdPlugin::DeepmdForce& cast(OpenMM::Force& force) {
            return dynamic_cast<DeepmdPlugin::DeepmdForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<DeepmdPlugin::DeepmdForce*>(&force) != NULL);
        }
    }
};

}
