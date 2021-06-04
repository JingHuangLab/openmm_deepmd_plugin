#include "openmm/Vec3.h"
#include "sfmt/SFMT.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "NNPInter.h"
#include <c_api.h>
#include <fstream>
#include <typeinfo>
#include <numeric>
#include <string>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <numeric>


#ifdef HIGH_PREC
#define VALUETYPE double
#else
#define VALUETYPE float
#endif

const double TOL = 1e-5;
const string graph = "../../../../frozen_model/lw_pimd.se_a_v1.2.0.float_step.1M_batchSize.1.pb";
const string op_path = "/home/dingye/.local/deepmd-kit-1.2.0/lib/libdeepmd_op.so";
const double coordUnitCoeff = 10;
const double forceUnitCoeff = 964.8792534459;
const double energyUnitCoeff = 96.48792534459;
const double temperature = 300;
const int randomSeed = 123456;
const double delta_T = 0.0001; // 0.1fs

using namespace std;
using namespace tensorflow;
using namespace OpenMM;


class atom{
public:
    atom(unsigned int index, int elementIndex, string elementSymbol, double mass): index(index), elementIndex(elementIndex), mass(mass), elementSymbol(elementSymbol){
        inverseMass = 1.0/mass;
    }
    unsigned int index;
    double mass, inverseMass;
    string elementSymbol;
    unsigned int elementIndex;
};

class Micromd {
    public:
        Micromd(unsigned int natoms): natoms(natoms){
            TF_Status* LoadOpStatus = TF_NewStatus();
            TF_LoadLibrary(op_path.c_str(), LoadOpStatus);
            TF_DeleteStatus(LoadOpStatus);

            nnp = NNPInter(graph, 0);
            dcoord = vector<VALUETYPE>(natoms * 3);
            dforce = vector<VALUETYPE>(natoms * 3);
            dvirial = vector<VALUETYPE>(9);
            dtype = vector<int>(natoms);
            dbox = vector<VALUETYPE>(9);
            positions = vector<Vec3>(natoms);
            xPrime = vector<Vec3>(natoms);
            velocities = vector<Vec3>(natoms);
            forces = vector<Vec3>(natoms);
        };
        void VerletStep(int nstep);
        void VelocityVerletStep(int num_step = 1);
        vector<Vec3> getPositions();
        vector<Vec3> getForces();
        vector<Vec3> getVelocities();
        double getTotalEnergy();
        double getPotentialEnergy();
        double getKineticEnergy();
        void setInitPositions(vector<Vec3> posi);
        void setInitVelocities(vector<Vec3> velocity);
        void setInitVelocitiesToTemperature(double temperature, int randomSeed);
        void setInitBox(vector<Vec3> box);
        void setTypes(vector<int> types);

        void addAtom(atom at);
        void setInitMass(vector<double> mass);


        vector<double> kinetic_energy, potential_energy, total_energy;
        int getNumAtoms();

        double calcEnergyAndForce(bool includeForces, bool includeEnergy);

    private:
        // System variable.
        vector<Vec3> positions;
        vector<Vec3> velocities;
        vector<Vec3> forces;
        vector<double> mass;
        vector<atom> atoms;
        unsigned int natoms;
        NNPInter nnp;
        
        // Variables for verlet step.
        vector<Vec3> xPrime;
        // Variables for VelocityVerletStep.
        int step_now = 0;

        // Input variables for NNPInter.
        vector<VALUETYPE> dcoord;
        vector<VALUETYPE> dforce;
        vector<VALUETYPE> dvirial;
        double dener;
        vector<int> dtype;
        vector<VALUETYPE> dbox;
        int nghost = 0;
};
