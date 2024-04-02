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

#include "ReferenceDeepmdKernels.h"
#include "DeepmdForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include <typeinfo>
#include <iostream>
#include <map>
#include <algorithm>
#include <limits>

using namespace DeepmdPlugin;
using namespace OpenMM;
using namespace std;

static vector<RealVec> &extractPositions(ContextImpl &context)
{
    ReferencePlatform::PlatformData *data = reinterpret_cast<ReferencePlatform::PlatformData *>(context.getPlatformData());
    return *((vector<RealVec> *)data->positions);
}

static vector<RealVec> &extractForces(ContextImpl &context)
{
    ReferencePlatform::PlatformData *data = reinterpret_cast<ReferencePlatform::PlatformData *>(context.getPlatformData());
    return *((vector<RealVec> *)data->forces);
}

static Vec3 *extractBoxVectors(ContextImpl &context)
{
    ReferencePlatform::PlatformData *data = reinterpret_cast<ReferencePlatform::PlatformData *>(context.getPlatformData());
    return (Vec3 *)data->periodicBoxVectors;
}

ReferenceCalcDeepmdForceKernel::~ReferenceCalcDeepmdForceKernel()
{

    return;
}

void ReferenceCalcDeepmdForceKernel::initialize(const System &system, const DeepmdForce &force)
{
    graph_file = force.getDeepmdGraphFile();
    type4EachParticle = force.getType4EachParticle();
    typesIndexMap = force.getTypesIndexMap();
    forceUnitCoeff = force.getForceUnitCoefficient();
    energyUnitCoeff = force.getEnergyUnitCoefficient();
    coordUnitCoeff = force.getCoordUnitCoefficient();
    lambda = force.getLambda();
    natoms = type4EachParticle.size();
    tot_atoms = system.getNumParticles();

    // Initialize DeepPot.
    this->dp.init<double>(graph_file);
    // string types_str = force.getTypesMap();
    // std::stringstream ss(types_str);
    // string token;
    // while (getline(ss, token, ' '))
    // {
    //     this->dp_types.push_back(token);
    // }

    // Check if DP region is defined with fixed particles.
    isFixedRegion = force.isFixedRegion();

    // Set up the dtype and input, output arrays.
    if (isFixedRegion)
    {
        std::cout << "Using fixed region for DP." << std::endl;
        std::cout << "Atoms Selected for DP Region: " << natoms << std::endl;
        std::cout << "Total Atoms in System: " << tot_atoms << std::endl;
        dener = 0.;
        dforce = vector<VALUETYPE>(natoms * 3, 0.);
        dvirial = vector<VALUETYPE>(9, 0.);
        dcoord = vector<VALUETYPE>(natoms * 3, 0.);
        dbox = vector<VALUETYPE>(9, 0.);

        for (std::map<int, string>::iterator it = type4EachParticle.begin(); it != type4EachParticle.end(); ++it)
        {
            dp_particles.push_back(it->first);
            dtype.push_back(typesIndexMap[it->second]);
        }
    }
    // else
    // {
    //     center_atoms = force.getCenterAtoms();
    //     radius = force.getRegionRadius();
    //     atom_names4dp_forces = force.getAtomNames4DPForces();
    //     sel_num4type = force.getSelNum4EachType();
    //     topology = force.getTopology();

    //     natoms = 0;
    //     // Fix the order of atom types.
    //     assert(typesIndexMap.size() == sel_num4type.size());
    //     assert(typesIndexMap.size() == dp_types.size());
    //     for (int i = 0; i < dp_types.size(); i++)
    //     {
    //         string type_name = dp_types[i];
    //         cum_sum4type[type_name] = vector<int>(2, 0);
    //         cum_sum4type[type_name][0] = natoms;
    //         natoms += sel_num4type[type_name];
    //         cum_sum4type[type_name][1] = natoms;
    //         for (int j = 0; j < sel_num4type[type_name]; j++)
    //         {
    //             dtype.push_back(typesIndexMap[type_name]);
    //         }
    //     }

    //     dener = 0.;
    //     dforce = vector<VALUETYPE>(natoms * 3, 0.);
    //     dvirial = vector<VALUETYPE>(9, 0.);
    //     dcoord = vector<VALUETYPE>(natoms * 3, 0.);
    //     dbox = {}; // Empty vector for adaptive region.
    //     daparam = vector<VALUETYPE>(natoms, 0.);
    //     dp_particles = vector<int>(natoms, -1);
    // }

    // AddedForces = vector<double>(tot_atoms * 3, 0.0);
}

double ReferenceCalcDeepmdForceKernel::execute(ContextImpl &context, bool includeForces, bool includeEnergy)
{
    vector<RealVec> &pos = extractPositions(context);
    vector<RealVec> &force = extractForces(context);

    if (isFixedRegion)
    {
        // Set box size.
        if (context.getSystem().usesPeriodicBoundaryConditions())
        {
            Vec3 *box = extractBoxVectors(context);
            // Transform unit from nanometers to angstrom.
            dbox[0] = box[0][0] * coordUnitCoeff;
            dbox[1] = box[0][1] * coordUnitCoeff;
            dbox[2] = box[0][2] * coordUnitCoeff;
            dbox[3] = box[1][0] * coordUnitCoeff;
            dbox[4] = box[1][1] * coordUnitCoeff;
            dbox[5] = box[1][2] * coordUnitCoeff;
            dbox[6] = box[2][0] * coordUnitCoeff;
            dbox[7] = box[2][1] * coordUnitCoeff;
            dbox[8] = box[2][2] * coordUnitCoeff;
        }
        else
        {
            dbox = {}; // No PBC.
        }
        // Set input coord.
        for (int ii = 0; ii < natoms; ++ii)
        {
            // Multiply by coordUnitCoeff means the transformation of the unit from nanometers to required input unit for positions in trained DP model.
            int atom_index = dp_particles[ii];
            dcoord[ii * 3 + 0] = pos[atom_index][0] * coordUnitCoeff;
            dcoord[ii * 3 + 1] = pos[atom_index][1] * coordUnitCoeff;
            dcoord[ii * 3 + 2] = pos[atom_index][2] * coordUnitCoeff;
        }
        dp.compute<double, double>(dener, dforce, dvirial, dcoord, dtype, dbox);
    }
    // else
    // {
    //     std::fill(dp_particles.begin(), dp_particles.end(), -1);
    //     std::fill(dcoord.begin(), dcoord.end(), 0.);
    //     std::fill(daparam.begin(), daparam.end(), 0.);

    //     vector<bool> addForcesSign(natoms, false);
    //     map<string, vector<bool>> addOrNot; // Whether to add the dp forces for selected atoms.

    //     map<string, vector<int>> dp_region_atoms = DeepmdPlugin::SearchAtomsInRegion(pos, center_atoms, radius, topology, atom_names4dp_forces, addOrNot);

    //     for (map<string, int>::iterator it = sel_num4type.begin(); it != sel_num4type.end(); ++it)
    //     {
    //         string atom_type = it->first;
    //         int max_atom_num = it->second;
    //         if (dp_region_atoms.find(atom_type) == dp_region_atoms.end())
    //         {
    //             // Selected atoms of this type are not found in the adaptive region. That's ok for adaptive region.
    //             continue;
    //         }

    //         vector<int> sel_atoms4type = dp_region_atoms[atom_type];
    //         // Checks whether the number of selected atoms exceeds the maximum number of input atoms allowed
    //         int sel_atom_num = sel_atoms4type.size();
    //         if (sel_atom_num > max_atom_num)
    //         {
    //             std::cout << "Atom type " << atom_type << " has " << sel_atom_num << " atoms in the adaptive region, which is larger than the maximum number of input atoms allowed " << max_atom_num << std::endl;
    //             throw OpenMMException("The number of atoms in the adaptive region is larger than the number of atoms selected for DP forces.");
    //         }

    //         int cum_sum_start = cum_sum4type[atom_type][0];
    //         for (int i = 0; i < sel_atom_num; i++)
    //         {
    //             int atom_index = sel_atoms4type[i];
    //             int dp_index = cum_sum_start + i;
    //             dcoord[dp_index * 3 + 0] = pos[atom_index][0] * coordUnitCoeff;
    //             dcoord[dp_index * 3 + 1] = pos[atom_index][1] * coordUnitCoeff;
    //             dcoord[dp_index * 3 + 2] = pos[atom_index][2] * coordUnitCoeff;
    //             daparam[dp_index] = 1;
    //             dp_particles[dp_index] = atom_index;
    //             addForcesSign[dp_index] = addOrNot[atom_type][i];
    //         }
    //     }
    //     vector<VALUETYPE> dfparam = {};

    //     // Calculate energy and forces.
    //     dp.compute<double, double>(dener, dforce, dvirial, dcoord, dtype, dbox, dfparam, daparam);

    //     // Filter the forces for atoms that can be add dp force.
    //     for (int ii = 0; ii < natoms; ii++)
    //     {
    //         if (!addForcesSign[ii])
    //         {
    //             dforce[ii * 3 + 0] = 0.;
    //             dforce[ii * 3 + 1] = 0.;
    //             dforce[ii * 3 + 2] = 0.;
    //         }
    //     }
    //     // Set dp ener to 0 since it is invalid in adaptive dp region.
    //     dener = 0.;
    // }

    // Add dp forces to the total forces.
    if (includeForces)
    {
        for (int ii = 0; ii < natoms; ii++)
        {
            int atom_index = dp_particles[ii];
            if (atom_index == -1)
            {
                continue;
            }

            force[atom_index][0] += lambda * dforce[ii * 3 + 0] * forceUnitCoeff;
            force[atom_index][1] += lambda * dforce[ii * 3 + 1] * forceUnitCoeff;
            force[atom_index][2] += lambda * dforce[ii * 3 + 2] * forceUnitCoeff;
        }
    }
    if (includeEnergy)
    {
        dener = lambda * dener * energyUnitCoeff;
    }
    else
    {
        dener = 0.;
    }

    // Return energy.
    return dener;
}
