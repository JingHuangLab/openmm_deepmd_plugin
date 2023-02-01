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

#include "DeepmdForce.h"
#include "internal/DeepmdForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <sys/stat.h>

using namespace DeepmdPlugin;
using namespace OpenMM;
using namespace std;

inline bool exists(const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

DeepmdForce::DeepmdForce(const string& GraphFile, const string& GraphFile_1, const string& GraphFile_2){
    graph_file  = GraphFile;
    graph_file_1 = GraphFile_1;
    graph_file_2 = GraphFile_2;
    //this->used4Alchemical = used4Alchemical;
    this->used4Alchemical = true;
    if(used4Alchemical){
        if (!exists(graph_file_1)){
            throw OpenMMException("Graph file 1 not found: "+graph_file_1);
        }
        if (!exists(graph_file_2)){
            throw OpenMMException("Graph file 2 not found: "+graph_file_2);
        }
    }
    if (!exists(graph_file)){
        throw OpenMMException("Graph file not found: "+graph_file);
    }
    // Initialize dp model
    DeepPot tmp_dp = DeepPot(graph_file);
    
    // Extract the model informations first.
    this->numb_types = tmp_dp.numb_types();
    this->cutoff = tmp_dp.cutoff();
    tmp_dp.get_type_map(this->type_map);
}

DeepmdForce::DeepmdForce(const string& GraphFile){
    graph_file  = GraphFile;
    this->used4Alchemical = false;
    if (!exists(graph_file)){
        throw OpenMMException("Graph file not found: "+graph_file);
    }
    // Initialize dp model
    DeepPot tmp_dp = DeepPot(graph_file);
    this->numb_types = tmp_dp.numb_types();
    this->cutoff = tmp_dp.cutoff();
    tmp_dp.get_type_map(this->type_map);
}

DeepmdForce::~DeepmdForce(){
    type4EachParticle.clear();
    particleGroup4EachType.clear();
    typesIndexMap.clear();
}


void DeepmdForce::setPBC(const bool use_PBC){
    // By default, use_pbc is set to be true.
    use_pbc = use_PBC;
}

void DeepmdForce::setUnitTransformCoefficients(const double coordCoefficient, const double forceCoefficient, const double energyCoefficient){
    coordCoeff = coordCoefficient;
    forceCoeff = forceCoefficient;
    energyCoeff = energyCoefficient;
}

double DeepmdForce::getCoordUnitCoefficient() const {return coordCoeff;}
double DeepmdForce::getForceUnitCoefficient() const {return forceCoeff;}
double DeepmdForce::getEnergyUnitCoefficient() const {return energyCoeff;}

double DeepmdForce::getCutoff() const {return cutoff;}
int DeepmdForce::getNumberTypes() const {return numb_types;}
string DeepmdForce::getTypesMap() const {return type_map;}

const string& DeepmdForce::getDeepmdGraphFile() const{return graph_file;}
const map<int, string>& DeepmdForce::getType4EachParticle() const{return type4EachParticle;}
const map<string, vector<int>>& DeepmdForce::getParticles4EachType() const{return particleGroup4EachType;}
const map<string, int>& DeepmdForce::getTypesIndexMap() const{return typesIndexMap;}

void DeepmdForce::addParticle(const int particleIndex, const string particleType){
    auto insertResult = type4EachParticle.insert(pair<int, string>(particleIndex, particleType));
    if(insertResult.second == false){
        throw OpenMMException("Failed to add in particle, duplicate key.");
    }
    auto it = particleGroup4EachType.find(particleType);
    if (it == particleGroup4EachType.end()){
        particleGroup4EachType[particleType] = vector<int>();
        particleGroup4EachType[particleType].push_back(particleIndex);
    }else{
        particleGroup4EachType[particleType].push_back(particleIndex);
    }
}

void DeepmdForce::addType(const int typeIndex, const string Type){
    auto it = typesIndexMap.find(Type);
    if(it == typesIndexMap.end()){
        typesIndexMap[Type] = typeIndex;
    }else{
        if(typeIndex != it->second){
            throw OpenMMException("Type Index duplicated.");
        }
    }
}

void DeepmdForce::addBond(const int particle1, const int particle2){
    bondsList.push_back(make_pair(particle1, particle2));
}

const vector<pair<int, int>> DeepmdForce::getBondsList() const{
    return bondsList;
}


ForceImpl* DeepmdForce::createImpl() const {
    return new DeepmdForceImpl(*this);
}

void DeepmdForce::updateParametersInContext(Context& context) {
    // Nothing to be done here.
    return;
}

void DeepmdForce::setAlchemical(const bool used4Alchemical){
    this->used4Alchemical = used4Alchemical;
}

void DeepmdForce::setAtomsIndex4Graph1(const vector<int> atomsIndex){
    atomsIndex4Graph1.clear();
    for(auto it = atomsIndex.begin(); it != atomsIndex.end();it++){
        atomsIndex4Graph1.push_back(*it);
    }
}

void DeepmdForce::setAtomsIndex4Graph2(const vector<int> atomsIndex){
    atomsIndex4Graph2.clear();
    for(auto it = atomsIndex.begin(); it != atomsIndex.end();it++){
        atomsIndex4Graph2.push_back(*it);
    }
}

void DeepmdForce::setLambda(const double lambda){
    this->lambda = lambda;
}

double DeepmdForce::getLambda() const {return lambda;}

vector<int> DeepmdForce::getAtomsIndex4Graph1() const {return atomsIndex4Graph1;}
vector<int> DeepmdForce::getAtomsIndex4Graph2() const {return atomsIndex4Graph2;}

const string DeepmdForce::getGraph1_4Alchemical() const {
    if(used4Alchemical)
    return graph_file_1;
    else{
        throw OpenMMException("This Deepmd Force is not used for alchemical simulation.");
    }
}

const string DeepmdForce::getGraph2_4Alchemical() const {
    if(used4Alchemical)
    return graph_file_2;
    else{
        throw OpenMMException("This Deepmd Force is not used for alchemical simulation.");
    }
}
