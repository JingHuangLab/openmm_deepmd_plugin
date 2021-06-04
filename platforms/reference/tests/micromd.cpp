#include "micromd.h"


void Micromd::VerletStep(int nstep){

    for(int kk = 0; kk < nstep; kk++){
        calcEnergyAndForce(true, false);
        // Update the velocities and positions of each particle with Verlet Integrator.
        // Get new velocities first.
        for(int ii=0; ii < natoms; ii++){
            for(int jj=0; jj < 3; jj++){
                velocities[ii][jj] += forces[ii][jj] * atoms[ii].inverseMass * delta_T;
                xPrime[ii][jj] = positions[ii][jj] + velocities[ii][jj] * delta_T;
            }
        }
        // Update the positions and velocities of each particles.
        for(int ii=0; ii < natoms; ii++){
            for(int jj = 0; jj < 3; jj++){
                //velocities[ii][jj] =  static_cast<double>(1.0/delta_T) * (xPrime[ii][jj] - positions[ii][jj]);
                positions[ii][jj] = xPrime[ii][jj];
            }
        }        
        step_now +=  1;
    }
}

void Micromd::VelocityVerletStep(int nstep){
    // Store the first step force.
    if (step_now == 0){
        for(int ii=0; ii < natoms; ii++){
        // Multiply 0.1 for unit transformation from nanometers to angstrom.
        dcoord[ii * 3 + 0] = static_cast<float>(positions[ii][0]) * 10;
        dcoord[ii * 3 + 1] = static_cast<float>(positions[ii][1]) * 10;
        dcoord[ii * 3 + 2] = static_cast<float>(positions[ii][2]) * 10;
        }
        nnp.compute(dener, dforce, dvirial, dcoord, dtype, dbox);
        for(int ii = 0; ii < natoms; ii++){
            for(int jj = 0; jj < 3; jj++){
                forces[ii][jj] = dforce[ii * 3 + jj] * forceUnitCoeff;
            }
        }
    }


    for (int kk = 0; kk < nstep; kk++){
        // Update system positions first.
        for(int ii = 0; ii < natoms; ii++){
            for(int jj = 0; jj < 3; jj++){
                positions[ii][jj] = positions[ii][jj] + velocities[ii][jj] * delta_T + 0.5 * forces[ii][jj] * atoms[ii].inverseMass * delta_T * delta_T;
            }
        }
        // Calculate the new force.
        for(int ii=0; ii < natoms; ii++){
            // Multiply 0.1 for unit transformation from nanometers to angstrom.
            dcoord[ii * 3 + 0] = static_cast<float>(positions[ii][0]) * 10;
            dcoord[ii * 3 + 1] = static_cast<float>(positions[ii][1]) * 10;
            dcoord[ii * 3 + 2] = static_cast<float>(positions[ii][2]) * 10;
        }
        nnp.compute(dener, dforce, dvirial, dcoord, dtype, dbox);

        // Update the velocity
        double kinetic = 0.0;
        for(int ii = 0; ii < natoms; ii++){
            for(int jj = 0; jj < 3; jj++){
                velocities[ii][jj] = velocities[ii][jj] + 0.5 * delta_T * (forces[ii][jj] + dforce[ii * 3 + jj] * forceUnitCoeff) * atoms[ii].inverseMass;
                kinetic +=  0.5 * atoms[ii].mass * velocities[ii][jj] * velocities[ii][jj];
            }
        }
        // Save the force for next step.
        for(int ii = 0; ii < natoms; ii++){
            for(int jj = 0; jj < 3; jj++){
                forces[ii][jj] = dforce[ii * 3 + jj] * forceUnitCoeff;
            }
        }

        // Update the energy.
        kinetic_energy.push_back(kinetic);
        potential_energy.push_back(dener * energyUnitCoeff);
        total_energy.push_back(kinetic + dener * energyUnitCoeff);

        step_now += 1;

    }
}

vector<Vec3> Micromd::getPositions(){return positions;}
vector<Vec3> Micromd::getForces(){
    calcEnergyAndForce(true, false);
    return forces;}
vector<Vec3> Micromd::getVelocities(){return velocities;}
double Micromd::getTotalEnergy(){
    double potential = getPotentialEnergy();
    double kinetic = getKineticEnergy();
    total_energy.push_back(kinetic + potential);
    return total_energy.back();
}
double Micromd::getPotentialEnergy(){
    calcEnergyAndForce(false, true);
    return potential_energy.back();
}
double Micromd::getKineticEnergy(){
    // Calculate the kinetic energy with shifted velocity.
    vector<Vec3> shiftedVelocity = vector<Vec3>(natoms, Vec3());
    double kinetic = 0.;
    for(int ii = 0; ii < natoms; ii ++){
        for(int jj = 0; jj < 3; jj++){
            shiftedVelocity[ii][jj] = velocities[ii][jj] + forces[ii][jj] * 0.5 * delta_T * atoms[ii].inverseMass;
            kinetic +=  atoms[ii].mass * shiftedVelocity[ii][jj] * shiftedVelocity[ii][jj];
        }
    }
    kinetic = 0.5 * kinetic;
    kinetic_energy.push_back(kinetic);
    return kinetic_energy.back();
}
int Micromd::getNumAtoms(){
    return atoms.size();
}

void Micromd::setInitBox(vector<Vec3> box){
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

void Micromd::setInitPositions(vector<Vec3> posi){
    if (posi.size() != natoms){
        cout<<"Error on positions size"<<endl;;
        exit(1);
    }
    for(int ii = 0; ii < natoms; ii++){
        for(int jj = 0; jj < 3; jj++){
            positions[ii][jj] = posi[ii][jj];
        }
    }
}

void Micromd::setInitVelocities(vector<Vec3> velocity){
    if (velocity.size() != natoms){
        cout<<"Error on velocities size"<<endl;;
        exit(1);
    }
    for(int ii = 0; ii < natoms; ii++){
        for(int jj = 0; jj < 3; jj++){
            velocities[ii][jj] = velocity[ii][jj];
        }
    }
}

void Micromd::setInitVelocitiesToTemperature(double temperature, int randomSeed){
    // Generate the list of Gaussian random numbers.
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(randomSeed, sfmt);
    std::vector<double> randoms;
    while (randoms.size() < natoms*3) {
        double x, y, r2;
        do {
            x = 2.0*genrand_real2(sfmt)-1.0;
            y = 2.0*genrand_real2(sfmt)-1.0;
            r2 = x*x + y*y;
        } while (r2 >= 1.0 || r2 == 0.0);
        double multiplier = sqrt((-2.0*std::log(r2))/r2);
        randoms.push_back(x*multiplier);
        randoms.push_back(y*multiplier);
    }

    // Assign the velocities.
    int nextRandom = 0;
    for (int i = 0; i < natoms; i++) {
        double mass = atoms[i].mass;
        if (mass != 0) {
            double velocityScale = sqrt(BOLTZ*temperature/mass);
            velocities[i] = Vec3(randoms[nextRandom++], randoms[nextRandom++], randoms[nextRandom++])*velocityScale;
        }
    }
    return;
}

void Micromd::setTypes(vector<int> types){
    for(int ii = 0; ii < natoms; ii ++){
        dtype[ii] = types[ii];
    }
}

void Micromd::addAtom(atom at){
    atoms.push_back(at);
}


double Micromd::calcEnergyAndForce(bool includeForces, bool includeEnergy){
    for(int ii=0; ii < natoms; ii++){
        // Multiply 10 for unit transformation from nanometers to angstrom.
        dcoord[ii * 3 + 0] = static_cast<float>(positions[ii][0]) * coordUnitCoeff;
        dcoord[ii * 3 + 1] = static_cast<float>(positions[ii][1]) * coordUnitCoeff;
        dcoord[ii * 3 + 2] = static_cast<float>(positions[ii][2]) * coordUnitCoeff;
    }
    nnp.compute(dener, dforce, dvirial, dcoord, dtype, dbox, nghost);

    if(includeForces){
        for(int ii = 0; ii < natoms; ii ++){
            for(int jj = 0; jj < 3; jj++){
                forces[ii][jj] = static_cast<double> (dforce[ii * 3 +jj]) * forceUnitCoeff;
            }
        }
    }

    if (includeEnergy){
        potential_energy.push_back(dener * energyUnitCoeff);
        return dener * energyUnitCoeff;
    }
    else
        return 0.0;
}