import json

try:
    import openmm as mm
    from openmm import NonbondedForce, CustomNonbondedForce, CustomBondForce
    from openmm import unit as u
    from openmm.unit import angstroms, nanometers 
    from openmm.app import CharmmPsfFile, CharmmCrdFile, CharmmParameterSet, AmberPrmtopFile, AmberInpcrdFile
except:
    import simtk.openmm as mm
    from simtk.openmm import NonbondedForce, CustomNonbondedForce, CustomBondForce
    from simtk import unit as u
    from simtk.unit import angstroms, nanometers
    from simtk.openmm.app import CharmmPsfFile, CharmmCrdFile, CharmmParameterSet, AmberPrmtopFile, AmberInpcrdFile

def read_top(filename, fftype='CHARMM'):
    if   fftype == 'CHARMM': top = CharmmPsfFile(filename)
    elif fftype == 'AMBER':  top = AmberPrmtopFile(filename)
    return top

def read_crd(filename, fftype='CHARMM'):
    if   fftype == 'CHARMM': crd = CharmmCrdFile(filename)
    elif fftype == 'AMBER':  crd = AmberInpcrdFile(filename)
    return crd

def read_params(filename):
    parFiles = ()
    for line in open(filename, 'r'):
        if '!' in line: line = line.split('!')[0]
        parfile = line.strip()
        if len(parfile) != 0: parFiles += ( parfile, )

    params = CharmmParameterSet( *parFiles )
    return params

def read_box(psf, filename):
    try:
        sysinfo = json.load(open(filename, 'r'))
        boxlx, boxly, boxlz = map(float, sysinfo['dimensions'][:3])
    except:
        for line in open(filename, 'r'):
            segments = line.split('=')
            if segments[0].strip() == "BOXLX": boxlx = float(segments[1])
            if segments[0].strip() == "BOXLY": boxly = float(segments[1])
            if segments[0].strip() == "BOXLZ": boxlz = float(segments[1])
    psf.setBox(boxlx*angstroms, boxly*angstroms, boxlz*angstroms)
    return psf

def restraints(system, crd, fc_bb, fc_sc, restrain_file):
    if fc_bb > 0 or fc_sc > 0:
        # positional restraints for protein
        posresPROT = mm.CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2;')
        posresPROT.addPerParticleParameter('k')
        posresPROT.addPerParticleParameter('x0')
        posresPROT.addPerParticleParameter('y0')
        posresPROT.addPerParticleParameter('z0')
        for line in open(restrain_file, 'r'):
            segments = line.strip().split()
            atom1 = int(segments[0])
            state = segments[1]
            xpos  = crd.positions[atom1].value_in_unit(nanometers)[0]
            ypos  = crd.positions[atom1].value_in_unit(nanometers)[1]
            zpos  = crd.positions[atom1].value_in_unit(nanometers)[2]
            if state == 'BB' and fc_bb > 0:
                fc_ppos = fc_bb
                posresPROT.addParticle(atom1, [fc_ppos, xpos, ypos, zpos])
            if state == 'SC' and fc_sc > 0:
                fc_ppos = fc_sc
                posresPROT.addParticle(atom1, [fc_ppos, xpos, ypos, zpos])
        system.addForce(posresPROT)

    return system


def vfswitch(system, psf, r_on, r_off):
    r_on = r_on
    r_off = r_off

    # custom nonbonded force for force-switch
    chknbfix = False
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            nonbonded = force
        if isinstance(force, CustomNonbondedForce):
            nbfix     = force
            chknbfix  = True

    # vfswitch
    vfswitch = CustomNonbondedForce('step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6) \
                                     +step(r-Ron)*step(Roff-r)*(cr12*rjunk6 - cr6*rjunk3) \
                                     -step(r-Ron)*step(Ron-r)*(cr12*rjunk6 - cr6*rjunk3); \
                                     cr6  = ccnbb*ofdif3*rjunk3; \
                                     cr12 = ccnba*ofdif6*rjunk6; \
                                     rjunk3 = r3-recof3; \
                                     rjunk6 = tr6-recof6; \
                                     r3 = r1*tr2; \
                                     r1 = sqrt(tr2); \
                                     tr6 = tr2 * tr2 * tr2; \
                                     tr2 = 1.0/s2; \
                                     s2 = r*r; \
                                     ccnbb = 4.0*epsilon*sigma^6; \
                                     ccnba = 4.0*epsilon*sigma^12; \
                                     sigma = sigma1+sigma2; \
                                     epsilon = epsilon1*epsilon2; \
                                     onoff3 = recof3/on3; \
                                     onoff6 = recof6/on6; \
                                     ofdif3 = off3/(off3 - on3); \
                                     ofdif6 = off6/(off6 - on6); \
                                     recof3 = 1.0/off3; \
                                     on6 = on3*on3; \
                                     on3 = c2onnb*Ron; \
                                     recof6 = 1.0/off6; \
                                     off6 = off3*off3; \
                                     off3 = c2ofnb*Roff; \
                                     c2ofnb = Roff*Roff; \
                                     c2onnb = Ron*Ron; \
                                     Ron  = %f; \
                                     Roff = %f;' % (r_on, r_off) )
    vfswitch.addPerParticleParameter('sigma')
    vfswitch.addPerParticleParameter('epsilon')
    vfswitch.setNonbondedMethod(vfswitch.CutoffPeriodic)
    vfswitch.setCutoffDistance(nonbonded.getCutoffDistance())
    for i in range(nonbonded.getNumParticles()):
        chg, sig, eps = nonbonded.getParticleParameters(i)
        nonbonded.setParticleParameters(i, chg, 0.0, 0.0) # zero-out LJ
        sig = sig*0.5
        eps = eps**0.5
        vfswitch.addParticle([sig, eps])
    for i in range(nonbonded.getNumExceptions()):
        atom1, atom2 = nonbonded.getExceptionParameters(i)[:2]
        vfswitch.addExclusion(atom1, atom2)
    vfswitch.setForceGroup(psf.NONBONDED_FORCE_GROUP)
    system.addForce(vfswitch)

    # vfswitch14
    vfswitch14 = CustomBondForce('step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6) \
                                  +step(r-Ron)*step(Roff-r)*(cr12*rjunk6 - cr6*rjunk3) \
                                  -step(r-Ron)*step(Ron-r)*(cr12*rjunk6 - cr6*rjunk3); \
                                  cr6  = ccnbb*ofdif3*rjunk3; \
                                  cr12 = ccnba*ofdif6*rjunk6; \
                                  rjunk3 = r3-recof3; \
                                  rjunk6 = tr6-recof6; \
                                  r3 = r1*tr2; \
                                  r1 = sqrt(tr2); \
                                  tr6 = tr2 * tr2 * tr2; \
                                  tr2 = 1.0/s2; \
                                  s2 = r*r; \
                                  ccnbb = 4.0*epsilon*sigma^6; \
                                  ccnba = 4.0*epsilon*sigma^12; \
                                  onoff3 = recof3/on3; \
                                  onoff6 = recof6/on6; \
                                  ofdif3 = off3/(off3 - on3); \
                                  ofdif6 = off6/(off6 - on6); \
                                  recof3 = 1.0/off3; \
                                  on6 = on3*on3; \
                                  on3 = c2onnb*Ron; \
                                  recof6 = 1.0/off6; \
                                  off6 = off3*off3; \
                                  off3 = c2ofnb*Roff; \
                                  c2ofnb = Roff*Roff; \
                                  c2onnb = Ron*Ron; \
                                  Ron  = %f; \
                                  Roff = %f;' % (r_on, r_off) )
    vfswitch14.addPerBondParameter('sigma')
    vfswitch14.addPerBondParameter('epsilon')
    for i in range(nonbonded.getNumExceptions()):
        atom1, atom2, chg, sig, eps = nonbonded.getExceptionParameters(i)
        nonbonded.setExceptionParameters(i, atom1, atom2, chg, 0.0, 0.0) # zero-out LJ14
        vfswitch14.addBond(atom1, atom2, [sig, eps])
    system.addForce(vfswitch14)

    # vfswitch_NBFIX
    if chknbfix:
        nbfix.setEnergyFunction('step(Ron-r)*(ccnba*tr6*tr6-ccnbb*tr6+ccnbb*onoff3-ccnba*onoff6) \
                                 +step(r-Ron)*step(Roff-r)*(cr12*rjunk6 - cr6*rjunk3) \
                                 -step(r-Ron)*step(Ron-r)*(cr12*rjunk6 - cr6*rjunk3); \
                                 cr6  = ccnbb*ofdif3*rjunk3; \
                                 cr12 = ccnba*ofdif6*rjunk6; \
                                 rjunk3 = r3-recof3; \
                                 rjunk6 = tr6-recof6; \
                                 r3 = r1*tr2; \
                                 r1 = sqrt(tr2); \
                                 tr6 = tr2 * tr2 * tr2; \
                                 tr2 = 1.0/s2; \
                                 s2 = r*r; \
                                 ccnbb = bcoef(type1, type2); \
                                 ccnba = acoef(type1, type2)^2; \
                                 onoff3 = recof3/on3; \
                                 onoff6 = recof6/on6; \
                                 ofdif3 = off3/(off3 - on3); \
                                 ofdif6 = off6/(off6 - on6); \
                                 recof3 = 1.0/off3; \
                                 on6 = on3*on3; \
                                 on3 = c2onnb*Ron; \
                                 recof6 = 1.0/off6; \
                                 off6 = off3*off3; \
                                 off3 = c2ofnb*Roff; \
                                 c2ofnb = Roff*Roff; \
                                 c2onnb = Ron*Ron; \
                                 Ron  = %f; \
                                 Roff = %f;' % (r_on, r_off) )

        # turn off long range correction (OpenMM Issues: #2353)
        nbfix.setUseLongRangeCorrection(False)

    return system