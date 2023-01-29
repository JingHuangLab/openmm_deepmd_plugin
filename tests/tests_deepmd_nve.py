from sys import stdout, exit
import sys
import os, platform
import numpy as np
from math import sqrt
import glob, random
try:
    from openmm import app
    import openmm as mm
    from openmm import unit as u
    import openmm as mm
    import openmm.app as app
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
except:
    from simtk.openmm import app
    import simtk.openmm as mm
    from simtk import unit as u
    import simtk.openmm as mm
    import simtk.openmm.app as app
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *

import time
import argparse

from OpenMMDeepmdPlugin import DeepmdForce
from OpenMMDeepmdPlugin import ForceReporter, Simulation4Deepmd



def test_deepmd_energy_forces():
    pass

def test_deepmd_nve():
    pass
