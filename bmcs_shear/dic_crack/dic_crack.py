
from .dic_grid import DICGrid
from .dic_strain_grid import DICStrainGrid
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
from .dic_cor import DICCOR

class DICCrack(bu.Model):
    '''
    Model of a shear crack with the representation of the kinematics
    evaluating the opening and sliding displacmeent
    '''
    name = 'crack'

    dic_cor = bu.Instance(DICCOR)

    tree = ['dic_cor']