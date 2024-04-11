import bmcs_utils.api as bu
from .dic_crack_list import DICCrackList
import traits.api as tr
from matplotlib import cm
import ibvpy.api as ib
import numpy as np
import copy
from scipy.interpolate import interp2d, LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from .cached_array import cached_array


class DICCompressionField(ib.TStepBC):
    name = 'compression field'

    cl = bu.Instance(DICCrackList)
    """List of identified cracks
    """

    depends_on = 'cl'

    data_dir = tr.DelegatesTo('cl')
    beam_param_file = tr.DelegatesTo('cl')

    bd = tr.DelegatesTo('cl')
    '''Beam design.
    '''

