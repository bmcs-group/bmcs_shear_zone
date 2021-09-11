

from .dic_cor import DICCOR
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
from scipy.optimize import minimize

class DICAlign(bu.Model):

    dic_cor = bu.Instance(DICCOR, ())
    dic_crack = tr.DelegatesTo('dic_cor')

    T_alpha = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_alpha(self):
        X_cor = self.dic_cor.X_cor
        # get the position of the upper left corner
        x_ul = self.dic_crack.X_ija[-1,-1,:]
        u_ul = self.dic_crack.u_ija[-1,-1,:]