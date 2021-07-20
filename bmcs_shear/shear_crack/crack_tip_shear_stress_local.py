

import traits.api as tr
import numpy as np
import sympy as sp
from bmcs_utils.api import Model, View, Item, Float, SymbExpr, InjectSymbExpr
from bmcs_shear.shear_crack.stress_profile import \
    SZStressProfile
from bmcs_shear.shear_crack.crack_tip_shear_stress import SZCrackTipShearStress


class SZCrackTipShearStressLocal(SZCrackTipShearStress):
    name = 'crack tip stress state'

    f_c = tr.Property
    def _get_f_c(self):
        return self.sz_cp.matrix_.f_c

    f_t = tr.Property
    def _get_f_t(self):
        return self.sz_cp.matrix_.f_t

    tau_x_tip_1k = tr.Property

    def _get_tau_x_tip_1k(self):  #Shear stress distribution in uncracked region?
        # calculate the biaxial stress
        pass

    tau_z = tr.Property

    def _get_tau_z(self):
        # check if it is necessary
        pass

    def subplots(self, fig):
        return fig.subplots(1, 1)

    def update_plot(self, axes):
        ax1, ax2 = axes
        # @todo
