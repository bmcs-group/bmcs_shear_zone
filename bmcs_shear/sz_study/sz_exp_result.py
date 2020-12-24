from bmcs_shear.shear_crack.deformed_state import \
    SZDeformedState
from bmcs_shear.shear_crack.stress_profile import SZStressProfile
import traits.api as tr
import numpy as np
from bmcs_utils.api import InteractiveModel, View, Item, mpl_align_xaxis

class SZShearStressDeflectionProfile(InteractiveModel):

    name = "Calculated Shear Stress Profile"

    sz_sp = tr.Instance(SZStressProfile, ())
    ds = tr.Instance(SZDeformedState, ())
    sz_cp = tr.DelegatesTo('ds')
    sz_bd = tr.DelegatesTo('sz_cp')

    _ITR = tr.DelegatesTo('sz_cp')
    _INC = tr.DelegatesTo('sz_cp')
    _MAT = tr.DelegatesTo('sz_cp')
    _GEO = tr.DelegatesTo('sz_cp')
    _DSC = tr.DelegatesTo('sz_cp')
    _ALL = tr.Event

    @tr.on_trait_change('_ITR, _INC, _GEO, _MAT, _DSC')
    def _reset_ALL(self):
        self._ALL = True

    ipw_view = View()

    tau_exp = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')

    @tr.cached_property
    def _get_tau_exp(self):
        B = self.ds.sz_bd.B
        D = self.ds.sz_bd.B
        F_La = self.sz_sp.F_La
        F_Na = self.sz_sp.F_Na
        V_exp = F_La[..., 1] + F_Na[...,1]
        print((V_exp )/ (B * D))
        return V_exp / (B * D)

    normalized_def = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')

    @tr.cached_property
    def _get_normalized_def(self):
        L = self.ds.sz_bd.L
        delta = self.sz_sp.u_La[...,1]
        print(delta / L)
        return (delta / L)

    def plot_tau_exp(self, ax, vot=1.0):
        ax.plot(self.normalized_def, self.tau_exp, lw=2, color='blue')
        ax.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax.set_ylabel(r'$\tau_{Exp}\;\;\mathrm{[MPa]}$')
        ax.set_title('Calculated Shear Stress')

    def subplots(self, fig):
        return fig.subplots(1,2)

    def update_plot(self, axes):
        ax1, ax2 = axes
        self.plot_tau_exp(ax1)

