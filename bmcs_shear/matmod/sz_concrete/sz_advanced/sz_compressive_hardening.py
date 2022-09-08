import numpy as np
import sympy as sp
import traits.api as tr
from bmcs_utils.api import InteractiveModel, View, Item, Float, SymbExpr, InjectSymbExpr
from bmcs_shear.matmod.i_matmod import IMaterialModel

class CompressiveHardeningBehaviorSymb(SymbExpr):
    eps_cp = sp.Symbol('\epsilon_{cp}', real=True)
    eps_p = sp.Symbol('\epsilon_{p}', real=True)
    f_t, f_c = sp.symbols(r'f_t, f_c', real=True)

    f_co = 2* f_t

    sigma_c = sp.Piecewise(
        (2 * f_t, eps_p == 0),
        (f_co + (f_c - f_co) * sp.sqrt(1 - ((eps_cp - eps_p) / (eps_cp)) ** 2), eps_p <= eps_cp),
        (f_c, eps_p > eps_cp)
    )

    symb_model_params = ['f_t', 'f_c'] #'eps_cp',

    symb_expressions = [('sigma_c', ('eps_p','eps_cp'))]

@tr.provides(IMaterialModel)
class CompressiveHardeningBehavior(InteractiveModel, InjectSymbExpr):
    name = 'Compressive Hardening Behavior'

    symb_class = CompressiveHardeningBehaviorSymb

    f_c = Float(41.8)  ## compressive strength of Concrete in MPa
    f_t = Float(3.2)  ## tensile strength of Concrete in MPa
    #eps_cp = Float(0.0014)  ## plastic strain corresponding to compressive strength

    ipw_view = View(
        Item('f_c'),
        Item('f_t')
        #Item('eps_cp')
    )

    def get_sigma_c(self, eps_p, eps_cp):
        return self.symb.get_sigma_c(eps_p,eps_cp)

    def subplots(self,fig):
        return fig.subplots(1,2)

    def update_plot(self,axes):
        ax_w, ax_s = axes
        eps_p_ = np.linspace(0, 0.0020)
        eps_cp_ = 0.0014
        sigma_c_ = self.get_sigma_c(eps_p_, eps_cp_)
        ax_w.plot(eps_p_, sigma_c_)
        ax_w.set_xlabel(r'$\epsilon_p\;\;\mathrm{[mm]}$')
        ax_w.set_ylabel(r'$\sigma_c}\;\;\mathrm{[MPa]}$')