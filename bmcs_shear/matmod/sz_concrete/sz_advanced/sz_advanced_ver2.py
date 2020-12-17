from bmcs_shear.matmod.i_matmod import IMaterialModel
from bmcs_utils.api import InteractiveModel, View, Item, Float, SymbExpr, InjectSymbExpr
import traits.api as tr
import numpy as np
import sympy as sp

class SZAdancedConModelSymb(SymbExpr):
    w, s = sp.symbols(r'w, s', real=True)
    f_c, f_t = sp.symbols(r'f_c, f_t', nonnegative=True)
    d_ag, E_c = sp.symbols(r'd_ag, E_c', nonnegative=True)
    L = sp.Symbol(r'L', nonnegative=True)
    c_1, c_2 = sp.symbols(r'c_1, c_2', nonnegative = True)


    G_f = 0.028 * f_c**0.18 * d_ag**0.32

    L_c = E_c * G_f / (f_t**2)

    w_cr = f_t / E_c * L_c

    eps_cp = w_cr / L_c

    eps_p = w / L_c

    f_co = 2 * f_t

    f_ce = f_c * (1 / (0.8))

    w_tc = 5.14 * 0.5 / f_t


    sigma_w = sp.Piecewise(
        (- f_c,  eps_p <  - eps_cp),
        # (- f_ce * (2*(eps_p / -0.002) - (eps_p / -0.002)**2 ), E_c * eps_p < f_t), #+ 170 * eps_cp)
        ( - (f_co + (f_c - f_co) * sp.sqrt(1 - ((eps_cp - eps_p) / (eps_cp)) ** 2)), eps_p < eps_cp),
        # (E_c * eps_p, E_c * eps_p <= f_t),
        (f_t * (1 + ((c_1 * w)/(w_tc))**3) * sp.exp((-c_2* w)/(w_tc)) - (w/w_tc) * (1 + c_1**3) * sp.exp(-c_2) \
             , w > w_cr) #E_c * eps_p > f_t
        )

    # r = s / w
    #
    # tau_0 = 0.25 * f_c
    #
    # a_3 = 2.45 / tau_0
    #
    # a_4 = 2.44 * (1 - (4 / tau_0))
    #
    # tau_ag = tau_0 * (1 - sp.sqrt((2 * w)/d_g)) * r * (a_3 + (a_4 * sp.Abs(r)**3)) / (1 + (a_4 *r**4))
    #
    # sigma_ag = -0.62 * sp.sqrt(w) * (r) / ((1 + r ** 2) ** 0.25) * tau_ag

    symb_model_params = ['f_c', 'f_t', 'd_ag', 'E_c', 'L', 'c_1', 'c_2']

    symb_expressions = [('sigma_w', ('w',))]
                        # ('sigma_ag', ('w','s',))]

@tr.provides(IMaterialModel)
class SZAdancedConModel(InteractiveModel, InjectSymbExpr):
    name = 'SZ Advanced Concrete Model'

    symb_class = SZAdancedConModelSymb

    f_c = Float(33.3)  ## (compressive strength of Concrete in MPa)
    f_t = Float(3)  ## (tensile strength of Concrete in MPa)
    d_ag = Float(22)  ##mm (size of aggregate)
    E_c = Float(28000)  ## (Elastic Modulus of Concrete in MPa)
    L = Float(1000) ## (Length of beam in mm)
    c_1 = Float(3) ## (Constants)
    c_2 = Float(6.93)  ## (Constants)

    ipw_view = View(
        Item('f_c'),
        Item('f_t'),
        Item('d_ag'),
        Item('E_c'),
        Item('L'),
        Item('c_1'),
        Item('c_2')
    )

    def get_sigma_w(self, w):
        # calculating the stress due to opening of crack
        return self.symb.get_sigma_w(w)

    def subplots(self,fig):
        return fig.subplots(1,2)

    def update_plot(self,axes):
        ax_w, ax_s = axes
        w_range = np.linspace(-0.1,0.1, 100)
        sigma_w_ = self.get_sigma_w(w_range)
        ax_w.plot(w_range, sigma_w_)
        ax_w.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        ax_w.set_ylabel(r'$\sigma_{\mathrm{w}}\;\;\mathrm{[MPa]}$')


