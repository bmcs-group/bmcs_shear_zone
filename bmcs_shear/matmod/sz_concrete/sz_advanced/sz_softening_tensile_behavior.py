import numpy as np
import sympy as sp
import traits.api as tr
from bmcs_utils.api import InteractiveModel, View, Item, Float, SymbExpr, InjectSymbExpr
from bmcs_shear.matmod.i_matmod import IMaterialModel

class TensileSofteningBehaviorSymb(SymbExpr):
    w_1, w_2 = sp.symbols(r'w_1, w_2')
    w = sp.symbols(r'w', nonnegative=True)
    lamda = sp.symbols(r'\lambda', nonnegative=True)
    d_a = sp.symbols(r'd_a', nonnegative=True)
    f_t, G_f = sp.symbols(r'f_t, G_f', nonnegative=True)
    c_2 = sp.Symbol('c_2', nonnegative=True)
    c_1 = sp.Symbol('c_1', nonnegative=True)

    alpha_f = lamda - d_a/8

    sigma_s = (f_t * (2 - f_t * (w_1 / G_f))) / alpha_f #w_1 = CTOD_c

    #
    sigma_w = sp.Piecewise(
            (f_t - (f_t - sigma_s) * (w / w_1), w <= w_1 ),
            (sigma_s * (w_2 - w) / (w_2 - w_1),  w <= w_2),
    )

    w_tc = 5.14 * G_f/f_t

    sigma_t = f_t * (1 + ((c_1 * w)/(w_tc))**3) * \
              sp.exp((-c_2* w)/(w_tc)) - (w/w_tc) * \
              (1 + c_1**3) * sp.exp(-c_2)

    sigma_t_diff = sigma_t.diff(w)

    symb_model_params = ['w_1', 'w_2', 'lamda', 'd_a', 'f_t', 'G_f', 'c_1', 'c_2']

    symb_expressions = [('sigma_w', ('w',)),
                        ('sigma_t', ('w',)),
                        ('sigma_t_diff', ('w',))]

@tr.provides(IMaterialModel)
class TensileSofteningBehavior(InteractiveModel, InjectSymbExpr):
    name = 'Tensile Softening Behavior'

    symb_class = TensileSofteningBehaviorSymb

    # delta = Float(0.1) ##mm (vertical displacement)
    w_1 = Float(0.3)  ##mm (critical opening)
    w_2 = Float(1)  ##opening mm
    lamda = Float(4)  ##calibration factor
    d_a = Float(22)  ## dia of steel mm
    f_t = Float(3)  ## tensile strength of Concrete in MPa
    G_f = Float(0.5)  ## Fracture Energy in N/m
    c_1 = Float(3)
    c_2 = Float(6.93)

    E_ct = Float(28000, desc='E modulus in tension')

    ipw_view = View(
        Item('w_1'),
        Item('w_2'),
        Item('lamda'),
        Item('d_a'),
        Item('f_t'),
        Item('G_f'),
        Item('c_1'),
        Item('c_2')
    )

    def get_sigma_w(self, w):
        sig = self.symb.get_sigma_w(w)


    def get_w_tc(self):
        return 5.14 * self.G_f/self.f_t

    def get_sigma_w_t(self, w):
        return self.symb.get_sigma_t(w)

    def get_sigma_w_t_diff(self,w):
        return self.symb.get_sigma_t_diff(w)

    def subplots(self,fig):
        return fig.subplots(1,2)

    def update_plot(self,axes):
        ax_w, ax_s = axes
        w_ = np.linspace(0, 1)
        w_t_ = np.linspace(0, self.get_w_tc(), 100)
        sigma_w_ = self.get_sigma_w(w_)
        sigma_t_ = self.get_sigma_w_t(w_t_)
        ax_w.plot(w_, sigma_w_)
        ax_s.plot(w_t_, sigma_t_)
        ax_w.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        ax_w.set_ylabel(r'$\sigma_w}\;\;\mathrm{[MPa]}$')
        ax_s.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        ax_s.set_ylabel(r'$\sigma_t}\;\;\mathrm{[MPa]}$')

