

import numpy as np
import sympy as sp
import traits.api as tr
from bmcs_utils.api import InteractiveModel, View, Item, Float, SymbExpr, InjectSymbExpr
from bmcs_shear.matmod.i_matmod import IMaterialModel
from bmcs_cross_section.api import ReinfLayer

class CrackBridgeSteelSymb(SymbExpr):
    E, A = sp.symbols(r'E, A', nonnegative = True )
    tau, p = sp.symbols(r'\bar{\tau}, p', nonnegative = True)
    P, w = sp.symbols('P, w')
    sig_y = sp.symbols('sig_y', positive=True)

    Pw_pull = sp.sqrt(2*w*tau*E*A*p)
    P_max = A * sig_y
    w_argmax = sp.solve(P_max - Pw_pull, w)[0]

    Pw_pull_y = sp.Piecewise((Pw_pull, w < w_argmax),
                             (P_max, w >= w_argmax))

    symb_model_params = ['E', 'A', 'p', 'tau', 'sig_y']

    symb_expressions = [('Pw_pull_y', ('w')),
                        ('w_argmax', ())]

@tr.provides(IMaterialModel)
class CrackBridgeSteel(ReinfLayer, InjectSymbExpr):
    name = 'Steel Bridge'
    tree = []

    symb_class = CrackBridgeSteelSymb

    E = Float(210000)
    d_s = Float(16)
    tau = Float(16)
    sig_y = Float(500)
    A = tr.Property(Float)
    def _get_A(self):
        return (self.d_s / 2) ** 2 * np.pi
    p = tr.Property(Float)
    def _get_p(self):
        return (self.d_s) * np.pi

    ipw_view = View(
        Item('E', latex=r'E'),
        Item('d_s', latex=r'd_s'),
        Item('tau', latex=r'\tau'),
        Item('sig_y', latex=r'\sigma_y'),
        Item('A', latex=r'A', readonly=True),
        Item('p', latex=r'p', readonly=True)
    )

    def get_F_a(self, u_a):
        F_w = self.get_sig_w_f(u_a[...,0])
        F_s = np.zeros_like(F_w)
        return np.array([F_w,F_s], dtype=np.float_).T

    def get_sig_w_f(self, w):
        # distinguish the crack width from the end slip of the pullout
        # which delivers the crack bridging force
        return self.symb.get_Pw_pull_y(w/2)

    def subplots(self,fig):
        return fig.subplots(1,2)

    def update_plot(self,axes):
        ax_w, ax_s = axes
        w_argmax = self.symb.get_w_argmax()
        w_range = np.linspace(0, 3*w_argmax)
        sig = self.get_sig_w_f(w_range)
        ax_w.plot(w_range, sig)
        ax_w.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        ax_w.set_ylabel(r'$\sigma\;\;\mathrm{[MPa]}$')


#=========================================================================
# Steel
#=========================================================================

w = sp.symbols('w')

L_f, E_f, f_s_t = sp.symbols('L_f, E_f, f_s_t')

sig_w_f = sp.Piecewise(
    (E_f * w / L_f, E_f * w / L_f <= f_s_t),
    (f_s_t, E_f * w / L_f > f_s_t)
)

@tr.provides(IMaterialModel)
class SteelMaterialModel(InteractiveModel):
    name = 'Steel behavior'
    #=========================================================================
    # Steel sig_eps
    #=========================================================================
    L_f = Float(200.0, MAT=True)
    E_f = Float(210000, MAT=True)
    f_s_t = Float(500, MAT=True)

    ipw_view = View(
        Item('L_f'),
        Item('E_f'),
        Item('f_s_t')
    )
    steel_law_data = tr.Property(depends_on='+MAT')

    @tr.cached_property
    def _get_steel_law_data(self):
        return dict(L_f=float(self.L_f),
                    E_f=float(self.E_f),
                    f_s_t=self.f_s_t)

    get_sig_w_f = tr.Property(depends_on='+MAT')

    @tr.cached_property
    def _get_get_sig_w_f(self):
        return sp.lambdify(w, sig_w_f.subs(self.steel_law_data), 'numpy')

    # get_d_sig_w_f = tr.Property(depends_on='+MAT')
    #
    # @tr.cached_property
    # def _get_get_d_sig_eps_f(self):
    #     return sp.lambdify(w, d_sig_w_f.subs(self.steel_law_data), 'numpy')

    def plot_sig_w_f(self, ax, vot=1.0):
        w_max_expr = (f_s_t / E_f * L_f * 2).subs(self.steel_law_data)
        w_min_expr = 0
        w_max = np.float_(w_max_expr)
        w_min = np.float_(w_min_expr)
        w_data = np.linspace(w_min, w_max, 50)
        ax.plot(w_data, self.get_sig_w_f(w_data), lw=2, color='darkred')
        ax.fill_between(w_data, self.get_sig_w_f(w_data),
                        color='darkred', alpha=0.2)
        ax.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        ax.set_ylabel(r'$\sigma\;\;\mathrm{[MPa]}$')
        ax.set_title('crack opening law')

    def update_plot(self, axes):
        ax = axes
        self.plot_sig_w_f(ax)

