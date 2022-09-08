import numpy as np
import sympy as sp
import traits.api as tr
from bmcs_utils.api import InteractiveModel, View, Item, Float, SymbExpr, InjectSymbExpr
from bmcs_shear.matmod.i_matmod import IMaterialModel


class PullOutFibSymb(SymbExpr):
    s_1, s_2 = sp.symbols(r's_1, s_2', nonnegative=True)
    s_3 = sp.symbols(r's_3', nonnegative=True)
    s, f_c = sp.symbols(r's, f_c', real=True)
    alpha = sp.Symbol(r'\alpha', nonnegative = True)

    tau_b_max = 2.5 * sp.sqrt(f_c)

    tau_bf = 0.4 * tau_b_max

    tau_b = sp.Piecewise(
        (tau_b_max * (s / s_1) ** alpha, s <= s_1),
        (tau_b_max, s <= s_2),
        (tau_b_max - ((tau_b_max - tau_bf) * (s - s_2) / (s_3 - s_2)), s <= s_3),
        (tau_bf, s > s_3)
    )

    d_tau_b = tau_b.diff(s)
    # print(d_tau_b)

    symb_model_params = ['s_1', 's_2', 's_3', 'f_c', 'alpha']

    symb_expressions = [('tau_b', ('s',)),
                        ('d_tau_b', ('s',))]


@tr.provides(IMaterialModel)
class PullOutFib(InteractiveModel, InjectSymbExpr):
    '''
    Explicitly defined Pullout using the Fib code
    '''
    name = 'Pull Out Fib'

    symb_class = PullOutFibSymb

    s_1 = Float(1)
    s_2 = Float(2)
    s_3 = Float(4)
    f_c = Float(37.9)  ## compressive strength of Concrete in MPa
    alpha = Float(0.4)

    ipw_view = View(
        Item('s_1'),
        Item('s_2'),
        Item('s_3'),
        Item('f_c'),
        Item('alpha')
    )

    def get_tau_b(self, s):
        # calculating bond stress
        return self.symb.get_tau_b(s)

    def get_d_tau_b(self, s):
        signs = np.sign(s)
        return signs * self.symb.get_d_tau_b(signs * s)

    def subplots(self, fig):
        return fig.subplots(1, 2)

    def update_plot(self, axes):
        ax_w, ax_s = axes
        s_range = np.linspace(0, 6)
        tau_b = self.get_tau_b(s_range)
        ax_w.plot(s_range, tau_b)
        ax_w.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax_w.set_ylabel(r'$\tau_{\mathrm{b}}\;\;\mathrm{[MPa]}$')
