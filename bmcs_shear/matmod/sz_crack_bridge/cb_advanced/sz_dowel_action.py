import numpy as np
import sympy as sp
import traits.api as tr
from bmcs_utils.api import InteractiveModel, View, Item, Float, SymbExpr, InjectSymbExpr
from bmcs_shear.matmod.i_matmod import IMaterialModel


class DowelActionSymb(SymbExpr):
    b_w, f_c = sp.symbols(r'b_w, f_c', nonnegative=True)
    n, d_s = sp.symbols(r'n, d_s', nonnegative=True)
    s = sp.symbols(r's', nonnegative=True)

    b_n = b_w - n * d_s

    V_d_max = 1.64 * b_n * d_s * f_c ** (1 / 3)

    V_da_1 = V_d_max * (s / 0.05) * (2 - (s / 0.05))

    V_da_2 = V_d_max * ((2.55 - s) / 2.5)

    V_da = sp.Piecewise(
        (V_da_1, s <= 0.05),
        (V_da_2, True))  # delta > 0.05

    symb_model_params = ['b_w', 'f_c', 'n', 'd_s']

    symb_expressions = [('V_da', ('s',))]


@tr.provides(IMaterialModel)
class DowelAction(InteractiveModel, InjectSymbExpr):
    name = 'Dowel Action'

    symb_class = DowelActionSymb

    # delta = Float(0.1) ##mm (vertical displacement)
    b_w = Float(250)  ##mm (width of the beam)
    n = Float(2)  ##number of bars
    d_s = Float(28)  ##dia of steel mm
    f_c = Float(33.3)  ## compressive strength of Concrete in MPa

    ipw_view = View(
        Item('b_w'),
        Item('n'),
        Item('d_s'),
        Item('f_c')
    )

    def get_sig_s_f(self, delta):
        # calculating the dowel action force
        return self.symb.get_V_da(delta)

    def subplots(self, fig):
        return fig.subplots(1, 2)

    def update_plot(self, axes):
        ax_w, ax_s = axes
        s_range = np.linspace(0, 2)
        V = self.get_sig_s_f(s_range)
        ax_w.plot(s_range, V)
        ax_w.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax_w.set_ylabel(r'$V_{\mathrm{da}}\;\;\mathrm{[N]}$')
