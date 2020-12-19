from bmcs_shear.matmod.i_matmod import IMaterialModel
from bmcs_utils.api import View, Item, Float
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
import sympy as sp
from bmcs_shear.matmod.sz_crack_bridge.cb_advanced.sz_pull_out_fib import PullOut
from bmcs_shear.matmod.sz_crack_bridge.cb_advanced.sz_dowel_action import DowelAction

class CrackBridgeModelAdvExpr(bu.SymbExpr):
    s_1, s_2 = sp.symbols(r's_1, s_2', nonnegative=True)
    s_3 = sp.symbols(r's_3', nonnegative=True)
    s, f_c = sp.symbols(r's, f_c', real=True)
    alpha = sp.Symbol(r'\alpha', nonnegative=True)
    B = sp.symbols(r'B', nonnegative=True)
    n, d_s = sp.symbols(r'n, d_s', nonnegative=True)

    tau_b_max = 2.5 * sp.sqrt(f_c)

    tau_bf = 0.4 * tau_b_max

    tau_b = sp.Piecewise(
        (tau_b_max * (s / s_1) ** alpha, s <= s_1),
        (tau_b_max, s <= s_2),
        (tau_b_max - ((tau_b_max - tau_bf) * (s - s_2) / (s_3 - s_2)), s <= s_3),
        (tau_bf, s > s_3)
    )

    d_tau_b = tau_b.diff(s)
    #print(d_tau_b)

    b_n = B - n * d_s

    V_d_max = 1.64 * b_n * d_s * f_c ** (1 / 3)

    V_da_1 = V_d_max * (s / 0.05) * (2 - (s / 0.05))

    V_da_2 = V_d_max * ((2.55 - s) / 2.5)

    V_da = sp.Piecewise(
        (V_da_1, s <= 0.05),
        (V_da_2, True))  # delta > 0.05

    symb_model_params = ['s_1', 's_2', 's_3', 'f_c', 'alpha', 'B', 'n', 'd_s']

    symb_expressions = [('tau_b', ('s',)),
                        ('d_tau_b', ('s',)),
                        ('V_da', ('s',))]
    pass

@tr.provides(IMaterialModel)
class CrackBridgeAdv(bu.InteractiveModel):

    name = 'Crack Bridge Adv'
    node_name = 'crack bridge model'

    pullout = tr.Instance(PullOut, ())
    dowelaction = tr.Instance(DowelAction, ())

    s_1 = Float(1)
    s_2 = Float(2)
    s_3 = Float(4)
    f_c = Float(37.9)  ## compressive strength of Concrete in MPa
    alpha = Float(0.4)
    B = Float(75)  ##mm (width of the beam)
    n = Float(4)  ##number of bars
    d_s = Float(16)  ##dia of steel mm

    ipw_view = View(
        Item('s_1', latex=r's_1'),
        Item('s_2', latex=r's_2'),
        Item('s_3', latex=r's_3'),
        Item('f_c', latex=r'f_c'),
        Item('alpha', latex=r'\alpha'),
        Item('B', latex=r'B'),
        Item('n', latex=r'n'),
        Item('d_s', latex=r'd_s')
    )

    def get_tau_b(self, s):
        '''Calculating bond stresses '''
        tau_b = self.symb.get_tau_b(s)
        return tau_b

    def get_df(self, s):
        '''Calculating dowel action force '''
        V_df = self.dowelaction.get_sig_s_f(s)
        return V_df

    def subplots(self,fig):
        return fig.subplots(1,2)

    def update_plot(self,axes):
        '''Plotting function '''
        ax_s, ax_f = axes
        s_ = np.linspace(0, 1, 100)
        tau_b_ = self.get_tau_b(s_)
        V_df_ = self.get_df(s_)
        ax_s.plot(s_, tau_b_)
        ax_f.plot(s_, V_df_)
        ax_s.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax_s.set_ylabel(r'$\tau_b\;\;\mathrm{[MPa]}$')
        ax_f.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax_f.set_ylabel(r'$V_{df}\;\;\mathrm{[N]}$')