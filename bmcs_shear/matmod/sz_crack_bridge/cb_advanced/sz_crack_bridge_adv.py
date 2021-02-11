from bmcs_shear.matmod.i_matmod import IMaterialModel
from bmcs_utils.api import View, Item, Float
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
import sympy as sp
from bmcs_shear.matmod.sz_crack_bridge.cb_advanced.sz_pull_out_fib import PullOutFib
from bmcs_shear.matmod.sz_crack_bridge.cb_advanced.sz_dowel_action import DowelAction

class CrackBridgeModelAdvExpr(bu.SymbExpr):
    w_1, w_2 = sp.symbols(r's_1, s_2', nonnegative=True)
    w_3 = sp.symbols(r's_3', nonnegative=True)
    w, f_c = sp.symbols(r's, f_c', real=True)
    alpha = sp.Symbol(r'\alpha', nonnegative=True)
    B = sp.symbols(r'B', nonnegative=True)
    n, d_s = sp.symbols(r'n, d_s', nonnegative=True)
    s = sp.Symbol('s', nonnegative = True)

    tau_b_max = 2.5 * sp.sqrt(f_c)

    tau_bf = 0.4 * tau_b_max

    tau_b = sp.Piecewise(
        (tau_b_max * ( w/ w_1) ** alpha, w <= w_1),
        (tau_b_max, w <= w_2),
        (tau_b_max - ((tau_b_max - tau_bf) * (w - w_2) / (w_3 - w_2)), w <= w_3),
        (tau_bf, w > w_3)
    )

    d_tau_b = tau_b.diff(w)
    #print(d_tau_b)

    b_n = B - n * d_s

    V_d_max = 1.64 * b_n * d_s * f_c ** (1 / 3)

    V_da_1 = V_d_max * (s / 0.05) * (2 - (s / 0.05))

    V_da_2 = V_d_max * ((2.55 - s) / 2.5)

    V_da = sp.Piecewise(
        (V_da_1, s <= 0.05),
        (V_da_2, True))  # delta > 0.05

    symb_model_params = ['w_1', 'w_2', 'w_3', 'f_c', 'alpha', 'B', 'n', 'd_s']

    symb_expressions = [('tau_b', ('w',)),
                        ('d_tau_b', ('w',)),
                        ('V_da', ('s',))]
    pass

@tr.provides(IMaterialModel)
class CrackBridgeAdv(bu.InteractiveModel, bu.InjectSymbExpr):

    name = 'Crack Bridge Adv'
    node_name = 'crack bridge model'

    symb_class = CrackBridgeModelAdvExpr

    pullout = tr.Instance(PullOutFib, ())
    dowelaction = tr.Instance(DowelAction, ())

    w_1 = Float(1)
    w_2 = Float(2)
    w_3 = Float(4)
    f_c = Float(37.9)  ## compressive strength of Concrete in MPa
    alpha = Float(0.4)
    B = Float(75)  ##mm (width of the beam)
    n = Float(4)  ##number of bars
    d_s = Float(16)  ##dia of steel mm

    ipw_view = View(
        Item('w_1', latex=r'w_1'),
        Item('w_2', latex=r'w_2'),
        Item('w_3', latex=r'w_3'),
        Item('f_c', latex=r'f_c'),
        Item('alpha', latex=r'\alpha'),
        Item('B', latex=r'B'),
        Item('n', latex=r'n'),
        Item('d_s', latex=r'd_s')
    )

    def get_sig_w_f(self, w):
        '''Calculating bond stresses '''
        tau_b = self.symb.get_tau_b(w)
        return tau_b

    def get_df(self, s):
        '''Calculating dowel action force '''
        V_df = self.dowelaction.get_sig_s_f(s)
        return V_df

    def get_F_a(self, u_a):
        F_w = self.get_sig_w_f(u_a[...,0])
        F_s = self.get_df(u_a[...,1])#np.zeros_like(F_w)
        return np.array([F_w,F_s], dtype=np.float_).T

    def subplots(self,fig):
        return fig.subplots(1,2)

    def update_plot(self,axes):
        '''Plotting function '''
        ax_w, ax_s = axes
        w_ = np.linspace(0, 1, 100)
        s_ = np.linspace(0, 1, 100)
        tau_b_ = self.get_sig_w_f(w_)
        V_df_ = self.get_df(s_)
        ax_w.plot(w_, tau_b_)
        ax_s.plot(s_, V_df_)
        ax_w.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        ax_w.set_ylabel(r'$\tau_b\;\;\mathrm{[MPa]}$')
        ax_s.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax_s.set_ylabel(r'$V_{df}\;\;\mathrm{[N]}$')