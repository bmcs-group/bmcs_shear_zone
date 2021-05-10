from bmcs_shear.matmod.i_matmod import IMaterialModel
from bmcs_utils.api import View, Item, Float
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
import sympy as sp
from bmcs_shear.matmod.sz_crack_bridge.cb_advanced.sz_pull_out_fib import PullOut
from bmcs_shear.matmod.sz_crack_bridge.cb_advanced.sz_dowel_action import DowelAction


class CrackBridgeModelAdvExpr(bu.SymbExpr):
    # w_1, w_2 = sp.symbols(r'w_1, w_2', nonnegative=True)
    # w_3 = sp.symbols(r'w_3', nonnegative=True)
    w, f_c = sp.symbols(r'w, f_c', real=True)
    # alpha = sp.Symbol(r'\alpha', nonnegative=True)
    B = sp.symbols(r'B', nonnegative=True)
    n, d_s = sp.symbols(r'n, d_s', nonnegative=True)
    s = sp.Symbol('s', nonnegative = True)
    E_f = sp.Symbol(r'E_\mathrm{f}', nonnegative=True)
    E_m, A_m = sp.symbols(r'E_\mathrm{m}, A_\mathrm{m}', nonnegative=True)
    p, P = sp.symbols(r'p, P', nonnegative=True)
    tau = sp.symbols(r'\bar{\tau}', nonnegative=True)
    sig_y = sp.symbols('\sigma_y', positive=True)
    A_f = sp.Symbol(r'A_f', nonnegative = True)

    # tau_b_max = 2.5 * sp.sqrt(f_c)
    #
    # tau_bf = 0.4 * tau_b_max
    #
    # tau_b = sp.Piecewise(
    #     (tau_b_max * ( w/ w_1) ** alpha, w <= w_1),
    #     (tau_b_max, w <= w_2),
    #     (tau_b_max - ((tau_b_max - tau_bf) * (w - w_2) / (w_3 - w_2)), w <= w_3),
    #     (tau_bf, w > w_3)
    # )

    Pw_pull = sp.sqrt(2 * w * tau * E_f * A_f * p)

    P_max = A_f * sig_y

    w_argmax = sp.solve(P_max - Pw_pull, w)[0]

    Pw_pull_y = sp.Piecewise(
        (Pw_pull, w < w_argmax),
        (P_max, w >= w_argmax))

    # d_tau_b = tau_b.diff(w)
    #print(d_tau_b)

    b_n = B - n * d_s

    V_d_max = 1.64 * b_n * d_s * f_c ** (1 / 3)

    V_da_1 = V_d_max * (s / 0.05) * (2 - (s / 0.05))

    V_da_2 = V_d_max * ((2.55 - s) / 2.5)

    V_da = sp.Piecewise(
        (V_da_1, s <= 0.05),
        (V_da_2, s > 0.05))  # delta > 0.05 True


    symb_model_params = [ 'f_c', 'B', 'n', 'd_s', 'E_f', 'A_f', 'p', 'tau', 'sig_y'] #'w_1', 'w_2', 'w_3', 'alpha',
                         # 'E_f', 'A_f', 'p', 'sig_y'] #, 'E_f'

    symb_expressions = [('Pw_pull_y', ('w')),
                        ('w_argmax', ()),
                        ('V_da', ('s',))] #('tau_b', ('w',)), #('d_tau_b', ('w',)),

@tr.provides(IMaterialModel)
class CrackBridgeAdv(bu.InteractiveModel, bu.InjectSymbExpr):

    name = 'Crack Bridge Adv'
    node_name = 'crack bridge model'

    symb_class = CrackBridgeModelAdvExpr

    pullout = tr.Instance(PullOut, ())
    dowelaction = tr.Instance(DowelAction, ())

    # w_1 = Float(1)
    # w_2 = Float(2)
    # w_3 = Float(4)
    f_c = Float(33.3)  ## compressive strength of Concrete in MPa
    # alpha = Float(0.4)
    B = Float(250)  ##mm (width of the beam)
    n = Float(2)  ##number of bars
    d_s = Float(28)  ##dia of steel mm
    E_f = Float(210000)
    A_f = tr.Property()

    def _get_A_f(self):
        return self.n * (self.d_s / 2) ** 2 * np.pi

    p = tr.Property

    def _get_p(self):
        return (self.d_s) * np.pi

    tau = Float(16)
    sig_y = Float(713)

    ipw_view = View(
        # Item('w_1', latex=r'w_1'),
        # Item('w_2', latex=r'w_2'),
        # Item('w_3', latex=r'w_3'),
        Item('f_c', latex=r'f_c'),
        #Item('alpha', latex=r'\alpha'),
        Item('B', latex=r'B'),
        Item('n', latex=r'n'),
        Item('d_s', latex=r'd_s'),
        Item('E_f', latex=r'E_f'),
        Item('tau', latex=r'\tau'),
        Item('sig_y', latex=r'\sigma_y')
    )

    def get_sig_w_f(self, w):
        # distinguish the crack width from the end slip of the pullout
        # which delivers the crack bridging force
        return self.symb.get_Pw_pull_y(w / 2)

    # def get_sig_w_f(self, w):
    #     '''Calculating bond stresses '''
    #     #tau_b = self.symb.Pw_pull_y(w/2)
    #     #Pw_pull = self.symb.get_Pw_pull(w)
    #     return self.symb.get_Pw_pull_y(w/2)

    # V_d_max = tr.Property
    #
    # @tr.cached_property
    # def _get_V_d_max(self):
    #     return self.symb.V_d_max

    def get_V_df(self, s):
        '''Calculating dowel action force '''
        V_df = self.symb.get_V_da(s)
        #print(V_df)
        return V_df


    def get_F_a(self, u_a):
        F_w = self.get_sig_w_f(u_a[...,0])
        #print(F_w)
        F_s = self.get_V_df(u_a[...,1])#np.zeros_like(F_w)
        return np.array([F_w,F_s], dtype=np.float_).T

    def subplots(self,fig):
        return fig.subplots(1,2)

    def update_plot(self,axes):
        '''Plotting function '''
        ax_w, ax_s = axes
        w_argmax = self.symb.get_w_argmax()
        w_range = np.linspace(0, 3 * w_argmax)
        #w_ = np.linspace(0,1, 100)
        s_ = np.linspace(0, 1, 100)
        #print(s_)
        tau_b_ = self.get_sig_w_f(w_range) / 1000
        V_df_ = self.get_V_df(s_) / 1000
        ax_w.plot(w_range, tau_b_)
        ax_s.plot(s_, V_df_)
        ax_w.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        ax_w.set_ylabel(r'$F_s\;\;\mathrm{[kN]}$')
        ax_s.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax_s.set_ylabel(r'$V_{da}\;\;\mathrm{[kN]}$')