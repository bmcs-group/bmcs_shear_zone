from bmcs_shear.matmod.i_matmod import IMaterialModel
from bmcs_utils.api import View, Item, Float, FloatRangeEditor
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
import sympy as sp
from bmcs_cross_section.api import ReinfLayer

class CrackBridgeModelAdvExpr(bu.SymbExpr):
    w, f_c = sp.symbols(r'w, f_c', real=True)
    B = sp.symbols(r'B', nonnegative=True)
    n, d_s = sp.symbols(r'n, d_s', nonnegative=True)
    s = sp.Symbol('s', nonnegative = True)
    E = sp.Symbol(r'E', nonnegative=True)
    p, P = sp.symbols(r'p, P', nonnegative=True)
    tau = sp.symbols(r'\bar{\tau}', nonnegative=True)
    sig_y = sp.symbols('\sigma_y', positive=True)
    A = sp.Symbol(r'A', nonnegative = True)

    Pw_pull = sp.sqrt(2 * w * tau * E * A * p)

    P_max = A * sig_y

    w_argmax = sp.solve(P_max - Pw_pull, w)[0]

    Pw_pull_y = sp.Piecewise(
        (Pw_pull, w < w_argmax),
        (P_max, w >= w_argmax))

    b_n = B - n * d_s

    V_d_max = 1.64 * b_n * d_s * f_c ** (1 / 3)

    V_da_1 = V_d_max * (s / 0.05) * (2 - (s / 0.05))

    V_da_2 = V_d_max * ((2.55 - s) / 2.5)

    V_da = sp.Piecewise(
        (V_da_1, s <= 0.05),
        (V_da_2, s > 0.05))  # delta > 0.05 True


    symb_model_params = [ 'f_c', 'B', 'n', 'd_s', 'E', 'A', 'p', 'tau', 'sig_y']

    symb_expressions = [('Pw_pull_y', ('w')),
                        ('w_argmax', ()),
                        ('V_da', ('s',))]

@tr.provides(IMaterialModel)
class CrackBridgeAdv(ReinfLayer, bu.InjectSymbExpr):

    name = 'crack bridge/dowel action'
    tree = []

    symb_class = CrackBridgeModelAdvExpr

    f_c = Float(33.3, MAT=True)  ## compressive strength of Concrete in MPa
    n = Float(1, MAT=True)  ##number of bars
    d_s = Float(28, MAT=True)  ##dia of steel mm
    E = Float(210000, MAT=True)
    tau = Float(16, MAT=True)
    sig_y = Float(713, MAT=True)

    dowel_factor = Float(1, MAT=True)

    A = tr.Property(Float, depends_on='state_changed')
    @tr.cached_property
    def _get_A(self):
        return self.n * (self.d_s / 2) ** 2 * np.pi

    B = tr.Property(Float, depends_on='state_changed')
    @tr.cached_property
    def _get_B(self):
        return self.cs_layout.cs_design.cross_section_shape_.B

    p = tr.Property(Float, depends_on='state_changed')
    '''Perimeter'''
    @tr.cached_property
    def _get_p(self):
        return self.n * (self.d_s) * np.pi

    ipw_view = View(
        Item('f_c', latex=r'f_c'),
        Item('n', latex=r'n'),
        Item('d_s', latex=r'd_s'),
        Item('E', latex=r'E'),
        Item('tau', latex=r'\tau'),
        Item('sig_y', latex=r'\sigma_y'),
        Item('dowel_factor', latex = r'\gamma_{dowel}', editor=FloatRangeEditor(low=0,high=1)),
        Item('B', latex=r'B', readonly=True),
        Item('A', latex=r'A', readonly=True),
        Item('p', latex=r'p', readonly=True),
    )

    def get_Pw_pull(self, w):
        '''Distinguish the crack width from the end slip of the pullout
            which delivers the crack bridging force
        '''
        return self.symb.get_Pw_pull_y(w / 2)

    def get_V_df(self, s):
        '''Calculating dowel action force '''
        V_df = self.symb.get_V_da(s) * self.dowel_factor
        #print(V_df)
        return V_df


    def get_F_a(self, u_a):
        F_w = self.get_Pw_pull(u_a[...,0])
        F_s = self.get_V_df(u_a[...,1])
        return np.array([F_w,F_s], dtype=np.float_).T

    def subplots(self,fig):
        return fig.subplots(1,2)

    def update_plot(self,axes):
        '''Plotting function '''
        ax_w, ax_s = axes
        w_argmax = self.symb.get_w_argmax()
        w_range = np.linspace(0, 3 * w_argmax)
        s_ = np.linspace(0, 1, 100)
        P_w = self.get_Pw_pull(w_range) / 1000
        V_df_ = self.get_V_df(s_) / 1000
        ax_w.plot(w_range, P_w)
        ax_s.plot(s_, V_df_)
        ax_w.set_xlabel(r'$\mathrm{w}\;\;\mathrm{[mm]}$', fontsize = 14)
        ax_w.set_ylabel(r'$\mathrm{F_s}\;\;\mathrm{[kN]}$', fontsize = 14)
        ax_s.set_xlabel(r'$s\;\;\mathrm{[mm]}$', fontsize = 14)
        ax_s.set_ylabel(r'$V_{da}\;\;\mathrm{[kN]}$', fontsize = 14)