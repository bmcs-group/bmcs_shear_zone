
from bmcs_shear.matmod.i_matmod import IMaterialModel
from bmcs_utils.api import InteractiveModel, View, Item, Float, SymbExpr, InjectSymbExpr
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

class ConcreteMaterialModelAdvExpr(bu.SymbExpr):
    # continue here
    w = sp.Symbol(r'w', real=True)
    s = sp.Symbol(r's', real = True)
    d_a, E_c = sp.symbols(r'd_a, E_c', nonnegative=True)
    f_t = sp.symbols(r'f_t', nonnegative=True)
    c_2 = sp.Symbol('c_2', nonnegative=True)
    c_1 = sp.Symbol('c_1', nonnegative=True)
    f_c = sp.Symbol('f_c', rnonnegative =True)
    L = sp.Symbol('L', nonnegative = True)

    G_f = 0.028 * f_c ** 0.18 * d_a ** 0.32

    L_c = E_c * G_f / f_t ** 2

    w_cr = f_t / E_c * L_c

    eps_cp = w_cr / L_c

    eps_p = w / L_c

    f_co = 2 * f_t

    w_tc = 5.14 * G_f / f_t

    f_ce = f_c * (1 / (0.8))

    sig_w = sp.Piecewise(
        (- f_c, w < - w_cr),
        (2 * f_t + (-f_c - 2 * f_t) * sp.sqrt(1 - ((w_cr - sp.Abs(w)) / (w_cr)) ** 2), w < 0),
        (E_c * w / L_c, w < w_cr),
        (f_t * (1 + ((c_1 * w) / (w_tc)) ** 3) * sp.exp((-c_2 * w) / (w_tc)) - (w / w_tc) * (1 + c_1 ** 3) * sp.exp(
            -c_2), w > w_cr)
    )

    r = s / w

    tau_0 = 0.25 * f_c

    a_3 = 2.45 / tau_0

    a_4 = 2.44 * (1 - (4 / tau_0))

    tau_s = tau_0 * (1 - sp.sqrt((2 * w) / d_a)) * r * (a_3 + (a_4 * sp.Abs(r) ** 3)) / (1 + (a_4 * r ** 4))
    #pass

    symb_model_params = ['d_a', 'E_c', 'f_t', 'c_1', 'c_2', 'f_c', 'L']

    symb_expressions = [('sig_w', ('w',)),
                        ('tau_s', ('w', 's',))]

@tr.provides(IMaterialModel)
class ConcreteMaterialModelAdv(bu.InteractiveModel, bu.InjectSymbExpr):

    name = 'Concrete behavior'
    node_name = 'material model'

    symb_class = ConcreteMaterialModelAdvExpr

    d_a = Float(22)  ## dia of steel mm
    E_c = Float(28000)  ## tensile strength of Concrete in MPa
    f_t = Float(3)  ## Fracture Energy in N/m
    c_1 = Float(3)
    c_2 = Float(6.93)
    f_c = Float(30)
    L = Float(3000)
    L_fps = Float(50, MAT=True)

    ipw_view = View(
        Item('d_a', latex=r'd_a'),
        Item('E_c', latex=r'E_c'),
        Item('f_t', latex=r'f_t'),
        Item('c_1', latex=r'c_1'),
        Item('c_2', latex=r'c_2'),
        Item('f_c', latex=r'f_c'),
        Item('L_fps', latex=r'L_{fps}')
    )

    def get_L_c(self):
        return self.E_c * self.get_G_f() / self.f_t ** 2

    w_cr = tr.Property

    def _get_w_cr(self):
        return self.f_t / self.E_c * self.get_L_c()

    def get_G_f(self):
        '''Calculating fracture energy '''
        return (0.028 * self.f_c ** 0.18 * self.d_a ** 0.32)

    def get_w_tc(self):
        '''Calculating point of softening curve resulting in 0 stress '''
        return ( 5.14 * self.get_G_f() / self.f_t)

    def get_sig_a(self, w, s):
        '''Calculating stresses '''
        #w, s = u_a[...,0], u_a[...,1]
        # w_ = np.sign(w + 1/34) * (w + 1e-9) #  1/2 * np.sign(w) * (w + 1e-9)
        # print(w_)
        sig_w = self.symb.get_sig_w(w)
        tau_s = self.symb.get_tau_s(w,s)
        #print(tau_s)
        return np.array([sig_w, tau_s])

    def subplots(self, fig):
        return fig.subplots(1, 2)

    w_max = bu.Float(1)

    def update_plot(self, axes):
        '''Plotting function '''
        ax_w, ax_s = axes
        w_ = np.linspace(1e-9, self.w_max, 100)
        s_ = np.linspace(-1,1, 100)
        s_2d, w_2d = np.meshgrid(s_,w_)
        #w_ag = np.linspace(0, 1, 100)
        #u_a_ = np.array([w_, s_])
        #sig_a_ = self.get_sig_a(w_, s_)
        sig_a_ = self.get_sig_a(w_, s_)
        #print(sig_a_)
        # ax_w.plot(w_, sig_a_[0,:], color='red')
        # ax_w.fill_between(w_, sig_a_[0, :], color='red', alpha=0.2)
        # ax_w.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        # ax_w.set_ylabel(r'$\sigma_w\;\;\mathrm{[MPa]}$')
        # #ax_s.plot(s_, sig_a_[1,:], color='red')
        ax_s = plt.axes(projection='3d')
        sig_a_1 = self.get_sig_a(w_2d, s_2d)
        sig_a_, tau_a_ = sig_a_1
        ax_s.plot_surface(w_2d, s_2d, tau_a_, cmap='viridis', edgecolor='none')
        # #ax_s.plot3D(w_, s_, sig_a_ag[1,:], 'red')
        ax_s.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        ax_s.set_ylabel(r'$s\;\;\mathrm{[mm]}$')
        ax_s.set_zlabel(r'$\tau\;\;\mathrm{[MPa]}$')
