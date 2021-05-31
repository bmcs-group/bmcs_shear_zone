
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
    a, b = sp.symbols(r'a, b', nonnegative = True)
    xi = sp.Symbol(r'\xi', nonnegative = True)
    sigma_z = sp.Symbol(r'\sigma_z', nonnegative = True)
    # mu, chi = sp.symbols(r'\mu, \chi')

    G_f = 0.028 * f_c ** 0.18 * d_a ** 0.32

    xi = sigma_z / f_c

    G_f_baz = (1 + (a/(1+b/xi)) - (1 + a + b)/(1 + b) * xi**8)

    L_c = E_c * G_f / f_t ** 2

    w_cr = f_t / E_c * L_c

    w_x = 5.14 * (G_f / f_t)

    # eps_cp = w_cr / L_c
    #
    # eps_p = w / L_c
    #
    # f_co = 2 * f_t

    #w_tc = 5.14 * G_f / f_t

    # f_ce = f_c * (1 / (0.8))

    r = s / w

    tau_0 = 0.25 * f_c

    a_3 = 2.45 / tau_0

    a_4 = 2.44 * (1 - (4 / tau_0))

    tau_s = sp.Piecewise(
        (0, w <= 0),
        (tau_0 * (1 - sp.sqrt((2 * w) / d_a)) * r * (a_3 + (a_4 * np.abs(r) ** 3)) / (1 + (a_4 * r ** 4)), w > 0)

    )

    tau_s_wal = sp.Piecewise(
        (0, w <= 0),
        ((- f_c / 30) + (1.8 * w**(-0.8) + (0.234 * w**(-0.707) - 0.2) * f_c ) * s, w > 0)
    )

    sigma_ag = sp.Piecewise(
        (0, w == w_cr),
        (-0.62 * sp.sqrt(w) * (r) / (1 + r ** 2) ** 0.25 * tau_s, w > w_cr)
    )
    #sp.Piecewise(
        #(0, w <= 0),
    #)

    f_w = f_t * sp.exp(-f_t * (w - w_cr) / G_f)

    sig_w = sp.Piecewise(
        (-f_c, E_c * w / L_c < -f_c),
        (E_c * w / L_c, w <= w_cr),
        (f_w, w > w_cr)
        #(f_w, sp.And(w > w_cr, w <= w_x)), #+ sigma_ag
        #(sigma_ag, w > w_x)  #f_w == 0
    )

    d_sig_w = sig_w.diff(w)



    # sig_w = sp.Piecewise(
    #     (- f_c, w < - w_cr),
    #     (2 * f_t + (-f_c - 2 * f_t) * sp.sqrt(1 - ((w_cr - sp.Abs(w)) / (w_cr)) ** 2), w < 0),
    #     (E_c * w / L_c, w < w_cr),
    #     (f_t * (1 + ((c_1 * w) / (w_tc)) ** 3) * sp.exp((-c_2 * w) / (w_tc)) - (w / w_tc) * (1 + c_1 ** 3) * sp.exp(
    #         -c_2), w > w_cr)
    # )

    #sigma = sig_w + sigma_ag

    symb_model_params = ['d_a', 'E_c', 'f_t', 'c_1', 'c_2', 'f_c', 'L'] #'mu', 'chi' , 'a', 'b'

    symb_expressions = [('sig_w', ('w', 's',)), #, 's'
                        ('tau_s', ('w', 's',)),
                        ('tau_s_wal', ('w', 's',))]
                        #('sigma_ag', ('w','s',))] #u_a

@tr.provides(IMaterialModel)
class ConcreteMaterialModelAdv(bu.InteractiveModel, bu.InjectSymbExpr):

    name = 'Concrete behavior'
    node_name = 'material model'

    symb_class = ConcreteMaterialModelAdvExpr

    d_a = Float(16)  ## dia of steel mm
    E_c = Float(28000)  ## tensile strength of Concrete in MPa
    f_t = Float(3)  ## Fracture Energy in N/m
    c_1 = Float(3)
    c_2 = Float(6.93)
    f_c = Float(33.3)
    L = Float(3850)
    L_fps = Float(50, MAT=True)
    a = Float(1.038)
    b = Float(0.245)

    ipw_view = View(
        Item('d_a', latex=r'd_a'),
        Item('E_c', latex=r'E_c'),
        Item('f_t', latex=r'f_t'),
        Item('c_1', latex=r'c_1'),
        Item('c_2', latex=r'c_2'),
        Item('f_c', latex=r'f_c'),
        Item('L_fps', latex=r'L_{fps}'),
        Item('a', latex = r'a'),
        Item('b', latex = r'b')
    )

    # G_f_baz = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    # @tr.cached_property
    # def _get_G_f_baz(self):
    #     xi = self.sz_crack_tip_orientation.
    #     return self.symb.G_f_baz

    L_c = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')

    @tr.cached_property
    def _get_L_c(self):
        return self.E_c * self.get_G_f() / self.f_t ** 2


    w_cr = tr.Property

    def _get_w_cr(self):
        return (self.f_t / self.E_c) * self._get_L_c()


    def get_G_f(self):
        '''''''Calculating fracture energy '''''''
        return (0.028 * self.f_c ** 0.18 * self.d_a ** 0.32)

    #def get_w_tc(self):
    #    '''''''Calculating point of softening curve resulting in 0 stress '''''''
    #    return ( 5.14 * self.G_f() / self.f_t)

    def get_sig_a(self, u_a): #w, s
        '''''''''Calculating stresses '''''''''
        sig_w = self.symb.get_sig_w(u_a[...,0], u_a[...,1])
        tau_s = self.symb.get_tau_s(u_a[...,0],u_a[...,1])
        #print(tau_s)
        #print(sig_w)
        return np.einsum('b...->...b', np.array([sig_w, tau_s], dtype=np.float_)) #, tau_s

    get_sig_w = tr.Property(depends_on='+MAT')

    def get_sig_w(self,w, s):
        return self.symb.get_sig_w(w, s)

    get_tau_s = tr.Property(depends_on='+MAT')

    def get_tau_s(self, w, s):
        return self.symb.get_tau_s(w, s)

    # get_sigma_ag = tr.Property(depends_on='+MAT')
    #
    # def get_sigma_ag(self, w, s):
    #     return self.symb.get_sigma_ag(w,s)

    w_min_factor = Float(1.2)
    w_max_factor = Float(3)
    def plot_sig_w(self, ax, vot=1.0):

        w_min_expr = -(self.f_c / self.E_c * self._get_L_c())
        w_max_expr = 3
        # w_max_expr = (sp.solve(self.f_w + self.f_w.diff(w) * w, w)
        #               [0]).subs(self.co_law_data)
        w_max = 0.5               #np.float_(w_max_expr) * self.w_max_factor
        w_min = np.float_(w_min_expr) * self.w_min_factor
        w_data = np.linspace(w_min, w_max, 100)
        s_max = 3
        s_data = np.linspace(-1.1 * s_max, 1.1 * s_max, 100)
        sig_w = self.get_sig_w(w_data, s_data)
        #print(sig_w)
        ax.plot(w_data, sig_w, lw=2, color='red')
        ax.fill_between(w_data, sig_w,
                        color='red', alpha=0.2)
        ax.set_xlabel(r'$w\;\;\mathrm{[mm]}$', fontsize=12)
        ax.set_ylabel(r'$\sigma\;\;\mathrm{[MPa]}$', fontsize=12)
        ax.set_title('crack opening law', fontsize=12)

    # def plot3d_Sig_Eps(self, ax3d):
    #     tau_x, tau_y = self.Sig_arr.T[:2, ...]
    #     tau = np.sqrt(tau_x ** 2 + tau_y ** 2)
    #     ax3d.plot3D(self.s_x_t, self.s_y_t, tau, color='orange', lw=3)

    def plot3d_tau_s(self, ax3d, vot=1.0):
        w_min = 0 #-1
        w_max = 3
        w_data = np.linspace(w_min, w_max, 100)
        s_max = 3
        s_data = np.linspace(0*s_max, 1.1*s_max, 100) #-1.1
        s_, w_ = np.meshgrid(s_data, w_data)
        tau_s = self.get_tau_s(w_, s_)
        ax3d.plot_surface(w_, s_, tau_s, cmap='viridis', edgecolor='none')
        ax3d.set_xlabel(r'$w\;\;\mathrm{[mm]}$', fontsize=12)
        ax3d.set_ylabel(r'$s\;\;\mathrm{[mm]}$', fontsize=12)
        ax3d.set_zlabel(r'$\tau\;\;\mathrm{[MPa]}$', fontsize=12)
        ax3d.set_title('aggregate interlock law', fontsize=12)

    def subplots(self, fig):
        ax_2d = fig.add_subplot(1, 2, 2)
        ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
        return ax_2d, ax_3d

    #w_max = bu.Float(1)

    def update_plot(self, axes):
        '''''''Plotting function '''''''
        ax_2d, ax_3d = axes
        self.plot_sig_w(ax_2d)
        self.plot3d_tau_s(ax_3d)










        # # self.plot_d_sig_w(ax1)
        # # self.plot_tau_s(ax2)
        # # self.plot_d_tau_s(ax2)
        # ax_w, ax_s = axes
        # w_ = np.linspace(-1, self.w_max, 100)
        # s_ = np.linspace(-1,1, 100)
        # s_2d, w_2d = np.meshgrid(s_,w_)
        # # #w_ag = np.linspace(0, 1, 100)
        # # #u_a_ = np.array([w_, s_])
        # # #sig_a_ = self.get_sig_a(w_, s_)
        # # sig_a_ = self.get_sig_a(w_, s_)
        # # #print(sig_a_)
        # # ax_w.plot(w_, sig_a_[0,:], color='red')
        # # ax_w.fill_between(w_, sig_a_[0, :], color='red', alpha=0.2)
        # # ax_w.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        # # ax_w.set_ylabel(r'$\sigma_w\;\;\mathrm{[MPa]}$')
        # # #ax_s.plot(s_, sig_a_[1,:], color='red')
        # # # ax_s = plt.axes(projection='3d')
        # sig_a_1 = self.get_sig_a(w_2d, s_2d)
        # sig_a_, tau_a_ = sig_a_1
        # ax_s.plot_surface(w_2d, s_2d, tau_a_, cmap='viridis', edgecolor='none')
        # # # # #ax_s.plot3D(w_, s_, sig_a_ag[1,:], 'red')
        # ax_s.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        # ax_s.set_ylabel(r'$s\;\;\mathrm{[mm]}$')
        # ax_s.set_zlabel(r'$\tau\;\;\mathrm{[MPa]}$')
