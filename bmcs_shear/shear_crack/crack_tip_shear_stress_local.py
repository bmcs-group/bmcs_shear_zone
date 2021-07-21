

import traits.api as tr
import numpy as np
import sympy as sp
from bmcs_utils.api import Model, View, Item, Float, SymbExpr, InjectSymbExpr
from bmcs_shear.shear_crack.stress_profile import \
    SZStressProfile
from bmcs_shear.shear_crack.crack_tip_shear_stress import SZCrackTipShearStress

tau_fps, sigma_x, sigma_y = sp.symbols(r'tau_fps, sigma_x, sigma_y')
sigma_1, sigma_2 = sp.symbols(r'sigma_1, sigma_2')
f_ct, f_cm = sp.symbols(r'f_ct, f_cm', nonnegative=True)

sigma_xy = sp.Matrix([[sigma_x, tau_fps],
                              [tau_fps, sigma_y]])
sigma_12 = sp.Matrix([[sigma_1, 0],
                              [0, sigma_2]])
P_xy, D_xy = sigma_xy.diagonalize()

Kupfer = sp.Eq(-sp.Rational(8, 10) * sigma_1 / f_cm + sigma_2 / f_ct, 1)

sigma_1_solved = sp.solve(Kupfer, sigma_1)[0]

sig_1_eq = sp.Eq(sigma_1_solved, D_xy[0, 0])

tau_fps_solved = sp.solve(sig_1_eq.subs(sigma_2, D_xy[1, 1]), tau_fps)[0]

get_tau_fps = sp.lambdify((sigma_x, sigma_y, f_ct, f_cm), tau_fps_solved, 'numpy')

class SZCrackTipShearStressLocal(SZCrackTipShearStress):
    name = 'crack tip stress state'

    f_c = tr.Property
    def _get_f_c(self):
        return self.sz_bd.matrix_.f_c

    f_t = tr.Property
    def _get_f_t(self):
        return self.sz_bd.matrix_.f_t

    tau_x_tip_1k = tr.Property

    def _get_tau_x_tip_1k(self):  # Shear stress distribution in uncracked region?
        # calculate the biaxial stress
        f_ct = self.f_t
        f_cm = self.f_c
        sigma_x = self.sig_x_tip_0
        sigma_y = self.sig_z_tip_1
        tau_x_tip_1k = get_tau_fps(sigma_x, sigma_y, f_ct, f_cm)#[0]
        #print(tau_x_tip_1k)
        return tau_x_tip_1k

    def subplots(self, fig):
        return fig.subplots(1, 2)

    def update_plot(self, axes):
       ax1, ax2 = axes
       # sig_x_var = np.linspace(0, 3, 100)
       # sig_y_fix = 3
       f_t = 3
       f_c = 33.3
       #
       #
       # tau_z_fps_sig_y_fixed = get_tau_fps(sig_x_var, sig_y_fix, f_t, f_c)
       #
       # ax1.plot(sig_x_var, tau_z_fps_sig_y_fixed, color='green');
       # ax1.set_xlabel(r'$\sigma_{\mathrm{x}}$');
       # ax1.set_ylabel(r'$\tau_{\mathrm{fpz}}$');
       # ax1.set_title(r'$\sigma_{\mathrm{y}} = constant$, and changing $\sigma_{\mathrm{x}}$')
       # ax1.legend()
       # ax1.fill_betweenx(z_fps_arr, tau_z_fps_arr, 0, color='green', alpha=0.1)

       sig_x_fix = 3
       sig_y_var = np.linspace(-3, 3, 100)

       tau_z_fps_sig_x_fixed = get_tau_fps(sig_x_fix, sig_y_var, f_t, f_c)

       ax1.plot(sig_y_var, tau_z_fps_sig_x_fixed, color='blue');
       ax1.set_xlabel(r'$\sigma_{\mathrm{y}}$');
       ax1.set_ylabel(r'$\tau_{\mathrm{fpz}}$');
       ax1.set_title(r'$\sigma_{\mathrm{y}} = constant$, and changing $\sigma_{\mathrm{x}}$')
       ax1.legend()

       sig_x_num = 100
       sig_x_var = np.linspace(0, 3, sig_x_num)
       sig_y_num = 100
       sig_y_var = np.linspace(0, 3, sig_y_num)
       tau_fps_val = np.zeros([sig_x_num, sig_y_num])
       for j in range(len(sig_y_var)):
           # print('sigma_z =', sigma_z[j])
           for i in range(len(sig_x_var)):
               # print('tau_fpz =', tau_fpz[i])
               tau_fps = get_tau_fps(sig_x_var[i], sig_y_var[j], f_t, f_c)
               tau_fps_val[j, i] = tau_fps
           ax2.plot(sig_y_var, tau_fps_val[j,:])#color='blue', label = r'$\sigma_{\mathrm{x}}[i]}$')
       ax2.set_xlabel(r'$\sigma_{\mathrm{y}}$')
       ax2.set_ylabel(r'$\tau_{\mathrm{fpz}}$')
       #ax2.set_title(r'$\sigma_{\mathrm{x}} = constant$, and changing $\sigma_{\mathrm{y}}$')
       ax2.legend()
       # ax2.fill_betweenx(z_arr, tau_z_arr, 0, color='blue', alpha=0.1)
       #  @todo
