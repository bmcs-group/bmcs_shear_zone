

import traits.api as tr
import numpy as np
import sympy as sp
from bmcs_utils.api import Model, View, Item, Float, SymbExpr, InjectSymbExpr
from bmcs_shear.shear_crack.stress_profile import \
    SZStressProfile
from bmcs_shear.shear_crack.crack_tip_shear_stress import SZCrackTipShearStress

tau_fps, sigma_x, sigma_y = sp.symbols(r'tau, sigma_x, sigma_y')
sigma_1, sigma_2 = sp.symbols(r'sigma_1, sigma_2')
f_ct, f_cm = sp.symbols(r'f_ct, f_cm', nonnegative=True)

sigma_xy = sp.Matrix([[sigma_x, tau_fps],
                     [tau_fps, sigma_y]])
sigma_12 = sp.Matrix([[sigma_1, 0],
                      [0, sigma_2]])

P_xy, D_xy = sigma_xy.diagonalize()

sigma_1_xy = D_xy[0,0]
sigma_2_xy = D_xy[1,1]

Kupfer_ct = sp.Eq(sigma_2 / f_ct - sp.Rational(8,10) * sigma_1 / f_cm, 1)

sigma_2_ct_solved = sp.solve(Kupfer_ct, sigma_2)[0]

sig_2_ct_eq = sp.Eq(sigma_2_ct_solved, sigma_2_xy)

sig_2_ct_eq_xy = sig_2_ct_eq.subs(sigma_1, sigma_1_xy)

tau_fps_ct_solved = sp.solve(sig_2_ct_eq_xy, tau_fps)[0]

get_tau_fps = sp.lambdify((sigma_x, sigma_y, f_cm, f_ct), tau_fps_ct_solved, 'numpy')

psi = sp.atan( sp.simplify(-P_xy[0,0] / P_xy[1,0])).subs(tau_fps, tau_fps_ct_solved)

get_psi = sp.lambdify((sigma_x, sigma_y, f_cm, f_ct), psi, 'numpy')

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
#        print('f_ct', f_ct)
        f_cm = self.f_c
#        print('f_cm', f_cm)
        sigma_x = min(self.sig_x_tip_0, f_ct)
        sigma_y = min(self.sig_z_tip_1, f_ct)
        tau_x_tip_1k = get_tau_fps(sigma_x, sigma_y, f_cm, f_ct)
        return tau_x_tip_1k

    sig_tip_ab_k = tr.Property
    def _get_sig_tip_k(self):
        return np.array()

    psi_k = tr.Property

    def _get_psi_k(self):
        # calculate the biaxial stress
        f_ct = self.f_t
        f_cm = self.f_c
        sigma_x = min(self.sig_x_tip_0, f_ct)
        sigma_y = min(self.sig_z_tip_1, f_ct)
        psi_k = get_psi(sigma_x, sigma_y, f_cm, f_ct)
        return psi_k

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

       # sig_x_fix = 3
       # sig_y_var = np.linspace(-3, 3, 100)
       #
       # tau_z_fps_sig_x_fixed = get_tau_fps(sig_x_fix, sig_y_var, f_t, f_c)
       #
       # ax1.plot(sig_y_var, tau_z_fps_sig_x_fixed, color='blue');
       # ax1.set_xlabel(r'$\sigma_{\mathrm{y}}$');
       # ax1.set_ylabel(r'$\tau_{\mathrm{fpz}}$');
       # ax1.set_title(r'$\sigma_{\mathrm{y}} = constant$, and changing $\sigma_{\mathrm{x}}$')
       # ax1.legend()

       # sig_x_num = 100
       # sig_x_var = np.linspace(0, 3, sig_x_num)
       # sig_y_num = 100
       # sig_y_var = np.linspace(0, 3, sig_y_num)
       # tau_fps_val = np.zeros([sig_x_num, sig_y_num])
       # for j in range(len(sig_y_var)):
       #     # print('sigma_z =', sigma_z[j])
       #     for i in range(len(sig_x_var)):
       #         # print('tau_fpz =', tau_fpz[i])
       #         tau_fps = get_tau_fps(sig_x_var[i], sig_y_var[j], f_t, f_c)
       #         tau_fps_val[j, i] = tau_fps
       #     ax2.plot(sig_y_var, tau_fps_val[j,:])#color='blue', label = r'$\sigma_{\mathrm{x}}[i]}$')
       # ax2.set_xlabel(r'$\sigma_{\mathrm{y}}$')
       # ax2.set_ylabel(r'$\tau_{\mathrm{fpz}}$')
       # #ax2.set_title(r'$\sigma_{\mathrm{x}} = constant$, and changing $\sigma_{\mathrm{y}}$')
       # #ax2.legend()
       # # ax2.fill_betweenx(z_arr, tau_z_arr, 0, color='blue', alpha=0.1)
       # #  @todo


       sigma_x_num = 100
       sigma_x_range = np.linspace(-33.3, 3.099999, sigma_x_num)
       sigma_y_num = 4
       sigma_y_range = np.linspace(0, 2.325, sigma_y_num)
       theta_ij = np.zeros([sigma_y_num, sigma_x_num])
       labels = ['$\sigma_z$ = 0', '$\sigma_z$ = 0.25$f_t$', '$\sigma_z$ = 0.5$f_t$', '$\sigma_z$ = 0.75$f_t$']
       for j in range(len(sigma_y_range)):
           for i in range(len(sigma_x_range)):
               theta = get_theta(sigma_x_range[i], sigma_y_range[j], 33.3, 3.1)
               theta_ij[j, i] = theta
           ax1.plot(theta_ij[j, :] * 180 / np.pi, sigma_x_range, 'o-', lw=2, label=labels[j])
           ax1.set_xlabel(r'$\theta$');
           ax1.set_ylabel(r'$\sigma_x$');
           ax1.legend()

       sigma_x_num = 100
       sigma_x_range = np.linspace(-33.3, 3.099999, sigma_x_num)
       sigma_y_num = 4
       sigma_y_range = np.linspace(0, 2.325, sigma_y_num)
       theta_ij = np.zeros([sigma_y_num, sigma_x_num])
       tau_ij = np.zeros([sigma_y_num, sigma_x_num])
       labels = ['$\sigma_z$ = 0', '$\sigma_z$ = 0.25$f_t$', '$\sigma_z$ = 0.5$f_t$', '$\sigma_z$ = 0.75$f_t$']
       for j in range(len(sigma_y_range)):
           for i in range(len(sigma_x_range)):
               theta = get_theta(sigma_x_range[i], sigma_y_range[j], 33.3, 3.1)
               tau = get_tau_fps(sigma_x_range[i], sigma_y_range[j], 33.3, 3.1)
               theta_ij[j, i] = theta
               tau_ij[j, i] = tau
           ax2.plot(theta_ij[j, :] * 180 / np.pi, tau_ij[j, :], 'o-', lw=2, label=labels[j])
           ax2.set_xlabel(r'$\theta$');
           ax2.set_ylabel(r'$\tau_{fps}$');
           ax2.legend()
