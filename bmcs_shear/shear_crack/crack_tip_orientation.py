import sympy as sp
import numpy as np
import bmcs_utils.api as bu
from bmcs_shear.shear_crack.crack_tip_shear_stress import SZCrackTipShearStress
from bmcs_shear.shear_crack.stress_profile import SZStressProfile
import traits.api as tr

tau_fps, sigma_x, sigma_z = sp.symbols(r'\tau_\mathrm{fps}, sigma_x, sigma_z')
f_ct = sp.Symbol('f_{\mathrm{ct}}', nonnegative=True)

sigma_xz = sp.Matrix([[sigma_x, tau_fps],
                      [tau_fps, sigma_z]])
sigma_x0 = sigma_xz.subs(sigma_z, 0)

P_xz, D_xz = sigma_xz.diagonalize()
P_x0, D_x0 = P_xz.subs(sigma_z, 0), D_xz.subs(sigma_z, 0)

subs_sigma_z = sp.solve({D_xz[1, 1] - f_ct}, {sigma_z})[0]
P_xf = P_xz.subs(subs_sigma_z)

psi_f = sp.atan(sp.simplify(-P_xf[0, 0] / P_xf[1, 0]))
psi_0 = sp.atan(sp.simplify(-P_x0[0, 0] / P_x0[1, 0]))
psi_z = sp.atan(sp.simplify(-P_xz[0, 0] / P_xz[1, 0]))

get_psi_f = sp.lambdify((tau_fps, sigma_x, f_ct), psi_f)
get_psi_0 = sp.lambdify((tau_fps, sigma_x), psi_0)
get_psi_z = sp.lambdify((tau_fps, sigma_x, sigma_z), psi_z)

class SZCrackTipOrientation(bu.InteractiveModel):
    """Given the global and local stress state around the crack
    tip determine the orientation of the crack orientation $\psi$
    for the next iteration. Possible inputs that can be included
    are the stress components defined in the vicinity of the crack.
    Shear stress $\tau_{\mathrm{xz}}$, horizontal stress $\sigma_x$.
    """
    name = "Orientation"

    crack_tip_shear_stress = tr.Instance(SZCrackTipShearStress, ())
    sz_stress_profile = tr.DelegatesTo('crack_tip_shear_stress')
    sz_cp = tr.DelegatesTo('crack_tip_shear_stress')
    sz_bd = tr.DelegatesTo('crack_tip_shear_stress')

    tree = ['crack_tip_shear_stress']

    def get_psi(self):
        ct_stress = self.crack_tip_shear_stress
        tau_x_tip_1 = ct_stress.tau_x_tip_1k
        f_t = self.sz_cp.sz_bd.matrix_.f_t
        sig_x_tip_0 = ct_stress.sig_x_tip_0
        sig_z1 = ct_stress.sig_z1
        psi_0 = get_psi_z(tau_x_tip_1, sig_x_tip_0, sig_z1)#sig_x_tip_0
        #psi_0 = get_psi_0(tau_x_tip_1, sig_x_tip_0)
        #print(psi_0)
        #print('sig_x_tip_0', sig_x_tip_0, psi_0)
        return psi_0

    def plot_crack_extension(self, ax):
        ct_tau = self.crack_tip_shear_stress
        x_tip_an = ct_tau.sz_cp.sz_ctr.x_tip_an[:, 0]
        L_fps = ct_tau.sz_cp.sz_ctr.L_fps
        psi = self.get_psi()
        s_psi, c_psi = np.sin(psi), np.cos(psi)
        x_fps_an = x_tip_an + np.array([-s_psi, c_psi]) * L_fps
        v_fps_an = np.array([x_tip_an, x_fps_an])
        ax.plot(*v_fps_an.T, '-o', color='magenta', lw=3)

    def plot(self, ax):
        sz_ctr = self.crack_tip_shear_stress.sz_cp.sz_ctr
        sz_ctr.plot_crack_tip_rotation(ax)
        self.plot_crack_extension(ax)
        ax.axis('equal')

    def update_plot(self, ax):
        self.plot(ax)

    ipw_view = bu.View()

class CrackStateAnimator(SZCrackTipOrientation):

    psi_slider = bu.Float(0, MAT=True)
    x_rot_1k_slider = bu.Float(0, MAT=True)
    w_slider = bu.Float(0, MAT = True)

    @tr.on_trait_change('w_slider')
    def reset_w(self):
        self.sz_cp.sz_ctr.w = self.w_slider

    @tr.on_trait_change('psi_slider')
    def reset_psi(self):
        self.sz_cp.sz_ctr.psi = self.psi_slider

    @tr.on_trait_change('x_rot_1k_slider')
    def reset_x_rot_1k(self):
        self.sz_cp.sz_ctr.x_rot_1k = self.x_rot_1k_slider

    ipw_view = bu.View(
        bu.Item('psi_slider'),
        bu.Item('x_rot_1k_slider'),
        bu.Item('w_slider'),
    )

    def subplots(self, fig):
        return self.sz_stress_profile.subplots(fig)

    def update_plot(self, ax):
        self.sz_stress_profile.update_plot(ax)

