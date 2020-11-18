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

get_psi_f = sp.lambdify((tau_fps, sigma_x, f_ct), psi_f)
get_psi_0 = sp.lambdify((tau_fps, sigma_x), psi_0)


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

    _ALL = tr.DelegatesTo('crack_tip_shear_stress')
    _GEO = tr.DelegatesTo('crack_tip_shear_stress')
    _MAT = tr.DelegatesTo('crack_tip_shear_stress')

    def get_psi(self):
        # TODO: check why the normal stress at tip is not
        #       exactly equal to f_t
        ct_tau = self.crack_tip_shear_stress
        tau_x_tip_1 = ct_tau.tau_x_tip_1k
        f_t = self.sz_cp.sz_bd.cmm.f_t
        sig_tip_1 = f_t
        psi_0 = get_psi_0(tau_x_tip_1, sig_tip_1)
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
