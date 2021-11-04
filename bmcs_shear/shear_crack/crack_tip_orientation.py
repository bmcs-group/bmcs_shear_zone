import sympy as sp
import numpy as np
import bmcs_utils.api as bu
from bmcs_shear.shear_crack.crack_tip_shear_stress_local import SZCrackTipShearStressLocal
import traits.api as tr


class SZCrackTipOrientation(bu.InteractiveModel):
    """Given the global and local stress state around the crack
    tip determine the orientation of the crack orientation $\psi$
    for the next iteration. Possible inputs that can be included
    are the stress components defined in the vicinity of the crack.
    Shear stress $\tau_{\mathrm{xz}}$, horizontal stress $\sigma_x$.
    """
    name = "Orientation"

    # sz_ctss = bu.EitherType(options=[
    #      ('global', SZCrackTipShearStressGlobal),
    #      ('local', SZCrackTipShearStressLocal)
    # ])

    sz_ctss = bu.Instance(SZCrackTipShearStressLocal, ())
    sz_sp = tr.DelegatesTo('sz_ctss')
    sz_cp = tr.DelegatesTo('sz_ctss')
    sz_bd = tr.DelegatesTo('sz_ctss')

    tree = ['sz_ctss']

    def get_psi(self):
        sz_ctss = self.sz_ctss #_
        return sz_ctss.psi_k

    def plot_crack_extension(self, ax):
        ct_tau = self.sz_ctss #_
        x_tip_an = ct_tau.sz_cp.sz_ctr.x_tip_an[:, 0]
        L_fps = ct_tau.sz_cp.sz_ctr.L_fps
        psi = self.get_psi()
        s_psi, c_psi = np.sin(psi), np.cos(psi)
        x_fps_an = x_tip_an + np.array([-s_psi, c_psi]) * L_fps
        v_fps_an = np.array([x_tip_an, x_fps_an])
        ax.plot(*v_fps_an.T, '-o', color='magenta', lw=3)

    def plot(self, ax):
        sz_ctr = self.sz_ctss.sz_cp.sz_ctr #_
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
        return self.sz_sp.subplots(fig)

    def update_plot(self, ax):
        self.sz_sp.update_plot(ax)

