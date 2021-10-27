import sympy as sp
import numpy as np
import bmcs_utils.api as bu
from bmcs_shear.shear_crack.crack_tip_shear_stress import SZCrackTipShearStress
from bmcs_shear.shear_crack.crack_tip_shear_stress_global import SZCrackTipShearStressGlobal
from bmcs_shear.shear_crack.crack_tip_shear_stress_local import SZCrackTipShearStressLocal
import traits.api as tr

tau_fps, sigma_x, sigma_y = sp.symbols(r'\tau_\mathrm{fps}, sigma_x, sigma_z')
sigma_1, sigma_2 = sp.symbols(r'sigma_1, sigma_2')
f_ct, f_cm = sp.symbols(r'f_ct, f_cm', nonnegative=True)

sigma_xy = sp.Matrix([[sigma_x, tau_fps],
                      [tau_fps, sigma_y]])
sigma_x0 = sigma_xy.subs(sigma_y, 0)

P_xy, D_xy = sigma_xy.diagonalize()

sigma_1_xy = D_xy[0,0]
sigma_2_xy = D_xy[1,1]

Kupfer_ct = sp.Eq(sigma_2 / f_ct - sp.Rational(8,10) * sigma_1 / f_cm, 1)

sigma_2_ct_solved = sp.solve(Kupfer_ct, sigma_2)[0]

sig_2_ct_eq = sp.Eq(sigma_2_ct_solved, sigma_2_xy)

sig_2_ct_eq_xy = sig_2_ct_eq.subs(sigma_1, sigma_1_xy)

tau_fps_ct_solved = sp.solve(sig_2_ct_eq_xy, tau_fps)[0]

P_x0, D_x0 = P_xy.subs(sigma_y, 0), D_xy.subs(sigma_y, 0)

subs_sigma_x = sp.solve({D_xy[0,0] - f_ct}, {sigma_x})[0]

#subs_sigma_z = sp.solve({D_xz[1, 1] - f_ct}, {sigma_z})[0]
#P_xf = P_xz.subs(subs_sigma_z)

P_xf = P_xy.subs(subs_sigma_x)

sigma_xf = sigma_xy.subs(subs_sigma_x)

psi_f = sp.atan(sp.simplify(-P_xf[0, 0] / P_xf[1, 0]))
psi_0 = sp.atan(sp.simplify(-P_x0[0, 0] / P_x0[1, 0]))
psi_z = sp.atan(sp.simplify(-P_xy[0, 0] / P_xy[1, 0]))

psi_tau = sp.atan( sp.simplify(-P_xy[0,0] / P_xy[1,0])).subs(tau_fps, tau_fps_ct_solved)

#get_psi_f = sp.lambdify((tau_fps, sigma_x, f_ct), psi_f)

get_psi_global = sp.lambdify((tau_fps, f_ct, sigma_y), psi_f)
get_psi_0 = sp.lambdify((tau_fps, sigma_x), psi_0)
get_psi_z = sp.lambdify((tau_fps, sigma_x, sigma_y), psi_z)
get_psi_tau = sp.lambdify((sigma_x, sigma_y, f_cm, f_ct), psi_tau)



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
        ct_stress = self.sz_ctss #_
        tau_x_tip_1 = ct_stress.tau_x_tip_1k
        #print('tau_x_tip_1', tau_x_tip_1)
        f_t = self.sz_cp.sz_bd.matrix_.f_t
        f_cm = self.sz_cp.sz_bd.matrix_.f_c
        sig_x_tip_0 = ct_stress.sig_x_tip_0
        sig_z_tip_1 = ct_stress.sig_z_tip_1
        #psi_0 = get_psi_global(tau_x_tip_1, f_t, sig_z_tip_1)#sig_x_tip_0 #Global Condition
        psi_0 = get_psi_tau(sig_x_tip_0, sig_z_tip_1, f_cm, f_t) #Local Condition
        #psi_0 = get_psi_0(tau_x_tip_1, sig_x_tip_0)
        #print('psi_0', psi_0 * 180/np.pi)
        #print('sig_x_tip_0', sig_x_tip_0, psi_0)
        return psi_0

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

