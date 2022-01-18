import traits.api as tr
import numpy as np
from bmcs_utils.api import Model, View, Item, mpl_align_xaxis
# from bmcs_shear.shear_crack.deformed_state import \
#     SZDeformedState
from scipy.interpolate import interp1d
from bmcs_utils.api import View, Bool, Item, Float, FloatRangeEditor
from .dic_crack import DICCrack

class DICStressProfile(Model):
    '''Stress profile calculation in an intermediate state
    '''
    name = "Profiles"

    dic_crack = tr.Instance(DICCrack, ())

    bd = tr.Property
    '''Beam design
    '''
    def _get_bd(self):
        return self.dic_crack.bd

    # bd = tr.DelegatesTo('dic_crack')
    #
    tree = ['dic_crack']

    show_stress = Bool(True)
    show_force = Bool(False)

    ipw_view = View(
        Item('show_stress'),
        Item('show_force')
    )

    X_La = tr.Property(depends_on='state_changed')
    '''Displacement of the segment midpoints '''
    @tr.cached_property
    def _get_X_La(self):
        return self.dic_crack.x_Na

    u_La = tr.Property(depends_on='state_changed')
    '''Displacement of the segment midpoints '''
    @tr.cached_property
    def _get_u_La(self):
        return self.dic_crack.u_Na

    u_Lb = tr.Property(depends_on='state_changed')
    '''Displacement of the segment midpoints '''
    @tr.cached_property
    def _get_u_Lb(self):
        return self.dic_crack.u_Nb

    # =========================================================================
    # Stress transformation and integration
    # =========================================================================

    S_Lb = tr.Property(depends_on='state_changed')
    '''Stress returned by the material model
    '''
    @tr.cached_property
    def _get_S_Lb(self):
        u_Lb = self.u_Lb
        cmm = self.bd.matrix_
        B = self.bd.B
        sig_La = cmm.get_sig_a(u_Lb)
        return sig_La * B

    S_La = tr.Property(depends_on='state_changed')
    '''Transposed stresses'''
    @tr.cached_property
    def _get_S_La(self):
        S_Lb = self.S_Lb
        S_La = np.einsum('Lb,Lab->La', S_Lb, self.dic_crack.T_Nab)
        return S_La

    # =========================================================================
    # Stress resultants
    # =========================================================================

    get_w_N = tr.Property(depends_on='state_changed')
    '''Get an interpolator function returning crack opening displacement 
    for a specified vertical coordinate of a ligament.
    '''
    @tr.cached_property
    def _get_get_w_N(self):
        return interp1d(self.X_La[:, 1], self.u_Lb[:, 0],
                        fill_value='extrapolate')

    get_s_N = tr.Property(depends_on='state_changed')
    '''Get an interpolator function returning slip displacement 
    component for a specified vertical coordinate of a ligament.
    '''
    @tr.cached_property
    def _get_get_s_N(self):
        # TODO - refine to handle the coordinate transformation correctly
        return interp1d(self.X_La[:, 1], self.u_La[:, 1],
                        fill_value='extrapolate')

    u_Na = tr.Property(depends_on='state_changed')
    '''Get an interpolator function returning slip displacement 
    component for a specified vertical coordinate of a ligament.
    '''
    @tr.cached_property
    def _get_u_Na(self):
        w_N = self.get_w_N(self.z_N)
        s_N = self.get_s_N(self.z_N)
        return np.array([w_N, s_N], dtype = np.float_).T

    z_N = tr.Property
    def _get_z_N(self):
        return self.bd.csl.z_j

    F_Na = tr.Property(depends_on='state_changed')
    '''Get the discrete force in the reinforcement z_N
    '''
    @tr.cached_property
    def _get_F_Na(self):
        u_Na = self.u_Na
        if len(u_Na) == 0:
            return np.zeros((0,2), dtype=np.float_)
        F_Na = np.array([r.get_F_a(u_a) for r, u_a in zip(self.bd.csl.items, u_Na)],
                 dtype=np.float_)
        return F_Na

    F_a = tr.Property(depends_on='state_changed')
    '''Integrated normal and shear force
    '''
    @tr.cached_property
    def _get_F_a(self):
        #F_La = self.F_La
        sum_F_a = np.trapz(self.S_La, self.X_La)
        F_Na = self.F_Na
        # sum_F_La = np.sum(F_La, axis=0)
        sum_F_Na = np.sum(F_Na, axis=0)
        return sum_F_a + sum_F_Na

    x_La = tr.Property(depends_on='state_changed')
    '''Midpoints within the crack segments.
    '''
    @tr.cached_property
    def _get_x_La(self):
        x_Ka = self.sz_ds.sz_cp.x_Ka
        K_Li = self.sz_ds.sz_cp.K_Li
        x_Lia = x_Ka[K_Li]
        x_La = np.sum(x_Lia, axis=1) / 2
        return x_La

    X_neutral_a = tr.Property(depends_on='state_changed')
    '''Vertical position of the neutral axis
    '''
    @tr.cached_property
    def _get_X_neutral_a(self):
        idx = np.argmax(self.u_La[:,0] < 0) - 1
        x_1, x_2 = self.X_La[(idx, idx + 1), 1]
        u_1, u_2 = self.u_La[(idx, idx + 1), 0]
        d_x = (x_2 - x_1) / (u_2 - u_1) * u_1
        y_neutral = x_1 + d_x
        x_neutral = self.X_La[idx + 1, 0]
        return np.array([x_neutral, y_neutral])

    M = tr.Property(depends_on='state_changed')
    '''Internal bending moment obtained by integrating the
    normal stresses with the lever arm rooted at the height of the neutral
    axis.
    '''
    @tr.cached_property
    def _get_M(self):
        # TODO - finish - by identifying the neutral axis position and the
        # horizontal distance between dowel action and crack tip.
        x_La = self.X_La
        #F_La = self.F_La
        x_rot_0k, x_rot_1k = self.X_neutral_a
        # M_L_0 = (x_La[:, 1] - x_rot_1k) * F_La[:, 0]
        # M_L_1 = (x_La[:, 0] - x_rot_0k) * F_La[:, 1]
        M_L_0 = np.trapz( (x_La[:, 1] - x_rot_1k) * self.S_La[:, 0], self.X_La[:, 0])
        M_L_1 = np.trapz( (x_La[:, 0] - x_rot_0k) * self.S_La[:, 1], self.X_La[:, 1])
        M = np.sum(M_L_0, axis=0) + np.sum(M_L_1, axis=0)
        M_z = np.einsum('i,i', (self.z_N - x_rot_1k), self.F_Na[:,0])
        # assuming that the horizontal position of the crack bridge
        # is almost equal to the initial position of the crack x_00
        # x_00 = np.ones_like(self.z_N) * self.sz_cp.x_00
        x_00 = self.dic_crack.C_cubic_spline(self.z_N)
        M_da = np.einsum('i,i', (x_00 - x_rot_0k), self.F_Na[:,1])
        return -(M + M_z + M_da)

    sig_x_tip_ak = tr.Property(depends_on='state_changed')
    '''Normal stress component in global $x$ direction in the fracture .
    process segment.
    '''
    def _get_sig_x_tip_ak(self):
        # TODO - use this to evaluate the crack propagation trend in the individual cracks along the specimen
        sz_cp = self.sz_cp
        x_tip_1 = sz_cp.sz_ctr.x_tip_ak[1]
        idx_tip = np.argmax(sz_cp.x_Ka[:, 1] >= x_tip_1)
        u_a = self.sz_ds.x1_Ka[idx_tip] - sz_cp.x_Ka[idx_tip]
        T_ab = sz_cp.T_tip_k_ab
        u_b = np.einsum('a,ab->b', u_a, T_ab)
        sig_b = self.bd.matrix_.get_sig_a(u_b)
        sig_a = np.einsum('b,ab->a', sig_b, T_ab)
        return sig_a

    sig_x_tip_0k = tr.Property(depends_on='state_changed')
    '''Check if this can be evaluated realistically.    
    Normal stress component in global $x$ direction in the fracture .
    process segment.
    '''
    @tr.cached_property
    def _get_sig_x_tip_0k(self):
        x_tip_1k = self.sz_cp.sz_ctr.x_tip_ak[1][0]
        return self.get_sig_x_tip_0k(x_tip_1k)

    get_sig_x_tip_0k = tr.Property(depends_on='state_changed')
    '''DEPRECATED - interpolation inprecise and not sufficient for the 
    crack orientation criterion.
    Get an interpolator function returning horizontal stress 
    component for a specified vertical coordinate of a ligament.
    '''
    @tr.cached_property
    def _get_get_sig_x_tip_0k(self):
        B = self.bd.B
        return interp1d(self.sz_cp.x_Lb[:, 1], self.S_La[:, 0] / B,
                        fill_value='extrapolate')

    def get_stress_resultant_and_position(self, irange):
        '''Helper function to determine the center of gravity of the force profile
        '''
        #F_L0 = self.F_La[:, 0]
        F_L0 = self.S_La[:, 0]
        range_F_L0 = F_L0[irange]
        range_x_L0 = self.X_La[irange, 1]
        int_F_L0 = np.trapz(range_F_L0, range_x_L0)
        range_normed_F_L0 = range_F_L0 / int_F_L0
        # x1 = self.x_La[:, 1][irange]
        return int_F_L0, np.sum(range_normed_F_L0 * range_x_L0)

    neg_F_y = tr.Property
    def _get_neg_F_y(self):
        #F_L0 = self.F_La[:, 0]
        F_L0 = self.S_La[:, 0]
        neg_range = F_L0 < 0
        return self.get_stress_resultant_and_position(neg_range)

    pos_F_y = tr.Property
    def _get_pos_F_y(self):
        #F_L0 = self.F_La[:, 0]
        F_L0 = self.S_La[:, 0]
        pos_range = F_L0 > 0
        return self.get_stress_resultant_and_position(pos_range)

    # =========================================================================
    # Plotting methods
    # =========================================================================

    def plot_u_Lc(self, ax, u_Lc, idx=0, color='black', label=r'$w$ [mm]'):
        x_La = self.X_La
        u_Lb_min = np.min(u_Lc[:, idx])
        u_Lb_max = np.max(u_Lc[:, idx])
    #    self.plot_hlines(ax, u_Lb_min, u_Lb_max)
        ax.plot(u_Lc[:, idx], x_La[:, 1], color=color, label=label)
        ax.fill_betweenx(x_La[:, 1], u_Lc[:, idx], 0, color=color, alpha=0.1)
        ax.set_xlabel(label)
        ax.legend(loc='lower left')

    # def plot_hlines(self, ax, h_min, h_max):
    #     y_tip = self.sz_cp.sz_ctr.x_tip_1n
    #     y_rot = self.sz_cp.sz_ctr.x_rot_1k
    #     z_fps = self.sz_cp.sz_ctr.x_fps_ak[1]
    #     ax.plot([h_min, h_max], [y_tip, y_tip],
    #             color='black', linestyle=':')
    #     ax.plot([h_min, h_max], [y_rot, y_rot],
    #             color='black', linestyle='--')
    #     ax.plot([h_min, h_max], [z_fps, z_fps],
    #             color='black', linestyle='-.')

    def plot_u_La(self, ax_w, ax_s, vot=1):
        '''Plot the displacement along the crack (w and s) in global coordinates
        '''
        self.plot_u_Lc(ax_w, self.u_La, 0, label=r'$u_x$ [mm]', color='blue')
        ax_w.set_xlabel(r'$u_x$ [mm]', fontsize=10)
        self.plot_u_Lc(ax_s, self.u_La, 1, label=r'$u_z$ [mm]', color='green')
        ax_s.set_xlabel(r'$u_z$ [mm]', fontsize=10)
        mpl_align_xaxis(ax_w, ax_s)

    def plot_u_Lb(self, ax_w, ax_s, vot=1):
        '''Plot the displacement (u_x, u_y) in local crack coordinates
        '''
        # plot the critical displacement
        # sz_ctr = self.sz_cp.sz_ctr
        # x_tip_1k = sz_ctr.x_tip_ak[1,0]
        # w = sz_ctr.w
        # ax_w.plot([0, w],[x_tip_1k, x_tip_1k], '-o', lw=2, color='red')
        self.plot_u_Lc(ax_w, self.u_Lb, 0, label=r'$w$ [mm]', color='blue')
        ax_w.set_xlabel(r'opening $w$ [mm]', fontsize=10)
        self.plot_u_Lc(ax_s, self.u_Lb, 1, label=r'$s$ [mm]', color='green')
        ax_s.set_xlabel(r'sliding $s$ [mm]', fontsize=10)
        mpl_align_xaxis(ax_w, ax_s)

    def plot_S_Lb(self, ax_sig, ax_tau, vot=1):
        '''Plot the stress components (sig, tau) in local crack coordinates
        '''
        # plot the critical displacement
        bd = self.bd
        cmm = bd.matrix_
        # sz_ctr = self.sz_cp.sz_ctr
        # x_tip_1k = sz_ctr.x_tip_ak[1,0]
        # S_t = cmm.f_t * bd.B
        # ax_sig.plot([0, S_t],[x_tip_1k, x_tip_1k], '-o', lw=2, color='red')
        self.plot_u_Lc(ax_sig, self.S_Lb, 0, label=r'$\sigma_\mathrm{N}$ [N/mm]', color='blue')
        ax_sig.set_xlabel(r'normal stress $\sigma_\mathrm{N}$ [N/mm]', fontsize=10)
        self.plot_u_Lc(ax_tau, self.S_Lb, 1, label=r'$\sigma_\mathrm{T}$ [N/mm]', color='green')
        ax_tau.set_xlabel(r'shear stress $\sigma_\mathrm{T}$ [N/mm]', fontsize=10)
        mpl_align_xaxis(ax_sig, ax_tau)

    def plot_S_La(self, ax_sig, ax_tau, vot=1):
        if self.show_stress:
            self.plot_u_Lc(ax_sig, self.S_La, 0, label=r'$f_x$ [N/mm]', color='blue')
            ax_sig.set_xlabel(r'horizontal stress $f_x$ [N/mm]', fontsize=10)
            self.plot_u_Lc(ax_tau, self.S_La, 1, label=r'$f_z$ [N/mm]', color='green')
            ax_tau.set_xlabel(r'vertical stress $f_z$ [N/mm]', fontsize=10)
            mpl_align_xaxis(ax_sig, ax_tau)
        if self.show_force:
            neg_F, neg_y = self.neg_F_y
            ax_sig.arrow(neg_F, neg_y, -neg_F, 0, color='red')
            pos_F, pos_y = self.pos_F_y
            ax_sig.arrow(pos_F, pos_y, -pos_F, 0, color='red')
        ax_sig.set_ylim(0, self.bd.H )

    def plot_N_a(self, ax_N):
        z_N = self.z_N
        F_N0 = self.F_Na[:,0]
        F_N = np.zeros_like(F_N0)
        ax_N.plot(np.array([F_N, F_N0]), np.array(([z_N, z_N])), color='green')

    def subplots(self, fig):
        ax_u_0, ax_w_0, ax_S_0, ax_F_0 = fig.subplots(1 ,4)
        ax_u_1 = ax_u_0.twiny()
        ax_w_1 = ax_w_0.twiny()
        ax_S_1 = ax_S_0.twiny()
        ax_F_1 = ax_F_0.twiny()
        return ax_u_0, ax_w_0, ax_S_0, ax_F_0, ax_u_1, ax_w_1, ax_S_1, ax_F_1

    def update_plot(self, axes):
        ax_u_0, ax_w_0, ax_S_0, ax_F_0, ax_u_1, ax_w_1, ax_S_1, ax_F_1 = axes
        self.plot_u_La(ax_u_0, ax_u_1)
        self.plot_u_Lb(ax_w_0, ax_w_1)
        self.plot_S_Lb(ax_S_0, ax_S_1)
        self.plot_S_La(ax_F_0, ax_F_1)
        self.plot_N_a(ax_S_0)
