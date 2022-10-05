import traits.api as tr
import numpy as np
import ibvpy.api as ib
from bmcs_utils.api import View, Item, mpl_align_xaxis
from scipy.interpolate import interp1d
from bmcs_utils.api import View, Bool, Item, Float, FloatRangeEditor
from .i_dic_crack import IDICCrack
import matplotlib.gridspec as gridspec
import sympy as sp
import bmcs_utils.api as bu


class DICStressProfile(bu.Model):
    """Stress profile calculation in an intermediate state
    """
    name = "Profiles"

    dic_crack = bu.Instance(IDICCrack)

    bd = tr.Property
    '''Beam design
    '''

    def _get_bd(self):
        return self.dic_crack.bd

    dic_grid = tr.Property()

    @tr.cached_property
    def _get_dic_grid(self):
        return self.dic_crack.dic_grid

    smeared_matmod = bu.Instance(ib.MATS2DMplDamageEEQ, ())

    depends_on = ['dic_crack']

    show_stress = Bool(True)
    show_force = Bool(False)

    ipw_view = View(
        Item('show_stress'),
        Item('show_force'),
        time_editor=bu.HistoryEditor(var='dic_crack.dic_grid.t')
    )

    x_1_La = tr.Property(depends_on='state_changed')
    '''Crack ligament along the whole cross section at the ultimate state '''

    @tr.cached_property
    def _get_x_1_La(self):
        x_1_Ka = self.dic_crack.x_1_Ka
        # add the top point
        x_top = x_1_Ka[-1, 0]
        return np.vstack([x_1_Ka, [[x_top, self.bd.H]]])

    x_t_unc_La = tr.Property(depends_on='state_changed')
    '''Crack ligament along the uncracked cross section at the intermediate state '''

    @tr.cached_property
    def _get_x_t_unc_La(self):
        if len(self.dic_crack.x_t_unc_Ka) == 0:
            return np.zeros_like(self.dic_crack.x_t_unc_Ka)
        else:
            x_t_unc_La = np.copy(self.dic_crack.x_t_unc_Ka)
            # add the top point
            x_top = x_t_unc_La[-1, 0]
            return np.vstack([x_t_unc_La, [[x_top, self.bd.H]]])

    x_t_crc_La = tr.Property(depends_on='state_changed')
    '''Displacement of the segment midpoints '''

    @tr.cached_property
    def _get_x_t_crc_La(self):
        # To correctly reflect the stress state, shift the
        # grid points by the offsets within the beam
        # X_min, Y_min, X_max, Y_max = self.dic_grid.X_frame
        x_t_crc_Ka = np.copy(self.dic_crack.x_t_crc_Ka)
        return x_t_crc_Ka

    eps_t_unc_Lab = tr.Property(depends_on='state_changed')
    '''Strain tensor along the compression zone'''

    @tr.cached_property
    def _get_eps_t_unc_Lab(self):
        eps_t_Kab = self.dic_crack.eps_t_Kab
        # Add the upper point by copying the top most measured strain tensor
        # This is a simplification. An extrapolation of strain should be used
        # instead or a local strain controlled IBVP along the ligament
        x_t_unc_La = self.dic_crack.x_t_unc_Ka
        if len(x_t_unc_La) == 0:
            # the crack
            return np.zeros_like(eps_t_Kab)
        else:
            y_1 = x_t_unc_La[-1, 1]
            y_0 = x_t_unc_La[0, 1]
            y_top = self.bd.H
            eps_0 = eps_t_Kab[0, 0, 0]
            eps_1 = eps_t_Kab[-1, 0, 0]
            eps_top = eps_1 + (eps_1 - eps_0) / (y_1 - y_0) * (y_top - y_1)
            eps_t_top_ab = np.zeros((2,2), dtype=np.float_)
            eps_t_top_ab[0, 0] = eps_top
            return np.vstack([eps_t_Kab, [eps_t_top_ab]])

    u_t_crc_Ka = tr.Property(depends_on='state_changed')
    '''Displacement of the segment midpoints '''

    @tr.cached_property
    def _get_u_t_crc_Ka(self):
        return self.dic_crack.u_t_crc_Ka

    u_t_crc_Kb = tr.Property(depends_on='state_changed')
    '''Displacement of the nodes'''

    @tr.cached_property
    def _get_u_t_crc_Kb(self):
        return self.dic_crack.u_t_crc_Kb

    # =========================================================================
    # Stress transformation and integration
    # =========================================================================

    sig_t_unc_Lab = tr.Property(depends_on='state_changed')
    '''Stress returned by the material model
    '''

    @tr.cached_property
    def _get_sig_t_unc_Lab(self):
        eps_t_unc_Lab = self.eps_t_unc_Lab
        cmm = self.bd.matrix_
        mdm = self.smeared_matmod
        mdm.trait_set(E=cmm.E_c, nu=0.2)
        n_K, _, _ = eps_t_unc_Lab.shape
        Eps = {
            name: np.zeros((n_K,) + shape)
            for name, shape in mdm.state_var_shapes.items()
        }
        sig_t_unc_Lab, _ = mdm.get_corr_pred(eps_t_unc_Lab, 1, **Eps)
        return sig_t_unc_Lab

    sig_t_crc_Lb = tr.Property(depends_on='state_changed')
    '''Stress returned by the material model
    '''

    @tr.cached_property
    def _get_sig_t_crc_Lb(self):
        u_t_crc_Kb = self.dic_crack.u_t_crc_Kb
        cmm = self.bd.matrix_
        sig_t_Kb = cmm.get_sig_a(u_t_crc_Kb)
        return sig_t_Kb
        # sig_t_top_b = sig_t_Kb[-1, :]
        # return np.vstack([sig_t_Kb, [sig_t_top_b]])

    sig_t_crc_La = tr.Property(depends_on='state_changed')
    '''Transposed stresses'''

    @tr.cached_property
    def _get_sig_t_crc_La(self):
        sig_t_crc_Lb = self.sig_t_crc_Lb
        T_t_crc_Kab = self.dic_crack.T_t_crc_Kab
        sig_t_crc_La = np.einsum('Lb,Lab->La', sig_t_crc_Lb, T_t_crc_Kab)
        return sig_t_crc_La

    # =========================================================================
    # Stress resultants
    # =========================================================================

    get_w_t_N = tr.Property(depends_on='state_changed')
    '''Get an interpolator function returning crack opening displacement 
    for a specified vertical coordinate of a ligament.
    '''

    @tr.cached_property
    def _get_get_w_t_N(self):
        return interp1d(self.x_t_crc_La[:, 1], self.u_t_crc_Kb[:, 0],
                        fill_value='extrapolate')

    get_s_t_N = tr.Property(depends_on='state_changed')
    '''Get an interpolator function returning slip displacement 
    component for a specified vertical coordinate of a ligament.
    '''

    @tr.cached_property
    def _get_get_s_t_N(self):
        # TODO - refine to handle the coordinate transformation correctly
        return interp1d(self.x_t_crc_La[:, 1], self.u_t_crc_Ka[:, 1],
                        fill_value='extrapolate')

    u_t_Na = tr.Property(depends_on='state_changed')
    '''Get an interpolator function returning slip displacement 
    component for a specified vertical coordinate of a ligament.
    '''

    @tr.cached_property
    def _get_u_t_Na(self):
        # handle the case that the crack did not cross the reinforcement yet
        z_tip = self.dic_crack.x_t_tip_a[1]
        self.z_N[z_tip < self.z_N] = z_tip
        w_t_N = self.get_w_t_N(self.z_N)
        s_t_N = self.get_s_t_N(self.z_N)
        return np.array([w_t_N, s_t_N], dtype=np.float_).T

    z_N = tr.Property

    def _get_z_N(self):
        return self.bd.csl.z_j

    F_t_Na = tr.Property(depends_on='state_changed')
    '''Get the discrete force in the reinforcement z_N
    '''

    @tr.cached_property
    def _get_F_t_Na(self):
        u_t_Na = self.u_t_Na
        if len(u_t_Na) == 0:
            return np.zeros((0, 2), dtype=np.float_)
        F_t_Na = np.array([r.get_F_a(u_a) for r, u_a in zip(self.bd.csl.items.values(), u_t_Na)],
                        dtype=np.float_)
        return F_t_Na

    F_t_a = tr.Property(depends_on='state_changed')
    '''Integrated normal and shear force
    '''

    @tr.cached_property
    def _get_F_t_a(self):
        sum_F_t_a = np.trapz(self.sig_t_crc_La, self.x_1_La[:, 0, np.newaxis], axis=0)
        F_t_Na = self.F_t_Na
        sum_F_t_Na = np.sum(F_t_Na, axis=0)
        return sum_F_t_a + sum_F_t_Na

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
        idx = np.argmax(self.u_t_crc_Ka[:, 0] < 0) - 1
        x_1, x_2 = self.x_1_La[(idx, idx + 1), 1]
        u_1, u_2 = self.u_t_crc_Ka[(idx, idx + 1), 0]
        d_x = -(x_2 - x_1) / (u_2 - u_1) * u_1
        y_neutral = x_1 + d_x
        x_neutral = self.x_1_La[idx + 1, 0]
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
        x_La = self.x_1_La
        # F_La = self.F_La
        x_rot_0k, x_rot_1k = self.X_neutral_a
        # M_L_0 = (x_La[:, 1] - x_rot_1k) * F_La[:, 0]
        # M_L_1 = (x_La[:, 0] - x_rot_0k) * F_La[:, 1]
        M_L_0 = np.trapz((x_La[:, 1] - x_rot_1k) * self.sig_t_crc_La[:, 0], self.x_1_La[:, 0])
        M_L_1 = np.trapz((x_La[:, 0] - x_rot_0k) * self.sig_t_crc_La[:, 1], self.x_1_La[:, 1])
        M = np.sum(M_L_0, axis=0) + np.sum(M_L_1, axis=0)
        M_z = np.einsum('i,i', (self.z_N - x_rot_1k), self.F_t_Na[:, 0])
        # assuming that the horizontal position of the crack bridge
        # is almost equal to the initial position of the crack x_00
        # x_00 = np.ones_like(self.z_N) * self.sz_cp.x_00
        x_00 = self.dic_crack.C_cubic_spline(self.z_N)
        M_da = np.einsum('i,i', (x_00 - x_rot_0k), self.F_t_Na[:, 1])
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
        return interp1d(self.sz_cp.x_Lb[:, 1], self.sig_t_crc_La[:, 0] / B,
                        fill_value='extrapolate')

    def get_stress_resultant_and_position(self, irange):
        '''Helper function to determine the center of gravity of the force profile
        '''
        # F_L0 = self.F_La[:, 0]
        F_L0 = self.sig_t_crc_La[:, 0]
        range_F_L0 = F_L0[irange]
        range_x_L0 = self.x_1_La[irange, 1]
        int_F_L0 = np.trapz(range_F_L0, range_x_L0)
        range_normed_F_L0 = range_F_L0 / int_F_L0
        # x1 = self.x_La[:, 1][irange]
        return int_F_L0, np.sum(range_normed_F_L0 * range_x_L0)

    get_SY = tr.Property

    @tr.cached_property
    def _get_get_SY(self):
        y, S1, S2, y1, y2 = sp.symbols('y, S1, S2, y1, y2')
        S = sp.integrate(S1 + (S2 - S1) / (y2 - y1) * (y - y1), (y, y1, y2))
        SY = sp.integrate((S1 + (S2 - S1) / (y2 - y1) * (y - y1)) * y, (y, y1, y2))
        Y = SY / S
        get_Y = sp.lambdify((S1, S2, y1, y2), Y)
        get_S = sp.lambdify((S1, S2, y1, y2), S)
        return get_S, get_Y

    neg_F_y = tr.Property

    def _get_neg_F_y(self):
        if len(self.x_t_unc_La) == 0:
            return 0, 0
        S_ = self.sig_t_unc_Lab[:, 0, 0]
        y_ = self.x_t_unc_La[:, 1] - self.x_t_unc_La[0, 1]
        get_S, get_Y = self.get_SY
        y_L = get_Y(S_[:-1], S_[1:], y_[:-1], y_[1:])
        S_L = get_S(S_[:-1], S_[1:], y_[:-1], y_[1:])
        sum_S = np.sum(S_L)
        sum_Sy = np.sum(S_L * y_L)
        if sum_S == 0:
            cg_y = 0
        else:
            cg_y = sum_Sy / sum_S
        return sum_S * self.bd.B, cg_y + self.x_t_unc_La[0, 1]

    pos_F_y = tr.Property

    def _get_pos_F_y(self):
        S_ = self.sig_t_crc_La[:, 0]
        y_ = self.x_t_crc_La[:, 1] - self.x_t_unc_La[-1, 1]
        get_S, get_Y = self.get_SY
        y_L = get_Y(S_[:-1], S_[1:], y_[:-1], y_[1:])
        S_L = get_S(S_[:-1], S_[1:], y_[:-1], y_[1:])
        sum_S = np.sum(S_L)
        sum_Sy = np.sum(S_L * y_L)
        if sum_S == 0:
            cg_y = 0
        else:
            cg_y = sum_Sy / sum_S
        return sum_S * self.bd.B, cg_y + self.x_t_unc_La[-1, 1]

    # =========================================================================
    # Plotting methods
    # =========================================================================

    def plot_u_Lc(self, ax, u_Lc, idx=0, color='black', label=r'$w$ [mm]'):
        x_t_crc_La = self.x_t_crc_La
        if len(u_Lc) > 0:
            u_t_crc_Kb_min = np.min(u_Lc[:, idx])
            u_t_crc_Kb_max = np.max(u_Lc[:, idx])
            #    self.plot_hlines(ax, u_t_crc_Kb_min, u_t_crc_Kb_max)
        ax.plot(u_Lc[:, idx], x_t_crc_La[:, 1], color=color, label=label)
        ax.fill_betweenx(x_t_crc_La[:, 1], u_Lc[:, idx], 0, color=color, alpha=0.1)
        ax.set_xlabel(label)

    def plot_u_t_crc_Ka(self, ax_w, vot=1):
        '''Plot the displacement along the crack (w and s) in global coordinates
        '''
        self.plot_u_Lc(ax_w, self.u_t_crc_Ka, 0, label=r'$u_x$ [mm]', color='blue')
        self.plot_u_Lc(ax_w, self.u_t_crc_Ka, 1, label=r'$u_z$ [mm]', color='green')
        ax_w.legend(loc='lower left')
        ax_w.set_xlabel(r'$u_x, u_y$ [mm]', fontsize=10)
        ax_w.set_ylim(0, self.bd.H)

    def plot_u_t_crc_Kb(self, ax_w):
        '''Plot the displacement (u_x, u_y) in local crack coordinates
        '''
        self.plot_u_Lc(ax_w, self.u_t_crc_Kb, 0, label=r'$w$ [mm]', color='blue')
        self.plot_u_Lc(ax_w, self.u_t_crc_Kb, 1, label=r'$s$ [mm]', color='green')
        ax_w.set_xlabel(r'sliding $w, s$ [mm]', fontsize=10)
        ax_w.legend(loc='lower left')
        ax_w.set_ylim(0, self.bd.H)

    def _plot_unc_sig_t(self, ax, sig_t_Lab, a=0, b=0, color='black', label=r'$\sigma$ [MPa]'):
        """Helper function for plotting the strain along the ligament
        at an intermediate state.
        """
        x_t_unc_La = self.x_t_unc_La
        ax.plot(sig_t_Lab[:, a, b], x_t_unc_La[:, 1], color=color, label=label)
        ax.fill_betweenx(x_t_unc_La[:, 1], sig_t_Lab[:, a, b], 0, color=color, alpha=0.1)
        ax.set_xlabel(label)
        ax.legend(loc='lower left')

    def plot_sig_t_unc_Lab(self, ax_sig):
        """Plot the displacement (u_x, u_y) in local crack coordinates
        at an intermediate state.
        """
        self._plot_unc_sig_t(ax_sig, self.sig_t_unc_Lab, 0, 0, color='blue')
        self._plot_unc_sig_t(ax_sig, self.sig_t_unc_Lab, 0, 1, color='green')
        ax_sig.set_xlabel(r'$\sigma$ [-]', fontsize=10)

    def plot_sig_t_crc_Lb(self, ax_sig):
        '''Plot the stress components (sig, tau) in local crack coordinates
        '''
        # plot the critical displacement
        bd = self.bd
        self.plot_u_Lc(ax_sig, self.sig_t_crc_Lb, 0, label=r'$\sigma_\mathrm{N}$ [N/mm]', color='blue')
        self.plot_u_Lc(ax_sig, self.sig_t_crc_Lb, 1, label=r'$\sigma_\mathrm{T}$ [N/mm]', color='green')
        ax_sig.set_xlabel(r'stress $\sigma_\mathrm{N,T}$ [N/mm]', fontsize=10)
        ax_sig.set_ylim(0, self.bd.H)

    def plot_sig_t_crc_La(self, ax_sig):
        self.plot_u_Lc(ax_sig, self.sig_t_crc_La, 0, label=r'$f_x$ [MPa]', color='blue')
        self.plot_u_Lc(ax_sig, self.sig_t_crc_La, 1, label=r'$f_z$ [MPa]', color='green')
        ax_sig.set_xlabel(r'stress $\sigma_{xx}, \sigma_{xy}$ [MPa]', fontsize=10)

    def plot_F_t_a(self, ax_F_a):
        # ax_F_a.plot(*self.x_1_La.T, color='black')
        ax_F_a.plot([0, 0], [0, self.bd.H], color='black', linewidth=0.4)
        # compression zone
        neg_F, neg_y = self.neg_F_y
        neg_F_kN = neg_F / 1000
        head_length = np.fabs(neg_F_kN * 0.2)
        head_width = 10
        if neg_F_kN != 0:
            ax_F_a.annotate('{0:.3g} kN'.format(neg_F_kN),
                            xy=(0, neg_y), xycoords='data',
                            xytext=(neg_F_kN, neg_y), textcoords='data',
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            color='blue',
                            # arrowprops=dict(arrowstyle="->",
                            #                 connectionstyle="arc3"),
                            )
        # tensile zone
        pos_F, pos_y = self.pos_F_y
        pos_F_kN = pos_F / 1000
        if pos_F_kN != 0:
            ax_F_a.annotate('{0:.3g} kN'.format(pos_F_kN),
                            xy=(0, pos_y), xycoords='data',
                            xytext=(pos_F_kN, pos_y), textcoords='data',
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            # arrowprops=dict(arrowstyle="->",
                            #                 connectionstyle="arc3"),
                            )
        # reinforcement
        y_N0 = self.z_N
        F_N0_kN = self.F_t_Na[:, 0] / 1000
        for F0_kN, y in zip(F_N0_kN, y_N0):
            ax_F_a.annotate("{0:.3g} kN".format(F0_kN),
                            xy=(0, float(y)), xycoords='data',
                            xytext=(float(F0_kN), float(y)), textcoords='data',
                            horizontalalignment='center',
                            verticalalignment='top',
                            # arrowprops=dict(arrowstyle="->",
                            #                 connectionstyle="arc3"),
                            )

        x = np.hstack([[neg_F_kN, pos_F_kN], F_N0_kN])
        y = np.hstack([[neg_y, pos_y], (y_N0)])
        x1 = np.hstack([[0, 0], np.zeros_like(F_N0_kN)])
        y1 = np.hstack([[neg_y, pos_y], (y_N0)])
        ax_F_a.quiver(x, y, x1 - x, y1 - y, scale=1, scale_units='x')

        ax_F_a.set_xlabel(r'$F$ [kN]')
        ax_F_a.set_ylabel(r'$y$ [mm]')
        max_F_N0_kN = np.max(F_N0_kN)
        ax_F_a.set_xlim(neg_F_kN * 1.3, np.max([pos_F_kN, max_F_N0_kN]) * 1.3)

    def subplots(self, fig):
        gs = gridspec.GridSpec(1, 4)
        ax_u_0 = fig.add_subplot(gs[0, 0])
        ax_w_0 = fig.add_subplot(gs[0, 1])
        ax_S_Lb = fig.add_subplot(gs[0, 2])
        ax_S_La = fig.add_subplot(gs[0, 3])
        ax_u_eps = ax_u_0.twiny()
        ax_w_eps = ax_w_0.twiny()
        return ax_u_0, ax_w_0, ax_S_Lb, ax_S_La, ax_u_eps, ax_w_eps

    def update_plot(self, axes):
        ax_u_0, ax_w_0, ax_S_Lb, ax_S_La, ax_u_eps, ax_w_eps = axes
        self.plot_u_t_crc_Ka(ax_u_0)
        self.dic_crack.plot_eps_t_Kab(ax_u_eps)
        bu.mpl_align_xaxis(ax_u_0, ax_u_eps)

        self.plot_u_t_crc_Kb(ax_w_0)
        self.dic_crack.plot_eps_t_Kab(ax_w_eps)
        bu.mpl_align_xaxis(ax_w_0, ax_w_eps)

        self.plot_sig_t_unc_Lab(ax_S_Lb)
        self.plot_sig_t_crc_Lb(ax_S_Lb)
        self.plot_sig_t_unc_Lab(ax_S_La)
        self.plot_sig_t_crc_La(ax_S_La)
        # self.plot_F_t_a(ax_F_a)
