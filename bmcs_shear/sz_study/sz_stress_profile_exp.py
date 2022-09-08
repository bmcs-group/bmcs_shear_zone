import traits.api as tr
import numpy as np
from bmcs_utils.api import InteractiveModel, View, Item, mpl_align_xaxis
from bmcs_shear.shear_crack.deformed_state import \
    SZDeformedState
from scipy.interpolate import interp1d
# # Stress profiles

# ## Stress resultants

# Normal force is obtained by evaluating the integral
# \begin{align}
# N = \int_{0}^{H} \sigma_{0}(x_1) \; \mathrm{d}x
# \end{align}

# Given a normal load $\bar{N}$ the equilibrium condition
# \begin{align}
# N = \bar{N}
# \end{align}
# must be fulfilled.

# At the same tome the shear force gets calculated as
# \begin{align}
# Q =
# \int_{\Gamma} \sigma_1(x_1) \; \mathrm{d}x
# \end{align}

# The moment around the center of rotation $x^\mathrm{rot}_a$ is obtained as
# \begin{align}
# M = \int \sigma_0 \left(x_1 - x^\mathrm{rot}_1\right) \; \mathrm{d}x_1
# + \int \sigma_1 \left(x_0 - x^\mathrm{rot}_0\right) \mathrm{d}x_0
# \end{align}


class SZStressProfile(InteractiveModel):
    '''
    @todo: check the update upon the discretization attributes
    '''
    name = "Profiles"

    ds = tr.Instance(SZDeformedState, ())
    sz_cp = tr.DelegatesTo('ds')
    sz_bd = tr.DelegatesTo('sz_cp')

    _ITR = tr.DelegatesTo('sz_cp')
    _INC = tr.DelegatesTo('sz_cp')
    _MAT = tr.DelegatesTo('sz_cp')
    _GEO = tr.DelegatesTo('sz_cp')
    _DSC = tr.DelegatesTo('sz_cp')
    _ALL = tr.Event

    @tr.on_trait_change('_ITR, _INC, _GEO, _MAT, _DSC')
    def _reset_ALL(self):
        self._ALL = True

    ipw_view = View()

    state_changed = tr.DelegatesTo('sz_cp')
    # =========================================================================
    # Displacement interpolation and transformation
    # =========================================================================

    u_La = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Displacement of the segment midpoints '''
    @tr.cached_property
    def _get_u_La(self):
        K_Li = self.sz_cp.K_Li
        u_Ka = self.ds.x1_Ka - self.sz_cp.x_Ka
        u_Lia = u_Ka[K_Li]
        u_La = np.sum(u_Lia, axis=1) / 2
        return u_La

    u_Ca = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Displacement of the corner nodes '''
    @tr.cached_property
    def _get_u_Ca(self):
        x_Ca = self.sz_bd.x_Ca
        x1_Ca = self.ds.x1_Ca
        u_Ca = x1_Ca - x_Ca
        return u_Ca

    u_Lb = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Displacement of the segment midpoints '''
    @tr.cached_property
    def _get_u_Lb(self):
        u_La = self.u_La
        T_Mab = self.sz_cp.T_Mab
        u_Lb = np.einsum(
            'La,Lab->Lb', u_La, T_Mab
        )
        return u_Lb

    # =========================================================================
    # Stress transformation and integration
    # =========================================================================

    S_Lb = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Stress returned by the material model
    '''
    @tr.cached_property
    def _get_S_Lb(self):
        u_Lb = self.u_Lb
        cmm = self.ds.sz_bd.cmm  #adv cmm_adv
        B = self.ds.sz_bd.B
        sig_La = cmm.get_sig_a(u_Lb)
        return sig_La * B
        # Sig_w = cmm.get_sig_w(u_a[..., 0]) * B #get_sig_w
        # Tau_w = cmm.get_tau_s(u_a[..., 1]) * B #get_tau_s get_tau_ag u_a[..., 0],
        # return np.einsum('b...->...b', np.array([Sig_w, Tau_w], dtype=np.float_))

    S_La = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Transposed stresses'''
    @tr.cached_property
    def _get_S_La(self):

        S_Lb = self.S_Lb
        S_La = np.einsum('Lb,Lab->La', S_Lb, self.ds.sz_cp.T_Mab)
        return S_La

    # =========================================================================
    # Stress resultants
    # =========================================================================

    F_La = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Integrated segment forces'''
    @tr.cached_property
    def _get_F_La(self):
        S_La = self.S_La
        F_La = np.einsum('La,L->La', S_La, self.ds.sz_cp.norm_n_vec_L)
        return F_La

    F_ag = tr.Property(depends_on = '_ITR, _INC, _GEO, _MAT, _DSC')
    '''Integrated Aggregate Interlock forces'''
    @tr.cached_property
    def _get_F_ag(self):
        S_La = self.S_La
        B = self.ds.sz_bd.B
        F_ag = S_La[...,1] * B * np.sin(self.sz_cp._get_beta())
        return F_ag

    get_w_N = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Get an interpolator function returning crack opening displacement 
    for a specified vertical coordinate of a ligament.
    '''
    @tr.cached_property
    def _get_get_w_N(self):
        return interp1d(self.sz_cp.x_Lb[:, 1], self.u_La[:, 0],
                        fill_value='extrapolate')

    get_s_N = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Get an interpolator function returning slip displacement 
    component for a specified vertical coordinate of a ligament.
    '''
    @tr.cached_property
    def _get_get_s_N(self):
        return interp1d(self.sz_cp.x_Lb[:, 1], self.u_La[:, 1],
                        fill_value='extrapolate')

    u_Na = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
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
        # @todo [RC]: adapt to the finished CS-design later
        return self.sz_bd.cross_section_layout.reinforcement.z_j

    A_N = tr.Property
    def _get_A_N(self):
        # @todo [RC]: adapt to the finished CS-design later
        return self.sz_bd.cross_section_layout.reinforcement.A_j

    E_N = tr.Property
    def _get_E_N(self):
        # @todo [RC]: adapt to the finished CS-design later
        return self.sz_bd.cross_section_layout.reinforcement.E_j

    F_Na = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Get the discrete force in the reinforcement z_N
    '''
    @tr.cached_property
    def _get_F_Na(self):
        u_Na = self.u_Na
        #F_N0 = self.A_N * self.E_N * w_N # self.sz_bd.get_sig_w_f(w_N)
        F_Na = self.sz_bd.smm.get_F_a(u_Na)
#        F_N0 = self.sz_bd.smm.get_sig_w_f(w_N)
#        F_N1 = self.sz_bd.smm.get_sig_s_f(s_N) #smm
#        F_N1 = np.zeros_like(F_N0)
        return F_Na
        # to transform into the global coordinates identify the
        # segment of the ligament L corresponding to the position
        # of the reinforcement N
        z_L = self.sz_bd.x_La[1]
        z_N = self.sz_bd.z_N
        L_N = np.argmax(z_L[np.newaxis, :] >= z_N[:, np.newaxis], axis=1)
        T_Nab = self.sz_cp.T_Lab[L_N,...]
        F_Na = np.einsum('...ab,...a->...a', T_Nab, F_Nb)
        return F_Na

    F_a = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Integrated normal and shear force
    '''
    @tr.cached_property
    def _get_F_a(self):
        F_La = self.F_La
        F_Na = self.F_Na
        #F_ag = self.F_ag
        sum_F_La = np.sum(F_La, axis=0)
        sum_F_Na = np.sum(F_Na, axis=0)
        #sum_F_ag = np.sum(F_ag, axis=0)
        return sum_F_La + sum_F_Na #+ sum_F_ag

    M = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Internal bending moment obtained by integrating the
    normal stresses with the lever arm rooted at the height of the neutral
    axis.
    '''
    @tr.cached_property
    def _get_M(self):
        x_Ka = self.ds.sz_cp.x_Ka
        K_Li = self.ds.sz_cp.K_Li
        x_Lia = x_Ka[K_Li]
        x_La = np.sum(x_Lia, axis=1) / 2
        F_La = self.F_La
        x_rot_1k = self.ds.sz_ctr.x_rot_1k
        M_L = (x_La[:, 1] - x_rot_1k) * F_La[:, 0]
        M = np.sum(M_L, axis=0)
        M_z = np.einsum('i,i', (self.z_N - x_rot_1k), self.F_Na[:,0])
        return -M - M_z

    sig_x_tip_0k = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Normal stress component in global $x$ direction in the fracture .
    process segment.
    '''
    @tr.cached_property
    def _get_sig_x_tip_0k(self):
        x_tip_1k = self.sz_cp.sz_ctr.x_tip_ak[1][0]
        return self.get_sig_x_tip_0k(x_tip_1k)

    #         S_a = self.S_La[self.xd.n_m_fps, ...]
    #         return S_a[0] / self.sim.B

    get_sig_x_tip_0k = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Get an interpolator function returning horizontal stress 
    component for a specified vertical coordinate of a ligament.
    '''
    @tr.cached_property
    def _get_get_sig_x_tip_0k(self):
        B = self.ds.sz_bd.B
        return interp1d(self.sz_cp.x_Lb[:, 1], self.S_La[:, 0] / B,
                        fill_value='extrapolate')

    tau_exp = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')

    @tr.cached_property
    def _get_tau_exp(self):
        B = self.ds.sz_bd.B
        D = self.ds.sz_bd.B
        F_La = self.F_La
        F_Na = self.F_Na
        V_exp = F_La[..., 1] + F_Na[...,1]
        print((V_exp )/ (B * D))
        return V_exp / (B * D)

    normalized_def = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')

    @tr.cached_property
    def _get_normalized_def(self):
        L = self.ds.sz_bd.L
        delta_ = self.u_La[:,1]
        #print(delta_)
        return (delta_)

    # =========================================================================
    # Plotting methods
    # =========================================================================

    def plot_u_Lc(self, ax, u_Lc, idx=0, color='black', label=r'$w$ [mm]'):
        x_La = self.sz_cp.x_Lb
        u_Lb_min = np.min(u_Lc[:, idx])
        u_Lb_max = np.max(u_Lc[:, idx])
        self.plot_hlines(ax, u_Lb_min, u_Lb_max)
        ax.plot(u_Lc[:, idx], x_La[:, 1], color=color, label=label)
        ax.fill_betweenx(x_La[:, 1], u_Lc[:, idx], 0, color=color, alpha=0.1)
        ax.set_xlabel(label)
        ax.legend(loc='lower left')

    def plot_hlines(self, ax, h_min, h_max):
        y_tip = self.sz_cp.sz_ctr.x_tip_1n
        y_rot = self.sz_cp.sz_ctr.x_rot_1k
        z_fps = self.sz_cp.sz_ctr.x_fps_ak[1]
        ax.plot([h_min, h_max], [y_tip, y_tip],
                color='black', linestyle=':')
        ax.plot([h_min, h_max], [y_rot, y_rot],
                color='black', linestyle='--')
        ax.plot([h_min, h_max], [z_fps, z_fps],
                color='black', linestyle='-.')

    def plot_u_La(self, ax_w, vot=1):
        '''Plot the displacement along the crack (w and s) in global coordinates
        '''
        ax_s = ax_w.twiny()
        self.plot_u_Lc(ax_w, self.u_La, 0, label=r'$u_x$ [mm]', color='blue')
        ax_w.set_xlabel(r'$u_x$ [mm]')
        self.plot_u_Lc(ax_s, self.u_La, 1, label=r'$u_z$ [mm]', color='green')
        ax_s.set_xlabel(r'$u_z$ [mm]')
        mpl_align_xaxis(ax_w, ax_s)

    def plot_u_Lb(self, ax_w, vot=1):
        '''Plot the displacement (u_x, u_y) in local crack coordinates
        '''
        ax_s = ax_w.twiny()
        # plot the critical displacement
        sz_ctr = self.sz_cp.sz_ctr
        x_tip_1k = sz_ctr.x_tip_ak[1,0]
        w = sz_ctr.w
        ax_w.plot([0, w],[x_tip_1k, x_tip_1k], '-o', lw=2, color='red')
        self.plot_u_Lc(ax_w, self.u_Lb, 0, label=r'$w$ [mm]', color='blue')
        ax_w.set_xlabel(r'opening $w$ [mm]')
        self.plot_u_Lc(ax_s, self.u_Lb, 1, label=r'$s$ [mm]', color='green')
        ax_s.set_xlabel(r'sliding $s$ [mm]')
        mpl_align_xaxis(ax_w, ax_s)

    def plot_S_Lb(self, ax_sig, vot=1):
        '''Plot the stress components (sig, tau) in local crack coordinates
        '''
        ax_tau = ax_sig.twiny()
        # plot the critical displacement
        bd = self.sz_cp.sz_bd
        cmm = bd.cmm
        sz_ctr = self.sz_cp.sz_ctr
        x_tip_1k = sz_ctr.x_tip_ak[1,0]
        S_t = cmm.f_t * bd.B
        ax_sig.plot([0, S_t],[x_tip_1k, x_tip_1k], '-o', lw=2, color='red')
        self.plot_u_Lc(ax_sig, self.S_Lb, 0, label=r'$\sigma_\mathrm{N}$ [N/mm]', color='blue')
        ax_sig.set_xlabel(r'normal stress flow $\sigma_\mathrm{N}$ [N/mm]')
        self.plot_u_Lc(ax_tau, self.S_Lb, 1, label=r'$\sigma_\mathrm{T}$ [N/mm]', color='green')
        ax_tau.set_xlabel(r'shear stress flow $\sigma_\mathrm{T}$ [N/mm]')
        mpl_align_xaxis(ax_sig, ax_tau)

    def plot_S_La(self, ax_sig, vot=1):
        ax_tau = ax_sig.twiny()
        self.plot_u_Lc(ax_sig, self.S_La, 0, label=r'$f_x$ [N/mm]', color='blue')
        ax_sig.set_xlabel(r'horizontal stress flow $f_x$ [N/mm]')
        self.plot_u_Lc(ax_tau, self.S_La, 1, label=r'$f_z$ [N/mm]', color='green')
        ax_tau.set_xlabel(r'vertical stress flow $f_z$ [N/mm]')
        mpl_align_xaxis(ax_sig, ax_tau)

    def plot_tau_exp(self, ax, vot=1.0):
        ax.plot(self.normalized_def, self.tau_exp, lw=2, color='blue')
        ax.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax.set_ylabel(r'$\tau_{Exp}\;\;\mathrm{[MPa]}$')
        ax.set_title('Calculated Shear Stress')

    def plot_N_a(self, ax_N):
        pass

    def subplots(self, fig):
        return fig.subplots(1 ,4)

    def update_plot(self, axes):
        ax_u_La, ax_u_Lb, ax_S_Lb, ax_S_La = axes
        #self.plot_u_La(ax_u_La)
        #self.plot_u_Lb(ax_u_Lb)
        #self.plot_S_Lb(ax_S_Lb)
        self.plot_tau_exp(ax_S_La)
        #self.plot_S_La(ax_S_La)

