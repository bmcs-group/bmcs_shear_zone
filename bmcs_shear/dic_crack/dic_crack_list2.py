from ctypes import alignment
import bmcs_utils.api as bu
from .dic_crack2 import DICCrack
from .dic_aligned_grid import DICAlignedGrid
from .dic_state_fields import DICStateFields
import traits.api as tr
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib import cm
from .cached_array import cached_array


class DICCrackList(bu.ModelDict):
    name = 'crack list'

    dsf = bu.Instance(DICStateFields)
    '''State field component providing the history of 
    displacement, strain and damage within the grid.
    '''
    depends_on = ['dsf']

    data_dir = tr.DelegatesTo('dsf')
    beam_param_file = tr.DelegatesTo('dsf')

    a_grid = tr.Property(depends_on='dsf')
    '''Grid aligned to a specified fixed frame - used 
    for the crack-centered evaluation of displacement.
    '''
    @tr.cached_property
    def _get_a_grid(self):
        return DICAlignedGrid(dsf=self.dsf)

    bd = tr.DelegatesTo('dsf')
    '''Beam design.
    '''


    T_t = tr.Property(bu.Int, depends_on='state_changed')
    @tr.cached_property
    def _get_T_t(self):
        return self.dsf.dic_grid.T_t

    delta_alpha_min = bu.Float(-np.pi/6, ALG=True)
    delta_alpha_max = bu.Float(np.pi/3, ALG=True)
    delta_s = bu.Float(10, ALG=True)
    n_G = bu.Int(40, ALG=True)
    x_boundary = bu.Float(20, ALG=True)

    crack_fraction = bu.Float(0.2, ALG=True)

    show_cracks = bu.Enum(options=('primary', 'secondary', 'all'), ALG=True)

    ipw_view = bu.View(
        bu.Item('delta_alpha_min'),
        bu.Item('delta_alpha_max'),
        bu.Item('delta_s'),
        bu.Item('n_G'),
        bu.Item('x_boundary'),
        bu.Item('crack_fraction'),
        bu.Item('show_cracks'),
        bu.Item('T_t', readonly=True),
        time_editor=bu.HistoryEditor(var='dsf.dic_grid.t')
    )

    y_range = tr.Property
    """Vertical limits of the zone
    """
    def _get_y_range(self):
        return self.dsf.X_ipl_bb_Ca[(0,1),(1,1)]


    cracks = tr.Property(depends_on='MESH, ALG')
    @tr.cached_property
    def _get_cracks(self):
        X_CKa = self.X_CKa
        K_tip_C = self.K_tip_C
        n_C = len(X_CKa)
        C_C = np.arange(n_C)
        return np.array([ DICCrack(cl=self, C=C, X_crc_1_Na=X_CKa[C, :K_tip_C[C], :])
            for C in C_C ])

    items = tr.Property(depends_on='MESH, ALG')
    @tr.cached_property
    def _get_items(self):
        return {
            str(C): crack for C, crack in enumerate(self.cracks)
        }
        # X_CKa = self.X_CKa
        # K_tip_C = self.K_tip_C
        # n_C = len(X_CKa)
        # C_C = np.arange(n_C)
        # items = {
        #     str(C): DICCrack(cl=self, C=C, X_crc_1_Na=X_CKa[C, :K_tip_C[C], :])
        #     for C in C_C
        # }
        # return items

    primary_cracks = tr.Property
    def _get_primary_cracks(self):
        primary_cracks, _ = self.sorted_cracks
        return primary_cracks

    secondary_cracks = tr.Property
    def _get_secondary_cracks(self):
        _, secondary_cracks = self.sorted_cracks
        return secondary_cracks

    sorted_cracks = tr.Property
    def _get_sorted_cracks(self):
        # get the array of crack tips
        x_tip_1_Ca = np.array([
            crack.X_tip_1_a for crack in self.cracks
        ])
        # constract the array of vectors connecting each with each crack tip
        diff_x_tip_1_CDa = x_tip_1_Ca[np.newaxis, :, :] - x_tip_1_Ca[:, np.newaxis, :]
        # get the distances
        delta_tip_1_CD = np.linalg.norm(diff_x_tip_1_CDa, axis=-1)
        # identify close crack tips below the threshold values
        close_CD = delta_tip_1_CD < 30
        # pick up the upper diagonal, the row index is smaller than col index
        triu_row, triu_col = np.triu_indices_from(close_CD)
        # select those that are close
        close_rc = close_CD[triu_row, triu_col]
        # realize that each crack tip is close to itself so that
        # its index will definitely appear in the active list
        # Other occurrences of the same index indicate that
        # there is a close crack that needs to be skipped.
        # Thus, the primary cracks can be obtained by subtracting
        # indexes col index from the row index of the close cracks.
        # If it is negative, it denotes the next crack on the right
        # with close crack tip. These cracks are facotred out from
        # the array
        secondary_C = triu_row[close_rc] - triu_col[close_rc] < 0
        secondary_D = np.unique(triu_col[close_rc][secondary_C])
        prim_C = np.ones_like(self.cracks, dtype=np.bool_)
        prim_C[secondary_D] = False
        return self.cracks[prim_C], self.cracks[np.invert(prim_C)]

    def get_X_C1a(self, X_C0a, C_r, alpha_C0):
        """
        X_C0a - initial positions of the crack tip
        alpha_C0 - previous angle of the crack behind the crack tip
        C_r - selection of running cracks
        """
        # active crack tips
        X_r0a = X_C0a[C_r]
        alpha_r0a = alpha_C0[C_r]
        # range of angles in each crack
        alpha_min_r1 = alpha_r0a + self.delta_alpha_min
        alpha_max_r1 = alpha_r0a + self.delta_alpha_max
        # avoid cracks turning downwards
        alpha_max_limit = 0.95 * np.pi/2
        alpha_max_r1[alpha_max_r1 >= alpha_max_limit] = alpha_max_limit
        alpha_gr1 = np.linspace(alpha_min_r1, alpha_max_r1, self.n_G)
        alpha_r1g = alpha_gr1.T
        # range of points around the crack tip
        delta_X_agr1 = np.array([-np.sin(alpha_gr1), np.cos(alpha_gr1)]) * self.delta_s
        delta_X_r1ga = np.einsum('agr->rga', delta_X_agr1)
        # global position of candidate crack tips
        X_r1ga = X_r0a[:, np.newaxis, :] + delta_X_r1ga
        x_r1g, y_r1g = np.einsum('...a->a...', X_r1ga)
        # damage values in candidate crack tips
        t_r1g = np.ones_like(x_r1g)
        args = (t_r1g, x_r1g, y_r1g)
        omega_r1g = self.dsf.f_omega_irn_txy(args)
        # index of candidate with maximum damage in each active tip
        arg_g_omega_r1 = np.argmax(omega_r1g, axis=-1)
        r_r = np.arange(len(arg_g_omega_r1))
        max_omega_r1 = omega_r1g[r_r, arg_g_omega_r1]
        alpha_r1 = alpha_r1g[r_r, arg_g_omega_r1]
        # Update active crack tips
        C_C = np.arange(len(X_C0a))
        r_running = max_omega_r1 > self.dsf.omega_threshold
        # new crack tip
        X_r1a = X_r1ga[r_r, arg_g_omega_r1]
        x_r1, y_r1 = np.einsum('...a->a...', X_r1a)
        # exclude cracks that are less than delta_s from the boundary
        d_s = self.delta_s * 1.01
        x_min, y_min, x_max, y_max = self.dsf.X_ipl_bb_Ca[(0,0,1,1),(0,1,0,1)]
        cross_conditions = (x_r1 < x_min + d_s, x_r1 > x_max - d_s, y_r1 > y_max - d_s)
        cross_cr = [np.array(np.where(cc)) for cc in cross_conditions]
        for cross_r in cross_cr:
            r_running[cross_r] = False
        # update global indexes of active cracks
        C_r = C_C[C_r[r_running]]
        X_C1a = np.copy(X_C0a)
        X_C1a[C_r] = X_r1a[r_running]
        # update last crack angle
        alpha_C1 = np.copy(alpha_C0)
        alpha_C1[C_r] = alpha_r1[r_running]
        return X_C1a, C_r, alpha_C1

    crack_paths = tr.Property#(depends_on='MESH, ALG')
    """Get the crack paths as an array `X_CKa` of a crack $C \in [0, n_C]$
    with defined by the points $K \in [0, n_K$ with coordinates $a \in [0,1]$
    """
    #@cached_array("beam_param_file",names=['X_CKa', 'X_tip_C'])
    #@tr.cached_property
    def _get_crack_paths(self):

        # spatial coordinates
        xx_MN, yy_MN, omega_irn_1_MN = self.dsf.omega_irn_1_MN
        # number of points to skip on the left and right side based on the x_boundary parameters
        d_x = xx_MN[1, 0] - xx_MN[0, 0]
        M_offset = int(self.x_boundary / d_x)
        # initial crack positions at the bottom of the zone
        M_C_with_offset = argrelextrema(omega_irn_1_MN[M_offset:-M_offset, 0], np.greater)[0]
        M_C = M_C_with_offset + M_offset
        # running and stopped cracks
        n_C = len(M_C)
        C_r = np.arange(n_C)
        # initial points
        x_C0, y_C0 = xx_MN[M_C, 0], yy_MN[M_C, 0]
        X_C0a = np.array([x_C0, y_C0]).T
        K_tip_C = np.zeros_like(x_C0, dtype=np.int_)
        X_KCa_ = [X_C0a]
        alpha_C0 = np.zeros((len(X_C0a),))
        K = 1
        while len(C_r) > 0:
            K += 1
            X_C1a, C_r, alpha_C0 = self.get_X_C1a(X_C0a, C_r, alpha_C0)
            K_tip_C[C_r] = K
            X_KCa_.append(X_C1a)
            X_C0a = X_C1a

        X_CKa = np.einsum('KCa->CKa', np.array(X_KCa_))
        C_C = np.arange(len(X_CKa))
        # discard two short cracks
        y_bot, y_top = self.y_range
        h = y_top - y_bot
        X_tip_Ca = X_CKa[C_C, K_tip_C]
        y_tip = X_tip_Ca[:,1] - y_bot
        C_long_enough = (y_tip / h) > self.crack_fraction
        return X_CKa[C_long_enough], K_tip_C[C_long_enough]

    X_CKa = tr.Property
    def _get_X_CKa(self):
        return self.crack_paths[0]

    K_tip_C = tr.Property
    def _get_K_tip_C(self):
        return self.crack_paths[1]

    def plot_all_cracks(self, ax_cracks):
        X_aKC = np.einsum('CKa->aKC', self.X_CKa)
        ax_cracks.plot(*X_aKC, color='black', linewidth=1)
        ax_cracks.axis('equal')

    def plot_primary_cracks(self, ax_cracks):
        for crack in self.primary_cracks:
            crack.plot_X_crc_1_Ka(ax_cracks)
            crack.plot_X_crc_t_Ka(ax_cracks)

    def plot_secondary_cracks(self, ax_cracks):
        for crack in self.secondary_cracks:
            crack.plot_X_crc_1_Ka(ax_cracks)
            crack.plot_X_crc_t_Ka(ax_cracks)

    critical_crack = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_critical_crack(self):
        u_1_max_C = []
        for dc in self.cracks:
            u_1_max = np.max(dc.u_crc_1_Ka[:, 1])
            u_1_max_C.append(u_1_max)
        critical_C = np.argmax(np.array(u_1_max_C))
        return self.items[str(critical_C)]

    def plot_MQ(self, ax_M, ax_Q):
        dg = self.dsf.dic_grid
        L_right = dg.sz_bd.L_right
        M_1_kN = -dg.M_1 / 1000
        ax_M.plot([0, L_right], [M_1_kN, 0], color='gray')
        ax_M.fill_between([0, L_right], [M_1_kN, 0], 0, color='gray', alpha=0.1)
        M_t_kN = -dg.M_t / 1000
        ax_M.plot([0, L_right], [M_t_kN, 0], color='brown')
        ax_M.fill_between([0, L_right], [M_t_kN, 0], 0, color='brown', alpha=0.2)

        x_CN = np.array([
            crack.x_N for crack in self.primary_cracks
        ])
        A_CN = np.array([
            crack.A_N for crack in self.primary_cracks
        ])
        F_t_CNa = np.array([
            crack.sp.F_t_Na for crack in self.primary_cracks
        ])
        sig_t_CN0 = F_t_CNa[:,0,0] / A_CN[:,0]
        ax_Q.plot(x_CN[:,0], sig_t_CN0, 'o-', color='orange')
        ax_Q.fill_between(x_CN[:,0], sig_t_CN0, 0, color='orange', alpha=0.1)

        X_mid_unc_t_Ca = np.array([
            crack.sp.X_mid_unc_t_a for crack in self.primary_cracks
        ])
        M_mid_unc_t_Cf = np.array([
            crack.sp.M_mid_unc_t_a for crack in self.primary_cracks
        ])
        M_cb_C, M_da_C, M_ag_C = M_mid_unc_t_Cf.T
        ax_M.plot(X_mid_unc_t_Ca[:,0], -M_cb_C/1e+6, 'o-', color='orange')

        bu.mpl_align_yaxis_to_zero(ax_M, ax_Q)
        ax_Q.get_shared_x_axes().join(ax_Q, ax_M)

    def subplots(self, fig):
        self.fig = fig
        gs = gridspec.GridSpec(ncols=3, nrows=2,
                               width_ratios=[1, 1, 1],
                               wspace=0.5,
                               # hspace=0.5,
                               height_ratios=[1, 1]
                               )
        ax_cl = fig.add_subplot(gs[0, :-1])
        ax_FU = fig.add_subplot(gs[:, -1])
        ax_M = fig.add_subplot(gs[1, :-1])
        ax_Q = ax_M.twinx()
        return ax_cl, ax_FU, ax_M, ax_Q

    def update_plot(self, axes):
        ax_cl, ax_FU, ax_M, ax_Q = axes
#        self.dsf.dic_grid.plot_bounding_box(ax_dsf)
        # self.dsf.dic_grid.plot_box_annotate(ax_dsf)
        self.bd.plot_sz_bd(ax_cl)
        self.dsf.plot_crack_detection_field(ax_cl, self.fig)
        #self.plot_cracking_hist2(ax_cl)
        #self.critical_crack.plot_x_t_crc_Ka(ax_cl, line_width=2, line_color='red', tip_color='red')
        if self.show_cracks in ['primary', 'all']:
            self.plot_primary_cracks(ax_cl)
        if self.show_cracks in ['secondary', 'all']:
            self.plot_secondary_cracks(ax_cl)
        ax_cl.axis('equal')
        ax_cl.axis('off');
        self.dsf.dic_grid.plot_load_deflection(ax_FU)

        self.plot_MQ(ax_M, ax_Q)
        return
        # plot the kinematic profile
        #self.critical_crack.plot_u_t_crc_Ka(ax_u)
        self.critical_crack.plot_eps_t_Kab(ax_eps)
        ax_eps.set_ylim(0, self.bd.H)
        ax_u.set_ylim(0, self.bd.H * 1.04)
        bu.mpl_align_xaxis(ax_u, ax_eps)

        # self.critical_crack.sp.plot_sig_t_unc_Lab(ax_sig)
        # self.critical_crack.sp.plot_sig_t_crc_La(ax_sig)
        # self.critical_crack.sp.plot_F_t_a(ax_F)
        # bu.mpl_align_xaxis(ax_sig, ax_F)

