from ctypes import alignment
import bmcs_utils.api as bu
from .dic_crack_deprecated import DICCrack
from .dic_aligned_grid_deprecated import DICAlignedGrid
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

    '''Cracks
    '''
    def identify_cracks(self):
        x_NC, y_NC, N_tip_C, M_NC = self.primary_cracks
        for C, (x_N, y_N, N_tip, M_N) in enumerate(zip(x_NC.T, y_NC.T, N_tip_C, M_NC.T)):
            self.items[str(C)] = DICCrack(cl=self,
                                          C=C, x_N=x_N, y_N=y_N, N_tip=N_tip, M_N=M_N)
#            self.__setitem__(str(C), DICCrack(cl=self, C=C, x_N=x_N, y_N=y_N, N_tip=N_tip, M_N=M_N))


    T_t = tr.Property(bu.Int, depends_on='state_changed')
    @tr.cached_property
    def _get_T_t(self):
        return self.dsf.dic_grid.T_t

    corridor_left = bu.Int(3, ALG=True)
    corridor_right = bu.Int(2, ALG=True)
    x_boundary = bu.Float(20, ALG=True)

    ipw_view = bu.View(
        bu.Item('corridor_left'),
        bu.Item('corridor_right'),
        bu.Item('x_boundary'),
        bu.Item('T_t', readonly=True),
        time_editor=bu.HistoryEditor(var='dsf.dic_grid.t')
    )

    def detect_cracks(self, M_C, xx_MN, yy_MN, cdf_MN):
        '''
        parameters:
        M_C: horizontal indexes of the starting cracks
        xx_MN: horizontal coordinates of the grid points (M, N)
        yy_MN: vertical coordinates of the grid points (M, N)
        cdf_MN: values of the crack detection field in the grid points (M, N)
                This field is scalar, with values representing the "proneness" to
                cracking. Either maximum principal tensile strain or damage
                are the choices at this place.
        returns:
        xx_NC: horizontal coordinates of N-th segment of C-th crack ligament
        yy_NC: vertical coordinates of N-th segment of C-th crack ligament
        N_tip_C: vertical index of the crack tip of the C-th ligament
        M_NC: horizontal indexes N of C-th ligament
        '''
        N_range = np.arange(yy_MN.shape[1])
        cdf_NM = cdf_MN.T
        n_N, n_M = cdf_NM.shape
        # smooth the landscape
        if len(M_C) == 0:
            return np.zeros((n_N, 0)), np.zeros((n_N, 0)), np.zeros((0,)), np.zeros((n_N, 0))
        # distance between right interval boundary and crack position
        M_NC_propag_ = []
        # list of crack horizontal indexes for each horizontal slice
        M_NC_ = [np.copy(M_C)]
        # crack tip N
        crack_tip_y = np.zeros_like(M_C, dtype=np.int_)
        C_fixed = np.zeros_like(M_C, dtype=np.bool_)
        for N1 in N_range[1:]:
            # horizontal indexes of midpoints between cracks
            M_C_left_ = M_C - self.corridor_left
            M_C_left_[M_C_left_ < 0] = 0
            M_C_right_ = M_C + self.corridor_right
            # array of intervals - first index - crack, second index (left, right)
            intervals_Cp = np.vstack([M_C_left_, M_C_right_]).T
            # index distance from the right boundary of the crack interval
            M_C_propag = np.array([
                np.argmax(cdf_NM[N1, interval_p[-1]:interval_p[0]:-1])
                for interval_p in intervals_Cp
            ])
            # Get the damage value of the maximum along the tested line
            omega_NC = cdf_NM[N1, M_C_right_ - M_C_propag]
            # if omega_NC is non-zero - the crack propagation is active
            C_propag = ((omega_NC > 0) & np.logical_not(C_fixed))
            C_fixed = np.logical_not(C_propag)
            # crack tips
            crack_tip_y[C_propag] = N1
            # next index position of the crack
            M_C[C_propag] = intervals_Cp[C_propag, -1] - M_C_propag[C_propag]
            M_NC_.append(np.copy(M_C))
            M_NC_propag_.append(M_C_propag)
        M_NC = np.array(M_NC_)
        n_C = M_NC.shape[1]
        N_C = np.arange(n_N)
        N_NC = np.repeat(N_C, n_C).reshape(n_N, -1)
        xx_NC = xx_MN[M_NC, N_NC]
        yy_NC = yy_MN[M_NC, N_NC]
        return xx_NC, yy_NC, crack_tip_y, M_NC

    primary_cracks = tr.Property(depends_on='MESH, ALG')
    """Get the cracks at the near-failure load level
    """
    @cached_array("beam_param_file",names=['x_NC', 'y_NC', 'N_tip_C', 'M_NC'])
    def _get_primary_cracks(self):
        # spatial coordinates
        xx_MN, yy_MN, omega_irn_1_MN = self.dsf.omega_irn_1_MN
        # number of points to skip on the left and right side based on the x_boundary parameters
        d_x = xx_MN[1,0] - xx_MN[0,0]
        M_offset = int(self.x_boundary / d_x)
        # initial crack positions at the bottom of the zone
        M_C = argrelextrema(omega_irn_1_MN[M_offset:-M_offset, 0], np.greater)[0]
        xx_NC, yy_NC, N_tip_C, M_NC = self.detect_cracks(
            M_C,
            xx_MN[M_offset:-M_offset,:],
            yy_MN[M_offset:-M_offset,:],
            omega_irn_1_MN[M_offset:-M_offset,:]
        )
        # remove secondary cracks and duplicate cracks
        n_N, n_C = M_NC.shape
        mid_N = int(n_N / 5)
        # boalean array marking the cracks that propagated more than 1 / 3 of the monitored zone
        C_mid_C = np.where(N_tip_C >= mid_N)[0]
        C_pri_C = C_mid_C
        # # identify cracks that joined at the level of the 1 / 3 of the monitored zone
        # M_mid_NC = M_NC[mid_N, C_mid_C]
        # _, M, dM = np.unique(M_mid_NC, return_index=True, return_counts=True)
        # C_pri_C = C_mid_C[M] + dM - 1
        return xx_NC[:, C_pri_C], yy_NC[:, C_pri_C], N_tip_C[C_pri_C], M_NC[:, C_pri_C] + M_offset

    critical_crack = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_critical_crack(self):
        u_1_max_C = []
        for dc in self.items.values():
            u_1_max = np.max(dc.u_1_crc_Ka[:, 1])
            u_1_max_C.append(u_1_max)
        critical_C = np.argmax(np.array(u_1_max_C))
        return self.items[str(critical_C)]


    def plot_primary_cracks(self, ax_cracks):
        xx_NC, yy_NC, N_tip_C, _ = self.primary_cracks
        for C, y_tip in enumerate(N_tip_C):
            ax_cracks.plot(xx_NC[:y_tip, C], yy_NC[:y_tip, C], color='black',
                           linewidth=1);
        ax_cracks.axis('equal')
#        ax_cracks.axis('off');

    def plot_cracking_hist2(self, ax_cracks):
        for crack in self.items.values():
            crack.plot_x_1_crc_Ka(ax_cracks)
            crack.plot_x_t_crc_Ka(ax_cracks)

    def subplots(self, fig):
        self.fig = fig
        gs = gridspec.GridSpec(ncols=3, nrows=2,
                               width_ratios=[1, 1, 1],
                               wspace=0.5,
                               # hspace=0.5,
                               height_ratios=[2, 1]
                               )
        ax_dsf = fig.add_subplot(gs[0, :])
        ax_FU = fig.add_subplot(gs[1, 0])
        ax_u = fig.add_subplot(gs[1, 1])
        ax_eps = ax_u.twiny()
        ax_F = fig.add_subplot(gs[1, 2])
        ax_sig = ax_F.twiny()
        return ax_dsf, ax_FU, ax_u, ax_eps, ax_F, ax_sig

    def update_plot(self, axes):
        ax_dsf, ax_FU, ax_u, ax_eps, ax_F, ax_sig = axes
#        self.dsf.dic_grid.plot_bounding_box(ax_dsf)
        # self.dsf.dic_grid.plot_box_annotate(ax_dsf)
        self.bd.plot_sz_bd(ax_dsf)
        self.dsf.plot_crack_detection_field(ax_dsf, self.fig)
        self.plot_cracking_hist2(ax_dsf)
        self.critical_crack.plot_x_t_crc_Ka(ax_dsf, line_width=2, line_color='red', tip_color='red')
        ax_dsf.axis('equal')
        ax_dsf.axis('on');
        self.dsf.dic_grid.plot_load_deflection(ax_FU)
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

