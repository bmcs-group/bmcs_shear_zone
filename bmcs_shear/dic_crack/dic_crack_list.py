
import bmcs_utils.api as bu
from .dic_crack import DICCrack
from .dic_crack_cor import DICCrackCOR
from .dic_stress_profile import DICStressProfile
from .dic_state_fields import DICStateFields
import traits.api as tr
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib import cm

class DICCrackList(bu.ModelList):
    name = 'crack list'

    dsf = bu.Instance(DICStateFields)

    bd = tr.DelegatesTo('dsf')

    items = tr.Property(bu.List(DICCrackCOR), depends_on='state_changed')
    @tr.cached_property
    def _get_items(self):
        x_NC, y_NC, N_tip_C, M_NC = self.primary_cracks
        dic_cracks = [DICCrack(cl=self, C=C, x_N=x_N, y_N=y_N, N_tip=N_tip, M_N=M_N)
            for C, (x_N, y_N, N_tip, M_N) in enumerate(zip(x_NC.T, y_NC.T, N_tip_C, M_NC.T))
        ]
        #return [ DICCrackCOR(dic_crack=dic_crack) for dic_crack in dic_cracks ]
        return [ DICStressProfile(dic_crack=dic_crack) for dic_crack in dic_cracks ]

    t = bu.Float(ALG=True)

    def _t_changed(self):
        n_t = self.dsf.dic_grid.n_t
        d_t = (1 / n_t)
        self.dsf.dic_grid.end_t = int((n_t - 1) * (self.t + d_t / 2))
        print('t_changed - ', self.dsf.dic_grid.end_t)
        self.t_idx = self.dsf.dic_grid.end_t

    t_idx = bu.Int(1)

    ipw_view = bu.View(
        bu.Item('t_idx', read_only=True),
        time_editor=bu.HistoryEditor(var='t')
    )

    def detect_cracks(self, M_C, xx_MN, yy_MN, cdf_MN):
        '''
        paramters:
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
            return np.zeros((n_N, 0)), np.zeros((n_N, 0)), np.zeros((0, )), np.zeros((n_N, 0))
        # distance between right interval boundary and crack position
        M_NC_shift_ = []
        # list of crack horizontal indexes for each horizontal slice
        M_NC_ = [np.copy(M_C)]
        # crack tip N
        crack_tip_y = np.zeros_like(M_C, dtype=np.int_)
        C_fixed = np.zeros_like(M_C, dtype=np.bool_)
        for N1 in N_range[1:]:
            # horizontal indexes of midpoints between cracks
            M_C_left_ = M_C - 3
            M_C_left_[M_C_left_ < 0] = 0
            M_C_right_ = M_C + 2
            # array of intervals - first index - crack, second index (left, right)
            intervals_Cp = np.vstack([M_C_left_, M_C_right_]).T
            # index distance from the right boundary of the crack interval
            M_C_shift = np.array([
                np.argmax(cdf_NM[N1, interval_p[-1]:interval_p[0]:-1])
                for interval_p in intervals_Cp
            ])
            # cracks, for which the next point could be identified
            C_shift = ((M_C_shift > 0) & np.logical_not(C_fixed))
            C_fixed = np.logical_not(C_shift)
            # crack tips
            crack_tip_y[C_shift] = N1
            # next index position of the crack
            M_C[C_shift] = intervals_Cp[C_shift, -1] - M_C_shift[C_shift]
            M_NC_.append(np.copy(M_C))
            M_NC_shift_.append(M_C_shift)
        M_NC = np.array(M_NC_)
        n_C = M_NC.shape[1]
        N_C = np.arange(n_N)
        N_NC = np.repeat(N_C, n_C).reshape(n_N, -1)
        xx_NC = xx_MN[M_NC, N_NC]
        yy_NC = yy_MN[M_NC, N_NC]
        return xx_NC, yy_NC, crack_tip_y, M_NC

    primary_cracks = tr.Property(depends_on='state_changed')
    '''Get the cracks at the near-failure load level
    '''
    @tr.cached_property
    def _get_primary_cracks(self):
        # spatial coordinates
        t_eta_idx = self.dsf.dic_grid.get_F_eta_dic_idx(0.95)
        self.dsf.dic_grid.end_t = t_eta_idx
        xx_MN, yy_MN, cd_field_irn_MN = self.dsf.crack_detection_field
        # initial crack positions at the bottom of the zone
        M_C = argrelextrema(cd_field_irn_MN.T[0, :], np.greater)[0]
        xx_NC, yy_NC, N_tip_C, M_NC = self.detect_cracks(M_C, xx_MN, yy_MN, cd_field_irn_MN)
        # remove secondary cracks and duplicate cracks
        n_N, n_C = M_NC.shape
        mid_N = int(n_N / 3)
        # boalean array marking the cracks that propagated more than 1 / 3 of the monitored zone
        C_mid_C = np.where(N_tip_C >= mid_N)[0]
        # identify cracks that joined at the level of the 1 / 3 of the monitored zone
        M_mid_NC = M_NC[mid_N, C_mid_C]
        _, M, dM = np.unique(M_mid_NC,
                             return_index=True, return_counts=True)
        C_pri_C = C_mid_C[M] + dM - 1
        return xx_NC[:, C_pri_C], yy_NC[:, C_pri_C], N_tip_C[C_pri_C], M_NC[:, C_pri_C]

    cracks_t = tr.Property(depends_on='MESH')
    '''Get the cracks at the near-failure load level
    '''
    def _get_cracks_t(self):
        # spatial coordinates
        _, _, _, M_pri_NC = self.primary_cracks
        xx_MN, yy_MN, cd_field_irn_MN = self.dsf.crack_detection_field
        M_C = M_pri_NC[0, :]
        print('MC - primary', M_C)
        xx_NC, yy_NC, N_tip_C, M_NC = self.detect_cracks(M_C, xx_MN, yy_MN, cd_field_irn_MN)
        print('MC - detected', M_NC[0,:], N_tip_C)
        return xx_NC, yy_NC, N_tip_C, M_NC

    def plot_detected_cracks(self, ax_cracks, fig):
        xx_MN, yy_MN, cd_field_irn_MN = self.dsf.crack_detection_field
        xx_NC, yy_NC, N_tip_C, _ = self.primary_cracks
        cs = ax_cracks.contour(xx_MN, yy_MN, cd_field_irn_MN, cmap=cm.coolwarm, antialiased=False)
        cbar_cracks = fig.colorbar(cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap),
                                   ax=ax_cracks, ticks=np.linspace(0, 1, 6),
                                   orientation='horizontal')
        cbar_cracks.set_label(r'$\omega = 1 - \min(\phi_I)$')
        for C, y_tip in enumerate(N_tip_C):
            ax_cracks.plot(xx_NC[:y_tip, C], yy_NC[:y_tip, C], color='black',
                           linewidth=1);
        # xx_NC, yy_NC, N_tip_C, _ = self.cracks_t
        # for C, y_tip in enumerate(N_tip_C):
        #     ax_cracks.plot(xx_NC[:y_tip, C], yy_NC[:y_tip, C], color='black', linewidth=1);

        ax_cracks.axis('equal')
        ax_cracks.axis('off');

    def subplots(self, fig):
        self.fig = fig
        gs = gridspec.GridSpec(ncols=2, nrows=1,
                               width_ratios=[2, 1], wspace=0.5,
                               # hspace=0.5, height_ratios=[1, 1]
                               )

        ax_dsf = fig.add_subplot(gs[0, 0])
        ax_FU = fig.add_subplot(gs[0, 1])
        return ax_dsf, ax_FU

    def update_plot(self, axes):
        ax_dsf, ax_FU = axes
        self.dsf.dic_grid.plot_bounding_box(ax_dsf)
        self.dsf.dic_grid.plot_box_annotate(ax_dsf)
        self.plot_detected_cracks(ax_dsf, self.fig)
        self.dsf.dic_grid.plot_load_deflection(ax_FU)