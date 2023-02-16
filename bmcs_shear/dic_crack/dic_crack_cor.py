
from hashlib import algorithms_guaranteed
from .i_dic_crack import IDICCrack
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

def rotate_around_ref(X_MNa, X_ref_a, T_ab):
    """Rotate the points around X_ref_a pull rotate and push back
    """
    X0_MNa = X_MNa - X_ref_a[np.newaxis, np.newaxis, :] # TODO - this can be done inplace
    # Rotate all points by the inclination of the vertical axis alpha
    x0_MNa = np.einsum('ba,...a->...b', T_ab, X0_MNa)
    # Return to the global coordinate system
    x_ref_mNa = x0_MNa + X_ref_a[np.newaxis, np.newaxis, :]
    return x_ref_mNa

class DICCrackCOR(bu.Model):
    '''Determination of the center of rotation.
    '''
    name = tr.Property
    @tr.cached_property
    def _get_name(self):
        return self.dic_crack.name

    dic_crack = bu.Instance(IDICCrack)

    cl = tr.DelegatesTo('dic_crack')

    dsf = tr.Property
    @tr.cached_property
    def _get_dsf(self):
        return self.dic_crack.cl.dsf

    a_grid = tr.Property
    @tr.cached_property
    def _get_a_grid(self):
        return self.dic_crack.cl.a_grid

    dic_grid = tr.Property()
    @tr.cached_property
    def _get_dic_grid(self):
        return self.dic_crack.dic_grid

    depends_on = ['dic_crack']

    delta_x = bu.Float(30, ALG=True)
    delta_y = bu.Float(30, ALG=True)

    X0_0_a = tr.Property(bu.Float, depends_on='state_changed')
    '''Point relative to the crack tip defining the rigid frame on left tooth
    '''
    @tr.cached_property
    def _get_X0_0_a(self):
        X_crc_1_0a, X_crc_1_1a = self.dic_crack.X_crc_1_Ka[(0, -1), :]
        if self.frame_position == 'vertical':
            X0_0_a = np.array([X_crc_1_1a[0] - self.delta_x,
                               X_crc_1_0a[1] + self.delta_y])
        elif self.frame_position == 'inclined':
            X0_0_a = X_crc_1_0a + np.array([-self.delta_x, self.delta_y])
        return X0_0_a

    X1_0_a = tr.Property(bu.Float, depends_on='state_changed')
    '''Point relative to the crack tip defining the rigid frame on left tooth
    '''
    @tr.cached_property
    def _get_X1_0_a(self):
        X_tip_1_a = self.dic_crack.X_crc_1_Ka[-1, :]
        return X_tip_1_a - np.array([self.delta_x, 0])


    # M_N = tr.Property(bu.Int, depends_on='state_changed')
    # '''Horizontal indexes of the crack segments.
    # '''
    # @tr.cached_property
    # def _get_M_N(self):
    #     return self.dic_crack.M_N
    #
    frame_position = bu.Enum(options=[
        'inclined', 'vertical'
    ], ALG=True)

    ipw_view = bu.View(
        bu.Item('delta_x'),
        bu.Item('delta_y'),
        # bu.Item('x0', readonly=True),
        # bu.Item('y0', readonly=True),
        # bu.Item('x1', readonly=True),
        # bu.Item('y1', readonly=True),
        bu.Item('step_N_COR'),
        bu.Item('frame_position'),
        bu.Item('phi_t', readonly=True),
        bu.Item('cov_phi_t', readonly=True),
        bu.Item('V_t', readonly=True),
        bu.Item('M_t', readonly=True),
        time_editor=bu.HistoryEditor(var='dic_crack.dic_grid.t')
    )

    step_N_COR = bu.Int(2, ALG=True)
    '''Vertical index distance between the markers included in the COR calculation
    '''


    x_bandwidth = bu.Float(70, ALG=True)
    x_spacing = bu.Float(20, ALG=True)
    y_spacing = bu.Float(20, ALG=True)
    y_offset = bu.Float(20, ALG=True)
    x_offset = bu.Float(20, ALG=True)

    X_OPa = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_X_OPa(self):
        if len(self.dic_crack.X_crc_t_Ka) == 0:
            return np.zeros((0, 0, 2), np.float_)
        X_start_a, X_end_a = self.dic_crack.X_crc_t_Ka[(0, -1), :]
        delta_y = X_end_a[1] - (X_start_a[1] + self.y_offset)
        n_P = int(delta_y / self.y_spacing) + 1
        y_P = X_start_a[1] + self.y_offset + np.arange(n_P) * self.y_spacing
        x_P = self.dic_crack.C_cubic_spline(y_P)

        n_O = int(self.x_bandwidth / self.x_spacing)
        x_O = self.x_offset + np.arange(n_O) * self.x_spacing
        y_O = np.zeros_like(x_O)

        x_OP = x_O[:, np.newaxis] + x_P[np.newaxis, :]
        y_OP = y_O[:, np.newaxis] + y_P[np.newaxis, :]
        X_aOP = np.array([x_OP, y_OP])
        return np.einsum('a...->...a', X_aOP)

    crack_exists = tr.Property
    def _get_crack_exists(self):
        return self.X_OPa.shape[1] > 0

    VW_rot_t_pa = tr.Property(depends_on='state_changed')
    '''Displacement vector and the normal to its midpoint 
    '''
    @tr.cached_property
    def _get_VW_rot_t_pa(self):
        if not self.crack_exists:
            return np.zeros((0, 2), np.float_), np.zeros((0, 2), np.float_)
        x0, y0 = self.X0_0_a
        x1, y1 = self.X1_0_a
        self.a_grid.trait_set(
            x0=x0, y0=y0, x1=x1, y1=y1,
            X_0_MNa = self.X_OPa
        )
        V_rot_t_pa, W_rot_t_pa = self.a_grid.VW_rot_t_MNa
        return V_rot_t_pa.reshape(-1,2), W_rot_t_pa.reshape(-1,2)

    X_cor_rot_t_pa_sol = tr.Property(depends_on='state_changed')
    '''Center of rotation determined for each patch point separately
    '''
    @tr.cached_property
    def _get_X_cor_rot_t_pa_sol(self):

        V_rot_pa, W_rot_pa = self.VW_rot_t_pa
        def get_X_cor_pa(eta_p):
            '''Get the points on the perpendicular lines with the sliders eta_p'''
            return V_rot_pa + np.einsum('p,pa->pa', eta_p, W_rot_pa)

        def get_R(eta_p):
            '''Residuum of the closest distance condition.
            '''
            x_cor_pa = get_X_cor_pa(eta_p)
            delta_x_cor_pqa = x_cor_pa[:, np.newaxis, :] - x_cor_pa[np.newaxis, :, :]
            R2 = np.einsum('pqa,pqa->', delta_x_cor_pqa, delta_x_cor_pqa)
            return np.sqrt(R2)

        eta0_p = np.zeros((V_rot_pa.shape[0],))
        min_eta_p_sol = minimize(get_R, eta0_p, method='BFGS')
        eta_p_sol = min_eta_p_sol.x
        X_cor_pa_sol = get_X_cor_pa(eta_p_sol)
        return X_cor_pa_sol

    X_cor_rot_t_a = tr.Property(depends_on='state_changed')
    '''Center of rotation of the patch related to the local 
    patch reference system.
    '''
    @tr.cached_property
    def _get_X_cor_rot_t_a(self):
        return np.average(self.X_cor_rot_t_pa_sol, axis=0)

    X_cor_t_pa = tr.Property(depends_on='state_changed')
    '''Center of rotation of the patch related to the local 
    patch reference system.
    '''
    @tr.cached_property
    def _get_X_cor_t_pa(self):
        x0, y0 = self.X0_0_a
        x1, y1 = self.X1_0_a
        self.a_grid.trait_set(
            x0=x0, y0=y0, x1=x1, y1=y1,
            X_0_MNa = self.X_OPa
        )
        X_cor_pull_t_pa = np.einsum('ba,...b->...a',
            self.a_grid.T_t_ab, self.X_cor_rot_t_pa_sol
        )
        X_cor_t_pa = X_cor_pull_t_pa + self.a_grid.X0_t_a
        return X_cor_t_pa

    X_cor_t_a = tr.Property(depends_on='state_changed')
    '''Center of rotation of the patch related to the local 
    patch reference system.
    '''
    @tr.cached_property
    def _get_X_cor_t_a(self):
        if not self.crack_exists:
            return np.array([0,0], dtype=np.float_)
        x0, y0 = self.X0_0_a
        x1, y1 = self.X1_0_a
        self.a_grid.trait_set(
            x0=x0, y0=y0, x1=x1, y1=y1,
            X_0_MNa = self.X_OPa
        )
        X_cor_pull_t_a = np.einsum('ba,...b->...a',
            self.a_grid.T_t_ab, self.X_cor_rot_t_a
        )
        X_cor_t_a = X_cor_pull_t_a + self.a_grid.X0_t_a
        return X_cor_t_a

    stat_phi_t = tr.Property(depends_on='state_changed')
    '''Statistics of the rotation angle around the current COR.
    '''
    @tr.cached_property
    def _get_stat_phi_t(self):
        if not self.crack_exists:
            return 0, 0
        V_rot_t_pa, W_rot_t_pa = self.VW_rot_t_pa
        norm_U2_t_MN = np.linalg.norm(self.a_grid.U_rot_t_MNa, axis=-1) / 2
        norm_U2_t_p = norm_U2_t_MN.flatten()
        norm_V2_t_p = np.linalg.norm(self.X_cor_rot_t_a - V_rot_t_pa, axis=-1)
        phi_p = 2 * np.arctan(norm_U2_t_p / norm_V2_t_p)
        mean_phi = np.mean(phi_p)
        std_phi = np.std(phi_p)
        cov_phi = std_phi / mean_phi
        return mean_phi, cov_phi

    phi_t = tr.Property(bu.Float, depends_on='state_changed')
    '''Average angle of rotation.
    '''
    @tr.cached_property
    def _get_phi_t(self):
        mean_phi, _ = self.stat_phi_t
        return mean_phi

    cov_phi_t = tr.Property(bu.Float, depends_on='state_changed')
    '''Coefficient of variation for the angles of rotation.
    '''
    @tr.cached_property
    def _get_cov_phi_t(self):
        _, cov_phi = self.stat_phi_t
        return cov_phi

    M_t = tr.Property(bu.Float, depends_on='state_changed')
    '''Coefficient of variation for the angles of rotation.
    '''
    @tr.cached_property
    def _get_M_t(self):
        if self.crack_exists:
            X_cor_r = self.X_cor_t_a[0]
        else:
            # if the crack does not exist yet, take it's initial position
            # at the bottom layer
            X_cor_r = self.dic_crack.X_crc_1_Ka[0,0]
        L_right = self.dic_grid.sz_bd.L_right
        L_left = self.dic_grid.sz_bd.L_left
        F_right = self.dic_grid.F_T_t * L_left / (L_left + L_right)
        M = (F_right * (L_right - X_cor_r)) / 1000
        return M

    V_t = tr.Property(bu.Float, depends_on='state_changed')
    '''Coefficient of variation for the angles of rotation.
    '''
    @tr.cached_property
    def _get_V_t(self):
        L_right = self.dic_grid.sz_bd.L_right
        L_left = self.dic_grid.sz_bd.L_left
        F_right = self.dic_grid.F_T_t * L_left / (L_left + L_right)
        return F_right

    def plot_X_cor_rot_t(self, ax):
        if not self.crack_exists:
            return
        ax.plot(*self.X_cor_rot_t_pa_sol.T, 'o', color = 'blue')
        ax.plot([self.X_cor_rot_t_a[0]], [self.X_cor_rot_t_a[1]], 'o', color='yellow')
        ax.axis('equal');


    cor_marker_size = bu.Int(10, ALG=True)
    cor_marker_color = bu.Str('magenta', ALG=True)
    def plot_X_cor_t(self, ax):
        if not self.crack_exists:
            return
        ax.plot(*self.X_cor_t_pa.T, 'o', color = 'blue')
        ax.plot([self.X_cor_t_a[0]], [self.X_cor_t_a[1]], 'o',
                color=self.cor_marker_color, markersize=self.cor_marker_size)
        ax.axis('equal');

    def plot_VW_rot_t(self, ax_x):
        if not self.crack_exists:
            return
        V_rot_t_pa, W_rot_pa = self.VW_rot_t_pa
        ax_x.scatter(*V_rot_t_pa.T, s=20, color='orange')

    def subplots(self, fig):
        return fig.subplots(1,1)

    def update_plot(self, axes):
        ax_x = axes
        self.dic_grid.plot_bounding_box(ax_x)
        self.dic_grid.plot_box_annotate(ax_x)

        if self.crack_exists:
            x0, y0 = self.X0_0_a
            x1, y1 = self.X1_0_a
            self.a_grid.trait_set(
                x0=x0, y0=y0, x1=x1, y1=y1,
                X_0_MNa = self.X_OPa
            )
            self.a_grid.plot_init(ax_x)
            self.plot_X_cor_t(ax_x)

        self.a_grid.plot_frame(ax_x)

        self.dic_crack.plot_X_1_Ka(ax_x)
        self.dic_crack.plot_X_t_Ka(ax_x)

        ax_x.axis('equal');
