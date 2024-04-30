
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
from .dic_grid import DICGrid

def rotate_around_ref(X_MNa, X_ref_a, T_ab):
    """Rotate the points around X_ref_a
    Pull rotate and push back
    """
    X0_MNa = X_MNa - X_ref_a[np.newaxis, np.newaxis, :] # TODO - this can be done inplace
    # Rotate all points by the inclination of the vertical axis alpha
    x0_MNa = np.einsum('ba,...a->...b', T_ab, X0_MNa)
    # Return to the global coordinate system
    x_ref_mNa = x0_MNa + X_ref_a[np.newaxis, np.newaxis, :]
    return x_ref_mNa


class DICAlignedGrid(bu.Model):
    """
    Define a grid rotated to local coordinate system.

    :param `**kwargs`

    :math: \alpha
    """
    name = 'rotated grid'

    dic_grid = bu.Instance(DICGrid)

    depends_on = ['dic_grid']

    # X0_a = bu.Float(0, ALG=True)
    # '''Origin of the reference system
    # '''
    #
    # X1_a = bu.Float(0, ALG=True)
    # '''Point on the vertical axis of the reference system
    # '''
    #
    x0 = bu.Float(0, ALG=True)
    '''Horizontal coordinate of the origin of the reference system
    '''

    y0 = bu.Float(0, ALG=True)
    '''Vertical coordinate of the origin of the reference system
    '''

    x1 = bu.Float(0, ALG=True)
    '''Horizontal coordinate of the point on the vertical axis of the reference system
    '''

    y1 = bu.Float(0, ALG=True)
    '''Vertical coordinate of the point on the vertical axis of the reference system
    '''

    U_factor = bu.Float(1, ALG=True)
    '''Rotation matrix
    '''

    show_init = bu.Bool(False, ALG=True)
    show_pull = bu.Bool(True, ALG=True)
    show_rot = bu.Bool(False, ALG=True)
    show_VW = bu.Bool(False, ALG=True)

    ipw_view = bu.View(
        bu.Item('x0'),
        bu.Item('y0'),
        bu.Item('x1'),
        bu.Item('y1'),
        bu.Item('show_init'),
        bu.Item('show_pull'),
        bu.Item('show_rot'),
        bu.Item('show_VW'),
        bu.Item('U_factor'),
        time_editor=bu.HistoryEditor(var='dic_grid.t')
    )

    X_0_MNa = tr.Array(dtype=np.float_, ALG=True)
    U_t_MNa = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_U_t_MNa(self):
        return self.dic_grid.f_U_IJ_xy(self.X_0_MNa)

    X_t_MNa = tr.Property(depends_on='state_changed')

    @tr.cached_property
    def _get_X_t_MNa(self):
        return self.X_0_MNa + self.U_t_MNa

    X0_0_a = tr.Property(depends_on='state_changed')
    '''Current position of the reference frame origin
    '''
    @tr.cached_property
    def _get_X0_0_a(self):
        return np.array([self.x0, self.y0])

    U0_t_a = tr.Property(depends_on='state_changed')
    '''Fixed frame rotation at intermediate state.
    '''
    @tr.cached_property
    def _get_U0_t_a(self):
        return self.dic_grid.f_U_IJ_xy(self.x0, self.y0)

    X0_t_a = tr.Property(depends_on='state_changed')
    '''Current position of the reference frame origin
    '''
    @tr.cached_property
    def _get_X0_t_a(self):
        return self.X0_0_a + self.U0_t_a

    X1_0_a = tr.Property(depends_on='state_changed')
    '''Current position of the reference frame origin
    '''
    @tr.cached_property
    def _get_X1_0_a(self):
        return np.array([self.x1, self.y1])

    U1_t_a = tr.Property(depends_on='state_changed')
    '''Fixed frame rotation at intermediate state.
    '''
    @tr.cached_property
    def _get_U1_t_a(self):
        return self.dic_grid.f_U_IJ_xy(self.x1, self.y1)

    X1_t_a = tr.Property(depends_on='state_changed')
    '''Current position of the point on the vertical axis of the reference frame 
    '''
    @tr.cached_property
    def _get_X1_t_a(self):
        return self.X1_0_a + self.U1_t_a

    X_pull_0_MNa = tr.Property(depends_on='state_changed')
    '''Position relative to the pull point
    '''
    @tr.cached_property
    def _get_X_pull_0_MNa(self):
        return self.X_0_MNa - self.X0_0_a

    X_t_MNa = tr.Property(depends_on='state_changed')
    '''Global positions of the grid at the current time
    '''
    @tr.cached_property
    def _get_X_t_MNa(self):
        return self.X_0_MNa + self.U_t_MNa

    X_t_MNa_scaled = tr.Property(depends_on='state_changed')
    '''Coordinates of pulled grid points relative to the origin X0_a.
    '''
    @tr.cached_property
    def _get_X_t_MNa_scaled(self):
        X_t_MNa = self.X_0_MNa + self.U_t_MNa * self.U_factor
        return X_t_MNa

    X_pull_t_MNa = tr.Property(depends_on='state_changed')
    '''Position relative to the pull point
    '''
    @tr.cached_property
    def _get_X_pull_t_MNa(self):
        return self.X_t_MNa - self.X0_t_a

    U_pull_t_MNa = tr.Property(depends_on='state_changed')
    '''Displacement relative to the pull point
    '''
    @tr.cached_property
    def _get_U_pull_t_MNa(self):
        return self.X_pull_t_MNa - self.X_pull_0_MNa

    U_pull_t_MNa_scaled = tr.Property(depends_on='state_changed')
    '''Displacement relative to the pull point
    '''
    @tr.cached_property
    def _get_U_pull_t_MNa_scaled(self):
        return self.U_pull_t_MNa * self.U_factor

    X_pull_t_MNa_scaled = tr.Property(depends_on='state_changed')
    '''Coordinates of pulled grid points relative to the origin X0_a.
    '''
    @tr.cached_property
    def _get_X_pull_t_MNa_scaled(self):
        return self.X_pull_0_MNa + self.U_pull_t_MNa_scaled

    alpha_0 = tr.Property(depends_on='state_changed')
    '''Fixed frame rotation at initial state.
    '''
    @tr.cached_property
    def _get_alpha_0(self):
        X01_a = self.X1_0_a - self.X0_0_a
        return np.arctan(X01_a[0] / X01_a[1])

    alpha_t = tr.Property(depends_on='state_changed')
    '''Fixed frame rotation at intermediate state.
    '''
    @tr.cached_property
    def _get_alpha_t(self):
        X01_a = self.X1_t_a - self.X0_t_a
        return np.arctan(X01_a[0] / X01_a[1])

    T_0_ab = tr.Property(depends_on='state_changed')
    '''Rotation matrix.
    '''
    @tr.cached_property
    def _get_T_0_ab(self):
        alpha = self.alpha_0
        sa, ca = np.sin(alpha), np.cos(alpha)
        return np.array([[ca,-sa],
                         [sa,ca]])

    T_t_ab = tr.Property(depends_on='state_changed')
    '''Rotation matrix at intermediate state.
    '''
    @tr.cached_property
    def _get_T_t_ab(self):
        alpha = self.alpha_t
        sa, ca = np.sin(alpha), np.cos(alpha)
        return np.array([[ca,-sa],
                         [sa,ca]])

    X_rot_0_MNa = tr.Property(depends_on='state_changed')
    '''Grid points rotated around X_0-X_1.
    '''
    @tr.cached_property
    def _get_X_rot_0_MNa(self):
        return np.einsum('ba,...a->...b', self.T_0_ab, self.X_pull_0_MNa)

    X_rot_t_MNa = tr.Property(depends_on='state_changed')
    '''Positions of grid points relative the line alpha_ref.
    '''
    @tr.cached_property
    def _get_X_rot_t_MNa(self):
        return np.einsum('ba,...a->...b', self.T_t_ab, self.X_pull_t_MNa)

    U_rot_t_MNa = tr.Property(depends_on='state_changed')
    '''Displacement increment relative to the rotated reference frame.
    '''
    @tr.cached_property
    def _get_U_rot_t_MNa(self):
        return self.X_rot_t_MNa - self.X_rot_0_MNa

    U_rot_t_MNa_scaled = tr.Property(depends_on='state_changed')
    '''Displacement increment relative to the rotated reference frame.
    '''
    @tr.cached_property
    def _get_U_rot_t_MNa_scaled(self):
        return self.U_rot_t_MNa * self.U_factor

    X_rot_t_MNa_scaled = tr.Property(depends_on='state_changed')
    '''Scaled positions of grid points relative the line X0-X1.
    '''
    @tr.cached_property
    def _get_X_rot_t_MNa_scaled(self):
        return self.X_rot_0_MNa + self.U_rot_t_MNa_scaled

    VW_rot_t_MNa = tr.Property(depends_on='state_changed')
    '''Get the midpoint on the displacement line and perpendicular
    vector w along which the search of the center of rotation can be defined.
    '''
    @tr.cached_property
    def _get_VW_rot_t_MNa(self):
        # get the midpoint on the line X_rot - X_rot_t
        V_rot_t_nMNa = np.array([self.X_rot_0_MNa, self.X_rot_t_MNa])
        V_rot_t_MNa = np.average(V_rot_t_nMNa, axis=0)
        # construct the perpendicular vector w
        U_rot_t_MNa = self.U_rot_t_MNa
        W_rot_t_aMN = np.array([U_rot_t_MNa[..., 1], -U_rot_t_MNa[..., 0]])
        W_rot_t_MNa = np.einsum('a...->...a', W_rot_t_aMN)
        return V_rot_t_MNa, W_rot_t_MNa

    VW_rot_t_MNa_scaled = tr.Property(depends_on='state_changed')
    '''Get the scaled vectors representing the relative displacement with respect
    to the line X0-X1 .
    '''
    @tr.cached_property
    def _get_VW_rot_t_MNa_scaled(self):
        # construct the scaled displacement vector v
        V_rot_t_nMNa_scaled = np.array([self.X_rot_0_MNa, self.X_rot_t_MNa_scaled])
        V_rot_t_anMN_scaled = np.einsum('n...a->an...', V_rot_t_nMNa_scaled)
        V_rot_anp_scaled = V_rot_t_anMN_scaled.reshape(2, 2, -1)
        # construct the perpendicular vector w
        V_rot_t_MNa_scaled = np.average(V_rot_t_nMNa_scaled, axis=0)
        U_rot_t_MNa = self.U_rot_t_MNa
        W_rot_t_aMN = np.array([U_rot_t_MNa[..., 1], -U_rot_t_MNa[..., 0]])
        W_rot_t_MNa = np.einsum('a...->...a', W_rot_t_aMN)
        W_rot_t_MNa_scaled = V_rot_t_MNa_scaled + W_rot_t_MNa * self.U_factor
        W_rot_t_nMNa_scaled = np.array([V_rot_t_MNa_scaled, W_rot_t_MNa_scaled])
        W_rot_t_aMNj_scaled = np.einsum('n...a->an...', W_rot_t_nMNa_scaled)
        W_rot_t_anp_scaled = W_rot_t_aMNj_scaled.reshape(2, 2, -1)
        return V_rot_anp_scaled, W_rot_t_anp_scaled

    def plot_selection_init(self, ax_u):
        X_aij = np.einsum('...a->a...', self.X_0_MNa)
        x_MN, y_MN = X_aij
        ax_u.scatter(x_MN, y_MN, s=15, marker='o', color='orange')
        X0_a = self.X0_t_a
        X1_a = self.X1_t_a
        X01_na = np.array([X0_a, X1_a])
        ax_u.plot(*X01_na.T, lw=2, color='green')

    def plot_frame(self, ax_u):
        """
        """
        X0_a = self.X0_0_a
        X1_a = self.X1_0_a
        U0_t_a = self.dic_grid.f_U_IJ_xy(self.x0, self.y0)
        U1_t_a = self.dic_grid.f_U_IJ_xy(self.x1, self.y1)
        X0_t_scaled_a = X0_a + self.U_factor * U0_t_a
        X1_t_scaled_a = X1_a + self.U_factor * U1_t_a
        X01_na = np.array([X0_a, X1_a])
        #X01_na = np.array([X0_t_scaled_a, X1_t_scaled_a])
        ax_u.plot(*X01_na.T, lw=3, color='green')

    def plot_init(self, ax_u):
        X_t_MNa_scaled = np.einsum('...a->a...', self.X_t_MNa_scaled)
        ax_u.scatter(*X_t_MNa_scaled.reshape(2,-1), s=15, marker='o', color='darkgray')

    def plot_pull(self, ax_u):
        X_t_aMN_scaled = np.einsum('...a->a...', self.X_pull_t_MNa_scaled)
        ax_u.scatter(*X_t_aMN_scaled.reshape(2,-1), s=15, marker='o', color='darkgray')

    def plot_rot(self, ax_u):
        X_rot_0_aMN = np.einsum('...a->a...', self.X_rot_0_MNa)
        ax_u.scatter(*X_rot_0_aMN.reshape(2,-1), s=15, marker='o', color='green')
        X_rot_t_aMN_scaled = np.einsum('...a->a...', self.X_rot_t_MNa_scaled)
        ax_u.scatter(*X_rot_t_aMN_scaled.reshape(2,-1), s=15, marker='o', color='brown')

    def plot_VW(self, ax_u):
        V_rot_anp_scaled, W_rot_t_anp_scaled = self.VW_rot_t_MNa_scaled
        ax_u.scatter(*V_rot_anp_scaled[:, -1, :], s=15, marker='o', color='brown')

    def subplots(self, fig):
        return fig.subplots(1, 1)

    def update_plot(self, axes):
        ax_u = axes


        if self.show_init:
            self.plot_frame(ax_u)
            self.plot_init(ax_u)

        if self.show_pull:
            self.plot_pull(ax_u)

        if self.show_rot:
            self.plot_rot(ax_u)

        if self.show_VW:
            self.plot_VW(ax_u)

        ax_u.axis('equal');
