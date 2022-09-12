
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
from .dic_state_fields import DICStateFields

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

    dsf = bu.Instance(DICStateFields)

    depends_on = ['dsf']
    # tree = ['dsf']

    M0 = bu.Int(0, ALG=True)
    '''Horizontal index of the origin marker
    '''

    N0 = bu.Int(0, ALG=True)
    '''Vertical index of the origin marker
    '''

    M1 = bu.Int(0, ALG=True)
    '''Horizontal index of the origin marker
    '''

    N1 = bu.Int(-1, ALG=True)
    '''Vertical index of the origin marker
    '''

    U_factor = bu.Float(1, ALG=True)
    '''Rotation matrix
    '''

    show_init = bu.Bool(False, ALG=True)
    show_pull = bu.Bool(True, ALG=True)
    show_rot = bu.Bool(False, ALG=True)

    ipw_view = bu.View(
        bu.Item('M0'),
        bu.Item('N0'),
        bu.Item('M1'),
        bu.Item('N1'),
        bu.Item('show_init'),
        bu.Item('show_pull'),
        bu.Item('show_rot'),
        bu.Item('U_factor'),
        time_editor=bu.HistoryEditor(var='dsf.dic_grid.t')
    )

    X_MNa = tr.DelegatesTo('dsf', 'X_ipl_MNa')
    U_t_MNa = tr.DelegatesTo('dsf', 'U_ipl_MNa')

    X0_t_a = tr.Property(depends_on='state_changed')
    '''Fixed frame rotation at intermediate state.
    '''
    @tr.cached_property
    def _get_X0_t_a(self):
        return self.X_t_MNa[self.M0, self.N0]

    U0_a = tr.Property(depends_on='state_changed')
    '''Displacement of the origin of the reference frame
    '''
    @tr.cached_property
    def _get_U0_a(self):
        return self.U_MNa[self.M0,self.N0,:]

    MN_selection = tr.Any(Ellipsis)

    X_pull_MNa = tr.Property(depends_on='state_changed')
    '''Position relative to the pull point
    '''
    @tr.cached_property
    def _get_X_pull_MNa(self):
        return self.X_MNa[self.MN_selection] - self.X_MNa[self.M0, self.N0]

    X_t_MNa = tr.Property(depends_on='state_changed')
    '''Displacement relative to the pull point
    '''
    @tr.cached_property
    def _get_X_t_MNa(self):
        return self.X_MNa + self.U_t_MNa

    X_t_MNa_scaled = tr.Property(depends_on='state_changed')
    '''Coordinates of pulled grid points relative to the origin X0_a.
    '''
    @tr.cached_property
    def _get_X_t_MNa_scaled(self):
        X_t_MNa = self.X_MNa + self.U_t_MNa * self.U_factor
        return X_t_MNa

    X_pull_t_MNa = tr.Property(depends_on='state_changed')
    '''Position relative to the pull point
    '''
    @tr.cached_property
    def _get_X_pull_t_MNa(self):
        return self.X_t_MNa[self.MN_selection] - self.X_t_MNa[self.M0, self.N0]

    U_pull_t_MNa = tr.Property(depends_on='state_changed')
    '''Displacement relative to the pull point
    '''
    @tr.cached_property
    def _get_U_pull_t_MNa(self):
        return self.U_t_MNa[self.MN_selection] - self.U_t_MNa[self.M0, self.N0]

    X_pull_t_MNa_scaled = tr.Property(depends_on='state_changed')
    '''Coordinates of pulled grid points relative to the origin X0_a.
    '''
    @tr.cached_property
    def _get_X_pull_t_MNa_scaled(self):
        return self.X_pull_MNa + self.U_pull_t_MNa * self.U_factor

    alpha = tr.Property(depends_on='state_changed')
    '''Fixed frame rotation at initial state.
    '''
    @tr.cached_property
    def _get_alpha(self):
        X0_a = self.X_MNa[self.M0, self.N0]
        X1_a = self.X_MNa[self.M1, self.N1]
        X01_a = X1_a - X0_a
        return np.arctan(X01_a[0] / X01_a[1])

    alpha_t = tr.Property(depends_on='state_changed')
    '''Fixed frame rotation at intermediate state.
    '''
    @tr.cached_property
    def _get_alpha_t(self):
        X0_a = self.X_t_MNa[self.M0, self.N0]
        X1_a = self.X_t_MNa[self.M1, self.N1]
        X01_a = X1_a - X0_a
        return np.arctan(X01_a[0] / X01_a[1])

    T_ab = tr.Property(depends_on='state_changed')
    '''Rotation matrix.
    '''
    @tr.cached_property
    def _get_T_ab(self):
        alpha = self.alpha
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

    X_rot_MNa = tr.Property(depends_on='state_changed')
    '''Grid points rotated around X_0-X_1.
    '''
    @tr.cached_property
    def _get_X_rot_MNa(self):
        return np.einsum('ba,...a->...b', self.T_ab, self.X_pull_MNa)

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
        return self.X_rot_t_MNa - self.X_rot_MNa

    X_rot_t_MNa_scaled = tr.Property(depends_on='state_changed')
    '''Scaled positions of grid points relative the line X0-X1.
    '''
    @tr.cached_property
    def _get_X_rot_t_MNa_scaled(self):
        return self.X_rot_MNa + self.U_rot_t_MNa * self.U_factor

    VW_rot_t_MNa = tr.Property(depends_on='state_changed')
    '''Get the midpoint on the displacement line and perpendicular
    vector w along which the search of the center of rotation can be defined.
    '''
    @tr.cached_property
    def _get_VW_rot_t_MNa(self):
        # get the midpoint on the line X_rot - X_rot_t
        V_rot_t_nMNa = np.array([self.X_rot_MNa, self.X_rot_t_MNa])
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
        V_rot_t_nMNa_scaled = np.array([self.X_rot_MNa, self.X_rot_t_MNa_scaled])
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
        X_aij = np.einsum('...a->a...', self.X_MNa)
        x_MN, y_MN = X_aij
        ax_u.scatter(x_MN[self.MN_selection], y_MN[self.MN_selection], s=15, marker='o', color='orange')
        X0_a = self.X_MNa[self.M0, self.N0, :]
        X1_a = self.X_MNa[self.M1, self.N1, :]
        X01_na = np.array([X0_a, X1_a])
        ax_u.plot(*X01_na.T, lw=2, color='green')

    def plot_init(self, ax_u):
        X_t_MNa_scaled = np.einsum('...a->a...', self.X_t_MNa_scaled)
        ax_u.scatter(*X_t_MNa_scaled.reshape(2,-1), s=15, marker='o', color='darkgray')
        X_aij = np.einsum('...a->a...', self.X_MNa)
#        ax_u.scatter(*X_aij.reshape(2,-1), s=15, marker='o', color='blue')
        x_MN, y_MN = X_aij
        ax_u.scatter(x_MN[self.MN_selection], y_MN[self.MN_selection], s=15, marker='o', color='orange')
        X0_a = self.X_t_MNa_scaled[self.M0, self.N0, :]
        X1_a = self.X_t_MNa_scaled[self.M1, self.N1, :]
        X01_na = np.array([X0_a, X1_a])
        ax_u.plot(*X01_na.T, lw=3, color='green')

    def plot_pull(self, ax_u):
        X_t_MNa_scaled = np.einsum('...a->a...', self.X_pull_t_MNa_scaled)
        ax_u.scatter(*X_t_MNa_scaled.reshape(2,-1), s=15, marker='o', color='darkgray')
        X_MNj = np.einsum('...a->a...', self.X_pull_MNa)
        ax_u.scatter(*X_MNj.reshape(2,-1), s=5, marker='o', color='blue')
        # X0_a = self.X_pull_t_MNa_scaled[self.M0, self.N0, :]
        # X1_a = self.X_pull_t_MNa_scaled[self.M1, self.N1, :]
        # X01_na = np.array([X0_a, X1_a])
        # ax_u.plot(*X01_na.T, lw=3, color='green')

    def plot_rot(self, ax_u):
        V_rot_anp_scaled, W_rot_t_anp_scaled = self.VW_rot_t_MNa_scaled
        ax_u.scatter(*V_rot_anp_scaled[:, -1, :], s=15, marker='o', color='silver')
        ax_u.plot(*V_rot_anp_scaled, color='silver', linewidth=0.5);
        # X0_a = self.X_rot_t_MNa_scaled[self.M0, self.N0, :]
        # X1_a = self.X_rot_t_MNa_scaled[self.M1, self.N1, :]
        # X01_na = np.array([X0_a, X1_a])
        # ax_u.plot(*X01_na.T, lw=3, color='green')

    def subplots(self, fig):
        return fig.subplots(1, 1)

    def update_plot(self, axes):
        ax_u = axes

        if self.show_init:
            self.plot_init(ax_u)

        if self.show_pull:
            self.plot_pull(ax_u)

        if self.show_rot:
            self.plot_rot(ax_u)

        ax_u.axis('equal');
