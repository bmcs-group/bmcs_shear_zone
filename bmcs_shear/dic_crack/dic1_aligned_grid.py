
from .dic_grid import DICGrid
from .dic1_strain_grid import DICStrainGrid
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np

class DICAlignedGrid(bu.Model):

    dic_grid = bu.Instance(DICGrid, ())

    tree = ['dic_grid']

    end_t = tr.DelegatesTo('dic_grid')
    start_t = tr.DelegatesTo('dic_grid')
    U_factor = tr.DelegatesTo('dic_grid')

    y_ref_i = bu.Int(-1, ALG=True)
    '''Horizontal index defining the y_axis position
    '''
    y_ref_j_min = bu.Int(1, ALG=True)
    '''Vertical index defining the start position of y axis
    '''
    y_ref_j_max = bu.Int(10, ALG=True)
    '''Vertical index defining the end position of y axis
    '''

    show_init = bu.Bool(False, ALG=True)
    show_xu = bu.Bool(True, ALG=True)
    show_xw = bu.Bool(False, ALG=True)

    t = bu.Float(1, ALG=True)
    '''Slider over the time history
    '''
    def _t_changed(self):
        n_t = self.dic_grid.n_t
        d_t = (1 / n_t)
        self.dic_grid.end_t = int((n_t - 1) * (self.t + d_t / 2))

    ipw_view = bu.View(
        bu.Item('y_ref_i'),
        bu.Item('y_ref_j_max'),
        bu.Item('show_init'),
        bu.Item('show_xu'),
        bu.Item('show_xw'),
        time_editor=bu.HistoryEditor(
            var='t'
        )
    )

    X_ija = tr.DelegatesTo('dic_grid')
    U_ija = tr.DelegatesTo('dic_grid')

    U0_a = tr.Property(depends_on='state_changed')
    '''Origin of the reference frame
    '''
    @tr.cached_property
    def _get_U0_a(self):
        return self.U_ija[self.y_ref_i,0,:]

    X0_a = tr.Property(depends_on='state_changed')
    '''Origin of the reference frame
    '''
    @tr.cached_property
    def _get_X0_a(self):
        return self.X_ija[self.y_ref_i,0,:] + self.U0_a

    U_ref_ija = tr.Property(depends_on='state_changed')
    '''Displacement increment relative to the reference point.
    '''
    @tr.cached_property
    def _get_U_ref_ija(self):
        U0_11a = self.U0_a[np.newaxis,np.newaxis,:]
        U_ref_ija = self.U_ija - U0_11a
        return U_ref_ija

    X_ref_ija = tr.Property(depends_on='state_changed')
    '''Displacement increment relative to the reference point.
    '''
    @tr.cached_property
    def _get_X_ref_ija(self):
        X_ref_ija = self.X_ija + self.U_ref_ija
        return X_ref_ija

    alpha_ref = tr.Property(depends_on='state_changed')
    '''Rotation of the reference boundary line.
    '''
    @tr.cached_property
    def _get_alpha_ref(self):
        # Put the origin of the coordinate system into the reference point
        X_ref_ja = (self.X_ija[self.y_ref_i, :self.y_ref_j_max,:] +
                    self.U_ref_ija[self.y_ref_i, :self.y_ref_j_max,:])
        X0_a = X_ref_ja[0, :]
        X0_ja = X_ref_ja - X0_a[np.newaxis, :]
        alpha_ref = np.arctan(X0_ja[1:, 0] / X0_ja[1:, 1])
        return np.average(alpha_ref)

    T_ab = tr.Property(depends_on='state_changed')
    '''Rotation matrix of the reference boundary line.
    '''
    @tr.cached_property
    def _get_T_ab(self):
        alpha_ref = self.alpha_ref
        sa, ca = np.sin(alpha_ref), np.cos(alpha_ref)
        return np.array([[ca,-sa],
                         [sa,ca]])

    x_ref_ija = tr.Property(depends_on='state_changed')
    '''Displacement increment relative to the reference frame.
    '''
    @tr.cached_property
    def _get_x_ref_ija(self):
        # Get the global displaced configuration without reference point displacement
        X_ref_ija = self.X_ija + self.U_ref_ija
        # Put the origin of the coordinate system into the reference point
        X0_a = X_ref_ija[self.y_ref_i, 0]
        X0_ija = X_ref_ija - X0_a[np.newaxis, np.newaxis, :]
        # Rotate all points by the inclination of the vertical axis alpha
        x0_ija = np.einsum('ba,...a->...b', self.T_ab, X0_ija)
        # Return to the global coordinate system
        x_ref_ija = x0_ija + X0_a[np.newaxis, np.newaxis, :]
        return x_ref_ija

    u_ref_ija = tr.Property(depends_on='state_changed')
    '''Displacement increment relative to the reference frame.
    '''
    @tr.cached_property
    def _get_u_ref_ija(self):
        return self.x_ref_ija - self.X_ija

    x_ref_ija_scaled = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_x_ref_ija_scaled(self):
        X_ija = self.X_ija
        return X_ija + self.u_ref_ija * self.U_factor

    xu_mid_w_ref_ija = tr.Property(depends_on='state_changed')
    '''Get the midpoint on the displacement line and perpendicular
    vector w along which the search of the center of rotation can be defined.
    '''
    @tr.cached_property
    def _get_xu_mid_w_ref_ija(self):
        X_ija = self.X_ija
        # find a midpoint on the line xu
        xu_ref_ija = self.x_ref_ija
        xu_ref_nija = np.array([X_ija, xu_ref_ija])
        xu_mid_ija = np.average(xu_ref_nija, axis=0)
        # construct the perpendicular vector w
        u_ref_ija = self.u_ref_ija
        w_ref_aij = np.array([u_ref_ija[..., 1], -u_ref_ija[..., 0]])
        w_ref_ija = np.einsum('aij->ija', w_ref_aij)
        return xu_mid_ija, w_ref_ija

    displ_grids_scaled = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_displ_grids_scaled(self):
        X_ija = self.X_ija
        xu_ref_ija_scaled = self.x_ref_ija_scaled
        # construct the displacement vector v
        xu_ref_nija_scaled = np.array([X_ija, xu_ref_ija_scaled])
        xu_ref_anij_scaled = np.einsum('nija->anij', xu_ref_nija_scaled)
        xu_ref_anp_scaled = xu_ref_anij_scaled.reshape(2, 2, -1)
        # construct the perpendicular vector w
        xu_mid_ija_scaled = np.average(xu_ref_nija_scaled, axis=0)
        u_ref_ija = self.u_ref_ija
        w_ref_aij = np.array([u_ref_ija[..., 1], -u_ref_ija[..., 0]])
        w_ref_ija = np.einsum('aij->ija', w_ref_aij)
        xw_ref_ija_scaled = xu_mid_ija_scaled + w_ref_ija * self.U_factor
        xw_ref_nija_scaled = np.array([xu_mid_ija_scaled, xw_ref_ija_scaled])
        xw_ref_anij_scaled = np.einsum('nija->anij', xw_ref_nija_scaled)
        xw_ref_anp_scaled = xw_ref_anij_scaled.reshape(2, 2, -1)
        return xu_ref_anp_scaled, xw_ref_anp_scaled

    def subplots(self, fig):
        return fig.subplots(1, 2)

    def update_plot(self, axes):
        ax_u, ax_load = axes

        if self.show_init:
            X_ref_aij = np.einsum('ija->aij', self.X_ref_ija)
            ax_u.scatter(*X_ref_aij.reshape(2,-1), s=15, marker='o', color='darkgray')
            X_aij = np.einsum('ija->aij', self.X_ija)
            ax_u.scatter(*X_aij.reshape(2,-1), s=15, marker='o', color='blue')

        xu_ref_anp_scaled, xw_ref_anp_scaled = self.displ_grids_scaled

        if self.show_xu:
            ax_u.scatter(*xu_ref_anp_scaled[:, -1, :], s=15, marker='o', color='silver')
            ax_u.plot(*xu_ref_anp_scaled, color='silver', linewidth=0.5);

        if self.show_xw:
            ax_u.plot(*xw_ref_anp_scaled, color='green', linewidth=0.5);

        y_ref_ja = self.x_ref_ija_scaled[self.y_ref_i, :self.y_ref_j_max]
        ax_u.scatter(*y_ref_ja.T, s=20, color='green')

        ax_u.axis('equal');

        deflection = self.dic_grid.ld_values[::50, 2]
        load = -self.dic_grid.ld_values[::50, 1]
        ax_load.plot(deflection, load, color='black')
        max_deflection = np.max(deflection)
        load_level = self.dic_grid.current_load
        ax_load.plot([0, max_deflection], [load_level, load_level],
                     color='green', lw=2)
