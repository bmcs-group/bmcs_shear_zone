
from .dic_grid import DICGrid
from .dic_strain_grid import DICStrainGrid
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

    show_rot = bu.Bool(False, ALG=True)
    show_perp = bu.Bool(False, ALG=True)
    show_init = bu.Bool(False, ALG=True)

    t = bu.Float(1, ALG=True)
    def _t_changed(self):
        n_t = self.dic_grid.n_t
        d_t = (1 / n_t)
        self.dic_grid.end_t = int((n_t - 1) * (self.t + d_t / 2))

    ipw_view = bu.View(
        bu.Item('y_ref_i'),
        bu.Item('y_ref_j_min'),
        bu.Item('y_ref_j_max'),
        bu.Item('show_rot'),
        bu.Item('show_perp'),
        bu.Item('show_init'),
        time_editor=bu.HistoryEditor(
            var='t'
        )
    )

    u_tija = tr.DelegatesTo('dic_grid')
    X_ija = tr.DelegatesTo('dic_grid')

    delta_u_ul_ija = tr.Property(depends_on='state_changed')
    '''Displacement increment relative to upper left corner.
    '''
    @tr.cached_property
    def _get_delta_u_ul_ija(self):
        delta_u_ija = self.u_tija[self.end_t] - self.u_tija[self.start_t]
        #print('delta_u_ija', delta_u_ija)
        #u_11a = delta_u_ija[-1:,:1,:]
        u_11a = delta_u_ija[self.y_ref_i,self.y_ref_j_min,:][np.newaxis,np.newaxis,:]
        #print('u_11a', u_11a)
        delta_u_ul_ija = delta_u_ija - u_11a
        #print('delta_u_ul_ija', delta_u_ul_ija)
        return delta_u_ul_ija

    delta_alpha = tr.Property(depends_on='state_changed')
    '''Rotation of the reference boundary line.
    '''
    @tr.cached_property
    def _get_delta_alpha(self):
        # slice the left boundary
        d_u_ul_1j1 = self.delta_u_ul_ija[-1,1:10,0] - self.delta_u_ul_ija[-1,:1,0]
        d_x_i0 = self.X_ija[-1,1:10,1] - self.X_ija[-1,:1,1]
        d_u_ul_1j1 = (self.delta_u_ul_ija[self.y_ref_i,self.y_ref_j_min:self.y_ref_j_max,0] -
                      self.delta_u_ul_ija[self.y_ref_i,:1,0])
        d_x_i0 = self.X_ija[self.y_ref_i,1:10,1] - self.X_ija[self.y_ref_i,:1,1]
        sin_delta_alpha = np.average(d_u_ul_1j1 / d_x_i0)
        return np.arcsin(sin_delta_alpha)

    T_ab = tr.Property(depends_on='state_changed')
    '''Rotation matrix of the reference boundary line.
    '''
    @tr.cached_property
    def _get_T_ab(self):
        delta_alpha = self.delta_alpha
        sa, ca = np.sin(delta_alpha), np.cos(delta_alpha)
        return np.array([[ca,-sa],
                         [sa,ca]])

    X_ref_a = tr.Property(depends_on='state_changed')
    '''Origin of the reference frame
    '''
    @tr.cached_property
    def _get_X_ref_a(self):
        XU_ija = self.X_ija + self.delta_u_ul_ija
        return XU_ija[self.y_ref_i, self.y_ref_j_min, :]

    delta_u0_ul_ija = tr.Property(depends_on='state_changed')
    '''Displacement increment relative to the reference frame.
    '''
    @tr.cached_property
    def _get_delta_u0_ul_ija(self):
        XU_ija = self.X_ija + self.delta_u_ul_ija
        XU_pull_ija = XU_ija - XU_ija[-1:,:1,:]
        XU_pull_ija = XU_ija - self.X_ref_a[np.newaxis,np.newaxis,:]
        XU0_ija = np.einsum('ba,...a->...b', self.T_ab, XU_pull_ija)
        XU_push_ija = XU0_ija + self.X_ref_a[np.newaxis,np.newaxis,:]
        return XU_push_ija - self.X_ija

    rot_Xu_ija = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_rot_Xu_ija(self):
        X_ija = self.X_ija
        delta_u_rot_ija = self.delta_u0_ul_ija
        return X_ija + delta_u_rot_ija * self.U_factor

    displ_grids = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_displ_grids(self):
        X_ija = self.X_ija
        delta_u_rot_ija = self.delta_u0_ul_ija
        rot_Xu_ija = self.rot_Xu_ija
        rot_vect_u_nija = np.array([X_ija, rot_Xu_ija])
        rot_vect_u_anij = np.einsum('nija->anij', rot_vect_u_nija)
        rot_vect_u_anp = rot_vect_u_anij.reshape(2, 2, -1)
        perp_u_aij = np.array([delta_u_rot_ija[..., 1], -delta_u_rot_ija[..., 0]])
        perp_u_ija = np.einsum('aij->ija', perp_u_aij)
        perp_Xu_ija = X_ija + perp_u_ija * self.U_factor
        perp_vect_u_nija = np.array([X_ija, perp_Xu_ija])
        perp_vect_u_anij = np.einsum('nija->anij', perp_vect_u_nija)
        perp_vect_u_anp = perp_vect_u_anij.reshape(2, 2, -1)
        return perp_u_ija, rot_vect_u_anp, perp_vect_u_anp

    def update_plot(self, axes):
        ax = axes

        if self.show_init:
            XU0_aij = np.einsum('ija->aij', self.X_ija + self.delta_u0_ul_ija)
            ax.scatter(*XU0_aij.reshape(2,-1), s=15, marker='o', color='darkgray')

        _, rot_vect_u_anp, perp_vect_u_anp = self.displ_grids

        ax.scatter(*rot_vect_u_anp[:,-1,:], s=15, marker='o', color='silver')
        if self.show_rot:
            ax.plot(*rot_vect_u_anp, color='silver', linewidth=0.5);

        if self.show_perp:
            ax.plot(*perp_vect_u_anp, color='green', linewidth=0.5);
        ax.axis('equal');

        y_ref_ja = self.rot_Xu_ija[self.y_ref_i, self.y_ref_j_min:self.y_ref_j_max]
        ax.scatter(*y_ref_ja.T, s=20, color='green')
