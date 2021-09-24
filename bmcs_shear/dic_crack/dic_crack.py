
import bmcs_utils.api as bu
import traits.api as tr
from os.path import join, expanduser
import os
import numpy as np

class DICCrack(bu.Model):

    name = 'DiC input'
    dir_name = bu.Str('<unnamed>', ALG=True)

    n_x = bu.Int(20, ALG=True)
    n_y = bu.Int(20, ALG=True)

    d_x = bu.Float(14, ALG=True)
    d_y = bu.Float(14, ALG=True)

    end_t = bu.Int(-1, ALG=True)
    start_t = bu.Int(-2, ALG=True)

    U_factor = bu.Float(100, ALG=True)

    show_rot = bu.Bool(True, ALG=True)
    show_perp = bu.Bool(False, ALG=True)
    show_init = bu.Bool(False, ALG=True)

    ipw_view = bu.View(
        bu.Item('n_x'),
        bu.Item('n_y'),
        bu.Item('d_x'),
        bu.Item('d_y'),
        bu.Item('start_t'),
        bu.Item('end_t'),
        bu.Item('U_factor'),
        bu.Item('show_rot'),
        bu.Item('show_perp'),
        bu.Item('show_init'),
    )

    data_dir = tr.Property
    def _get_data_dir(self):
        home_dir = expanduser('~')
        data_dir = join(home_dir, 'simdb', 'data', 'shear_zone', self.dir_name)
        return data_dir

    u_tija = tr.Property(depends_on='state_changed')
    '''Read the displacement data from the individual csv files'''
    tr.cached_property
    def _get_u_tija(self):
        files = [join(self.data_dir, each)
               for each in sorted(os.listdir(self.data_dir))
               if each.endswith('.csv')]
        u_tpa = np.array([
            np.loadtxt(csv_file, dtype=float,
                       skiprows=1, delimiter=',', usecols=(2,3), unpack=False)
            for csv_file in files
        ], dtype=np.float_)
        n_t, n_e, n_a = u_tpa.shape  # get the dimensions of the time and entry dimensions
        n_x, n_y = self.n_x, self.n_y
        u_tjia = u_tpa.reshape(n_t, n_y, n_x, 2)
        u_tija = np.einsum('tjia->tija', u_tjia)
        return u_tija

    X_ija = tr.Property(depends_on='state_changed')
    '''Read the displacement data from the individual csv files'''
    tr.cached_property
    def _get_X_ija(self):
        n_x, n_y = self.n_x, self.n_y
        x_range = np.arange(n_x)[::-1] * self.d_x
        y_range = np.arange(n_y) * self.d_y
        y_ij, x_ij = np.meshgrid(y_range, x_range)
        X_aij = np.array([x_ij, y_ij])
        X_ija = np.einsum('aij->ija', X_aij)
        return X_ija

    U_ija = tr.Property(depends_on='state_changed')
    '''Total displacement
    '''
    @tr.cached_property
    def _get_U_ija(self):
        return self.u_tija[self.end_t] - self.u_tija[0]

    delta_u_rot_ija = tr.Property(depends_on='state_changed')
    '''Displacement increment with subtracted rigid body motion
    within that increment.
    '''
    @tr.cached_property
    def _get_delta_u_rot_ija(self):
        delta_u_ija = self.u_tija[self.end_t] - self.u_tija[self.start_t]
        avg_a = np.average(delta_u_ija, axis=(0, 1))
        delta_u_rot_ija = delta_u_ija - avg_a[np.newaxis, np.newaxis, :]
        return delta_u_rot_ija

    delta_u_ul_ija = tr.Property(depends_on='state_changed')
    '''Displacement increment relative to upper left corner.
    '''
    @tr.cached_property
    def _get_delta_u_ul_ija(self):
        delta_u_ija = self.u_tija[self.end_t] - self.u_tija[self.start_t]
        #print('delta_u_ija', delta_u_ija)
        u_11a = delta_u_ija[-1:,:1,:]
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
        sin_delta_alpha = np.average(d_u_ul_1j1 / d_x_i0)
        return np.arcsin(sin_delta_alpha)

    T_ab = tr.Property(depends_on='state_changed')
    '''Rotation of the reference boundary line.
    '''
    @tr.cached_property
    def _get_T_ab(self):
        delta_alpha = self.delta_alpha
        sa, ca = np.sin(delta_alpha), np.cos(delta_alpha)
        return np.array([[ca,-sa],
                         [sa,ca]])

    delta_u0_ul_ija = tr.Property(depends_on='state_changed')
    '''Displacement increment relative to the reference line.
    '''
    @tr.cached_property
    def _get_delta_u0_ul_ija(self):
        XU_ija = self.X_ija + self.delta_u_ul_ija
        XU_pull_ija = XU_ija - XU_ija[-1:,:1,:]
        XU0_ija = np.einsum('ba,...a->...b', self.T_ab, XU_pull_ija)
        XU_push_ija = XU0_ija + XU_ija[-1:,:1,:]
        return XU_push_ija - self.X_ija

    displ_grids = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_displ_grids(self):
        X_ija = self.X_ija
        delta_u_rot_ija = self.delta_u0_ul_ija
        rot_Xu_ija = X_ija + delta_u_rot_ija * self.U_factor
        rot_vect_u_nija = np.array([X_ija, rot_Xu_ija])
        rot_vect_u_anij = np.einsum('nija->anij', rot_vect_u_nija)
        rot_vect_u_anp = rot_vect_u_anij.reshape(2, 2, -1)
        perp_u_aij = np.array([delta_u_rot_ija[..., 1], -delta_u_rot_ija[..., 0]])
        perp_u_ija = np.einsum('aij->ija', perp_u_aij)
        perp_Xu_ija = X_ija + perp_u_ija * self.U_factor
        perp_vect_u_nija = np.array([X_ija, perp_Xu_ija])
        perp_vect_u_anij = np.einsum('nija->anij', perp_vect_u_nija)
        perp_vect_u_anp = perp_vect_u_anij.reshape(2, 2, -1)
        #print('perp_vect_u_anp', perp_vect_u_anp)
        # perp_vect_u_anp
        return perp_u_ija, rot_vect_u_anp, perp_vect_u_anp

    def update_plot(self, axes):
        ax = axes
        # XU_aij = np.einsum('ija->aij', self.X_ija + self.delta_u_ul_ija)
        # ax.plot(*XU_aij.reshape(2,-1), 'o', color='black')
        if self.show_init:
            XU0_aij = np.einsum('ija->aij', self.X_ija + self.delta_u0_ul_ija)
            ax.plot(*XU0_aij.reshape(2,-1), 'o', color='blue')

        _, rot_vect_u_anp, perp_vect_u_anp = self.displ_grids

        ax.plot(*rot_vect_u_anp[:,-1,:], 'o', color='grey')
        if self.show_rot:
            ax.plot(*rot_vect_u_anp, color='grey', linewidth=0.5);

        if self.show_perp:
            ax.plot(*perp_vect_u_anp, color='green', linewidth=0.5);
        ax.axis('equal');

