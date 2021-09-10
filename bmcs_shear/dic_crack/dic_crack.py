
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
    show_perp = bu.Bool(True, ALG=True)

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

    displ_grids = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_displ_grids(self):
        delta_u_ija = self.u_tija[self.end_t] - self.u_tija[self.start_t]
        # delta_u_ija = u1_tija[1] - u1_tija[0]
        avg_a = np.average(delta_u_ija, axis=(0, 1))
        u_rot_ija = delta_u_ija - avg_a[np.newaxis, np.newaxis, :]
        rot_Xu_ija = self.X_ija + u_rot_ija * self.U_factor
        rot_vect_u_nija = np.array([self.X_ija, rot_Xu_ija])
        rot_vect_u_anij = np.einsum('nija->anij', rot_vect_u_nija)
        rot_vect_u_anp = rot_vect_u_anij.reshape(2, 2, -1)
        perp_u_aij = np.array([u_rot_ija[..., 1], -u_rot_ija[..., 0]])
        perp_u_ija = np.einsum('aij->ija', perp_u_aij)
        perp_Xu_ija = self.X_ija + perp_u_ija * self.U_factor
        perp_vect_u_nija = np.array([self.X_ija, perp_Xu_ija])
        perp_vect_u_anij = np.einsum('nija->anij', perp_vect_u_nija)
        perp_vect_u_anp = perp_vect_u_anij.reshape(2, 2, -1)
        # perp_vect_u_anp
        return perp_u_ija, rot_vect_u_anp, perp_vect_u_anp

    def update_plot(self, axes):
        ax = axes
        _, rot_vect_u_anp, perp_vect_u_anp = self.displ_grids
        if self.show_rot:
            ax.plot(*rot_vect_u_anp, color='blue', linewidth=0.5);

        if self.show_perp:
            ax.plot(*perp_vect_u_anp, color='green', linewidth=0.5);
        ax.axis('equal');

