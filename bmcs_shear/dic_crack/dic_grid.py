
import bmcs_utils.api as bu
import traits.api as tr
from os.path import join, expanduser
import os
import numpy as np

class DICGrid(bu.Model):

    name = 'DiC input'
    dir_name = bu.Str('<unnamed>', ALG=True)

    n_x = bu.Int(20, ALG=True)
    n_y = bu.Int(20, ALG=True)

    d_x = bu.Float(14, ALG=True)
    d_y = bu.Float(14, ALG=True)

    end_t = bu.Int(-1, ALG=True)
    start_t = bu.Int(0, ALG=True)

    U_factor = bu.Float(100, ALG=True)

    t = bu.Float(1, ALG=True)
    def _t_changed(self):
        d_t = (1 / self.n_t)
        self.end_t = int( (self.n_t-1) * (self.t + d_t/2))

    ipw_view = bu.View(
        bu.Item('n_x'),
        bu.Item('n_y'),
        bu.Item('d_x'),
        bu.Item('d_y'),
        bu.Item('start_t', readonly=True),
        bu.Item('end_t', readonly=True),
        bu.Item('U_factor'),
        time_editor=bu.HistoryEditor(
            var='t'
        )
    )

    L_x = tr.Property
    def _get_L_x(self):
        return self.d_x * (self.n_x-1)

    L_y = tr.Property
    def _get_L_y(self):
        return self.d_y * (self.n_y-1)

    data_dir = tr.Property
    def _get_data_dir(self):
        home_dir = expanduser('~')
        data_dir = join(home_dir, 'simdb', 'data', 'shear_zone', self.dir_name)
        return data_dir

    grid_column_first = bu.Bool(True)

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
        n_t, n_e, n_a = u_tpa.shape # get the dimensions of the time and entry dimensions
        n_x, n_y = self.n_x, self.n_y
        if self.grid_column_first:
            u_tjia = u_tpa.reshape(n_t, n_x, n_y, 2)  # for numbering from top right to bottom right
            u_val = u_tjia
            #u_tjia = u_tpa.reshape(n_t, n_y, n_x, 2) # for numbering from bottom right to left
        else:
            u_tjia = u_tpa.reshape(n_t, n_y, n_x, 2) # for numbering from bottom right to left
            u_tija = np.einsum('tjia->tija', u_tjia)
            u_val = u_tija
        #u_tjia = u_tpa.reshape(n_t, n_x, n_y, 2) # for numbering from top right to bottom right
        return  u_val #u_tjia #u_tjia #

    n_t = tr.Property(depends_on='state_changed')
    '''Read the displacement data from the individual csv files'''
    @tr.cached_property
    def _get_n_t(self):
        return len(self.u_tija)


    #grid_x_slice = tr.Any(slice(None,None,-1))
    #grid_y_slice = tr.Any(slice(None,None,-1))

    grid_number_vertical = bu.Bool(True)

    X_ija = tr.Property(depends_on='state_changed')
    '''Read the displacement data from the individual csv files'''
    tr.cached_property
    def _get_X_ija(self):
        n_x, n_y = self.n_x, self.n_y
        #x_range = np.arange(n_x)[self.grid_x_slice] * self.d_x
        x_range = np.arange(n_x)[::-1] * self.d_x
        #y_range = np.arange(n_y) * self.d_y #for beams having grid number from bottom right to left
        if self.grid_number_vertical:
            y_range = np.arange(n_y)[::-1] * self.d_y #for beams having grid number from right top to bottom
        else:
            y_range = np.arange(n_y) * self.d_y #for beams having grid number from bottom right to left
        #y_range = np.arange(n_y)[self.grid_y_slice] * self.d_y
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

    def update_plot(self, axes):
        ax = axes
        XU0_aij = np.einsum('ija->aij', self.X_ija + self.U_ija * self.U_factor)
        ax.scatter(*XU0_aij.reshape(2,-1), s=15, marker='o', color='darkgray')
        ax.axis('equal');
