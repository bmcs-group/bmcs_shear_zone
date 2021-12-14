
import bmcs_utils.api as bu
import traits.api as tr
from os.path import join, expanduser
import os
import numpy as np
import pandas as pd

class DICGrid(bu.Model):

    name = 'DIC grid history'
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
        bu.Item('end_t', readonly=True),
        bu.Item('U_factor'),
        time_editor=bu.HistoryEditor(
            var='t'
        )
    )

    L_x = tr.Property
    '''Width of the domain'''
    def _get_L_x(self):
        return self.d_x * (self.n_x-1)

    L_y = tr.Property
    '''Height of the domain'''
    def _get_L_y(self):
        return self.d_y * (self.n_y-1)

    data_dir = tr.Property
    '''Directory with the data'''
    def _get_data_dir(self):
        home_dir = expanduser('~')
        data_dir = join(home_dir, 'simdb', 'data', 'shear_zone', self.dir_name)
        return data_dir

    dic_data_dir = tr.Property
    '''Directory with the DIC data'''
    def _get_dic_data_dir(self):
        return join(self.data_dir, 'dic_data')

    ld_data_dir = tr.Property
    '''Directory with the load deflection data'''
    def _get_ld_data_dir(self):
        return join(self.data_dir, 'load_deflection')

    files = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_files(self):
        return [join(self.dic_data_dir, each)
         for each in sorted(os.listdir(self.dic_data_dir))
         if each.endswith('.csv')]

    load_levels = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_load_levels(self):
        return np.array([float(os.path.basename(file_name).split('_')[-2])
                         for file_name in self.files ], dtype=np.float_ )

    grid_column_first = bu.Bool(True)

    U_tija = tr.Property(depends_on='state_changed')
    '''Read the displacement data from the individual csv files'''
    @tr.cached_property
    def _get_U_tija(self):
        files = self.files
        U_tpa = np.array([
            np.loadtxt(csv_file, dtype=float,
                       skiprows=1, delimiter=',', usecols=(2,3), unpack=False)
            for csv_file in files
        ], dtype=np.float_)
        n_t, n_e, n_a = U_tpa.shape # get the dimensions of the time and entry dimensions
        n_x, n_y = self.n_x, self.n_y
        if self.grid_column_first:
            U_tija = U_tpa.reshape(n_t, n_x, n_y, 2)  # for numbering from top right to bottom right
        else:
            U_tjia = U_tpa.reshape(n_t, n_y, n_x, 2) # for numbering from bottom right to left
            U_tija = np.einsum('tjia->tija', U_tjia)
        if self.grid_number_vertical:
            return U_tija[:,:,::-1,:]
        else:
            return U_tija


    n_t = tr.Property(depends_on='state_changed')
    '''Read the displacement data from the individual csv files'''
    @tr.cached_property
    def _get_n_t(self):
        return len(self.U_tija)

    grid_number_vertical = bu.Bool(True)

    X_ija = tr.Property(depends_on='state_changed')
    '''Read the displacement data from the individual csv files'''
    @tr.cached_property
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
        return self.U_tija[self.end_t] - self.U_tija[self.start_t]

    ld_file_name = tr.Str('load_deflection.csv')
    
    ld_values = tr.Property(depends_on='state_changed')
    '''Read the load displacement values from the individual csv files from the test'''

    @tr.cached_property
    def _get_ld_values(self):
        ld_file = join(self.ld_data_dir, self.ld_file_name)
        ld_values = np.array(pd.read_csv(ld_file, decimal=",", skiprows=1, delimiter=None), dtype=np.float_)
        return ld_values

    def subplots(self, fig):
        return fig.subplots(1,2)

    current_load = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_current_load(self):
        return self.load_levels[self.end_t]

    def update_plot(self, axes):
        ax_u, ax_load = axes
        self.plot_grid(ax_u)
        self.plot_load_deflection(ax_load)

    def plot_grid(self, ax_u):
        XU_aij = np.einsum('ija->aij', self.X_ija + self.U_ija * self.U_factor)
        ax_u.scatter(*XU_aij.reshape(2, -1), s=15, marker='o', color='darkgray')
        ax_u.axis('equal')

    def plot_load_deflection(self, ax_load):
        deflection = self.ld_values[::50, 2]
        load = -self.ld_values[::50,1]
        ax_load.plot(deflection, load, color='black')
        max_deflection = np.max(deflection)
        load_level = self.current_load
        ax_load.plot([0, max_deflection], [load_level, load_level], color='green', lw=2)
