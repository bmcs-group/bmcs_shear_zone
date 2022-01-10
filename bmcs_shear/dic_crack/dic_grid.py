
import bmcs_utils.api as bu
import traits.api as tr
from os.path import join, expanduser
import os
import numpy as np
import pandas as pd
from bmcs_shear.beam_design import RCBeamDesign

def convert_to_bool(str_bool):
    '''helper method for the parsing of input file with boalean values
    given as strings 'True' and 'False'
    '''
    value_map = {'True': True,
                 'False': False,
                 '1' : True,
                 '0' : False,
                 'true' : True,
                 'false' : False}
    return value_map[str_bool]


class DICGrid(bu.Model):

    tree = ['sz_bd']

    sz_bd = tr.Instance(RCBeamDesign ,())
    '''Beam design object provides geometrical data and material data.
    '''

    name = 'DIC grid history'

    dir_name = bu.Str('<unnamed>', ALG=True)
    def _dir_name_change(self):
        self.name = 'DIC grid %s' % self.name
        self._set_grid_params()

    grid_param_file_name = bu.Str('grid_params.txt', ALG=True)

    n_x = tr.Property(bu.Int, depends_on='state_changed')
    @tr.cached_property
    def _get_n_x(self):
        return self.grid_params['n_x']

    n_y = tr.Property(bu.Int, depends_on='state_changed')
    @tr.cached_property
    def _get_n_y(self):
        return self.grid_params['n_y']

    d_x = tr.Property(bu.Int, depends_on='state_changed')
    @tr.cached_property
    def _get_d_x(self):
        return self.grid_params['d_x']

    d_y = tr.Property(bu.Int, depends_on='state_changed')
    @tr.cached_property
    def _get_d_y(self):
        return self.grid_params['d_y']

    column_first_enum = tr.Property(bu.Bool, depends_on='state_changed')
    @tr.cached_property
    def _get_column_first_enum(self):
        return self.grid_params['column_first_enum']

    top_down_enum = tr.Property(bu.Bool, depends_on='state_changed')
    @tr.cached_property
    def _get_top_down_enum(self):
        return self.grid_params['top_down_enum']

    end_t = bu.Int(-1, ALG=True)
    start_t = bu.Int(0, ALG=True)

    U_factor = bu.Float(100, ALG=True)

    L = bu.Float(10, MAT=True)

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
        bu.Item('column_first_enum'),
        bu.Item('top_down_enum'),
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

    grid_param_file = tr.Property
    '''File containing the parameters of the grid'''
    def _get_grid_param_file(self):
        return join(self.dic_data_dir, self.grid_param_file_name)

    param_types = {'n_x' : int,
                   'n_y' : int,
                   'd_x' : int,
                   'd_y' : int,
                   'column_first_enum' : convert_to_bool,
                   'top_down_enum' : convert_to_bool}

    grid_params = tr.Property(depends_on='dir_name')
    def _get_grid_params(self):
        params_str = {}
        f = open(self.grid_param_file)
        data = f.readlines()
        for line in data:
            # parse input, assign values to variables
            key, value = line.split(":")
            params_str[key.strip()] = value.strip()
        f.close()
        # convert the strings to the paramater types specified in the param_types table
        params = { key : type_(params_str[key]) for key, type_ in self.param_types.items()  }
        return params

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
        if self.column_first_enum:
            U_tija = U_tpa.reshape(n_t, n_x, n_y, 2)  # for numbering from top right to bottom right
        else:
            U_tjia = U_tpa.reshape(n_t, n_y, n_x, 2) # for numbering from bottom right to left
            U_tija = np.einsum('tjia->tija', U_tjia)
        if self.top_down_enum:
            return U_tija[:,:,::-1,:]
        else:
            return U_tija


    n_t = tr.Property(depends_on='state_changed')
    '''Read the displacement data from the individual csv files'''
    @tr.cached_property
    def _get_n_t(self):
        return len(self.U_tija)

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
        ax_load.set_ylabel(r'$F$ [kN]')
        ax_load.set_xlabel(r'$w$ [mm]')

        max_deflection = np.max(deflection)
        argmax_load_idx = np.argmax(load)
        # define an interpolation function
        argmax_load_dic_idx = np.argmax(self.load_levels)

        # show available load levels
        F_levels = self.load_levels[:argmax_load_dic_idx]
        w_levels = np.interp(F_levels, load[:argmax_load_idx], deflection[:argmax_load_idx])
        ax_load.plot(w_levels, F_levels, 'o', markersize=3, color='orange')

        # show the current load marker
        F_idx = self.end_t
        F_level = self.load_levels[F_idx]
        if F_idx < argmax_load_dic_idx:
            w_level = np.interp(F_level, load[:argmax_load_idx], deflection[:argmax_load_idx])
            ax_load.plot(w_level, F_level, marker='o',
                         markersize=6, color='green')

        # annotate the maximum load level
        max_load = load[argmax_load_idx]
        argmax_deflection = deflection[argmax_load_idx]
        ax_load.annotate(f'[$F_\max=${max_load:.1f}, w={argmax_deflection:.2f}]',
                    xy=(argmax_deflection, max_load), xycoords='data',
                    xytext=(0.05, 0.95), textcoords='axes fraction',
                    # arrowprops=dict(arrowstyle="->", connectionstyle="arc3",
                    #                 facecolor='blue'),
                    horizontalalignment='left', verticalalignment='top',
                    )
