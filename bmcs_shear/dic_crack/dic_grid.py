
import bmcs_utils.api as bu
import traits.api as tr
from os.path import join, expanduser
import os
import numpy as np
import pandas as pd
from bmcs_shear.beam_design import RCBeamDesign
from bmcs_shear.matmod import CrackBridgeAdv

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

    name = 'DIC grid history'

    dir_name = bu.Str('<unnamed>', ALG=True)
    def _dir_name_change(self):
        self.name = 'DIC grid %s' % self.name
        self._set_grid_params()

    grid_param_file_name = bu.Str('grid_params.txt', ALG=True)

    beam_param_file_name = bu.Str('beam_params.txt', ALG=True)

    beam_param_types = {'L' : float,
                        'B' : float,
                        'H' : float,
                        'n_s' : float,
                        'y_s' : float,
                        'd_s' : float}

    beam_param_file = tr.Property
    '''File containing the parameters of the beam'''
    def _get_beam_param_file(self):
        return join(self.data_dir, self.beam_param_file_name)

    # sz_bd = tr.Property(bu.Instance(RCBeamDesign), depends_on='dir_name')
    # '''Beam design object provides geometrical data and material data.
    # '''
    # @tr.cached_property
    # def _get_sz_bd(self):
    #     params_str = {}
    #     f = open(self.beam_param_file)
    #     data = f.readlines()
    #     for line in data:
    #         key, value = line.split(":")
    #         params_str[key.strip()] = value.strip()
    #     f.close()
    #     # convert the strings to the paramater types specified in the param_types table
    #     params = {key : type_(params_str[key]) for key, type_ in self.beam_param_types.items()}
    #     sz_bd = RCBeamDesign(**{key: params[key] for key in ['H', 'B', 'L']})
    #     sz_bd.Rectangle = True
    #     sz_bd.csl.add_layer(CrackBridgeAdv(z=params['y_s'], n=params['n_s'], d_s=params['d_s']))
    #     return sz_bd

    sz_bd = bu.Instance(RCBeamDesign)
    '''Beam design object provides geometrical data and material data.
    '''
    def _sz_bd_default(self):
        return RCBeamDesign()

    def read_beam_design(self):
        params_str = {}
        f = open(self.beam_param_file)
        data = f.readlines()
        for line in data:
            key, value = line.split(":")
            params_str[key.strip()] = value.strip()
        f.close()
        # convert the strings to the paramater types specified in the param_types table
        params = {key : type_(params_str[key]) for key, type_ in self.beam_param_types.items()}
        self.sz_bd.trait_set(**{key: params[key] for key in ['H', 'B', 'L']})
        self.sz_bd.Rectangle = True
        self.sz_bd.csl.add_layer(CrackBridgeAdv(z=params['y_s'], n=params['n_s'], d_s=params['d_s']))

    n_I = tr.Property(bu.Int, depends_on='state_changed')
    ''' Number of horizontal nodes of the DIC input displacement grid
    '''
    @tr.cached_property
    def _get_n_I(self):
        return self.grid_params['n_x']

    n_J = tr.Property(bu.Int, depends_on='state_changed')
    ''' Number of vertical nodes of the DIC input displacement grid
    '''
    @tr.cached_property
    def _get_n_J(self):
        return self.grid_params['n_y']

    d_x = tr.Property(bu.Float, depends_on='state_changed')
    ''' Horizontal spacing between nodes of the DIC input displacement grid
    '''
    @tr.cached_property
    def _get_d_x(self):
        return self.grid_params['d_x']

    d_y = tr.Property(bu.Float, depends_on='state_changed')
    ''' Vertical spacing between nodes of the DIC input displacement grid
    '''
    @tr.cached_property
    def _get_d_y(self):
        return self.grid_params['d_y']

    x_offset = tr.Property(bu.Float, depends_on='state_changed')
    ''' Horizontal offset of the DIC input displacement grid from the left
        boundary of the beam
    '''
    @tr.cached_property
    def _get_x_offset(self):
        return self.grid_params['x_offset']

    y_offset = tr.Property(bu.Float, depends_on='state_changed')
    ''' Vertical offset of the DIC input displacement grid from the bottom
        boundary of the beam
    '''
    @tr.cached_property
    def _get_y_offset(self):
        return self.grid_params['y_offset']

    column_first_enum = tr.Property(bu.Bool, depends_on='state_changed')
    @tr.cached_property
    def _get_column_first_enum(self):
        return self.grid_params['column_first_enum']

    top_down_enum = tr.Property(bu.Bool, depends_on='state_changed')
    @tr.cached_property
    def _get_top_down_enum(self):
        return self.grid_params['top_down_enum']

    T0 = bu.Int(0, ALG=True)

    T1 = bu.Int(-1, ALG=True)

    U_factor = bu.Float(100, ALG=True)

    L = bu.Float(10, MAT=True)

    t = bu.Float(1, ALG=True)

    def _t_changed(self):
        d_t = (1 / self.n_dic_T)
        self.T1 = int( (self.n_dic_T - 1) * (self.t + d_t/2))

    ipw_view = bu.View(
        bu.Item('n_I', readonly=True),
        bu.Item('n_J', readonly=True),
        bu.Item('d_x', readonly=True),
        bu.Item('d_y', readonly=True),
        bu.Item('x_offset', readonly=True),
        bu.Item('y_offset', readonly=True),
        bu.Item('T1', readonly=True),
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
        return self.d_x * (self.n_I-1)

    L_y = tr.Property
    '''Height of the domain'''
    def _get_L_y(self):
        return self.d_y * (self.n_J-1)

    X_frame = tr.Property
    '''Define the bottom left and top right corners'''
    def _get_X_frame(self):
        L_x, L_y = self.L_x, self.L_y
        x_offset, y_offset = self.x_offset, self.y_offset
        X_min, X_max = x_offset, L_x + x_offset
        Y_min, Y_max = y_offset, L_y + y_offset
        return X_min, Y_min, X_max, Y_max


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

    Fw_data_dir = tr.Property
    '''Directory with the load deflection data'''
    def _get_Fw_data_dir(self):
        return join(self.data_dir, 'load_deflection')

    grid_param_file = tr.Property
    '''File containing the parameters of the grid'''
    def _get_grid_param_file(self):
        return join(self.dic_data_dir, self.grid_param_file_name)

    grid_param_types = {'n_x' : int,
                       'n_y' : int,
                       'd_x' : float,
                       'd_y' : float,
                       'x_offset' : float,
                       'y_offset' : float,
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
        # convert the strings to the parameter types specified in the param_types table
        params = { key : type_(params_str[key]) for key, type_ in self.grid_param_types.items()  }
        return params

    files = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_files(self):
        return [join(self.dic_data_dir, each)
         for each in sorted(os.listdir(self.dic_data_dir))
         if each.endswith('.csv')]

    F_DIC_T = tr.Property(depends_on='state_changed')
    '''DIC load levels'''
    @tr.cached_property
    def _get_F_DIC_T(self):
        return np.array([float(os.path.basename(file_name).split('_')[-2])
                         for file_name in self.files ], dtype=np.float_ )

    argmax_F_dic_T = tr.Property(depends_on='state_changed')
    '''Time index of DIC snapshots at maximum load
    '''
    def _get_argmax_F_dic_T(self):
        return np.argmax(self.F_DIC_T)

    F_dic_T = tr.Property(depends_on='state_changed')
    '''Load levels of ascending DIC snapshots
    '''
    @tr.cached_property
    def _get_F_dic_T(self):
        return self.F_DIC_T[:self.argmax_F_dic_T]

    U_TIJa = tr.Property(depends_on='state_changed')
    '''Read the displacement data from the individual csv files'''
    @tr.cached_property
    def _get_U_TIJa(self):
        files = self.files
        # indexes: T - load level, P - point, a - dimension
        U_TPa = np.array([
            np.loadtxt(csv_file, dtype=float,
                       skiprows=1, delimiter=',', usecols=(2,3), unpack=False)
            for csv_file in files
        ], dtype=np.float_)
        n_T, n_e, n_a = U_TPa.shape # get the dimensions of the time and entry dimensions
        n_I, n_J = self.n_I, self.n_J
        if self.column_first_enum:
            U_TIJa = U_TPa.reshape(n_T, n_I, n_J, 2)  # for numbering from top right to bottom right
        else:
            U_TJIa = U_TPa.reshape(n_T, n_J, n_I, 2) # for numbering from bottom right to left
            U_TIJa = np.einsum('TJIa->TIJa', U_TJIa)
        if self.top_down_enum:
            return U_TIJa[:,::-1,::-1,:]
#            return U_TIJa[:,:,::-1,:]
        else:
            return U_TIJa[:,::-1,:,:]

    n_dic_T = tr.Property(depends_on='state_changed')
    '''Number of dic snapshots up to the maximum load'''
    @tr.cached_property
    def _get_n_dic_T(self):
        return self.argmax_F_dic_T

    def get_T_eta(self, eta = 0.9):
        '''Get the dic index correponding to the specified fraction
        of ultimate load.
        '''
        F = -self.Fw_T[::50,1]
        F_max = np.max(F)
        F_eta = eta * F_max
        F_dic_T = self.F_dic_T
        dic_T = np.arange(len(F_dic_T))
        T_eta = np.interp(F_eta, F_dic_T, dic_T)
        return int(T_eta)

    X_IJa = tr.Property(depends_on='state_changed')
    '''Coordinates of the DIC markers in the grid'''
    @tr.cached_property
    def _get_X_IJa(self):
        n_I, n_J = self.n_I, self.n_J
#        x_range = np.arange(n_I)[::-1] * self.d_x + self.x_offset
        x_range = np.arange(n_I) * self.d_x + self.x_offset
        y_range = np.arange(n_J) * self.d_y + self.y_offset
        y_IJ, x_IJ = np.meshgrid(y_range, x_range)
        X_aIJ = np.array([x_IJ, y_IJ])
        X_IJa = np.einsum('aIJ->IJa', X_aIJ)
        return X_IJa

    U_IJa = tr.Property(depends_on='state_changed')
    '''Total displacement at step T1 w.r.t. T0
    '''
    @tr.cached_property
    def _get_U_IJa(self):
        return self.U_TIJa[self.T1] - self.U_TIJa[self.T0]

    Fw_file_name = tr.Str('load_deflection.csv')
    '''Name of the file with the measured load deflection data
    '''
    
    Fw_T = tr.Property(depends_on='state_changed')
    '''Read the load displacement values from the individual 
    csv files from the test
    '''
    @tr.cached_property
    def _get_Fw_T(self):
        Fw_file = join(self.Fw_data_dir, self.Fw_file_name)
        Fw_T = np.array(pd.read_csv(Fw_file, decimal=",", skiprows=1, delimiter=None), dtype=np.float_)
        return Fw_T

    F_T1 = tr.Property(depends_on='state_changed')
    '''Current load
    '''
    @tr.cached_property
    def _get_F_T1(self):
        return self.F_DIC_T[self.T1]

    def plot_grid(self, ax_u):
        XU_aIJ = np.einsum('IJa->aIJ', self.X_IJa + self.U_IJa * self.U_factor)
        ax_u.scatter(*XU_aIJ.reshape(2, -1), s=15, marker='o', color='darkgray')
        ax_u.axis('equal')

    def plot_bounding_box(self, ax):
        X_Ca = self.X_IJa[(0, 0, -1, -1, 0), (0, -1, -1, 0, 0), :]
        X_iLa = np.array([X_Ca[:-1], X_Ca[1:]], dtype=np.float_)
        X_aiL = np.einsum('iLa->aiL', X_iLa)
        ax.plot(*X_aiL, color='black', lw=0.5)

    def plot_box_annotate(self, ax):
        X_Ca = self.X_IJa[(0, 0, -1, -1, 0), (0, -1, -1, 0, 0), :]
        X_iLa = np.array([X_Ca[:-1], X_Ca[1:]], dtype=np.float_)
        X_La = np.sum(X_iLa, axis=0) / 2
        x, y = X_La[0, :]
        ax.annotate(f'{self.L_y} mm',
                    xy=(x, y), xytext=(5, 0), xycoords='data',
                    textcoords='offset pixels',
                    horizontalalignment='left',
                    verticalalignment='center',
                    rotation=90
                    )
        x, y = X_La[1, :]
        ax.annotate(f'{self.L_x} mm',
                    xy=(x, y), xytext=(0, 1), xycoords='data',
                    textcoords='offset pixels',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    )
        x, y = X_Ca[1, :]
        ax.annotate(f'{self.dir_name}',
                    xy=(x, y), xytext=(-2, -2), xycoords='data',
                    textcoords='offset pixels',
                    horizontalalignment='right',
                    verticalalignment='top',
                    )

    def plot_load_deflection(self, ax_load):
        w = self.Fw_T[::50, 2]
        F = -self.Fw_T[::50,1]

        argmax_F_T = np.argmax(F)

        ax_load.plot(w[:argmax_F_T], F[:argmax_F_T], color='black')
        ax_load.set_ylabel(r'$F$ [kN]')
        ax_load.set_xlabel(r'$w$ [mm]')

        # plot the markers of dic levels
        w_dic_T = np.interp(self.F_dic_T, F[:argmax_F_T], w[:argmax_F_T])
        ax_load.plot(w_dic_T, self.F_dic_T, 'o', markersize=3, color='orange')

        # show the current load marker
        F_T1 = self.F_dic_T[self.T1]
        w_T1 = np.interp(F_T1, F[:argmax_F_T], w[:argmax_F_T])
        ax_load.plot(w_T1, F_T1, marker='o', markersize=6, color='green')

        # annotate the maximum load level
        max_F = F[argmax_F_T]
        argmax_F_w = w[argmax_F_T]
        ax_load.annotate(f'$F_\max=${max_F:.1f} kN, w={argmax_F_w:.2f} mm',
                    xy=(argmax_F_w, max_F), xycoords='data',
                    xytext=(0.05, 0.95), textcoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top',
                    )

    def subplots(self, fig):
        return fig.subplots(1,2)

    def update_plot(self, axes):
        ax_u, ax_load = axes
        self.plot_grid(ax_u)
        self.plot_load_deflection(ax_load)
