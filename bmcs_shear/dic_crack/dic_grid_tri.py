
import bmcs_utils.api as bu
import traits.api as tr
from os.path import join, expanduser
import os
import numpy as np
import pandas as pd
from bmcs_shear.beam_design import RCBeamDesign
from bmcs_shear.matmod import CrackBridgeAdv
from scipy.spatial import Delaunay
import scipy.interpolate
from .i_dic_grid import IDICGrid

def convert_to_bool(str_bool):
    """Helper method for the parsing of input file with boalean values
    given as strings 'True' and 'False'
    """
    value_map = {'True': True,
                 'False': False,
                 '1' : True,
                 '0' : False,
                 'true' : True,
                 'false' : False}
    return value_map[str_bool]

@tr.provides(IDICGrid)
class DICGridTri(bu.Model):
    """
    History of displacment grids imported from the DIC measurement.
    """
    depends_on = ['sz_bd']
    tree = ['sz_bd']

    name = 'DIC grid history'

    dir_name = bu.Str('<unnamed>', ALG=True)
    """Directory name containing the test data.
    """
    def _dir_name_change(self):
        self.name = 'DIC grid %s' % self.name
        self._set_dic_params()

    dic_param_file_name = bu.Str('dic_params.txt', ALG=True)
    """Default name of the file with the parameters of the grid.
    """

    beam_param_file_name = bu.Str('beam_params.txt', ALG=True)
    """Default name of the file with the parameters of the beam.
    L_right - length of the beam at the side with DIC measurement
    L_left - length of the beam at th side without DIC measurement  
    """

    beam_param_types = {'L_right' : float,
                        'L_left' : float,
                        'B' : float,
                        'H' : float,
                        'n_s' : float,
                        'y_s' : float,
                        'd_s' : float}
    """Parameters of the test specifying the length, width and depth of the beam.
    """

    beam_param_file = tr.Property
    """File containing the parameters of the beam
    """
    def _get_beam_param_file(self):
        return join(self.data_dir, self.beam_param_file_name)

    L_left = bu.Float(1, ALG=True)
    L_right = bu.Float(2, ALG=True)

    sz_bd = bu.Instance(RCBeamDesign)
    """Beam design object provides geometrical data and material data.
    """
    def _sz_bd_default(self):
        return RCBeamDesign()

    def read_beam_design(self):
        """Read the file with the input data using the input configuration
        including the beam param types.
        """
        params_str = {}
        f = open(self.beam_param_file)
        data = f.readlines()
        for line in data:
            key, value = line.split(":")
            params_str[key.strip()] = value.strip()
        f.close()
        # convert the strings to the paramater types specified in the param_types table
        params = {key : type_(params_str[key]) for key, type_ in self.beam_param_types.items()}
        self.sz_bd.trait_set(**{key: params[key] for key in ['H', 'B', 'L_right', 'L_left']})
        self.sz_bd.L = self.sz_bd.L_right
        self.sz_bd.Rectangle = True
        self.sz_bd.csl.add_layer(CrackBridgeAdv(z=params['y_s'], n=params['n_s'], d_s=params['d_s']))

    n_I = tr.Property(bu.Int, depends_on='state_changed')
    """Number of horizontal nodes of the DIC input displacement grid.
    """
    @tr.cached_property
    def _get_n_I(self):
        return int(self.L_x / self.d_x)

    n_J = tr.Property(bu.Int, depends_on='state_changed')
    """Number of vertical nodes of the DIC input displacement grid
    """
    @tr.cached_property
    def _get_n_J(self):
        return int(self.L_y / self.d_y)

    d_x = bu.Float(5 , ALG=True)
    """Horizontal spacing between nodes of the DIC input displacement grid.
    """

    d_y = bu.Float(5 , ALG=True)
    """Vertical spacing between nodes of the DIC input displacement grid.
    """

    x_offset = tr.Property(bu.Float, depends_on='state_changed')
    """Horizontal offset of the DIC input displacement grid from the left
    boundary of the beam.
    """
    @tr.cached_property
    def _get_x_offset(self):
        return self.dic_params['x_offset']

    y_offset = tr.Property(bu.Float, depends_on='state_changed')
    """ Vertical offset of the DIC input displacement grid from the bottom
        boundary of the beam
    """
    @tr.cached_property
    def _get_y_offset(self):
        return self.dic_params['y_offset']

    pad_t = tr.Property(bu.Float, depends_on='state_changed')
    """ Top pad width
    """
    @tr.cached_property
    def _get_pad_t(self):
        return self.dic_params['pad_t']

    pad_b = tr.Property(bu.Float, depends_on='state_changed')
    """ Bottom pad width
    """
    @tr.cached_property
    def _get_pad_b(self):
        return self.dic_params['pad_b']

    pad_l = tr.Property(bu.Float, depends_on='state_changed')
    """ Left pad width
    """
    @tr.cached_property
    def _get_pad_l(self):
        return self.dic_params['pad_l']

    pad_r = tr.Property(bu.Float, depends_on='state_changed')
    """ Right pad width
    """
    @tr.cached_property
    def _get_pad_r(self):
        return self.dic_params['pad_r']

    T0 = bu.Int(0, ALG=True)

    T_t = bu.Int(-1, ALG=True)

    U_factor = bu.Float(100, ALG=True)

    L = bu.Float(10, MAT=True)

    t = bu.Float(1, ALG=True)

    def _t_changed(self):
        d_t = (1 / self.n_T)
        #self.T_t = self.get_T_t(self.t)
        self.T_t = int( (self.n_T - 1) * (self.t + d_t/2))

    t_dic_T = tr.Property(depends_on='state_changed')
    """Time steps of ascending DIC snapshots
    """
    @tr.cached_property
    def _get_t_dic_T(self):
        return np.linspace(0, 1, self.n_T)
        #return self.F_dic_T / self.F_dic_T[-1]

    ipw_view = bu.View(
        bu.Item('pad_t', readonly=True),
        bu.Item('pad_b', readonly=True),
        bu.Item('pad_l', readonly=True),
        bu.Item('pad_r', readonly=True),
        bu.Item('n_I', readonly=True),
        bu.Item('n_J', readonly=True),
        bu.Item('d_x'),
        bu.Item('d_y'),
        bu.Item('x_offset', readonly=True),
        bu.Item('y_offset', readonly=True),
        bu.Item('T_t', readonly=True),
        bu.Item('U_factor'),
        time_editor=bu.HistoryEditor(
            var='t'
        )
    )

    X_outer_frame = tr.Property(depends_on='state_changed')
    def _get_X_outer_frame(self):
        return np.min(self.X_Qa, axis=0), np.max(self.X_Qa, axis=0)

    L_x = tr.Property
    """Width of the domain"""
    def _get_L_x(self):
        X_min, X_max = self.X_outer_frame
        return X_max[0] - X_min[0]

    L_y = tr.Property
    """Height of the domain"""
    def _get_L_y(self):
        X_min, X_max = self.X_outer_frame
        return X_max[1] - X_min[1]

    X_frame = tr.Property
    """Define the bottom left and top right corners"""
    def _get_X_frame(self):
        L_x, L_y = self.L_x, self.L_y
        x_offset, y_offset = self.x_offset, self.y_offset
        X_min, X_max = x_offset, L_x + x_offset
        Y_min, Y_max = y_offset, L_y + y_offset
        return X_min, Y_min, X_max, Y_max


    data_dir = tr.Property
    """Directory with the data"""
    def _get_data_dir(self):
        home_dir = expanduser('~')
        data_dir = join(home_dir, 'simdb', 'data', 'shear_zone', self.dir_name)
        return data_dir

    dic_data_dir = tr.Property
    """Directory with the DIC data"""
    def _get_dic_data_dir(self):
        return join(self.data_dir, 'dic_point_data')

    Fw_data_dir = tr.Property
    """Directory with the load deflection data"""
    def _get_Fw_data_dir(self):
        return join(self.data_dir, 'load_deflection')

    dic_param_file = tr.Property
    """File containing the parameters of the grid"""
    def _get_dic_param_file(self):
        return join(self.dic_data_dir, self.dic_param_file_name)

    dic_param_types = {'x_offset' : float,
                       'y_offset' : float,
                       'pad_t': float,
                       'pad_b' : float,
                       'pad_l' : float,
                       'pad_r' : float,
                       }

    dic_params = tr.Property(depends_on='dir_name')
    def _get_dic_params(self):
        params_str = {}
        f = open(self.dic_param_file)
        data = f.readlines()
        for line in data:
            # parse input, assign values to variables
            key, value = line.split(":")
            params_str[key.strip()] = value.strip()
        f.close()
        # convert the strings to the parameter types specified in the param_types table
        params = { key : type_(params_str[key]) for key, type_ in self.dic_param_types.items()  }
        return params

    all_files = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_all_files(self):
        return np.array([join(self.dic_data_dir, each)
         for each in sorted(os.listdir(self.dic_data_dir))
         if each.endswith('.csv')])

    asc_F_files = tr.Property(depends_on='state_changed')
    """DIC files with ascending levels of force

    T - index of the time step
    t - slider from 0 to 1 where 1 is the ultimate state
    ts - true time stamp of the DIC figure
    """
    @tr.cached_property
    def _get_asc_F_files(self):
        ts_DIC_T = np.array([float(os.path.basename(file_name).replace('.csv','').split('_')[-2])
                         for file_name in self.all_files ], dtype=np.float_ )
        F_DIC_T = self.F_ts(ts_DIC_T)
        dF_ST = F_DIC_T[np.newaxis, :] - F_DIC_T[:, np.newaxis]
        dic_asc_T = np.unique(np.argmax(np.triu(dF_ST, 0) > 0, axis=1))
        return F_DIC_T[dic_asc_T], self.all_files[dic_asc_T]

    F_dic_T = tr.Property(depends_on='state_changed')
    """Load levels of ascending DIC snapshots
    """
    @tr.cached_property
    def _get_F_dic_T(self):
        F_dic_T, _ = self.asc_F_files
        return F_dic_T

    def F_ts(self, ts):
        ts_ext = self.Fw_T[::50,0]
        F_ext = -self.Fw_T[::50,1]
        argmax_F_T = np.argmax(F_ext)
        return np.interp(ts, F_ext[:argmax_F_T], ts_ext[:argmax_F_T])

    w_T = tr.Property(depends_on='state_changed')
    """Displacement levels of ascending DIC snapshots
    """
    @tr.cached_property
    def _get_w_T(self):
        w = self.Fw_T[::50,2]
        F = -self.Fw_T[::50,1]
        argmax_F_T = np.argmax(F)
        return np.interp(self.F_dic_T, F[:argmax_F_T], w[:argmax_F_T])

    w_dic_T = tr.Property(depends_on='state_changed')
    """Displacement levels of ascending DIC snapshots
    """
    @tr.cached_property
    def _get_w_dic_T(self):
        return -self.U_TIJa[:len(self.F_dic_T),0,-1,1]

    X_Qa__U_TQa = tr.Property(depends_on='state_changed')
    """Read the displacement data from the individual csv files"""
    @tr.cached_property
    def _get_X_Qa__U_TQa(self):
        _, pxyz_files = self.asc_F_files
        # pxyz_files = [op.join(input_dir, each)
        #               for each in sorted(os.listdir(files))
        #               if each.endswith('.csv')]
        pxyz_list = [
            np.loadtxt(csv_file, dtype=np.float_,
                       skiprows=6, delimiter=';')
            for csv_file in pxyz_files
        ]
        # Identify the points that are included in all time steps.
        P_list = [np.array(pxyz[:, 0], dtype=np.int_)
                  for pxyz in pxyz_list
                  ]
        # Maximum number of points ocurring in one of the time steps to allocate the space
        max_n_P = np.max(np.array([np.max(P_) for P_ in P_list])) + 1
        P_Q = P_list[0]
        for P_next in P_list[1:]:
            P_Q = np.intersect1d(P_Q, P_next)
        # Define the initial configuration
        X_TPa = np.zeros((self.n_T, max_n_P, 3), dtype=np.float_)
        for T in range(self.n_T):
            X_TPa[T, P_list[T]] = pxyz_list[T][:, 1:]
        U_TPa = np.zeros_like(X_TPa)
        for T in range(1, self.n_T):
            U_TPa[T, P_Q] = np.array(X_TPa[T, P_Q] - X_TPa[0, P_Q])
        return X_TPa[0, P_Q], U_TPa[:, P_Q]

    X_Qa = tr.Property(depends_on='state_changed')
    """Initial coordinates"""
    @tr.cached_property
    def _get_X_Qa(self):
        X_Qa, _ = self.X_Qa__U_TQa
        return X_Qa

    U_TQa = tr.Property(depends_on='state_changed')
    """Read the displacement data from the individual csv files"""

    @tr.cached_property
    def _get_U_TQa(self):
        _, U_TQa = self.X_Qa__U_TQa
        return U_TQa

    X0_IJa = tr.Property(depends_on='state_changed')
    """Coordinates of the DIC markers in the grid"""
    @tr.cached_property
    def _get_X0_IJa(self):
        n_I, n_J = self.n_I, self.n_J
        X_min_a, X_max_a = self.X_outer_frame
        min_x, min_y, _ = X_min_a
        max_x, max_y, _ = X_max_a
        X_aIJ = np.mgrid[
                min_x + self.pad_l:max_x - self.pad_r:complex(n_I),
                min_y + self.pad_b:max_y - self.pad_t:complex(n_J)]
        x_IJ, y_IJ = X_aIJ
        # x_IJ -= min_x
        # y_IJ -= min_y
        X0_IJa = np.einsum('aIJ->IJa', np.array([x_IJ, y_IJ]))
        return X0_IJa

    U_TIJa = tr.Property(depends_on='state_changed')
    """Read the displacement data from the individual csv files"""
    @tr.cached_property
    def _get_U_TIJa(self):
        points = self.X_Qa[:, :-1]
        delaunay = Delaunay(points)
        triangles = delaunay.simplices
        x0_IJ, y0_IJ = np.einsum('IJa->aIJ', self.X0_IJa)
        U_IJa_list = []
        for T in range(self.n_T):
            values = self.U_TQa[T, :, :]
            get_U = scipy.interpolate.LinearNDInterpolator(delaunay, values)
            U_IJa = get_U(x0_IJ, y0_IJ)
            U_IJa_list.append(U_IJa)
        U_TIJa = np.array(U_IJa_list)
        return U_TIJa[...,:-1]

    X_IJa = tr.Property(depends_on='state_changed')
    """Coordinates of the DIC markers in the grid"""
    @tr.cached_property
    def _get_X_IJa(self):
        X0_IJa = self.X0_IJa
        x_min, y_min = X0_IJa[0,0,(0, 1)]
        x0_IJ, y0_IJ = np.einsum('...a->a...', X0_IJa)
        X_aIJ = np.array([x0_IJ-x_min+self.x_offset, y0_IJ-y_min+self.y_offset])
        return np.einsum('a...->...a', X_aIJ)

    n_T = tr.Property(depends_on='state_changed')
    """Number of dic snapshots up to the maximum load"""
    @tr.cached_property
    def _get_n_T(self):
        return len(self.F_dic_T)

    def get_T_t(self, t = 0.9):
        """Get the dic index correponding to the specified fraction
        of ultimate load.
        """
        F = -self.Fw_T[::50,1]
        F_max = np.max(F)
        F_t = t * F_max
        F_dic_T = self.F_dic_T
        dic_T = np.arange(len(F_dic_T))
        T_t = np.interp(F_t, F_dic_T, dic_T)
        return int(T_t)

    U_IJa = tr.Property(depends_on='state_changed')
    """Total displacement at step T_t w.r.t. T0
    """
    @tr.cached_property
    def _get_U_IJa(self):
        return self.U_TIJa[self.T_t] - self.U_TIJa[self.T0]

    Fw_file_name = tr.Str('load_deflection.csv')
    """Name of the file with the measured load deflection data
    """
    
    Fw_T = tr.Property(depends_on='state_changed')
    """Read the load displacement values from the individual 
    csv files from the test
    """
    @tr.cached_property
    def _get_Fw_T(self):
        Fw_file = join(self.Fw_data_dir, self.Fw_file_name)
        Fw_T = np.array(pd.read_csv(Fw_file, decimal=",", skiprows=1, delimiter=None), dtype=np.float_)
        return Fw_T

    F_T_t = tr.Property(depends_on='state_changed')
    """Current load
    """
    @tr.cached_property
    def _get_F_T_t(self):
        return self.F_dic_T[self.T_t]

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
        w = self.Fw_T[::50,2]
        F = -self.Fw_T[::50,1]

        argmax_F_T = np.argmax(F)

        ax_load.plot(w[:argmax_F_T], F[:argmax_F_T], color='black')
        ax_load.set_ylabel(r'$F$ [kN]')
        ax_load.set_xlabel(r'$w$ [mm]')

        # plot the markers of dic levels
        w_dic_T = np.interp(self.F_dic_T, F[:argmax_F_T], w[:argmax_F_T])
        ax_load.plot(self.w_T, self.F_dic_T, 'o', markersize=3, color='orange')
        ax_load.plot(self.w_dic_T, self.F_dic_T, '-o', markersize=3, color='blue')

        # show the current load marker
        F_T_t = self.F_dic_T[self.T_t]
        w_T_t = np.interp(F_T_t, F[:argmax_F_T], w[:argmax_F_T])
        ax_load.plot(w_T_t, F_T_t, marker='o', markersize=6, color='green')

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
