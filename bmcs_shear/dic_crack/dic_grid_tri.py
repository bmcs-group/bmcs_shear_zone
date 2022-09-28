
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
        self.T_t = int( (self.n_T - 1) * (self.t + d_t/2))

    t_dic_T = tr.Property(depends_on='state_changed')
    """Time steps of ascending DIC snapshots
    """
    @tr.cached_property
    def _get_t_dic_T(self):
        return np.linspace(0, 1, self.n_T)

    ipw_view = bu.View(
        bu.Item('d_x'),
        bu.Item('d_y'),
        bu.Item('n_T'),
        bu.Item('U_factor'),
        bu.Item('T_t', readonly=True),
        bu.Item('pad_t', readonly=True),
        bu.Item('pad_b', readonly=True),
        bu.Item('pad_l', readonly=True),
        bu.Item('pad_r', readonly=True),
        bu.Item('n_I', readonly=True),
        bu.Item('n_J', readonly=True),
        bu.Item('n_m', readonly=True),
        bu.Item('n_dic', readonly=True),
        bu.Item('x_offset', readonly=True),
        bu.Item('y_offset', readonly=True),
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
        x_min, y_min = self.X_IJa[0,0,(0,1)] #x_offset + self.pad_l
        #x_max = x_min + L_x - self.pad_r
        x_max, y_max = self.X_IJa[-1,-1,(0,1)]
        # y_max = y_min + L_y - self.pad_t
        return x_min, y_min, x_max, y_max


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

    time_F_w_data_dir = tr.Property
    """Directory with the load deflection data"""
    def _get_time_F_w_data_dir(self):
        return join(self.data_dir, 'load_deflection')

    time_F_w_file_name = tr.Str('load_deflection.csv')
    """Name of the file with the measured load deflection data
    """

    time_F_w_m = tr.Property(depends_on='state_changed')
    """Read the load displacement values from the individual 
    csv files from the test
    """
    @tr.cached_property
    def _get_time_F_w_m(self):
        time_F_w_file = join(self.time_F_w_data_dir, self.time_F_w_file_name)
        time_F_w_m = np.array(pd.read_csv(time_F_w_file, decimal=",", skiprows=1,
                              delimiter=None), dtype=np.float_)
        time_m, F_m, w_m = time_F_w_m[::50, (0,1,2)].T
        F_m *= -1
        dF_mn = F_m[np.newaxis, :] - F_m[:, np.newaxis]
        dF_up_mn = np.triu(dF_mn > 0)
        argmin_m = np.argmin(dF_up_mn, axis=0)
        for n in range(len(argmin_m)):
            dF_up_mn[argmin_m[n]:, n] = False
        asc_m = np.unique(np.argmax(np.triu(dF_up_mn, 0) > 0, axis=1))
        F_asc_m = F_m[asc_m]
        time_asc_m = time_m[asc_m]
        w_asc_m = w_m[asc_m]
        return time_asc_m, F_asc_m, w_asc_m

    time_F_m = tr.Property(depends_on='state_changed')
    """time and force
    """
    @tr.cached_property
    def _get_time_F_m(self):
        time_m, F_m, _ = self.time_F_w_m
        return time_m, F_m

    w_m = tr.Property(depends_on='state_changed')
    """time and force
    """
    @tr.cached_property
    def _get_w_m(self):
        _, _, w_m = self.time_F_w_m
        return w_m

    n_m = tr.Property(bu.Int, depends_on='state_changed')
    """Number of machine time sampling points up to the peak load 
    """
    def _get_n_m(self):
        time_m, _ = self.time_F_m
        return len(time_m)

    argmax_F_m = tr.Property(depends_on='state_changed')
    """times and forces for all snapshots with machine clock (m index)
    """
    @tr.cached_property
    def _get_argmax_F_m(self):
        time_m, F_m = self.time_F_m
        argmax_F_m = np.argmax(F_m)
        return argmax_F_m

    time_F_dic_file = tr.Property(depends_on='state_changed')
    """File specifying the dic snapshot times and forces
    """
    @tr.cached_property
    def _get_time_F_dic_file(self):
        return os.path.join(self.dic_data_dir, 'Kraft.DIM.csv')

    tstring_time_F_dic = tr.Property(depends_on='state_changed')
    """times and forces for all snapshots with camera clock (dic index)
    """
    @tr.cached_property
    def _get_tstring_time_F_dic(self):
        rows_ = []
        for row in open(self.time_F_dic_file):
            rows_.append(row)
        rows = np.array(rows_)
        time_entries_dic = rows[0].replace('name;type;attribute;id;', '').split(';')
        tstring_dic = np.array([time_entry_dic.split(' ')[0] for time_entry_dic in time_entries_dic])
        time_dic = np.array(tstring_dic, dtype=np.float_)
        F_entries_dic = rows[1].replace('Kraft.DIM;deviation;dimension;;', '')
        F_dic = -np.fromstring(F_entries_dic, sep=';', dtype=np.float_)
        dF_dic = F_dic[np.newaxis, :] - F_dic[:, np.newaxis]
        dF_up_dic = np.triu(dF_dic > 0, 0)
        argmin_dic = np.argmin(dF_up_dic, axis=0)
        for n in range(len(argmin_dic)):
            dF_up_dic[argmin_dic[n]:, n] = False
        asc_dic = np.unique(np.argmax(np.triu(dF_up_dic, 0) > 0, axis=1))
        return tstring_dic[asc_dic], time_dic[asc_dic], F_dic[asc_dic]

    time_F_dic = tr.Property
    def _get_time_F_dic(self):
        _, time_dic, F_dic = self.tstring_time_F_dic
        return time_dic, F_dic

    tstring_dic = tr.Property
    def _get_tstring_dic(self):
        tstrings_dic, _, _ = self.tstring_time_F_dic
        return tstrings_dic

    n_dic = tr.Property(bu.Int, depends_on='state_changed')
    """Number of camera time sampling points up to the peak load 
    """
    def _get_n_dic(self):
        time_dic, _ = self.time_F_dic
        return len(time_dic)

    argmax_F_dic = tr.Property(depends_on='state_changed')
    """times and forces for all snapshots with camera clock (dic index)
    """
    @tr.cached_property
    def _get_argmax_F_dic(self):
        _, F_dic = self.time_F_dic
        return np.argmax(F_dic)

    tstring_time_F_T = tr.Property(depends_on='state_changed')
    """synchronized times and forces for specified resolution n_T
    """
    def _get_tstring_time_F_T(self):
        time_m, F_m = self.time_F_m
        argmax_F_m = self.argmax_F_m
        time_dic, F_dic = self.time_F_dic
        argmax_F_dic = self.argmax_F_dic
        argmax_time_F_dic = time_dic[argmax_F_dic]
        max_F_dic = F_dic[argmax_F_dic]
        m_max_F_dic = argmax_F_m - np.argmax(F_m[argmax_F_m:0:-1] < max_F_dic)
        time_m_F_dic_max = time_m[m_max_F_dic]
        delta_time_dic_m = argmax_time_F_dic - time_m_F_dic_max
        time_dic_ = time_dic - delta_time_dic_m
        dic_0 = np.argmax(time_dic_ > 0) - 1
        delta_T = int(len(time_dic_[dic_0:argmax_F_dic]) / self.n_T)
        T_slice = slice(dic_0, argmax_F_dic, delta_T)
        time_T = time_dic_[T_slice]
        F_T = F_dic[T_slice]
        time_T[0] = 0
        tstring_T = self.tstring_dic[T_slice]
        tstring_T[0] = self.tstring_dic[0]
        return tstring_T, time_T, F_T

    time_F_T = tr.Property(depends_on='state_changed')
    """synchronized times and forces for specified resolution n_T
    """
    def _get_time_F_T(self):
        _, time_T, F_T = self.tstring_time_F_T
        return time_T, F_T

    tstring_T = tr.Property(depends_on='state_changed')
    """synchronized times and forces for specified resolution n_T
    """
    def _get_tstring_T(self):
        tstring_T, _, _ = self.tstring_time_F_T
        return tstring_T

    w_T = tr.Property(depends_on='state_changed')
    """Displacement levels of ascending DIC snapshots
    """
    @tr.cached_property
    def _get_w_T(self):
        w_m = self.w_m
        _, F_m = self.time_F_m
        _, F_T = self.time_F_T
        argmax_F_m = np.argmax(F_m)
        return np.interp(F_T, F_m[:argmax_F_m], w_m[:argmax_F_m])

    pxyz_file_T = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_pxyz_file_T(self):
        return [os.path.join(self.dic_data_dir, r'Flächenkomponente 1_{} s.csv'.format(tstring)) for
                      tstring in self.tstring_T]

    asc_time_F_T = tr.Property(depends_on='state_changed')
    """Ascending force sequence
    """
    @tr.cached_property
    def _get_asc_time_F_T(self):
        time_T, F_T = self.time_F_T
        dF_ST = F_T[np.newaxis, :] - F_T[:, np.newaxis]
        asc_T = np.unique(np.argmax(np.triu(dF_ST, 0) > 0, axis=1))
        return time_T[asc_T], F_T[asc_T]

    u_T = tr.Property(depends_on='state_changed')
    """Displacement levels of ascending DIC snapshots
    """
    @tr.cached_property
    def _get_u_T(self):
        return -self.U_TIJa[:,0,-1,1]

    X_Qa__U_TQa = tr.Property(depends_on='state_changed')
    """Read the displacement data from the individual csv files"""
    @tr.cached_property
    def _get_X_Qa__U_TQa(self):
        pxyz_file_T = self.pxyz_file_T
        pxyz_list = [
            np.loadtxt(pxyz_file, dtype=np.float_,
                       skiprows=6, delimiter=';', usecols=(0, 1, 2, 3))
            for pxyz_file in pxyz_file_T
        ]
        # Identify the points that are included in all time steps.
        P_list = [np.array(pxyz[:, 0], dtype=np.int_)
                  for pxyz in pxyz_list]
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
        X0_IJa = np.einsum('aIJ->IJa', np.array([x_IJ, y_IJ]))
        return X0_IJa

    delaunay = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_delaunay(self):
        points = self.X_Qa[:, :-1]
        return Delaunay(points)

    U_TIJa = tr.Property(depends_on='state_changed')
    """Read the displacement data from the individual csv files"""
    @tr.cached_property
    def _get_U_TIJa(self):
        x0_IJ, y0_IJ = np.einsum('IJa->aIJ', self.X0_IJa)
        U_IJa_list = []
        for T in range(self.n_T):
            values = self.U_TQa[T, :, :]
            get_U = scipy.interpolate.LinearNDInterpolator(self.delaunay, values)
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

    n_T = bu.Int(30, ALG=True)
    """Number of dic snapshots up to the maximum load"""

    def _n_T_changed(self):
        if self.n_T > self.n_dic:
            raise ValueError('n_T = {} larger than n_dic {}'.format(self.n_T, self.n_dic))

    def get_T_t(self, t = 0.9):
        """Get the T index correponding to the specified fraction
        of ultimate load.
        """
        _, F_m = self.time_F_m
        max_F_m = np.max(F_m)
        F_t = t * max_F_m
        _, F_T = self.time_F_T
        T_T = np.arange(self.n_T)
        T = np.interp(F_t, F_T, T_T)
        return int(T)

    U_IJa = tr.Property(depends_on='state_changed')
    """Total displacement at step T_t w.r.t. T0
    """
    @tr.cached_property
    def _get_U_IJa(self):
        return self.U_TIJa[self.T_t] - self.U_TIJa[self.T0]

    F_T_t = tr.Property(depends_on='state_changed')
    """Current load
    """
    @tr.cached_property
    def _get_F_T_t(self):
        return self.F_dic_T[self.T_t]

    def plot_grid_on_triangulation(self, ax):
        triangles = self.delaunay.simplices
        x, y = self.delaunay.points.T
        ax.triplot(x, y, triangles, linewidth=0.5)
        X0_aIJ = np.einsum('IJa->aIJ', self.X0_IJa)
        ax.scatter(*X0_aIJ.reshape(2, -1), s=15, marker='o', color='orange')
        ax.axis('equal')

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
        w_m = self.w_m
        _, F_m = self.time_F_m
        argmax_F_m = self.argmax_F_m

        ax_load.plot(w_m[:argmax_F_m], F_m[:argmax_F_m], color='black')
        ax_load.set_ylabel(r'$F$ [kN]')
        ax_load.set_xlabel(r'$w$ [mm]')

        # plot the markers of dic levels
        _, F_T = self.time_F_T
        ax_load.plot(self.w_T, F_T, 'o', markersize=3, color='orange')

        # show the current load marker
        F_t = F_T[self.T_t]
        w_t = np.interp(F_t, F_T, self.w_T)
        ax_load.plot(w_t, F_t, marker='o', markersize=6, color='green')

        # annotate the maximum load level
        max_F = F_m[argmax_F_m]
        argmax_w_F = w_m[argmax_F_m]
        ax_load.annotate(f'$F_\max=${max_F:.1f} kN, w={argmax_w_F:.2f} mm',
                    xy=(argmax_w_F, max_F), xycoords='data',
                    xytext=(0.05, 0.95), textcoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top',
                    )

    def subplots(self, fig):
        return fig.subplots(1,2)

    def update_plot(self, axes):
        ax_u, ax_load = axes
        self.plot_grid(ax_u)
        self.plot_load_deflection(ax_load)
