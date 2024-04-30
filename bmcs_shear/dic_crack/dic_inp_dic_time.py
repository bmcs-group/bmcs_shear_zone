import bmcs_utils.api as bu
import traits.api as tr
from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
from bmcs_shear.beam_design import RCBeamDesign
from bmcs_shear.matmod import CrackBridgeAdv
from .i_dic_inp import IDICInp
from .dic_inp_time_sync import DICInpTimeSync
from .cached_array import cached_array

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

@tr.provides(IDICInp)
class DICInpDICTime(bu.Model):
    """
    History slice of displacement grids imported from the DIC measurement.
    """
    name = 'DIC history'

    force_array_refresh = bu.Bool(False)

    time_F_w = bu.Instance(DICInpTimeSync, ())
    
    depends_on = ['time_F_w']
    tree = ['time_F_w']

    time_0 = tr.Property
    def _get_time_0(self):
        return self.time_F_w.ld_time.time_0

    time_1 = tr.Property
    def _get_time_1(self):
        return self.time_F_w.ld_time.time_1

    dir_name = tr.DelegatesTo('time_F_w')
    dic_data_dir = tr.DelegatesTo('time_F_w')
    time_shift = tr.DelegatesTo('time_F_w')

    dic_param_file_name = bu.Str('dic_params.txt', GEO=True)
    """Default name of the file with the parameters of the grid.
    """

    dic_param_file = tr.Property
    """File containing the parameters of the grid"""
    def _get_dic_param_file(self):
        return self.dic_data_dir / self.dic_param_file_name

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
        with open(self.dic_param_file) as f:
            data = f.readlines()
            for line in data:
                # parse input, assign values to variables
                key, value = line.split(":")
                params_str[key.strip()] = value.strip()
        return {
            key: type_(params_str[key])
            for key, type_ in self.dic_param_types.items()
        }

    x_offset = tr.Property(bu.Float, depends_on='+GEO')
    """Horizontal offset of the DIC input displacement grid from the left
    boundary of the beam.
    """
    @tr.cached_property
    def _get_x_offset(self):
        return self.dic_params['x_offset']

    y_offset = tr.Property(bu.Float, depends_on='+GEO')
    """ Vertical offset of the DIC input displacement grid from the bottom
        boundary of the beam
    """
    @tr.cached_property
    def _get_y_offset(self):
        return self.dic_params['y_offset']

    pad_t = tr.Property(bu.Float, depends_on='+GEO')
    """ Top pad width
    """
    @tr.cached_property
    def _get_pad_t(self):
        return self.dic_params['pad_t']

    pad_b = tr.Property(bu.Float, depends_on='+GEO')
    """ Bottom pad width
    """
    @tr.cached_property
    def _get_pad_b(self):
        return self.dic_params['pad_b']

    pad_l = tr.Property(bu.Float, depends_on='+GEO')
    """ Left pad width
    """
    @tr.cached_property
    def _get_pad_l(self):
        return self.dic_params['pad_l']

    pad_r = tr.Property(bu.Float, depends_on='+GEO')
    """ Right pad width
    """
    @tr.cached_property
    def _get_pad_r(self):
        return self.dic_params['pad_r']

    time_t = tr.Property(bu.Float, depends_on='t')
    @tr.cached_property
    def _get_time_t(self):
        return (self.time_1 - self.time_0) * self.t

    T_t = tr.Property(bu.Int, depends_on='+TIME,+ALG')
    @tr.cached_property
    def _get_T_t(self):
        return self._find_nearest_index(self.time_S, self.time_t)
    
    @staticmethod
    def _find_nearest_index(arr, value):
        return np.argmin(np.abs(arr - value))

    U_factor = bu.Float(0, ALG=True)

    t = bu.Float(1, ALG=True)

    ipw_view = bu.View(
        bu.Item('U_factor'),
        bu.Item('pad_t', readonly=True),
        bu.Item('pad_b', readonly=True),
        bu.Item('pad_l', readonly=True),
        bu.Item('pad_r', readonly=True),
        bu.Item('n_S', readonly=True),
        bu.Item('x_offset', readonly=True),
        bu.Item('y_offset', readonly=True),
        # bu.Item('time_0'),
        # bu.Item('time_1'),
        bu.Item('time_t', readonly=True),
        time_editor=bu.HistoryEditor(
            var='t'
        )
    )

    base_dir = tr.Property
    def _get_base_dir(self):
        return Path().home() /  'simdb' / 'data' / 'shear_zone'

    data_dir = tr.Property
    """Directory with the data"""
    def _get_data_dir(self):
        return self.base_dir / self.dir_name

    dic_data_dir = tr.Property
    """Directory with the DIC data"""
    def _get_dic_data_dir(self):
        return self.data_dir / 'dic_point_data'

    time_pxyz_T = tr.Property(depends_on='+GEO')
    """Times of the DIC snapshots in ms and the files along the index T"""
    @tr.cached_property
    def _get_time_pxyz_T(self):
        # Assuming dic.pxyz_time is a list of filenames
        pattern = r'(\d+\.\d+)\s'

        csv_files = [file for file in os.listdir(self.dic_data_dir) 
                    if file.endswith('.csv')]

        time_T, pxyz_T = [], []
        for filename in csv_files:
            if match := re.search(pattern, filename):
                time_in_ms_str = match[1]
                time_in_ms_float = float(time_in_ms_str)
                time_T.append(time_in_ms_float)
                pxyz_T.append(self.dic_data_dir / filename)

        time_T = np.array(time_T, dtype=np.float_)
        pxyz_T = np.array(pxyz_T)

        argsort_time_T = np.argsort(time_T)
        time_T = time_T[argsort_time_T] + self.time_shift
        pxyz_T = pxyz_T[argsort_time_T]

        return (time_T, pxyz_T)

    time_T = tr.Property
    def _get_time_T(self): return self.time_pxyz_T[0]

    pxyz_T = tr.Property
    def _get_pxyz_T(self): return self.time_pxyz_T[1]

    time_S = tr.Property(bu.Array(np.int_), depends_on='state_changed')
    """Select the time steps for the DIC data"""
    @tr.cached_property
    def _get_time_S(self):
        return self.time_F_w.ld_time.time_T

    pxyz_S = tr.Property(depends_on='+TIME, +GEO')
    @tr.cached_property
    def _get_pxyz_S(self):
        T_S = np.argmin(np.abs(self.time_T[:, None] - self.time_S[None, :]), axis=0)
        return self.pxyz_T[T_S]
        
    n_S = tr.Property(bu.Int, depends_on="+TIME, +GEO")
    @tr.cached_property
    def _get_n_S(self): 
        return len(self.pxyz_S)

    X_SQa = tr.Property(depends_on='+TIME, +GEO')
    """Read the displacement data from the individual csv files"""
    @tr.cached_property
    def _get_X_SQa(self):

        pxyz_S = self.pxyz_S
        pxyz_list = [
            np.loadtxt(pxyz, dtype=np.float_,
                       skiprows=6, delimiter=';', usecols=(0, 1, 2, 3))
            for pxyz in pxyz_S
        ]
        # Identify the points that are included in all time steps.
        P_list = [np.array(pxyz[:, 0], dtype=np.int_)
                  for pxyz in pxyz_list]
        # Maximum number of points occurring in one of the time steps to allocate the space
        max_n_P = np.max(np.array([np.max(P_) for P_ in P_list])) + 1
        P_Q = P_list[0]
        for P_next in P_list[1:]:
            P_Q = np.intersect1d(P_Q, P_next)
        # Define the initial configuration
        n_S = self.n_S
        X_SPa = np.zeros((n_S, max_n_P, 3), dtype=np.float_)
        for S in range(n_S):
            X_SPa[S, P_list[S]] = pxyz_list[S][:, 1:]

        return X_SPa[:, P_Q]

    U_SQa = tr.Property(depends_on='+TIME, +GEO')
    """Get the displacement history"""
    @tr.cached_property
    def _get_U_SQa(self):
        X_SPa = self.X_SQa
        X_0Pa = X_SPa[0]
        return X_SPa - X_0Pa[None, ...]

    X_0Qa = tr.Property(depends_on='+TIME, +GEO')
    """Initial coordinates"""
    @tr.cached_property
    def _get_X_0Qa(self):
        return self.X_SQa[0]

    X_outer_frame = tr.Property(depends_on='+TIME, +GEO')
    def _get_X_outer_frame(self):
        return np.min(self.X_0Qa, axis=0), np.max(self.X_0Qa, axis=0)

    X_inner_frame = tr.Property(depends_on='+TIME, +GEO')
    def _get_X_inner_frame(self):
        X_min_a, X_max_a = self.X_outer_frame
        pad_l, pad_r, pad_b, pad_t = (
            self.pad_l, self.pad_r, self.pad_b, self.pad_t
        )
        min_x, min_y, _ = X_min_a
        max_x, max_y, _ = X_max_a
        return (
            np.array([min_x + pad_l, min_y + pad_b]),
            np.array([max_x - pad_r, max_y - pad_t])
        )

    def plot_points(self, ax):
        U_t_Qa = self.U_SQa[self.T_t] * self.U_factor
        X_t_Qa = self.X_0Qa + U_t_Qa
        ax.scatter(*X_t_Qa[:,:-1].T, s=15, marker='o', color='orange')
        ax.axis('equal')
        ax.axis('off')

    X_Ca = tr.Property
    def _get_X_Ca(self):
        min_X_a, max_X_a = self.X_outer_frame
        return np.array(
            [
                [min_X_a[0], min_X_a[1]],
                [max_X_a[0], min_X_a[1]],
                [max_X_a[0], max_X_a[1]],
                [min_X_a[0], max_X_a[1]],
                [min_X_a[0], min_X_a[1]],
            ]
        )

    def plot_bounding_box(self, ax):
        X_Ca = self.X_Ca
        X_iLa = np.array([X_Ca[:-1], X_Ca[1:]], dtype=np.float_)
        X_aiL = np.einsum('iLa->aiL', X_iLa)
        ax.plot(*X_aiL, color='black', lw=0.5)

    def plot_box_annotate(self, ax):
        X_Ca = self.X_Ca
        X_iLa = np.array([X_Ca[:-1], X_Ca[1:]], dtype=np.float_)
        X_La = np.sum(X_iLa, axis=0) / 2
        x, y = X_La[2, :]

        L_inner_Ca = self.X_inner_frame
        X_0a, X_1a = L_inner_Ca
        L_x, L_y = X_1a - X_0a
        ax.annotate('{:.0f} mm'.format(L_x),
                    xy=(x, y), xytext=(0, 1), xycoords='data',
                    textcoords='offset pixels',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    )
        x, y = X_La[3, :]
        ax.annotate("{:.0f} mm".format(L_y),
                    xy=(x, y), xytext=(-17, 0), xycoords='data',
                    textcoords='offset pixels',
                    horizontalalignment='left',
                    verticalalignment='center',
                    rotation=90
                    )
        x, y = X_Ca[2, :]
        ax.annotate(f'{self.dir_name}',
                    xy=(x, y), xytext=(-2, -2), xycoords='data',
                    textcoords='offset pixels',
                    horizontalalignment='right',
                    verticalalignment='top',
                    )

    def subplots(self, fig):
        return fig.subplots(1,)

    def update_plot(self, axes):
        ax_u = axes
        self.plot_points(ax_u)
        self.plot_bounding_box(ax_u)
        self.plot_box_annotate(ax_u)
        ax_u.axis('equal')


    def get_latex_dic_params(self):
        sz_bd = self.sz_bd
        names = [name.replace('_', r'\_') for name in self.dic_params.keys()]
        values = [str(value) for value in self.dic_params.values()]

        table_body = r" & ".join(values)

        return r'''
    \begin{center}
    \begin{tabular}{|c|''' + 'c|' * len(names) + r'''}
    \hline
    ''' + ' & '.join(names) + r'''\\
    \hline
    ''' + table_body + r'''\\
    \hline
    \end{tabular}
    \end{center}
    '''

