import bmcs_utils.api as bu
import traits.api as tr
from pathlib import Path
import os
import numpy as np
import pandas as pd
from .i_dic_inp import IDICInp
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
class DICInpUnstructuredPoints(bu.Model):
    """
    History of displacment grids imported from the DIC measurement.
    """
    name = 'DIC grid history'

    force_array_refresh = bu.Bool(False)

    dir_name = bu.Str('<unnamed>', ALG=True)
    """Directory name containing the test data.
    """

    dic_param_file_name = bu.Str('dic_params.txt', ALG=True)
    """Default name of the file with the parameters of the grid.
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

    T_t = tr.Property(bu.Int, depends_on='t, t_final, t_ref')
    def _get_T_t(self):
        return self._find_nearest_index(self.t_T, self.t)
    
    U_factor = bu.Float(100, ALG=True)

    L = bu.Float(10, MAT=True)

    t_final = bu.Float(1.1, ALG=True)
    """Value of the time slider representing to the last data value in the monitored history.
    This value must be larger than the reference value.
    """

    t_ref = bu.Float(1, ALG=True)
    """Value of the time slider representing to reference point in the response, for example the 
    ultimate load on a sliding scale. Relative to this scale, other stages of the test can be introduced
    at which structural changes appear and at which focused detection schemes can be defined.
    """

    t = bu.Float(1, ALG=True)

    @staticmethod
    def _find_nearest_index(arr, value):
        return np.argmin(np.abs(arr - value))

    ipw_view = bu.View(
        bu.Item('n_T_max'),
        bu.Item('U_factor'),
        bu.Item('T_stepping'),
        bu.Item('T_t', readonly=True),
        bu.Item('pad_t', readonly=True),
        bu.Item('pad_b', readonly=True),
        bu.Item('pad_l', readonly=True),
        bu.Item('pad_r', readonly=True),
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

    X_inner_frame = tr.Property(depends_on='state_changed')
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

    L_x = tr.Property
    """Width of the domain"""
    def _get_L_x(self):
        X_min, X_max = self.X_inner_frame
        return X_max[0] - X_min[0]

    L_y = tr.Property
    """Height of the domain"""
    def _get_L_y(self):
        X_min, X_max = self.X_inner_frame
        return X_max[1] - X_min[1]

    base_dir = tr.Directory
    def _base_dir_default(self):
        return Path.home() / 'simdb' / 'data' / 'shear_zone'

    data_dir = tr.Property
    """Directory with the data"""
    def _get_data_dir(self):
        return Path(self.base_dir) / self.dir_name

    dic_data_dir = tr.Property
    """Directory with the DIC data"""
    def _get_dic_data_dir(self):
        return self.data_dir / 'dic_point_data'

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

    time_F_w_data_dir = tr.Property
    """Directory with the load deflection data"""
    def _get_time_F_w_data_dir(self):
        return self.data_dir / 'load_deflection'

    time_F_w_file_name = tr.Str('load_deflection.csv')
    """Name of the file with the measured load deflection data
    """

    time_m_skip = bu.Int(50, ALG=True )

    time_F_w_m_all = tr.Property(depends_on='state_changed')
    """Read the load displacement values from the individual 
    csv files from the test
    """
    @tr.cached_property
    def _get_time_F_w_m_all(self):
        time_F_w_file = self.time_F_w_data_dir / self.time_F_w_file_name
        time_F_w_m = np.array(pd.read_csv(time_F_w_file, decimal=",", skiprows=1,
                              delimiter=None), dtype=np.float_)
        time_m, F_m, w_m = time_F_w_m[::self.time_m_skip, (0,1,2)].T
        F_m *= -1
        return time_m, F_m, w_m
    
    time_F_w_m = tr.Property(depends_on='state_changed')
    """Return the ascending branch of the monitored F-w response
    """
    @tr.cached_property
    def _get_time_F_w_m(self):
        time_m, F_m, w_m = self.time_F_w_m_all
        dF_mn = F_m[np.newaxis, :] - F_m[:, np.newaxis]
        dF_up_mn = np.triu(dF_mn > 0)
        argmin_m = np.argmin(dF_up_mn, axis=0)
        for n in range(len(argmin_m)):
            dF_up_mn[argmin_m[n]:, n] = False
        asc_m = np.unique(np.argmax(np.triu(dF_up_mn, 0) > 0, axis=1))
        F_asc_m = F_m[asc_m]
        time_asc_m = time_m[asc_m]
        w_asc_m = w_m[asc_m]

        # Append the post reference part
        _, time_dic_dsc, F_dic_dsc = self.tstring_time_F_dic_dsc
        time_asc_m = np.append(time_asc_m, time_dic_dsc[-1])
        F_asc_m = np.append(F_asc_m, F_dic_dsc[-1])
        w_asc_m = np.append(w_asc_m, self.t_final * w_asc_m[-1])

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
        return np.argmax(F_m)

    time_F_dic_file = tr.Property(depends_on='state_changed')
    """File specifying the dic snapshot times and forces
    """
    @tr.cached_property
    def _get_time_F_dic_file(self):
        return self.dic_data_dir / 'Kraft.DIM.csv'

    tstring_time_F_dic_all = tr.Property(depends_on='state_changed')
    """times and forces for all snapshots with camera clock (dic index)
    """
    @tr.cached_property
    def _get_tstring_time_F_dic_all(self):
        rows_ = list(open(self.time_F_dic_file))
        rows = np.array(rows_)
        time_entries_dic = rows[0].replace('name;type;attribute;id;', '').split(';')
        tstring_dic = np.array([time_entry_dic.split(' ')[0] for time_entry_dic in time_entries_dic])
        time_dic = np.array(tstring_dic, dtype=np.float_)
        F_entries_dic = rows[1].replace('Kraft.DIM;deviation;dimension;;', '')
        F_dic = -np.fromstring(F_entries_dic, sep=';', dtype=np.float_)
        return tstring_dic, time_dic, F_dic

    time_F_dic_all = tr.Property
    def _get_time_F_dic_all(self):
        _, time_dic, F_dic = self.tstring_time_F_dic_all
        return time_dic, F_dic

    tstring_time_F_dic = tr.Property(depends_on='state_changed')
    """times and forces for all snapshots with camera clock (dic index)
    """
    @tr.cached_property
    def _get_tstring_time_F_dic(self):
        tstring_dic, time_dic, F_dic = self.tstring_time_F_dic_all
        dF_dic = F_dic[np.newaxis, :] - F_dic[:, np.newaxis]
        dF_up_dic = np.triu(dF_dic > 0, 0)
        argmin_dic = np.argmin(dF_up_dic, axis=0)
        for n in range(len(argmin_dic)):
            dF_up_dic[argmin_dic[n]:, n] = False
        asc_dic = np.unique(np.argmax(np.triu(dF_up_dic, 0) > 0, axis=1))
        return tstring_dic[asc_dic], time_dic[asc_dic], F_dic[asc_dic]

    F_dsc_cutoff_fraction = bu.Float(0.98)
    def argcut(self, arr):
        specified_value = arr[0] * self.F_dsc_cutoff_fraction
        index = np.argmax(arr < specified_value)
        if np.all(arr >= specified_value):
            index = len(arr) - 1
        return index
        #return np.argmax(arr < arr[0] * self.F_dsc_cutoff_fraction)

    tstring_time_F_dic_dsc = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_tstring_time_F_dic_dsc(self):
        """Times and forces for all snapshots with camera clock (dic index)
        descending branch down to the specified cutoff fraction 
        `F_dsc_cutoff_fraction` of the peak load.

        Returns:
            tuple: A tuple containing the subset of tstring, time, 
            and force arrays corresponds to the branch with descending
            load (post peak branch) 
        """
        tstring_dic_all, time_dic_all, F_dic_all = self.tstring_time_F_dic_all
        argmax_F_dic_all = np.argmax(F_dic_all)
        F_dic_dsc = F_dic_all[argmax_F_dic_all:]
        argcut_F_dic_all = argmax_F_dic_all + self.argcut(F_dic_dsc)
        F_dic_dsc = F_dic_all[argmax_F_dic_all:argcut_F_dic_all]
        time_dic_dsc = time_dic_all[argmax_F_dic_all:argcut_F_dic_all]
        tstring_dic_dsc = tstring_dic_all[argmax_F_dic_all:argcut_F_dic_all]
        return tstring_dic_dsc, time_dic_dsc, F_dic_dsc

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

    T_stepping = bu.Enum(options=['delta_n', 'delta_T'], ALG=True)

    tstring_time_F_T = tr.Property(depends_on='state_changed')
    """synchronized times and forces for specified resolution n_T
    """
    @cached_array(names=['tstring', 'time', 'F'])
    def _get_tstring_time_F_T(self):

        time_m, F_m, _ = self.time_F_w_m # machine time and force
        argmax_F_m = self.argmax_F_m
        time_dic, F_dic = self.time_F_dic # dic time and force
        argmax_F_dic = self.argmax_F_dic
        argmax_time_F_dic = time_dic[argmax_F_dic]
        max_F_dic = F_dic[argmax_F_dic]
        m_max_F_dic = argmax_F_m - np.argmax(F_m[argmax_F_m:0:-1] < max_F_dic)
        time_m_F_dic_max = time_m[m_max_F_dic]
        delta_time_dic_m = argmax_time_F_dic - time_m_F_dic_max
        time_dic_ = time_dic - delta_time_dic_m
        dic_0 = np.argmax(time_dic_ > 0) - 1
        if self.T_stepping == 'delta_n':
            delta_T = int(len(time_dic_[dic_0:argmax_F_dic+1]) / self.n_T_max)
            T_slice = slice(dic_0, argmax_F_dic, delta_T)
        elif self.T_stepping == 'delta_T':
            time_ = time_dic_[dic_0:argmax_F_dic+1]
            idx_ = np.arange(len(time_))
            time1_T = np.linspace(0, time_[-1], self.n_T_max)
            T_slice = np.unique(np.array(np.interp(time1_T, time_, idx_), dtype=np.int_))
        time_T = time_dic_[T_slice]
        F_T = F_dic[T_slice]
        time_T[0] = 0
        tstring_T = self.tstring_dic[T_slice]
        tstring_T[0] = self.tstring_dic[0]

        # append the descending part 
        tstring_dic_dsc, time_dic_dsc, F_dic_dsc = self.tstring_time_F_dic_dsc
        tstring_T = np.append(tstring_T, tstring_dic_dsc[-1])
        time_T = np.append(time_T, time_dic_dsc[-1])
        F_T = np.append(F_T, F_dic_dsc[-1])

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

        
        time_T, F_T = self.time_F_T
        argmax_F_T = np.argmax(F_T)
        F_T_asc = F_T[:argmax_F_T+1]

        _, F_m = self.time_F_m
        argmax_F_m = np.argmax(F_m)
        F_m = F_m[:argmax_F_m+1]

        w_m = self.w_m[:argmax_F_m+1]
        w_T = np.interp(F_T_asc, F_m, w_m)
        w_T = np.append(w_T, w_m[-1])

        return w_T


        # w_m = self.w_m
        # _, F_m = self.time_F_m
        # _, F_T = self.time_F_T
        # argmax_F_m = np.argmax(F_m)
        # return np.interp(F_T, F_m[:argmax_F_m+1], w_m[:argmax_F_m+1])

    pxyz_file_T = tr.Property(depends_on='state_changed')
    """List of file names corresponding to the synchronized time index T
    """
    @tr.cached_property
    def _get_pxyz_file_T(self):
        return [self.dic_data_dir / r'Flächenkomponente 1_{} s.csv'.format(tstring) for
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

    X_TQa = tr.Property(depends_on='dir_name')
    """Read the displacement data from the individual csv files"""
    #@tr.cached_property
    @cached_array('X_TQa')
    def _get_X_TQa(self):

        pxyz_file_T = self.pxyz_file_T
        pxyz_list = [
            np.loadtxt(pxyz_file, dtype=np.float_,
                       skiprows=6, delimiter=';', usecols=(0, 1, 2, 3))
            for pxyz_file in pxyz_file_T
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
        X_TPa = np.zeros((self.n_T, max_n_P, 3), dtype=np.float_)
        for T in range(self.n_T):
            X_TPa[T, P_list[T]] = pxyz_list[T][:, 1:]

        return X_TPa[:, P_Q]

    U_TQa = tr.Property(depends_on='state_changed')
    """Get the displacement history"""
    @tr.cached_property
    def _get_U_TQa(self):
        X_TPa = self.X_TQa
        X_0Pa = X_TPa[0]
        return X_TPa - X_0Pa[None, ...]

    X_Qa = tr.Property(depends_on='state_changed')
    """Initial coordinates"""
    @tr.cached_property
    def _get_X_Qa(self):
        return self.X_TQa[0]

    n_T_max = bu.Int(30, ALG=True)
    """Number of dic snapshots up to the maximum load"""

    def _n_T_max_changed(self):
        if self.n_T_max > self.n_dic:
            raise ValueError(f'n_T_max = {self.n_T_max} larger than n_dic {self.n_dic}')

    n_T = tr.Property(bu.Int, depends_on='state_changed')
    """Number of camera time sampling points up to the peak load 
    """
    def _get_n_T(self):
        time_T, _ = self.time_F_T
        return len(time_T)

    t_T = tr.Property(bu.Array, depends_on='state_changed')
    """Slider values matching the DIC data snapshots in the range t \in (0,1)  
    """
    def _get_t_T(self):
        argmax_F_T = np.argmax(self.F_T)
        t_T = self.F_T[:argmax_F_T+1] / self.F_T[argmax_F_T] 
        t_T = np.append(t_T, self.t_final)
        t_T[t_T < 0] = 0 # avoid negative time
        t_T[0] = 0  # ensure that the interpolators include zero valued slider
        return t_T

    F_T_t = tr.Property(depends_on='state_changed')
    """Current load
    """
    @tr.cached_property
    def _get_F_T_t(self):
        return self.F_T[self.T_t]

    F_T = tr.Property(depends_on='state_changed')
    """Loading history
    """
    @tr.cached_property
    def _get_F_T(self):
        _, F_T = self.time_F_T
        return F_T

    def plot_points(self, ax):
        U_Qa = self.U_TQa[self.T_t] * self.U_factor
        X_Qa = self.X_Qa + U_Qa
        ax.scatter(*X_Qa[:,:-1].T, s=15, marker='o', color='orange')
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
        ax.annotate('{:.0f} mm'.format(self.L_x),
                    xy=(x, y), xytext=(0, 1), xycoords='data',
                    textcoords='offset pixels',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    )
        x, y = X_La[3, :]
        ax.annotate("{:.0f} mm".format(self.L_y),
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


    def plot_load_deflection(self, ax_load):
        w_m = self.w_m
        _, F_m = self.time_F_m
        argmax_F_m = self.argmax_F_m

        # ax_load.plot(w_m[:argmax_F_m], F_m[:argmax_F_m], color='black')
        ax_load.plot(w_m, F_m, color='black')
        ax_load.set_ylabel(r'$F$ [kN]')
        ax_load.set_xlabel(r'$w$ [mm]')

        # plot the markers of dic levels
        _, F_T = self.time_F_T
        ax_load.plot(self.w_T, F_T, 'o', markersize=3, color='orange')


        # show the current load marker
        F_t = F_T[self.T_t]
        w_t = self.w_T[self.T_t]
        ax_load.plot(w_t, F_t, marker='o', markersize=6, color='green')

        # annotate the maximum load level
        max_F = F_m[argmax_F_m]
        argmax_w_F = w_m[argmax_F_m]
        ax_load.annotate(f'$F_{{\max}}=${max_F:.1f} kN,\nw={argmax_w_F:.2f} mm',
                    xy=(argmax_w_F, max_F), xycoords='data',
                    xytext=(0.05, 0.95), textcoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top',
                    )

    def subplots(self, fig):
        return fig.subplots(2,)

    def update_plot(self, axes):
        ax_u, ax_load = axes
        self.plot_points(ax_u)
        self.plot_bounding_box(ax_u)
        self.plot_box_annotate(ax_u)
        self.plot_load_deflection(ax_load)

