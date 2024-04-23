import bmcs_utils.api as bu
import traits.api as tr
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class DICTimeSync(bu.Model):
    """
    Load cell input channel.
    """
    name = 'DIC grid history'

    dir_name = bu.Str('<unnamed>', ALG=True)
    """Directory name containing the test data.
    """
    def _dir_name_change(self):
        self.name = f'Test {self.name}'
        self._set_dic_params()

    time_0 = bu.Float(0, ALG=True)
    """Initial time  
    """

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
        bu.Item('time_0'),
        bu.Item('n_m', readonly=True),
        time_editor=bu.HistoryEditor(
            var='t'
        )
    )

    base_dir = tr.Directory
    def _base_dir_default(self):
        return Path.home() / 'simdb' / 'data' / 'shear_zone'

    data_dir = tr.Property
    """Directory with the data"""
    def _get_data_dir(self):
        return Path(self.base_dir) / self.dir_name

    time_F_w_data_dir = tr.Property
    """Directory with the load deflection data"""
    def _get_time_F_w_data_dir(self):
        return self.data_dir / 'load_deflection'

    time_F_w_file_name = tr.Str('load_deflection.csv')
    """Name of the file with the measured load deflection data
    """

    time_m_skip = bu.Int(50, ALG=True )

    time_F_w_m = tr.Property(depends_on='state_changed')
    """Read the load displacement values from the individual 
    csv files from the test
    """
    @tr.cached_property
    def _get_time_F_w_m(self):
        time_F_w_file = self.time_F_w_data_dir / self.time_F_w_file_name
        time_F_w_m = np.array(pd.read_csv(time_F_w_file, decimal=",", skiprows=1,
                              delimiter=None), dtype=np.float_)
        time_m, F_m, w_m = time_F_w_m[::self.time_m_skip, (0,1,2)].T
        F_m *= -1

        argtime_0 = np.argmax(time_m > self.time_0)
        time_m, F_m, w_m = time_m[argtime_0:], F_m[argtime_0:], w_m[argtime_0:]

        # End data criterion - generalize to introduce a logical condition to identify the final index
        argmax_w_m = np.argmax(w_m)
        return time_m[:argmax_w_m+1], F_m[:argmax_w_m+1], w_m[:argmax_w_m+1]
    
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

    f_F_time = tr.Property(depends_on="state_changed")
    @tr.cached_property
    def _get_f_F_time(self):
        """Return the load for a specified time"""
        time_m, F_m, _ = self.time_F_w_m
        return interp1d(time_m, F_m, kind='linear', bounds_error=True)

    f_time_F = tr.Property(depends_on="state_changed")
    """Return the times array for ascending load from zero to maximum"""
    @tr.cached_property
    def _get_f_time_F(self):
        time_m, F_m, _ = self.time_F_w_m
        argmax_F_m = np.argmax(F_m)
        return interp1d(F_m[:argmax_F_m+1], time_m[:argmax_F_m+1], kind='linear', bounds_error=True) 

    f_w_time = tr.Property(depends_on="state_changed")
    @tr.cached_property
    def _get_f_w_time(self):
        time_m, _, w_m = self.time_F_w_m
        return interp1d(time_m, w_m, kind='linear', bounds_error=True)

    f_time_w = tr.Property(depends_on="state_changed")
    """Return the times array for ascending deflection from zero to maximum"""
    @tr.cached_property
    def _get_f_time_w(self):
        time_m, _, w_m = self.time_F_w_m
        return interp1d(w_m, time_m, kind='linear', bounds_error=True)

    n_F = bu.Int(30, ALG=True)

    F_range = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_F_range(self):
        time_m, F_m, w_m = self.time_F_w_m
        argmax_F_m = np.argmax(F_m)
        return np.linspace(F_m[0], F_m[argmax_F_m], self.n_w)

    n_w = bu.Int(30, ALG=True)

    w_range = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_w_range(self):
        time_m, _, w_m = self.time_F_w_m
        return np.linspace(w_m[0], w_m[-1], self.n_w)

    def plot_load_deflection(self, ax_load):
        w_m = self.w_m
        _, F_m = self.time_F_m
        argmax_F_m = np.argmax(F_m)

        # ax_load.plot(w_m[:argmax_F_m], F_m[:argmax_F_m], color='black')
        ax_load.plot(w_m, F_m, color='black')
        ax_load.set_ylabel(r'$F$ [kN]')
        ax_load.set_xlabel(r'$w$ [mm]')

        # annotate the maximum load level
        max_F = F_m[argmax_F_m]
        argmax_w_F = w_m[argmax_F_m]
        ax_load.annotate(f'$F_{{\max}}=${max_F:.1f} kN,\nw={argmax_w_F:.2f} mm',
                    xy=(argmax_w_F, max_F), xycoords='data',
                    xytext=(0.05, 0.95), textcoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top',
                    )

    def subplots(self, fig):
        ax_time_F, ax_Fw = fig.subplots(1, 2)
        ax_time_w = ax_time_F.twinx()
        return ax_Fw, ax_time_F, ax_time_w

    def update_plot(self, axes):
        ax_Fw, ax_time_F, ax_time_w = axes
        self.plot_load_deflection(ax_Fw)
        time, F, w = self.time_F_w_m
        ax_time_F.plot(time, F, color='red', label='F')
        ax_time_F.set_ylabel(r'$F$/mm')
        ax_time_F.set_xlabel(r'time/ms')
        ax_time_w.plot(time, w, color='blue', label='w')
        ax_time_w.set_ylabel(r'$w$/mm')
        ax_time_F.legend()
        ax_time_w.legend()

