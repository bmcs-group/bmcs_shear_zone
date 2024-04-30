import bmcs_utils.api as bu
import traits.api as tr
from pathlib import Path
from .dic_inp_ld_time import DICInpLDTime

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# This was not directly working
class DICInpTimeSync(bu.Model):
    """
    Time synchronizer.

    This class is used to derive the time shift for secondary input channels. 
    There are multiple criteria possible, including 
     - manually specified time shift
     - time shift derived based on an identical functional dependence 
       stored both in the primary and secondary channels
     - qualitative inspection of a profile of a displacement function
       in the primary and secondary channel.

    This class implements the second case assuming that the GOM chanel includes 
    the time->load mapping. Alternative synchronizations can be produced by subclassing.

    Normally, the primary chanel contains a displacement or strain profile. By defining
    the idea is to define qualitatively comparable monitor in the DIC or FOS data which 
    for which the functional dependency can be defined in a certain time frame. By constructing 
    the derivative of the time function and matching the zero points of this function, the 
    maxima and minima of the response variable can be matched.
    """
    name = 'DIC time synchronizer'
    
    ld_time = bu.Instance(DICInpLDTime, ())

    ipw_view = bu.View(
        bu.Item('time_shift', readonly=True),
    )

    depends_on = ['ld_time']
    tree = ['ld_time']

    dir_name = tr.DelegatesTo('ld_time')
    base_dir = tr.DelegatesTo('ld_time')
    data_dir = tr.DelegatesTo('ld_time')
    dic_data_dir = tr.DelegatesTo('ld_time')

    dic_data_dir = tr.Property
    """Directory with the DIC data"""
    def _get_dic_data_dir(self):
        return self.data_dir / 'dic_point_data'
    
    time_F_dic_file = tr.Property(depends_on='state_changed')
    """File specifying the dic snapshot times and forces
    """
    @tr.cached_property
    def _get_time_F_dic_file(self):
        return self.dic_data_dir / 'Kraft.DIM.csv'

    time_F_m = tr.Property(depends_on='state_changed')
    """Read the load values from the single dic file exported by GOM
    """
    @tr.cached_property
    def _get_time_F_m(self):
        rows_ = list(open(self.time_F_dic_file))
        rows = np.array(rows_)
        time_entries_dic = rows[0].replace('name;type;attribute;id;', '').split(';')
        tstring_dic = np.array([time_entry_dic.split(' ')[0] for time_entry_dic in time_entries_dic])
        time_dic = np.array(tstring_dic, dtype=np.float_)
        F_entries_dic = rows[1].replace('Kraft.DIM;deviation;dimension;;', '')
        F_dic = -np.fromstring(F_entries_dic, sep=';', dtype=np.float_)
        return time_dic, F_dic
    
    time_shift = tr.Property(bu.Float)
    def _get_time_shift(self):
        return self.ld_time.argmax_F_time - self.argmax_F_time

    argmax_F_m = tr.Property(depends_on="state_changed")
    @tr.cached_property
    def _get_argmax_F_m(self):
        _, F_m = self.time_F_m
        return np.argmax(F_m)

    argmax_F_time = tr.Property(depends_on="state_changed")
    @tr.cached_property
    def _get_argmax_F_time(self):
        """Return the time for the maximum load"""
        time_m, _ = self.time_F_m
        argmax_F_m = self.argmax_F_m
        return time_m[argmax_F_m]
    
    def plot_time_F(self, ax):
        time_m, F_m = self.time_F_m
        ax.plot(time_m + self.time_shift, F_m, '-', color='blue')
        ax.set_ylabel(r'$F$/kN')
        ax.set_xlabel(r'$time$/-')
        ax.scatter([self.argmax_F_time] + self.time_shift,[F_m[self.argmax_F_m]], color='orange')

    def subplots(self, fig):
        ax_time_F = fig.subplots(1,)
        return ax_time_F

    def update_plot(self, axes):
        ax_time_F = axes
        self.plot_time_F(ax_time_F)
