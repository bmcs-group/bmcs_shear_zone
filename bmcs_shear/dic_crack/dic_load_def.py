import bmcs_utils.api as bu
import traits.api as tr
from os.path import join, expanduser
import os
import numpy as np
import pandas as pd

class DICLoadDeflection(bu.Model):

    name = 'DIC load deflection'

    dir_name = bu.Str('<unnamed>', ALG=True)

    data_dir = tr.Property
    '''Directory with the data'''

    def _get_data_dir(self):
        home_dir = expanduser('~')
        data_dir = join(home_dir, 'simdb', 'data', 'shear_zone', 'load_deflection', self.dir_name)
        return data_dir


    ld_values = tr.Property(depends_on='state_changed')
    '''Read the load displacement values from the individual csv files from the test'''

    @tr.cached_property
    def _get_ld_values(self):
        files = [join(self.data_dir, each)
             for each in sorted(os.listdir(self.data_dir))
             if each.endswith('.csv')]
        ld_values = np.array([
            pd.read_csv(csv_file, decimal=",", skiprows=1, delimiter=None)
        for csv_file in files
    ], dtype=np.float_)
        #print(ld_values[:,:,2])
        return ld_values

    def plot_ld(self, axes):
        ax = axes
        ax.plot(self.ld_values[:,:,2].flatten(), -self.ld_values[:,:,1].flatten(), color='black')

    def update_plot(self, ax):
        self.plot_ld(ax)


