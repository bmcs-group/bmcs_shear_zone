import bmcs_utils.api as bu
import traits.api as tr
from os.path import join, expanduser
import os
import numpy as np
import pandas as pd

class DICLoadRotation(bu.Model):

    name = 'DIC load rotation'

    dir_name = bu.Str('<unnamed>', ALG=True)

    data_dir = tr.Property
    '''Directory with the data'''

    def _get_data_dir(self):
        home_dir = expanduser('~')
        data_dir = join(home_dir, 'simdb', 'data', 'shear_zone', 'load', self.dir_name)
        return data_dir


    load_values = tr.Property(depends_on='state_changed')
    '''Read the load values from the individual csv files'''

    @tr.cached_property
    def _get_ld_values(self):
        files = [join(self.data_dir, each)
             for each in sorted(os.listdir(self.data_dir))
             if each.endswith('.csv')]
        load_values = np.array([
            np.loadtxt(csv_file, dtype=float, delimiter=',', usecols=(0), unpack=False)
        for csv_file in files
    ], dtype=np.float_)
        #print(ld_values[:,:,2])
        return load_values

    phi = tr.Property(depends_on='state_changed')
    '''Calculate of angle of rotation'''

    @tr.cached_property
    def _get_phi(self):
        end_t_arr = np.arange(1, 31, 1)
        phi_arr = []
        for end_t in end_t_arr[::1]:
            print('evaluating step', end_t)

            self.dic_grid.end_t = end_t

            # selected points for rotation
            XU_ija = self.dic_cor.dic_aligned_grid.rot_Xu_ija  # check the refence system and verify
            XU_ija_sel = (XU_ija[self.dic_cor.n_x_min:self.dic_cor.n_x_max:self.dic_cor.n_x_step,
                      self.dic_cor.n_y_min:self.dic_cor.n_y_max:self.dic_cor.n_y_step])
            XU_pr = XU_ija_sel.reshape(-1, 2)

            self.dic_cor.dic_grid.X_ija
            # selection of grid of points
            X_ija_sel = self.dic_cor.dic_grid.X_ija[self.dic_cor.n_x_min:self.dic_cor.n_x_max:self.dic_cor.n_x_step,
                    self.dic_cor.n_y_min:self.dic_cor.n_y_max:self.dic_cor.n_y_step]
            X_pr = X_ija_sel.reshape(-1, 2)

            # evaluating distances using distance formula
            X_cor_r = self.dic_cor.X_cor
            XU_mid_pr = (XU_pr + X_pr) / 2

            V_X_XU_mid_pr = X_cor_r[np.newaxis, :] - XU_mid_pr
            V_XU_XU_mid_pr = XU_pr - XU_mid_pr

            len_d_0c = np.sqrt(np.einsum('...i,...i->...', V_X_XU_mid_pr, V_X_XU_mid_pr))
            len_d_0t = np.sqrt(np.einsum('...i,...i->...', V_XU_XU_mid_pr, V_XU_XU_mid_pr))

            phi = 2 * np.arctan(len_d_0t / len_d_0c)
            phi_avg = np.average(phi)
            phi_arr.append(phi_avg)

            print('phi_avg', phi_avg)

            return phi_arr.append(phi_avg)

    def plot_lr(self, axes):
            ax = axes
            ax.plot(self.phi, self.load_values, color='black')

    def update_plot(self, ax):
            self.plot_lr(ax)

