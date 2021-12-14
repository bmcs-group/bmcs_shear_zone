
from .dic_aligned_grid import DICAlignedGrid
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
from scipy.optimize import minimize

class DICCOR(bu.Model):

    name = 'COR detector'

    dic_aligned_grid = bu.Instance(DICAlignedGrid, ())

    dic_grid = tr.DelegatesTo('dic_aligned_grid')

    n_x_min = bu.Int(0, ALG=True)
    n_x_max = bu.Int(-1, ALG=True)
    n_x_step = bu.Int(5, ALG=True)

    n_y_min = bu.Int(0, ALG=True)
    n_y_max = bu.Int(-1, ALG=True)
    n_y_step = bu.Int(5, ALG=True)

    crack_idx = bu.Int

    tree = ['dic_aligned_grid']

    t = bu.Float(1, ALG=True)
    def _t_changed(self):
        n_t = self.dic_grid.n_t
        d_t = (1 / n_t)
        self.dic_grid.end_t = int((n_t - 1) * (self.t + d_t / 2))

    ipw_view = bu.View(
        bu.Item('n_x_min'),
        bu.Item('n_x_max'),
        bu.Item('n_x_step'),
        bu.Item('n_y_min'),
        bu.Item('n_y_max'),
        bu.Item('n_y_step'),
        time_editor=bu.HistoryEditor(
            var='t'
        )
    )

    X_pa = tr.Property(depends_on='state_changed')
    '''Nodal coordinates of the crack rotation patch - flattened'''
    @tr.cached_property
    def _get_X_pa(self):
        X_ija = self.dic_grid.X_ija[
            self.n_x_min:self.n_x_max:self.n_x_step,
            self.n_y_min:self.n_y_max:self.n_y_step, :]
        X_pa = X_ija.reshape(-1, 2)
        return X_pa

    X_cor_pa_sol = tr.Property(depends_on='state_changed')
    '''Center of rotation determined for each patch point separately
    '''
    @tr.cached_property
    def _get_X_cor_pa_sol(self):
        xu_mid_ija, w_ref_ija = self.dic_aligned_grid.xu_mid_w_ref_ija
        xu_mid_ija = xu_mid_ija[
                self.n_x_min:self.n_x_max:self.n_x_step,
                self.n_y_min:self.n_y_max:self.n_y_step,:]
        w_ref_ija = w_ref_ija[
                self.n_x_min:self.n_x_max:self.n_x_step,
                self.n_y_min:self.n_y_max:self.n_y_step,:]

        xu_mid_pa = xu_mid_ija.reshape(-1, 2)
        w_ref_pa = w_ref_ija.reshape(-1, 2)

        def get_X_cor_pa(eta_p):
            '''Get the points on the perpendicular lines with the sliders eta_p'''
            return xu_mid_pa + np.einsum('p,pa->pa', eta_p, w_ref_pa)

        def get_R(eta_p):
            '''Residuum of the closest distance condition'''
            x_cor_pa = get_X_cor_pa(eta_p)
            delta_x_cor_pqa = x_cor_pa[:, np.newaxis, :] - x_cor_pa[np.newaxis, :, :]
            R2 = np.einsum('pqa,pqa->', delta_x_cor_pqa, delta_x_cor_pqa)
            return np.sqrt(R2)

        eta0_p = np.zeros((xu_mid_pa.shape[0],))
        min_eta_p_sol = minimize(get_R, eta0_p, method='BFGS')
        eta_p_sol = min_eta_p_sol.x
        X_cor_pa_sol = get_X_cor_pa(eta_p_sol)
        return X_cor_pa_sol

    X_cor = tr.Property(depends_on='state_changed')
    '''Center of rotation of the patch related 
    to the local patch reference system
    '''
    @tr.cached_property
    def _get_X_cor(self):
        return np.average(self.X_cor_pa_sol, axis=0)

    X_cor_b = tr.Property(depends_on='state_changed')
    '''Center of rotation within the global reference system
    '''
    @tr.cached_property
    def _get_X_cor_b(self):
        X_a = self.X_cor
        X_pull_a = X_a - self.dic_aligned_grid.X0_a
        X_b = np.einsum('ba,a->b', self.dic_aligned_grid.T_ab, X_pull_a)
        X_push_b = X_b + self.dic_aligned_grid.X0_a
        return X_push_b

    phi = tr.Property(depends_on='state_changed')
    '''Calculate of angle of rotation'''

    @tr.cached_property
    def _get_phi(self):
        end_t_arr = np.arange(0, self.dic_grid.end_t, 1)
        phi_arr = []
        for end_t in end_t_arr[::1]:
            #print('evaluating step', end_t)

            self.dic_grid.end_t = end_t

            # selected points for rotation
            XU_ija = self.dic_aligned_grid.x_ref_ija_scaled  # check the refence system and verify
            XU_ija_sel = (XU_ija[self.n_x_min:self.n_x_max:self.n_x_step,
                          self.n_y_min:self.n_y_max:self.n_y_step])
            XU_pr = XU_ija_sel.reshape(-1, 2)

            self.dic_grid.X_ija
            # selection of grid of points
            X_ija_sel = self.dic_grid.X_ija[self.n_x_min:self.n_x_max:self.n_x_step,
                        self.n_y_min:self.n_y_max:self.n_y_step]
            X_pr = X_ija_sel.reshape(-1, 2)

            # evaluating distances using distance formula
            X_cor_r = self.X_cor
            XU_mid_pr = (XU_pr + X_pr) / 2

            V_X_XU_mid_pr = X_cor_r[np.newaxis, :] - XU_mid_pr
            V_XU_XU_mid_pr = XU_pr - XU_mid_pr

            len_d_0c = np.sqrt(np.einsum('...i,...i->...', V_X_XU_mid_pr, V_X_XU_mid_pr))
            len_d_0t = np.sqrt(np.einsum('...i,...i->...', V_XU_XU_mid_pr, V_XU_XU_mid_pr))

            phi = 2 * np.arctan(len_d_0t / len_d_0c)
            phi = np.where(np.isnan(phi), 0, phi)
            # if phi.any == np.nan:
            #     phi = 0
            # else:
            #     phi = 2 * np.arctan(len_d_0t / len_d_0c)

            phi_avg = np.average(phi)
            phi_arr.append(phi_avg)

            #print('phi_avg', phi_avg)

        return phi_arr


    def subplots(self, fig):
        return fig.subplots(1,2)

    def plot_cor(self, axes):
        ax_u, ax_lr = axes
        x_ref_ija_scaled = self.dic_aligned_grid.x_ref_ija_scaled
        Xu_ija = x_ref_ija_scaled[
                self.n_x_min:self.n_x_max:self.n_x_step,
                self.n_y_min:self.n_y_max:self.n_y_step, :]
        Xu_aij = np.einsum('ija->aij', Xu_ija)
        ax_u.scatter(*Xu_aij.reshape(2,-1), s=7, marker='o', color='black')

        X0_b = self.dic_aligned_grid.X0_a
        ax_u.scatter([X0_b[0]],[X0_b[1]], s=20, color='green')

        # self.dic_grid.update_plot(axes)

        # ax.plot(*rot_vect_u_anp, color='blue', linewidth=0.5);
        # ax.plot(*perp_vect_u_anp, color='green', linewidth=0.5);
        ax_u.plot(*self.X_cor_pa_sol.T, 'o', color = 'blue')
        ax_u.plot([self.X_cor[0]], [self.X_cor[1]], 'o', color='red')
        ax_u.axis('equal');
        ax_lr.plot(self.phi[:-1], self.dic_grid.load_levels[:self.dic_grid.end_t], label='crack {}'.format(self.crack_idx))
        ax_lr.set_xlabel(r'$\varphi$'), ax_lr.set_ylabel('Load [KN]')
        ax_lr.legend()

    def update_plot(self, ax):
        ax = ax
        self.dic_aligned_grid.update_plot(ax)
        self.plot_cor(ax)
