
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

    x_cor_pa_sol = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_x_cor_pa_sol(self):
        X_ija = self.dic_grid.X_ija[
                self.n_x_min:self.n_x_max:self.n_x_step,
                self.n_y_min:self.n_y_max:self.n_y_step,:]
        perp_u_ija, _, _ = self.dic_aligned_grid.displ_grids
        perp_u_ija = perp_u_ija[
                self.n_x_min:self.n_x_max:self.n_x_step,
                self.n_y_min:self.n_y_max:self.n_y_step,:]

        X_pa = X_ija.reshape(-1, 2)
        perp_u_pa = perp_u_ija.reshape(-1, 2)

        def get_x_cor_pa(eta_p):
            '''Get the points on the perpendicular lines with the sliders eta_p'''
            return X_pa + np.einsum('p,pa->pa', eta_p, perp_u_pa)

        def get_R(eta_p):
            '''Residuum of the closest distance condition'''
            x_cor_pa = get_x_cor_pa(eta_p)
            delta_x_cor_pqa = x_cor_pa[:, np.newaxis, :] - x_cor_pa[np.newaxis, :, :]
            R2 = np.einsum('pqa,pqa->', delta_x_cor_pqa, delta_x_cor_pqa)
            return np.sqrt(R2)

        eta0_p = np.zeros((X_pa.shape[0],))
        min_eta_p_sol = minimize(get_R, eta0_p, method='BFGS')
        eta_p_sol = min_eta_p_sol.x
        x_cor_pa_sol = get_x_cor_pa(eta_p_sol)
        return x_cor_pa_sol

    X_cor = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_X_cor(self):
        return np.average(self.x_cor_pa_sol, axis=0)


    X_cor_b = tr.Property(depends_on='state_changed')
    '''Global coordinates of COR.
    '''
    @tr.cached_property
    def _get_X_cor_b(self):
        X_a = self.X_cor
        X_pull_a = X_a - self.dic_aligned_grid.X_ref_a
        X_b = np.einsum('ba,a->b', self.dic_aligned_grid.T_ab, X_pull_a)
        X_push_b = X_b + self.dic_aligned_grid.X_ref_a
        return X_push_b

    def plot_cor(self, ax):
        rot_Xu_ija = self.dic_aligned_grid.rot_Xu_ija
        Xu_ija = rot_Xu_ija[
                self.n_x_min:self.n_x_max:self.n_x_step,
                self.n_y_min:self.n_y_max:self.n_y_step, :]
        Xu_aij = np.einsum('ija->aij', Xu_ija)
        ax.scatter(*Xu_aij.reshape(2,-1), s=7, marker='o', color='black')

        X_ref_b = self.dic_aligned_grid.X_ref_a
        ax.scatter([X_ref_b[0]],[X_ref_b[1]], s=20, color='green')

        # ax.plot(*rot_vect_u_anp, color='blue', linewidth=0.5);
        # ax.plot(*perp_vect_u_anp, color='green', linewidth=0.5);
        ax.plot(*self.x_cor_pa_sol.T, 'o', color = 'blue')
        ax.plot([self.X_cor[0]], [self.X_cor[1]], 'o', color='red')
        ax.axis('equal');

    def update_plot(self, axes):
        ax = axes
        _, rot_vect_u_anp, perp_vect_u_anp = self.dic_aligned_grid.displ_grids
        self.dic_aligned_grid.update_plot(axes)
        self.plot_cor(ax)
