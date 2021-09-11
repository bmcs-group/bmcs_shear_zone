
from .dic_crack import DICCrack
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
from scipy.optimize import minimize

class DICCOR(bu.Model):

    name = 'COR detector'

    dic_crack = bu.Instance(DICCrack, ())

    n_x_min = bu.Int(0, ALG=True)
    n_x_max = bu.Int(-1, ALG=True)
    n_x_step = bu.Int(5, ALG=True)

    n_y_min = bu.Int(0, ALG=True)
    n_y_max = bu.Int(-1, ALG=True)
    n_y_step = bu.Int(5, ALG=True)

    tree = ['dic_crack']

    ipw_view = bu.View(
        bu.Item('n_x_min'),
        bu.Item('n_x_max'),
        bu.Item('n_x_step'),
        bu.Item('n_y_min'),
        bu.Item('n_y_max'),
        bu.Item('n_y_step'),
    )

    x_cor_pa_sol = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_x_cor_pa_sol(self):
        dc = self.dic_crack
        perp_u_ija, _, _ = dc.displ_grids
        n_x, n_y = dc.n_x, dc.n_y
        X_ija = self.dic_crack.X_ija[
                self.n_x_min:self.n_x_max:self.n_x_step,
                self.n_y_min:self.n_y_max:self.n_y_step,:]
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

    def update_plot(self, axes):
        ax = axes
        _, rot_vect_u_anp, perp_vect_u_anp = self.dic_crack.displ_grids
        self.dic_crack.update_plot(axes)
        # ax.plot(*rot_vect_u_anp, color='blue', linewidth=0.5);
        # ax.plot(*perp_vect_u_anp, color='green', linewidth=0.5);
        ax.plot(*self.x_cor_pa_sol.T, 'o')
        ax.plot([self.X_cor[0]], [self.X_cor[1]], 'o', color='red')
        ax.axis('equal');
