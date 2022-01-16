
from .dic_state_fields import DICStateFields
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
from scipy.interpolate import CubicSpline
from bmcs_shear.shear_crack.crack_path import get_T_Lab
import matplotlib.gridspec as gridspec

def get_f_ironed_weighted(x_, y_, r=10):
    '''Averaging of a function using a bell-shaped ironing function
    '''
    x = np.hstack([x_, 2 * x_[-1] - x_[::-1]])
    y = np.hstack([y_, 2 * y_[-1] - y_[::-1]])
    RR = r
    delta_x = x[None, ...] - x[..., None]
    r2_n = (delta_x ** 2) / (2 * RR ** 2)
    alpha_r_MJ = np.exp(-r2_n)
    a_M = np.trapz(alpha_r_MJ, x, axis=-1)
    normed_a_MJ = np.einsum('MJ,M->MJ', alpha_r_MJ, 1 / a_M)
    y_MJ = np.einsum('MJ,J->MJ', normed_a_MJ, y)
    y_smooth = np.trapz(y_MJ, x, axis=-1)
    return y_smooth[:len(y_)]

EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1
Z = np.array([0, 0, 1], dtype=np.float_)

def get_T_Lab(line_vec_La):
    '''
    Given a sequence of nodes, I with the coordinates a return
    the transformation matrices into the local coordinates of the lines.
    '''
    norm_line_vec_L = np.sqrt(np.einsum('...a,...a->...',
                                        line_vec_La, line_vec_La))
    normed_line_vec_La = np.einsum('...a,...->...a',
                                   line_vec_La, 1. / norm_line_vec_L)
    t_vec_La = np.einsum('ijk,...j,k->...i',
                         EPS[:-1, :-1, :], normed_line_vec_La, Z);
    T_bLa = np.array([t_vec_La, normed_line_vec_La])
    T_Lab = np.einsum('bLa->Lab', T_bLa)
    return T_Lab

class DICCrack(bu.Model):
    '''
    Model of a shear crack with the representation of the kinematics
    evaluating the opening and sliding displacmeent
    '''
    name = tr.Property(depends_on='C')
    @tr.cached_property
    def _get_name(self):
        return f'crack #{self.C}'

    cl = tr.WeakRef

    C = bu.Int(0, ALG=True)

    x_N = tr.Array(np.float_)
    y_N = tr.Array(np.float_)
    N_tip = tr.Int(ALG=True)
    M_N = tr.Array(np.int_)

    R = bu.Float(20, ALG=True)
    tip_delta_N = bu.Int(3, ALG=True)
    w_H_plot_ratio = bu.Float(0.3, ALG=True)
    plot_grid_markers = bu.Bool(False, ALG=True)

    ipw_view = bu.View(
        bu.Item('C'),
        bu.Item('R'),
        bu.Item('d_x'),
        bu.Item('n_N_ligament'),
        bu.Item('w_H_plot_ratio'),
        bu.Item('plot_grid_markers'),
        bu.Item('t_idx', read_only=True),
        time_editor=bu.HistoryEditor(var='t')
    )

    dic_grid = tr.Property
    def _get_dic_grid(self):
        return self.cl.dsf.dic_grid

    t = bu.Float(1, ALG=True)
    def _t_changed(self):
        n_t = self.dic_grid.n_t
        d_t = (1 / n_t)
        self.dic_grid.end_t = int((n_t - 1) * (self.t + d_t / 2))
        self.t_idx = self.dic_grid.end_t

    t_idx = bu.Int(1)

    C_cubic_spline = tr.Property(depends_on='state_changed')
    '''Smoothed crack profile
    '''
    @tr.cached_property
    def _get_C_cubic_spline(self):
        x_N, y_N = self.x_N, self.y_N
        x_U_N, y_U_N = x_N[:self.N_tip+1], y_N[:self.N_tip+1]
        x_U_N_irn = get_f_ironed_weighted(y_U_N, x_U_N, self.R)
        return CubicSpline(y_U_N, x_U_N_irn, bc_type='natural')

    X_tip_a = tr.Property(depends_on='state_changed')
    '''Resolution of the spline approximation over the whole height of the uncracked section
    '''
    @tr.cached_property
    def _get_X_tip_a(self):
        x_N, y_N = self.x_N, self.y_N
        return np.array([x_N[self.N_tip], y_N[self.N_tip]], dtype=np.float_)

    n_N_ligament = bu.Int(50, ALG=True)
    '''Resolution of the ligament points over the height
    '''

    H_ligament = tr.Property(depends_on='state_changed')
    '''Height of the ligament - information about positioning of the ligament
    within the cross section needed!
    '''
    @tr.cached_property
    def _get_H_ligament(self):
        return self.dic_grid.L_y

    n_N_cracked = tr.Property(depends_on='state_changed')
    '''Resolution of the spline approximation over the height of the cracked
    '''
    @tr.cached_property
    def _get_n_N_cracked(self):
        _, y_tip = self.X_tip_a
        return int(y_tip / self.H_ligament * self.n_N_ligament)

    n_N_uncracked = tr.Property(depends_on='state_changed')
    '''Resolution of the spline approximation over the whole height of the uncracked section
    '''
    @tr.cached_property
    def _get_n_N_uncracked(self):
        _, y_tip = self.X_tip_a
        return int((self.H_ligament - y_tip) / self.H_ligament * self.n_N_ligament)

    x_C_Na = tr.Property(depends_on='state_changed')
    '''Cracked ligament points
    '''
    @tr.cached_property
    def _get_x_C_Na(self):
        y_N = self.y_N
        _, y_tip = self.X_tip_a
        y_cracked_range = np.linspace(y_N[0], y_tip, self.n_N_cracked)
        cracked_range = np.array([self.C_cubic_spline(y_cracked_range), y_cracked_range], dtype=np.float_).T
        return cracked_range

    x_U_Na = tr.Property(depends_on='state_changed')
    '''Uncracked ligament points
    '''
    @tr.cached_property
    def _get_x_U_Na(self):
        y_N = self.y_N
        _, y_tip = self.X_tip_a
        y_uncracked_range = np.linspace(y_tip, y_N[-1], self.n_N_uncracked)
        uncracked_range = np.array([
            np.ones_like(y_uncracked_range)*self.C_cubic_spline(y_tip),
            y_uncracked_range], dtype=np.float_).T
        return uncracked_range

    x_Na = tr.Property(depends_on='state_changed')
    '''All ligament points.
    '''
    @tr.cached_property
    def _get_x_Na(self):
        return np.vstack([self.x_C_Na, self.x_U_Na])

    T_Nab = tr.Property(depends_on='state_changed')
    '''Smoothed crack profile
    '''
    @tr.cached_property
    def _get_T_Nab(self):
        f_prime = self.C_cubic_spline.derivative()
        # cracked part
        d_C_x = -f_prime(self.x_C_Na[:, 1])
        e_C_Na = np.array([d_C_x, np.ones_like(d_C_x)]).T
        T_C_Nab = get_T_Lab(e_C_Na)
        # uncracked part
        e_U_Na = np.array([0, 1])[np.newaxis,:] * np.ones((len(self.x_U_Na),), np.float_)[:,np.newaxis]
        T_U_Nab = get_T_Lab(e_U_Na)
        return np.vstack([T_C_Nab, T_U_Nab])

    d_x = bu.Float(30, ALG=True)
    '''Distance between the points across the crack to evaluate sliding and opening
    '''

    u_Na = tr.Property(depends_on='state_changed')
    '''Global relative displacement of points along the crack'''
    @tr.cached_property
    def _get_u_Na(self):
        d_x = self.d_x / 2
        x_N, y_N = self.x_Na.T
        X_right = np.array([x_N + d_x, y_N], dtype=np.float_).T
        X_left = np.array([x_N - d_x, y_N], dtype=np.float_).T
        return self.cl.dsf.interp_U(X_right) - self.cl.dsf.interp_U(X_left)

    u_Nb = tr.Property(depends_on='state_changed')
    '''Local relative displacement of points along the crack'''
    @tr.cached_property
    def _get_u_Nb(self):
        u_Na = self.u_Na
        u_Nb = np.einsum('...ab,...b->...a', self.T_Nab, u_Na)
        return u_Nb

    #----------------------------------------------------------
    # Plot functions
    #----------------------------------------------------------
    def plot_u_Nib(self, ax):
        w_max = np.max(self.u_Nb[:, 0])
        w_max_plot_size = self.H_ligament * self.w_H_plot_ratio
        w_plot_factor = w_max_plot_size / w_max
        u_Nb = self.u_Nb[:, 0, np.newaxis] * self.T_Nab[..., 0, :] * w_plot_factor
        xu_Nb = self.x_Na + u_Nb
        l_iNb = np.array([self.x_Na, xu_Nb], dtype=np.float_)
        l_biN = np.einsum('iNb->biN', l_iNb)
        ax.plot(*l_biN, color='orange', alpha=0.5)
        ax.plot(*xu_Nb.T, color='orange')

    def plot_x_Na(self, ax_x):
        '''Plot crack geometry
        '''
        if self.plot_grid_markers:
            ax_x.plot(self.x_N, self.y_N, 'o')
        ax_x.plot(*self.x_Na.T, color='black');
        ax_x.plot(*self.X_tip_a[:,np.newaxis], 'o', color='red')
        ax_x.set_title(r'smoothed ligament & opening')
        ax_x.set_xlabel(r'$x$ [mm]')
        ax_x.set_ylabel(r'$y$ [mm]')
        ax_x.axis('equal');
        ax_x.set_ylim(self.y_N[0], self.y_N[-1])

    def _plot_u(self, ax, u_Na, idx=0, color='black', label=r'$w$ [mm]'):
        x_Na = self.x_Na
        ax.plot(u_Na[:, idx], x_Na[:, 1], color=color, label=label)
        ax.fill_betweenx(x_Na[:, 1], u_Na[:, idx], 0, color=color, alpha=0.1)
        ax.set_xlabel(label)
        ax.legend(loc='lower left')

    def plot_u_Na(self, ax_w):
        '''Plot the displacement along the crack (w and s) in global coordinates
        '''
        self._plot_u(ax_w, self.u_Na, 0, label=r'$u_x$ [mm]', color='blue')
        ax_w.set_xlabel(r'$u_x, u_y$ [mm]', fontsize=10)
        ax_w.plot([0], [self.X_tip_a[1]], 'o', color='red')
        self._plot_u(ax_w, self.u_Na, 1, label=r'$u_y$ [mm]', color='green')
        ax_w.set_title(r'displacement jump')
        ax_w.legend()
        ax_w.set_ylim(self.y_N[0], self.y_N[-1])

    def plot_u_Nb(self, ax_w):
        '''Plot the displacement (u_x, u_y) in local crack coordinates
        '''
        self._plot_u(ax_w, self.u_Nb, 0, label=r'$w$ [mm]', color='blue')
        ax_w.plot([0], [self.X_tip_a[1]], 'o', color='red')
        ax_w.set_xlabel(r'$w, s$ [mm]', fontsize=10)
        self._plot_u(ax_w, self.u_Nb, 1, label=r'$s$ [mm]', color='green')
        ax_w.set_title(r'opening and sliding')
        ax_w.legend()
        ax_w.set_ylim(self.y_N[0], self.y_N[-1])

    def subplots(self, fig):
        self.fig = fig
        gs = gridspec.GridSpec(2, 3)
        ax_cl = fig.add_subplot(gs[0, :2])
        ax_FU = fig.add_subplot(gs[0, 2])
        ax_x = fig.add_subplot(gs[1, 0])
        ax_u_0 = fig.add_subplot(gs[1, 1])
        ax_w_0 = fig.add_subplot(gs[1, 2])
        return ax_cl, ax_FU, ax_x, ax_u_0, ax_w_0

    def update_plot(self, axes):
        ax_cl, ax_FU, ax_x, ax_u_0, ax_w_0 = axes
        self.dic_grid.plot_bounding_box(ax_cl)
        self.dic_grid.plot_box_annotate(ax_cl)
        self.cl.plot_detected_cracks(ax_cl, self.fig)
        self.dic_grid.plot_load_deflection(ax_FU)
        self.plot_x_Na(ax_x)
        self.plot_u_Nib(ax_x)
        self.plot_u_Na(ax_u_0)
        self.plot_u_Nb(ax_w_0)
