from .dic_state_fields import DICStateFields
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
from scipy.interpolate import CubicSpline
from bmcs_shear.shear_crack.crack_path import get_T_Lab
import matplotlib.gridspec as gridspec
from .dic_stress_profile import DICStressProfile
from .i_dic_crack import IDICCrack

def get_f_ironed_weighted(x_, y_, r=10):
    """Averaging of a function using a bell-shaped ironing function
    """
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
    """Given a sequence of nodes, I with the coordinates a return
    the transformation matrices into the local coordinates of the lines.
    """
    norm_line_vec_L = np.sqrt(np.einsum('...a,...a->...',
                                        line_vec_La, line_vec_La))
    normed_line_vec_La = np.einsum('...a,...->...a',
                                   line_vec_La, 1. / norm_line_vec_L)
    t_vec_La = np.einsum('ijk,...j,k->...i',
                         EPS[:-1, :-1, :], normed_line_vec_La, Z);
    T_bLa = np.array([t_vec_La, normed_line_vec_La])
    T_Lab = np.einsum('bLa->Lab', T_bLa)
    return T_Lab

@tr.provides(IDICCrack)
class DICCrack(bu.Model):
    """
    Model of a shear crack with the representation of the kinematics
    evaluating the opening and sliding displacmeent
    """
    name = tr.Property(depends_on='C')

    @tr.cached_property
    def _get_name(self):
        return f'crack #{self.C}'

    cl = tr.WeakRef
    '''Reference to containing crack list
    '''

    bd = tr.DelegatesTo('cl')
    '''Access to the beam design available through crack list
    '''

    sp = bu.Instance(DICStressProfile)

    def _sp_default(self):
        return DICStressProfile(dic_crack=self)

    dic_grid = tr.Property()
    @tr.cached_property
    def _get_dic_grid(self):
        return self.cl.dsf.dic_grid

    tree = ['sp', 'dic_grid']

    C = bu.Int(0, ALG=True)
    '''Crack index within the crack list
    '''

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
        bu.Item('n_K_ligament'),
        bu.Item('w_H_plot_ratio'),
        bu.Item('plot_grid_markers'),
        bu.Item('plot_field'),
        bu.Item('T1', readonly=True),
        time_editor=bu.HistoryEditor(var='dic_grid.t')
    )

    dic_grid = tr.Property

    def _get_dic_grid(self):
        return self.cl.dsf.dic_grid

    T1 = tr.Property(bu.Int)

    def _get_T1(self):
        return self.dic_grid.T1

    C_cubic_spline = tr.Property(depends_on='state_changed')
    '''Smoothed crack profile
    '''

    @tr.cached_property
    def _get_C_cubic_spline(self):
        x_N, y_N = self.x_N, self.y_N
        x_U_N, y_U_N = x_N[:self.N_tip + 1], y_N[:self.N_tip + 1]
        x_U_N_irn = get_f_ironed_weighted(y_U_N, x_U_N, self.R)
        return CubicSpline(y_U_N, x_U_N_irn, bc_type='natural')

    X_tip_a = tr.Property(depends_on='state_changed')
    '''Resolution of the spline approximation over the whole height of the uncracked section
    '''

    @tr.cached_property
    def _get_X_tip_a(self):
        x_N, y_N = self.x_N, self.y_N
        return np.array([x_N[self.N_tip], y_N[self.N_tip]], dtype=np.float_)

    n_K_ligament = bu.Int(50, ALG=True)
    '''Resolution of the ligament points over the height
    '''

    H_ligament = tr.Property(depends_on='state_changed')
    '''Height of the ligament - information about positioning of the ligament
    within the cross section needed!
    '''

    @tr.cached_property
    def _get_H_ligament(self):
        return self.dic_grid.L_y

    # Return the ligament discretization given the crack tip X_tip_a

    def get_crack_ligament(self, X_tip_a):
        """Discretize the crack path along the identified spline
        up to X_tip_a and complete it with the vertical ligament
        returns:
        N_ligament, N_K_ligament, X_tip_a, C_spline
        """
        _, y_tip = X_tip_a
        d_y = y_tip / self.H_ligament
        n_K_crc = int(d_y * self.n_K_ligament)
        n_K_unc = int((self.H_ligament - y_tip) / self.H_ligament * self.n_K_ligament)
        y_N = self.y_N
        y_crc_range = np.linspace(y_N[0], y_tip, n_K_crc)
        X_crc_Ka = np.array([self.C_cubic_spline(y_crc_range), y_crc_range], dtype=np.float_).T
        y_unc_range = np.linspace(y_tip + d_y, y_N[-1], n_K_unc)
        X_unc_Ka = np.array([
            np.ones_like(y_unc_range) * self.C_cubic_spline(y_tip),
            y_unc_range], dtype=np.float_).T
        X_Ka = np.vstack([X_crc_Ka, X_unc_Ka])
        f_prime = self.C_cubic_spline.derivative()
        # cracked part
        d_C_x = -f_prime(X_crc_Ka[:, 1])
        e_C_Na = np.array([d_C_x, np.ones_like(d_C_x)]).T
        T_C_Nab = get_T_Lab(e_C_Na)
        # uncracked part
        e_U_Na = np.array([0, 1])[np.newaxis, :] * np.ones((len(X_unc_Ka),), np.float_)[:, np.newaxis]
        T_U_Nab = get_T_Lab(e_U_Na)
        T_Kab = np.vstack([T_C_Nab, T_U_Nab])
        return X_Ka, T_Kab, X_unc_Ka, X_crc_Ka, T_Kab[:n_K_crc]

    crack_ligament = tr.Property(depends_on='state_changed')
    '''Crack ligament nodes and transformation matrixes X_Ka, T_Kab
    at ultimate load
    '''

    @tr.cached_property
    def _get_crack_ligament(self):
        return self.get_crack_ligament(self.X_tip_a)

    crack_ligament_T1 = tr.Property(depends_on='state_changed')
    '''Crack ligament nodes and transformation matrixes X_Ka, T_Kab
    at the currently selected state
    '''

    @tr.cached_property
    def _get_crack_ligament_T1(self):
        return self.get_crack_ligament(self.X1_tip_a)

    # Crack ligament at ultimate state

    X_Ka = tr.Property(depends_on='state_changed')
    '''All ligament points at ultimate state.
    '''

    @tr.cached_property
    def _get_X_Ka(self):
        X_Ka, _, _, _, _ = self.crack_ligament
        return X_Ka

    T_Kab = tr.Property(depends_on='state_changed')
    '''Smoothed crack profile
    '''

    @tr.cached_property
    def _get_T_Kab(self):
        _, T_Kab, _, _, _ = self.crack_ligament
        return T_Kab

    # Crack ligament at an intermediate state T1

    X1_Ka = tr.Property(depends_on='state_changed')
    '''All ligament points at ultimate state.
    '''

    @tr.cached_property
    def _get_X1_Ka(self):
        X1_Ka, _, _, _, _ = self.crack_ligament_T1
        return X1_Ka

    T1_Kab = tr.Property(depends_on='state_changed')
    '''Transformation matrices into and from ligament coordinates
    '''

    @tr.cached_property
    def _get_T1_Kab(self):
        _, T1_Kab, _, _, _ = self.crack_ligament_T1
        return T1_Kab

    # Displacement jump evaluation

    def get_eps_Kab(self, t, X_Ka):
        d_x = self.d_x / 2
        x_K, y_K = X_Ka.T
        t_K = np.ones_like(x_K) * t
        tX_mid_K = np.array([t_K, x_K, y_K], dtype=np.float_).T
        eps_Kab = self.cl.dsf.f_eps_ipl_txy(tX_mid_K)
        eps_Ka, _ = np.linalg.eig(eps_Kab)
        max_eps_K = np.max(eps_Ka, axis=-1)
        K_eps = max_eps_K < self.bd.matrix_.eps_cr
        return eps_Kab, K_eps

    eps1_Kab = tr.Property(depends_on='state_changed')
    '''Global strain displacement of points along the crack at an intermediate state
    '''

    @tr.cached_property
    def _get_eps1_Kab(self):
        return self.get_eps_Kab(self.dic_grid.t, self.X1_Ka)

    eps1_Kcd = tr.Property(depends_on='state_changed')
    '''Local strain of points along the crack
    '''

    @tr.cached_property
    def _get_eps1_Kcd(self):
        return np.einsum('...ca,...ab,...bd->...cd',
                         self.T1_Kab, self.eps1_Kab, self.T1_Kab)


    d_x = bu.Float(30, ALG=True)
    '''Distance between the points across the crack to evaluate sliding and opening
    '''

    def get_U_Ka(self, t, X_Ka, X_tip_a):
        d_x = self.d_x / 2
        x_K, y_K = X_Ka.T
        t_K = np.ones_like(x_K) * t
        tX_right_K = np.array([t_K, x_K + d_x, y_K], dtype=np.float_).T
        tX_left_K = np.array([t_K, x_K - d_x, y_K], dtype=np.float_).T
        U_Ka = self.cl.dsf.f_U_ipl_txy(tX_right_K) - self.cl.dsf.f_U_ipl_txy(tX_left_K)

        eps_Kab, K_eps = self.get_eps_Kab(t, X_Ka)
        U_Ka[K_eps,0] = eps_Kab[K_eps, 0, 0] * self.bd.matrix_.L_cr
        U_Ka[K_eps,1] = eps_Kab[K_eps, 0, 1] * self.bd.matrix_.L_cr
        return U_Ka

    U_Ka = tr.Property(depends_on='state_changed')
    '''Global relative displacement of points along the crack at the ultimate state
    '''

    @tr.cached_property
    def _get_U_Ka(self):
        return self.get_U_Ka(1, self.X_Ka, self.X_tip_a)

    U1_Ka = tr.Property(depends_on='state_changed')
    '''Global relative displacement of points along the crack at an intermediate state
    '''

    @tr.cached_property
    def _get_U1_Ka(self):
        return self.get_U_Ka(self.dic_grid.t, self.X1_Ka, self.X1_tip_a)

    U1_Kb = tr.Property(depends_on='state_changed')
    '''Local relative displacement of points along the crack
    '''

    @tr.cached_property
    def _get_U1_Kb(self):
        U1_Ka = self.U1_Ka
        U1_Kb = np.einsum('...ab,...b->...a', self.T1_Kab, U1_Ka)
        return U1_Kb

    # Time-space interpolation of the damage function along the ligament

    t_T = tr.Property(depends_on='state_changed')
    '''Time slider values for the load levels
    '''

    @tr.cached_property
    def _get_t_T(self):
        n_dic_T = self.cl.dsf.dic_grid.n_dic_T
        return np.linspace(0, 1, n_dic_T)

    omega_TK = tr.Property(depends_on='state_changed')
    '''History of damage along the crack ligament
    '''

    @tr.cached_property
    def _get_omega_TK(self):
        X_Ka = self.X_Ka
        t_TKa, x_TKa = np.broadcast_arrays(self.t_T[:, np.newaxis, np.newaxis], X_Ka[np.newaxis, :, :])
        tx_Pb = np.hstack([t_TKa[..., 0].reshape(-1, 1), x_TKa.reshape(-1, 2)])
        # reshape the dimension of the result array back to TK
        return self.cl.dsf.f_omega_ipl_TMN(tx_Pb).reshape(len(self.t_T), -1)

    K_tip_T = tr.Property(depends_on='state_changed')
    '''Vertical index of the crack tip at time index T
    '''

    @tr.cached_property
    def _get_K_tip_T(self):
        return np.argmax(self.omega_TK < 0.1, axis=-1)

    X1_tip_a = tr.Property(depends_on='state_changed')
    '''Position of the crack tip at time index T1
    '''

    @tr.cached_property
    def _get_X1_tip_a(self):
        return self.X_Ka[self.K_tip_T[self.T1]]

    omega1_N = tr.Property(depends_on='state_changed')
    '''Damage along the ligament at current time t
    '''

    @tr.cached_property
    def _get_omega1_N(self):
        print('updating omega', self.dic_grid.t)
        return self.omega_TK[self.dic_grid.T1]

    # ----------------------------------------------------------
    # Plot functions
    # ----------------------------------------------------------
    def plot_X_Ka(self, ax_x):
        """Plot crack geometry at ultimate state
        """
        if self.plot_grid_markers:
            ax_x.plot(self.x_N, self.y_N, 'o')
        ax_x.plot(*self.X_Ka.T, color='black');
        ax_x.plot(*self.X_tip_a[:, np.newaxis], 'o', color='red')

    def plot_X1_Ka(self, ax):
        """Plot geometry at current state
        """
        ax.plot(*self.X1_Ka.T, linestyle='dashed', color='black');
        ax.plot(*self.X1_tip_a[:, np.newaxis], 'o', color='magenta')

    def _plot_U1(self, ax, U1_Ka, idx=0, color='black', label=r'$w$ [mm]'):
        """Helper function for plotting the displacement along the ligament
        """
        X1_Ka = self.X1_Ka
        ax.plot(U1_Ka[:, idx], X1_Ka[:, 1], color=color, label=label)
        ax.fill_betweenx(X1_Ka[:, 1], U1_Ka[:, idx], 0, color=color, alpha=0.1)
        ax.set_xlabel(label)
        ax.legend(loc='lower left')
        ax.set_xlim(np.min(self.U_Ka) * 1.04, np.max(self.U_Ka) * 1.04)

    def plot_U1_Ka(self, ax):
        """Plot the displacement along the crack (w and s) in global coordinates
        """
        self._plot_U1(ax, self.U1_Ka, 0, label=r'$u_x$ [mm]', color='blue')
        ax.set_xlabel(r'$u_x, u_y$ [mm]', fontsize=10)
        ax.plot([0], [self.X1_tip_a[1]], 'o', color='magenta')
        self._plot_U1(ax, self.U1_Ka, 1, label=r'$u_y$ [mm]', color='green')
        ax.set_title(r'displacement jump')
        ax.legend()
        ax.set_ylim(self.y_N[0], self.y_N[-1])
        ax.set_xlim(np.min(self.U_Ka) * 1.04, np.max(self.U_Ka) * 1.04)

    def plot_U1_Kb(self, ax_w):
        """Plot the displacement (u_x, u_y) in local crack coordinates
        """
        self._plot_U1(ax_w, self.U1_Kb, 0, label=r'$w$ [mm]', color='blue')
        ax_w.plot([0], [self.X1_tip_a[1]], 'o', color='magenta')
        ax_w.set_xlabel(r'$w, s$ [mm]', fontsize=10)
        self._plot_U1(ax_w, self.U1_Kb, 1, label=r'$s$ [mm]', color='green')
        ax_w.set_title(r'opening and sliding')
        ax_w.legend()
        ax_w.set_ylim(self.y_N[0], self.y_N[-1])

    def plot_U1_Nib(self, ax):
        """Plot the opening displacement in ligament coordinates at intermediate state
        """
        w_max = np.max(self.U1_Kb[:, 0])
        w_max_plot_size = self.H_ligament * self.w_H_plot_ratio
        w_plot_factor = w_max_plot_size / w_max
        U1_Kb = self.U1_Kb[:, 0, np.newaxis] * self.T1_Kab[..., 0, :] * w_plot_factor
        xU1_Kb = self.X1_Ka + U1_Kb
        l_iNb = np.array([self.X1_Ka, xU1_Kb], dtype=np.float_)
        l_biN = np.einsum('iNb->biN', l_iNb)
        ax.plot(*l_biN, color='orange', alpha=0.5)
        ax.plot(*xU1_Kb.T, color='orange')

    def plot_omega1_Ni(self, ax):
        """Plot the damage in ligament coordinates at ultimate state
        """
        w_max_plot_size = self.H_ligament * self.w_H_plot_ratio
        w_plot_factor = w_max_plot_size
        omega_Nb = self.omega1_N[:, np.newaxis] * self.T_Kab[..., 0, :] * w_plot_factor
        X_omega_Kb = self.X_Ka + omega_Nb
        l_iNb = np.array([self.X_Ka, X_omega_Kb], dtype=np.float_)
        l_biN = np.einsum('iNb->biN', l_iNb)
        ax.plot(*l_biN, color='blue', alpha=0.5)
        ax.plot(*X_omega_Kb.T, color='blue')

    def subplots(self, fig):
        self.fig = fig
        gs = gridspec.GridSpec(2, 3)
        ax_cl = fig.add_subplot(gs[0, :2])
        ax_FU = fig.add_subplot(gs[0, 2])
        ax_x = fig.add_subplot(gs[1, 0])
        ax_u = fig.add_subplot(gs[1, 1])
        ax_F = fig.add_subplot(gs[1, 2])
        ax_sig = ax_F.twiny()
        return ax_cl, ax_FU, ax_x, ax_u, ax_F, ax_sig

    plot_field = bu.Enum(options=['damage','strain','stress','--'])

    def update_plot(self, axes):
        ax_cl, ax_FU, ax_x, ax_u, ax_F, ax_sig = axes
        self.dic_grid.plot_bounding_box(ax_cl)
        self.dic_grid.plot_box_annotate(ax_cl)
        self.bd.plot_sz_bd(ax_cl)
        if self.plot_field == 'damage':
            self.cl.plot_crack_detection_field(ax_cl, self.fig)
        elif self.plot_field == 'stress':
            self.cl.dsf.plot_sig_field(ax_cl, self.fig)
        elif self.plot_field == 'strain':
            self.cl.dsf.plot_eps_field(ax_cl, self.fig)

        self.cl.plot_primary_cracks(ax_cl)
        self.plot_omega1_Ni(ax_cl)
        self.dic_grid.plot_load_deflection(ax_FU)
        self.plot_X_Ka(ax_x)
        self.plot_X1_Ka(ax_x)
        ax_x.set_title(r'smoothed ligament & opening')
        ax_x.set_xlabel(r'$x$ [mm]')
        ax_x.set_ylabel(r'$y$ [mm]')
        ax_x.axis('equal');
        ax_x.set_ylim(self.y_N[0], self.y_N[-1])
        # self.plot_U1_Nib(ax_x)
        self.plot_omega1_Ni(ax_x)
        self.plot_U1_Ka(ax_u)
        self.sp.plot_S_La(ax_sig)
        self.sp.plot_F_a(ax_F)
        bu.mpl_align_xaxis(ax_sig, ax_F)

        #self.plot_U1_Kb(ax_sig_0)
