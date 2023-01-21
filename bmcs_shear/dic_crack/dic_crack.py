from .dic_state_fields import DICStateFields
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
from scipy.interpolate import CubicSpline
from bmcs_shear.shear_crack.crack_path import get_T_Lab
import matplotlib.gridspec as gridspec
from .dic_stress_profile import DICStressProfile
from .dic_crack_cor import DICCrackCOR
from .i_dic_crack import IDICCrack
from .dic_grid import DICGrid


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
    evaluating the opening and sliding displacement.
    """

    cl = tr.WeakRef
    '''Reference to containing crack list.
    '''
    def _cl_changed(self):
        self.dic_grid = self.cl.dsf.dic_grid

    bd = tr.DelegatesTo('cl')
    '''Access to the beam design available through crack list.
    '''

    sp = bu.Instance(DICStressProfile)
    '''Stress profile evaluating with evaluation of stress resultants.
    '''

    def _sp_default(self):
        return DICStressProfile(dic_crack=self)

    cor = bu.Instance(DICCrackCOR)
    '''Center of rotation evaluation.
    '''

    def _cor_default(self):
        return DICCrackCOR(dic_crack=self)

    dic_grid = bu.Instance(DICGrid)
    '''Input data grid.
    '''

    tree = [
        'sp',
        'cor'
    ]
    depends_on = ['dic_grid']

    C = bu.Int(0, ALG=True)
    '''Crack index within the crack list.
    '''

    x_N = tr.Array(np.float_)
    '''Horizontal coordinates of the crack path points.
    '''

    y_N = tr.Array(np.float_)
    '''Vertical coordinates of the crack path points.
    '''

    N_tip = tr.Int(ALG=True)
    '''Vertical index of the crack tip near failure load
    To obtain the vertical coordinate of the crack tip
    we can write self.y_N[self.N_tip].
    '''

    M_N = tr.Array(np.int_)
    '''Horizontal indexes of grid nodes corresponding 
    to the crack path.
    '''

    R = bu.Float(10, ALG=True)
    '''Ironing radius used to smoothen the crack path
    before constructing its spline representation.
    '''

    n_K_ligament = bu.Int(100, ALG=True)
    '''Resolution of the ligament points over the height.
    '''

    d_x = bu.Float(30, ALG=True)
    '''Distance between the points across the crack 
    to evaluate sliding and opening.
    '''

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
        bu.Item('T_t', readonly=True),
        time_editor=bu.HistoryEditor(var='dic_grid.t')
    )

    omega_threshold = tr.Property
    def _get_omega_threshold(self):
        return self.cl.dsf.omega_threshold * 0.5 # TODO - make transparent as a parameter

    T_t = tr.Property(bu.Int, depends_on='state_changed')
    @tr.cached_property
    def _get_T_t(self):
        return self.dic_grid.T_t

    C_cubic_spline = tr.Property(depends_on='state_changed')
    '''Smoothed crack profile.
    '''
    @tr.cached_property
    def _get_C_cubic_spline(self):
        x_N, y_N = self.x_N, self.y_N
        x_U_N, y_U_N = x_N[:self.N_tip + 1], y_N[:self.N_tip + 1]
        # Iron the shape of the crack
        x_U_N_irn = get_f_ironed_weighted(y_U_N, x_U_N, self.R)
        return CubicSpline(y_U_N, x_U_N_irn, bc_type='natural')

    x_1_tip_a = tr.Property(depends_on='state_changed')
    '''Coordinates of the crack tip near failure load.
    '''
    @tr.cached_property
    def _get_x_1_tip_a(self):
        x_N, y_N = self.x_N, self.y_N
        return np.array([x_N[self.N_tip], y_N[self.N_tip]], dtype=np.float_)

    H_ligament = tr.Property(depends_on='state_changed')
    '''Height of the ligament - information about positioning of the ligament
    within the cross section needed!
    '''
    @tr.cached_property
    def _get_H_ligament(self):
        # Distance between the lower and upper point
        return self.y_N[-1] - self.y_N[0]

    K_tip_1 = tr.Property(depends_on='state_changed')
    '''Indes of the tip in the failure state.
    '''
    @tr.cached_property
    def _get_K_tip_1(self):
        return np.argmax(self.x_1_Ka[:, 1] > self.x_1_tip_a[1]) + 1

    def get_crack_ligament(self, x_tip_a):
        """Discretize the crack path along the identified spline
        up to X_tip_a and complete it with the vertical ligament
        returns:
        X_Ka
        T_Kab
        X_unc_Ka
        X_crc_Ka
        T_Kab[:n_K_crc]
        """
        _, y_tip = x_tip_a - self.y_N[0]
        # Cracked fraction of cross-section
        y_tip_ratio = y_tip / self.H_ligament
        # number of discretization nodes in the cracked part
        n_K_crc = int(y_tip_ratio * self.n_K_ligament)
        d_y = self.H_ligament / self.n_K_ligament
        # y_K = np.linspace(self.y_N[0], self.y_N[-1], self.H_ligament)
        # n_K_crc = np.argmax(y_K > y_tip)
        # number of discretization nodes in the uncracked part
        n_K_unc = self.n_K_ligament - n_K_crc
        # n_K_unc = int((self.H_ligament - y_tip) / self.H_ligament * self.n_K_ligament)
        # vertical coordinates of the whole ligament
        y_N = self.y_N
        # discretize the ligament from the bottom to the crack tip
        y_crc_range = np.linspace(y_N[0], y_N[0] + y_tip, n_K_crc)
        # horizontal coordinates of the ligament nodes cracked
        X_crc_Ka = np.array([self.C_cubic_spline(y_crc_range), y_crc_range], dtype=np.float_).T
        # discretize the ligament from the crack tip to the top
        y_unc_range = np.linspace(y_N[0] + y_tip + d_y, y_N[-1], n_K_unc)
        # horizontal coordinates of the ligament nodes uncracked
        X_unc_Ka = np.array([
            np.ones_like(y_unc_range) * self.C_cubic_spline(y_N[0] + y_tip),
            y_unc_range], dtype=np.float_).T
        # horizontal coordinates of the whole ligament
        X_Ka = np.vstack([X_crc_Ka, X_unc_Ka])
        # derivative of the crack spline representation
        f_prime = self.C_cubic_spline.derivative()
        # array of derivatives along the ligament as horizontal component of the direction vector
        d_crc_x = -f_prime(X_crc_Ka[:, 1])
        # vector of unit values as a vertical component of the direction vector
        e_crc_Na = np.array([d_crc_x, np.ones_like(d_crc_x)]).T
        # use the direction vector to construct the transformation matrix
        T_crc_Nab = get_T_Lab(e_crc_Na)
        # uncracked part
        e_unc_Na = np.array([0, 1])[np.newaxis, :] * np.ones((len(X_unc_Ka),), np.float_)[:, np.newaxis]
        # use the direction vector to construct the transformation matrix
        T_unc_Nab = get_T_Lab(e_unc_Na)
        # arrays of transformation matrices
        T_Kab = np.vstack([T_crc_Nab, T_unc_Nab])
        return X_Ka, T_Kab, X_unc_Ka, X_crc_Ka, T_Kab[:n_K_crc]

    crack_ligament_1 = tr.Property(depends_on='state_changed')
    '''Crack ligament nodes and transformation matrixes x_1_Ka, T_Kab
    at ultimate load.
    '''
    @tr.cached_property
    def _get_crack_ligament_1(self):
        return self.get_crack_ligament(self.x_1_tip_a)

    crack_ligament_t = tr.Property(depends_on='state_changed')
    '''Crack ligament nodes and transformation matrixes X_Ka, T_Kab
    at the currently selected state.
    '''
    @tr.cached_property
    def _get_crack_ligament_t(self):
        return self.get_crack_ligament(self.x_t_tip_a)

    # Crack ligament at ultimate state

    x_1_Ka = tr.Property(depends_on='state_changed')
    '''All ligament points at ultimate state.
    '''
    @tr.cached_property
    def _get_x_1_Ka(self):
        x_1_Ka, _, _, _, _ = self.crack_ligament_1
        return x_1_Ka

    T_1_Kab = tr.Property(depends_on='state_changed')
    '''Smoothed crack profile.
    '''
    @tr.cached_property
    def _get_T_1_Kab(self):
        _, T_1_Kab, _, _, _ = self.crack_ligament_1
        return T_1_Kab

    x_1_crc_Ka = tr.Property(depends_on='state_changed')
    '''All ligament points at ultimate state.
    '''
    @tr.cached_property
    def _get_x_1_crc_Ka(self):
        _, _, _, x_1_crc_Ka, _ = self.crack_ligament_1
        return x_1_crc_Ka

    T_1_crc_Kab = tr.Property(depends_on='state_changed')
    '''Smoothed crack profile.
    '''
    @tr.cached_property
    def _get_T_1_crc_Kab(self):
        _, _, _, _, T_1_crc_Kab = self.crack_ligament_1
        return T_1_crc_Kab

    # Crack ligament at an intermediate state T_t

    x_t_Ka = tr.Property(depends_on='state_changed')
    '''All ligament points at intermediate state.
    '''
    @tr.cached_property
    def _get_x_t_Ka(self):
        x_t_Ka, _, _, _, _ = self.crack_ligament_t
        return x_t_Ka

    T_t_Kab = tr.Property(depends_on='state_changed')
    '''Transformation matrices into and from ligament coordinates.
    '''
    @tr.cached_property
    def _get_T_t_Kab(self):
        _, T_t_Kab, _, _, _ = self.crack_ligament_t
        return T_t_Kab

    x_t_unc_Ka = tr.Property(depends_on='state_changed')
    '''Uncracked ligament points at intermediate state.
    '''
    @tr.cached_property
    def _get_x_t_unc_Ka(self):
        _, _, x_t_unc_Ka, _, _ = self.crack_ligament_t
        return x_t_unc_Ka

    x_t_crc_Ka = tr.Property(depends_on='state_changed')
    '''All ligament points at intermediate state.
    '''
    @tr.cached_property
    def _get_x_t_crc_Ka(self):
        _, _, _, x_t_crc_Ka, _ = self.crack_ligament_t
        return x_t_crc_Ka

    T_t_crc_Kab = tr.Property(depends_on='state_changed')
    '''Smoothed crack profile.
    '''
    @tr.cached_property
    def _get_T_t_crc_Kab(self):
        _, _, _, _, T_t_crc_Kab = self.crack_ligament_t
        return T_t_crc_Kab

    # Strain evaluation

    def get_eps_Kab(self, t, x_Ka):
        """Get strain along the ligament by picking up the interpolated
        strain in the state field model.
        """
        x_K, y_K = x_Ka.T
        t_K = np.ones_like(x_K) * t
        tX_mid_K = np.array([t_K, x_K, y_K], dtype=np.float_).T
        eps_Kab = self.cl.dsf.f_eps_fe_txy(tX_mid_K)
        return eps_Kab

    eps_t_Kab = tr.Property(depends_on='state_changed')
    '''Global strain displacement of points along the 
    crack at an intermediate state.
    '''
    @tr.cached_property
    def _get_eps_t_Kab(self):
        return self.get_eps_Kab(self.dic_grid.t, self.x_t_unc_Ka)

    min_eps_1 = tr.Property(depends_on='state_changed')
    '''Global strain displacement of points along the 
    crack at an intermediate state.
    '''
    @tr.cached_property
    def _get_min_eps_1(self):
        eps_1_Kab = self.get_eps_Kab(1, self.x_1_Ka)
        min_eps_1 = np.min(eps_1_Kab[:,0,:])
        return min_eps_1

    max_u_1 = tr.Property(depends_on='state_changed')
    '''Global strain displacement of points along the 
    crack at an intermediate state.
    '''
    @tr.cached_property
    def _get_max_u_1(self):
        u_1_Ka = self.get_u_crc_Ka(1, self.x_1_Ka)
        u_1 = np.max(u_1_Ka[:,:])
        return u_1

    # Displacement jump evaluation

    def get_u_crc_Ka(self, t, X_crc_Ka):
        """Displacement jump across the crack.
        """
        d_x = self.d_x / 2
        x_K, y_K = X_crc_Ka.T
        t_K = np.ones_like(x_K) * t
        tX_right_K = np.array([t_K, x_K + d_x, y_K], dtype=np.float_).T
        tX_left_K = np.array([t_K, x_K - d_x, y_K], dtype=np.float_).T
        # handle the situation with coordinates outside the bounding box
        self.cl.dsf
        u_Ka = self.cl.dsf.f_U_ipl_txy(tX_right_K) - self.cl.dsf.f_U_ipl_txy(tX_left_K)

        # eps_Kab, K_eps = self.get_eps_Kab(t, X_Ka)
        # U_Ka[K_eps, 0] = eps_Kab[K_eps, 0, 0] * self.bd.matrix_.L_cr
        # U_Ka[K_eps, 1] = eps_Kab[K_eps, 0, 1] * self.bd.matrix_.L_cr
        return u_Ka

    u_1_crc_Ka = tr.Property(depends_on='state_changed')
    '''Global relative displacement of points along the crack 
    at the ultimate state.
    '''
    @tr.cached_property
    def _get_u_1_crc_Ka(self):
        return self.get_u_crc_Ka(1, self.x_1_crc_Ka)

    u_t_crc_Ka = tr.Property(depends_on='state_changed')
    '''Global relative displacement of points along the crack 
    at an intermediate state.
    '''
    @tr.cached_property
    def _get_u_t_crc_Ka(self):
        return self.get_u_crc_Ka(self.dic_grid.t, self.x_t_crc_Ka)

    u_t_crc_Kb = tr.Property(depends_on='state_changed')
    '''Local relative displacement of points along the crack 
    in an intermediate state.
    '''
    @tr.cached_property
    def _get_u_t_crc_Kb(self):
        u_t_crc_Ka = self.u_t_crc_Ka
        u_t_crc_Kb = np.einsum('...ab,...b->...a', self.T_t_crc_Kab, u_t_crc_Ka)
        return u_t_crc_Kb

    # Time-space interpolation of the damage function along the ligament

    t_T = tr.Property(depends_on='state_changed')
    '''Time slider values for the load levels.
    '''
    @tr.cached_property
    def _get_t_T(self):
        n_T = self.cl.dsf.dic_grid.n_T
        return np.linspace(0, 1, n_T)

    omega_TK = tr.Property(depends_on='state_changed')
    '''History of damage along the crack ligament.
    '''
    @tr.cached_property
    def _get_omega_TK(self):
        x_1_Ka = self.x_1_Ka
        # construct a grid with time and space dimensions
        t_TKa, x_TKa = np.broadcast_arrays(
            self.t_T[:, np.newaxis, np.newaxis], x_1_Ka[np.newaxis, :, :]
        )
        # flatten the time space such that points are flattened
        tx_Pb = np.hstack([t_TKa[..., 0].reshape(-1, 1), x_TKa.reshape(-1, 2)])
        # reshape the dimension of the result array back to TK
        return self.cl.dsf.f_omega_irn_txy(tx_Pb).reshape(len(self.t_T), -1)

    K_tip_T = tr.Property(depends_on='state_changed')
    '''Vertical index of the crack tip at time index T.
    '''
    @tr.cached_property
    def _get_K_tip_T(self):
        # array of tip indexes for each dic time step
        K_tip_T = np.zeros((self.dic_grid.n_T,), dtype=np.int_)
        # consider only steps with at least one non-zero damage value
        K_omega_T = np.where(np.sum(self.omega_TK, axis=-1) > 0)[0]
        # search from the top of the ligament the first occurrence of damage
        # larger than threshold. The search starts from the crack tip identified
        # for the ultimate state and goes downwards to the point where the
        # damage reaches the overall damage threshold omega_threshold.
        K_omega_tip_T = np.argmax(self.omega_TK[K_omega_T][:, :self.K_tip_1+1] <
                            self.omega_threshold, axis=-1)
        # identify the fully cracked ligaments - the argmax search did not identify them
        # since they did not drop below the omega_threshold
        fully_cracked = np.where(self.omega_TK[K_omega_T][:,self.K_tip_1] >= self.omega_threshold)
        K_omega_tip_T[fully_cracked] = self.K_tip_1
        # place the found indexes into the time array
        K_tip_T[K_omega_T] = K_omega_tip_T
        return K_tip_T

    N_tip_T = tr.Property(depends_on='state_changed')
    """Vertical index of the crack tip at time index T
    """
    def _get_N_tip_T(self):
        K_tip = self.K_tip_T[self.T_t]
        n_K = self.n_K_ligament
        n_N = len(self.M_N)
        N_tip = int(K_tip / n_K * n_N)
        return N_tip

    x_t_tip_a = tr.Property(depends_on='state_changed')
    '''Position of the crack tip at time index T_t.
    '''
    @tr.cached_property
    def _get_x_t_tip_a(self):
        return self.x_1_Ka[self.K_tip_T[self.T_t]]

    omega_t_N = tr.Property(depends_on='state_changed')
    '''Damage along the ligament at current time t.
    '''
    @tr.cached_property
    def _get_omega_t_N(self):
        return self.omega_TK[self.T_t]

    # ----------------------------------------------------------
    # Plot functions
    # ----------------------------------------------------------
    def plot_x_1_crc_Ka(self, ax_x, line_width=1, line_color='gray', tip_color='gray'):
        """Plot crack geometry at ultimate state.
        """
        ax_x.plot(*self.x_1_crc_Ka.T, linewidth=line_width, color=line_color);
        ax_x.plot(*self.x_1_tip_a[:, np.newaxis], 'o', color=tip_color)

    def plot_x_1_Ka(self, ax_x, line_width=1, line_color='gray', tip_color='gray'):
        """Plot crack geometry at ultimate state.
        """
        if self.plot_grid_markers:
            ax_x.plot(self.x_N, self.y_N, 'o')
        ax_x.plot(*self.x_1_Ka.T, linewidth=line_width, color=line_color);
        ax_x.plot(*self.x_1_tip_a[:, np.newaxis], 'o', color=tip_color)

    def plot_x_t_crc_Ka(self, ax, line_width=1, line_color='black', tip_color='red'):
        """Plot geometry at current state.
        """
        ax.plot(*self.x_t_crc_Ka.T, linewidth=line_width, color=line_color);
        ax.plot(*self.x_t_tip_a[:, np.newaxis], 'o', color=tip_color)

    def plot_x_t_Ka(self, ax):
        """Plot geometry at current state.
        """
        ax.plot(*self.x_t_Ka.T, linewidth=1, linestyle='dotted', color='black');
        ax.plot(*self.x_t_tip_a[:, np.newaxis], 'o', color='red')

    def _plot_eps_t(self, ax, eps_t_Kab, a=0, b=0,
                    linestyle='dotted', color='black', label=r'$\varepsilon$ [-]'):
        """Helper function for plotting the strain along the ligament
        at an intermediate state.
        """
        x_t_unc_La = self.sp.x_t_unc_La
        ax.plot(eps_t_Kab[:, a, b], x_t_unc_La[:, 1], linestyle=linestyle, color=color, label=label)
        ax.fill_betweenx(x_t_unc_La[:, 1], eps_t_Kab[:, a, b], 0, color=color, alpha=0.1)
        ax.set_xlabel(label)
        ax.legend(loc='lower left')
        min_eps_1 = self.min_eps_1
        ax.set_xlim(xmin=min_eps_1 * 1.04, xmax=-min_eps_1 * 1.04)
        # eps_K = self.eps_Kab[:, idx, idx]
        # ax.set_xlim(np.min(eps_K) * 1.04, np.max(self.u_1_crc_Ka) * 1.04)

    def plot_eps_t_Kab(self, ax_eps):
        """Plot the displacement (u_x, u_y) in local crack coordinates
        at an intermediate state.
        """
        self._plot_eps_t(ax_eps, self.sp.eps_t_unc_Lab, 0, 0, linestyle='dotted', color='blue')
        self._plot_eps_t(ax_eps, self.sp.eps_t_unc_Lab, 0, 1, linestyle='dotted', color='green')
        ax_eps.set_xlabel(r'$\varepsilon$ [-]', fontsize=10)
        # ax_eps.legend()
        # ax_eps.set_ylim(self.y_N[0], self.y_N[-1])

    def _plot_u_t(self, ax, u_t_crc_Ka, idx=0, color='black', label=r'$w$ [mm]'):
        """Helper function for plotting the displacement along the ligament
        at an intermediate state.
        """
        x_t_crc_Ka = self.x_t_crc_Ka
        ax.plot(u_t_crc_Ka[:, idx], x_t_crc_Ka[:, 1], color=color, label=label)
        ax.fill_betweenx(x_t_crc_Ka[:, 1], u_t_crc_Ka[:, idx], 0, color=color, alpha=0.1)
        ax.set_xlabel(label)
        ax.legend(loc='lower left')
        max_u_1 = self.max_u_1
        ax.set_xlim(xmin=-max_u_1 * 1.04, xmax=max_u_1 * 1.04)
        # ax.set_xlim(np.min(self.u_1_crc_Ka) * 1.04, np.max(self.u_1_crc_Ka) * 1.04)

    def plot_u_t_crc_Ka(self, ax):
        """Plot the displacement along the crack (w and s)
        in global coordinates at an intermediate state.
        """
        self._plot_u_t(ax, self.u_t_crc_Ka, 0, label=r'$u_x$ [mm]', color='blue')
        ax.set_xlabel(r'$u_x, u_y$ [mm]', fontsize=10)
        ax.plot([0], [self.x_t_tip_a[1]], 'o', color='magenta')
        self._plot_u_t(ax, self.u_t_crc_Ka, 1, label=r'$u_y$ [mm]', color='green')
        ax.legend()
        ax.set_ylim(self.y_N[0], self.y_N[-1])
        #ax.set_xlim(np.min(self.u_1_crc_Ka) * 1.04, np.max(self.u_1_crc_Ka) * 1.04)

    def plot_u_t_crc_Kb(self, ax_w):
        """Plot the displacement (u_x, u_y) in local crack coordinates
        at an intermediate state.
        """
        self._plot_u_t(ax_w, self.u_t_crc_Kb, 0, label=r'$w$ [mm]', color='blue')
        ax_w.plot([0], [self.x_t_tip_a[1]], 'o', color='magenta')
        ax_w.set_xlabel(r'$w, s$ [mm]', fontsize=10)
        self._plot_u_t(ax_w, self.u_t_crc_Kb, 1, label=r'$s$ [mm]', color='green')
        ax_w.legend()
        ax_w.set_ylim(self.y_N[0], self.y_N[-1])

    def plot_u_t_Nib(self, ax):
        """Plot the opening displacement in ligament coordinates
        at an intermediate state.
        """
        w_max = np.max(self.u_t_crc_Kb[:, 0])
        w_max_plot_size = self.H_ligament * self.w_H_plot_ratio
        w_plot_factor = w_max_plot_size / w_max
        u_t_crc_Kb = self.u_t_crc_Kb[:, 0, np.newaxis] * self.T_t_crc_Kab[..., 0, :] * w_plot_factor
        xu_t_crc_Kb = self.x_t_crc_Ka + u_t_crc_Kb
        l_iNb = np.array([self.x_t_crc_Ka, xu_t_crc_Kb], dtype=np.float_)
        l_biN = np.einsum('iNb->biN', l_iNb)
        ax.plot(*l_biN, color='orange', alpha=0.5)
        ax.plot(*xu_t_crc_Kb.T, color='orange')

    def plot_omega_t_Ni(self, ax):
        """Plot the damage in ligament coordinates
        at an ultimate state.
        """
        w_max_plot_size = self.H_ligament * self.w_H_plot_ratio
        w_plot_factor = w_max_plot_size
        omega_Nb = self.omega_t_N[:, np.newaxis] * self.T_1_Kab[..., 0, :] * w_plot_factor
        x_1_omega_Kb = self.x_1_Ka + omega_Nb
        l_iNb = np.array([self.x_1_Ka, x_1_omega_Kb], dtype=np.float_)
        l_biN = np.einsum('iNb->biN', l_iNb)
        ax.plot(*l_biN, color='blue', alpha=0.5)
        ax.plot(*x_1_omega_Kb.T, color='blue')

    def subplots(self, fig):
        self.fig = fig
        gs = gridspec.GridSpec(ncols=3, nrows=2,
                               width_ratios=[1, 1, 1],
                               wspace=0.5,
                               # hspace=0.5,
                               height_ratios=[2, 1]
                               )
        ax_dsf = fig.add_subplot(gs[0, :])
        ax_FU = fig.add_subplot(gs[1, 0])
        ax_u = fig.add_subplot(gs[1, 1])
        ax_eps = ax_u.twiny()
        ax_F = fig.add_subplot(gs[1, 2])
        ax_sig = ax_F.twiny()
        return ax_dsf, ax_FU, ax_u, ax_eps, ax_F, ax_sig

    plot_field = bu.Enum(options=['damage', 'strain', 'stress', '--'])

    def update_plot(self, axes):
        ax_cl, ax_FU, ax_u, ax_eps, ax_F, ax_sig = axes
        self.dic_grid.plot_bounding_box(ax_cl)
        self.dic_grid.plot_box_annotate(ax_cl)
        self.bd.plot_sz_bd(ax_cl)
        if self.plot_field == 'damage':
            self.cl.dsf.plot_crack_detection_field(ax_cl, self.fig)
        elif self.plot_field == 'stress':
            self.cl.dsf.plot_sig_field(ax_cl, self.fig)
        elif self.plot_field == 'strain':
            self.cl.dsf.plot_eps_field(ax_cl, self.fig)

        self.cl.plot_primary_cracks(ax_cl)
        # self.cor.plot_X_cor_t(ax_cl)

        self.plot_omega_t_Ni(ax_cl)
        self.dic_grid.plot_load_deflection(ax_FU)
        self.plot_x_1_Ka(ax_cl)
        self.plot_x_t_Ka(ax_cl)
        ax_cl.axis('equal');
        # ax_x.set_ylim(self.y_N[0], self.y_N[-1])
        # self.plot_u_t_Nib(ax_x)
        self.plot_omega_t_Ni(ax_cl)
        self.plot_u_t_crc_Ka(ax_u)
        self.plot_eps_t_Kab(ax_eps)
        ax_eps.set_ylim(0, self.bd.H)
        ax_u.set_ylim(0, self.bd.H * 1.04)
        bu.mpl_align_xaxis(ax_u, ax_eps)

        if 'sp' in self.tree:
            self.sp.plot_sig_t_unc_Lab(ax_sig)
            self.sp.plot_sig_t_crc_La(ax_sig)

            self.sp.plot_F_t_a(ax_F)
            bu.mpl_align_xaxis(ax_sig, ax_F)

        # self.plot_u_t_Kb(ax_sig_0)
