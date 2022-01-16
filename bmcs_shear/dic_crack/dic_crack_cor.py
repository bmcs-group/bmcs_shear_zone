
from .dic_crack import DICCrack
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

class DICCrackCOR(bu.Model):
    '''Determination of the center of rotation
    '''
    name = tr.Property
    @tr.cached_property
    def _get_name(self):
        return self.dic_crack.name

    dic_crack = bu.Instance(DICCrack, ())

    cl = tr.DelegatesTo('dic_crack')

    dsf = tr.Property
    def _get_dsf(self):
        return self.dic_crack.cl.dsf

    dic_grid = tr.Property
    def _get_dic_grid(self):
        return self.dsf.dic_grid

    # tree = ['dic_crack']

    M0 = tr.Property(bu.Int, depends_on='state_changed')
    '''Horizontal index defining the y_axis position
    '''
    @tr.cached_property
    def _get_M0(self):
        return self.dic_crack.M_N[self.dic_crack.N_tip] - 3

    N0_min = bu.Int(1, ALG=True)
    '''Vertical index defining the start position of y axis
    '''

    N0_max = bu.Int(15, ALG=True)
    '''Vertical index defining the end position of y axis
    '''

    M_N = tr.Property(bu.Int, depends_on='state_changed')
    '''Horizontal indexes of the crack segments
    '''
    @tr.cached_property
    def _get_M_N(self):
        return self.dic_crack.M_N

    show_init = bu.Bool(False, ALG=True)
    show_xu = bu.Bool(True, ALG=True)
    show_xw = bu.Bool(False, ALG=True)

    t = bu.Float(1, ALG=True)
    '''Slider over the time history
    '''
    def _t_changed(self):
        n_t = self.dic_grid.n_t
        d_t = (1 / n_t)
        self.dic_grid.end_t = int((n_t - 1) * (self.t + d_t / 2))

    ipw_view = bu.View(
        bu.Item('M0'),
        bu.Item('N0_max'),
        bu.Item('show_init'),
        bu.Item('show_xu'),
        bu.Item('show_xw'),
        time_editor=bu.HistoryEditor(
            var='t'
        )
    )

    X_MNa = tr.Property(depends_on='state_changed')
    '''Marker positions of an interpolation grid. 
    '''
    @tr.cached_property
    def _get_X_MNa(self):
        return self.dsf.X_ipl_MNa

    U_MNa = tr.Property(depends_on='state_changed')
    '''Displacement field on an interpolation grid.
    '''
    @tr.cached_property
    def _get_U_MNa(self):
        return self.dsf.U_ipl_MNa

    X0_a = tr.Property(depends_on='state_changed')
    '''Position of the crack frame origin (CFO) 
    in DIC coordinate system.
    '''
    @tr.cached_property
    def _get_X0_a(self):
        return self.X_MNa[self.M0,0,:] + self.U0_a

    U0_a = tr.Property(depends_on='state_changed')
    '''Displacement of the crack frame origin (CFO).
    '''
    @tr.cached_property
    def _get_U0_a(self):
        return self.U_MNa[self.M0,0,:]

    U0_MNa = tr.Property(depends_on='state_changed')
    '''Displacement relative to the crack frame origin (CFO).
    '''
    @tr.cached_property
    def _get_U0_MNa(self):
        U0_11a = self.U0_a[np.newaxis,np.newaxis,:]
        U0_MNa = self.U_MNa - U0_11a
        return U0_MNa

    X0_MNa = tr.Property(depends_on='state_changed')
    '''Positions relative to crack frame origin (CFO).
    '''
    @tr.cached_property
    def _get_X0_MNa(self):
        X0_MNa = self.X_MNa + self.U0_MNa
        return X0_MNa

    alpha_ref = tr.Property(depends_on='state_changed')
    '''Crack frame rotation (CFR).
    '''
    @tr.cached_property
    def _get_alpha_ref(self):
        # Put the origin of the coordinate system into the reference point
        X_ref_Na = (self.X_MNa[self.M0, :self.N0_max,:] +
                    self.U0_MNa[self.M0, :self.N0_max,:])
        X0_a = X_ref_Na[0, :]
        X0_Na = X_ref_Na - X0_a[np.newaxis, :]
        alpha_ref = np.arctan(X0_Na[1:, 0] / X0_Na[1:, 1])
        return np.average(alpha_ref)

    T_ab = tr.Property(depends_on='state_changed')
    '''Rotation matrix of the crack frame.
    '''
    @tr.cached_property
    def _get_T_ab(self):
        alpha_ref = self.alpha_ref
        sa, ca = np.sin(alpha_ref), np.cos(alpha_ref)
        return np.array([[ca,-sa],
                         [sa,ca]])

    x_ref_MNa = tr.Property(depends_on='state_changed')
    '''Displacement increment relative to the rotated crack frame.
    '''
    @tr.cached_property
    def _get_x_ref_MNa(self):
        # Get the global displaced configuration without reference point displacement
        X0_MNa = self.X_MNa + self.U0_MNa
        # Put the origin of the coordinate system into the reference point
        X0_a = self.X0_a
        X0_MNa = X0_MNa - X0_a[np.newaxis, np.newaxis, :]
        # Rotate all points by the inclination of the vertical axis alpha
        x0_MNa = np.einsum('ba,...a->...b', self.T_ab, X0_MNa)
        # Return to the global coordinate system
        x_ref_MNa = x0_MNa + X0_a[np.newaxis, np.newaxis, :]
        return x_ref_MNa

    u_ref_MNa = tr.Property(depends_on='state_changed')
    '''Displacement increment relative to the crack frame.
    '''
    @tr.cached_property
    def _get_u_ref_MNa(self):
        return self.x_ref_MNa - self.X_MNa

    x_ref_MNa_scaled = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_x_ref_MNa_scaled(self):
        X_MNa = self.X_MNa
        return X_MNa + self.u_ref_MNa * self.dic_grid.U_factor

    xu_mid_w_ref_MNa = tr.Property(depends_on='state_changed')
    '''Get the midpoint on the displacement line and perpendicular
    vector w along which the search of the center of rotation can be defined.
    '''
    @tr.cached_property
    def _get_xu_mid_w_ref_MNa(self):
        X_MNa = self.X_MNa
        # find a midpoint on the line xu
        xu_ref_MNa = self.x_ref_MNa
        xu_ref_nija = np.array([X_MNa, xu_ref_MNa])
        xu_mid_MNa = np.average(xu_ref_nija, axis=0)
        # construct the perpendicular vector w
        u_ref_MNa = self.u_ref_MNa
        w_ref_aij = np.array([u_ref_MNa[..., 1], -u_ref_MNa[..., 0]])
        w_ref_MNa = np.einsum('aij->ija', w_ref_aij)
        return xu_mid_MNa, w_ref_MNa

    #####################################################################################
    X_mNa = tr.Property(depends_on='state_changed')
    '''Right crack plane positions.
    '''
    @tr.cached_property
    def _get_X_mNa(self):
        return np.array([self.X_MNa[self.M_N + i, np.arange(len(self.M_N))] for i in range(1,6)])

    U0_mNa = tr.Property(depends_on='state_changed')
    '''Displacement of the right crack plane relative to CFO
    '''
    @tr.cached_property
    def _get_U0_mNa(self):
        return np.array([self.U0_MNa[self.M_N + i, np.arange(len(self.M_N))] for i in range(1,6)])

    x_ref_mNa = tr.Property(depends_on='state_changed')
    '''Positions of the right plane within the rotated CF.
    '''
    @tr.cached_property
    def _get_x_ref_mNa(self):
        # Get the global displaced configuration without reference point displacement
        X0_mNa = self.X_mNa + self.U0_mNa
        # Put the origin of the coordinate system into the reference point
        X0_a = self.X0_a
        X0_mNa = X0_mNa - X0_a[np.newaxis, np.newaxis, :]
        # Rotate all points by the inclination of the vertical axis alpha
        x0_mNa = np.einsum('ba,...a->...b', self.T_ab, X0_mNa)
        # Return to the global coordinate system
        x_ref_mNa = x0_mNa + X0_a[np.newaxis, np.newaxis, :]
        return x_ref_mNa

    u_ref_mNa = tr.Property(depends_on='state_changed')
    '''Displacement increment relative to the crack frame.
    '''
    @tr.cached_property
    def _get_u_ref_mNa(self):
        return self.x_ref_mNa - self.X_mNa

    x_ref_mNa_scaled = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_x_ref_mNa_scaled(self):
        return self.X_mNa + self.u_ref_mNa * self.dic_grid.U_factor

    xu_mid_w_ref_mNa = tr.Property(depends_on='state_changed')
    '''Get the midpoint on the displacement line and perpendicular
    vector w along which the search of the center of rotation can be defined.
    '''
    @tr.cached_property
    def _get_xu_mid_w_ref_mNa(self):
        X_mNa = self.X_mNa
        # find a midpoint on the line xu
        xu_ref_mNa = self.x_ref_mNa
        xu_ref_nija = np.array([X_mNa, xu_ref_mNa])
        xu_mid_mNa = np.average(xu_ref_nija, axis=0)
        # construct the perpendicular vector w
        u_ref_mNa = self.u_ref_mNa
        w_ref_aij = np.array([u_ref_mNa[..., 1], -u_ref_mNa[..., 0]])
        w_ref_mNa = np.einsum('aij->ija', w_ref_aij)
        return xu_mid_mNa, w_ref_mNa

    displ_grids_scaled = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_displ_grids_scaled(self):
        X_mNa = self.X_mNa
        xu_ref_mNa_scaled = self.x_ref_mNa_scaled
        # construct the displacement vector v
        xu_ref_nija_scaled = np.array([X_mNa, xu_ref_mNa_scaled])
        xu_ref_anij_scaled = np.einsum('nija->anij', xu_ref_nija_scaled)
        xu_ref_anp_scaled = xu_ref_anij_scaled.reshape(2, 2, -1)
        # construct the perpendicular vector w
        xu_mid_mNa_scaled = np.average(xu_ref_nija_scaled, axis=0)
        u_ref_mNa = self.u_ref_mNa
        w_ref_amN = np.array([u_ref_mNa[..., 1], -u_ref_mNa[..., 0]])
        w_ref_mNa = np.einsum('amN->mNa', w_ref_amN)
        xw_ref_mNa_scaled = xu_mid_mNa_scaled + w_ref_mNa * self.dic_grid.U_factor
        xw_ref_nmNa_scaled = np.array([xu_mid_mNa_scaled, xw_ref_mNa_scaled])
        xw_ref_anmN_scaled = np.einsum('nmNa->anmN', xw_ref_nmNa_scaled)
        xw_ref_anp_scaled = xw_ref_anmN_scaled.reshape(2, 2, -1)
        return xu_ref_anp_scaled, xw_ref_anp_scaled

    ################################################################################

    X_cor_pa_sol = tr.Property(depends_on='state_changed')
    '''Center of rotation determined for each patch point separately
    '''
    @tr.cached_property
    def _get_X_cor_pa_sol(self):
        xu_mid_mNa, w_ref_mNa = self.xu_mid_w_ref_mNa
        xu_mid_mNa = xu_mid_mNa#[:, self.dic_crack.N_tip]
        w_ref_mNa = w_ref_mNa#[:, self.dic_crack.N_tip]
        xu_mid_pa = xu_mid_mNa.reshape(-1, 2)
        w_ref_pa = w_ref_mNa.reshape(-1, 2)

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
        X_pull_a = X_a - self.X0_a
        X_b = np.einsum('ba,a->b', self.dic_aligned_grid.T_ab, X_pull_a)
        X_push_b = X_b + self.X0_a
        return X_push_b

    def plot_COR(self, ax):
        ax.plot(*self.X_cor_pa_sol.T, 'o', color = 'blue')
        ax.plot([self.X_cor[0]], [self.X_cor[1]], 'o', color='red')
        ax.axis('equal');

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
        self.dic_crack.plot_x_Na(ax_x)
        self.dic_crack.plot_u_Nib(ax_x)
        self.dic_crack.plot_u_Na(ax_u_0)
        self.dic_crack.plot_u_Nb(ax_w_0)

        if self.show_init:
            X0_aMN = np.einsum('MNa->aMN', self.X0_MNa)
            ax_x.scatter(*X0_aMN.reshape(2,-1), s=15, marker='o', color='darkgray')
            X_aMN = np.einsum('MNa->aMN', self.X_MNa)
            ax_x.scatter(*X_aMN.reshape(2,-1), s=15, marker='o', color='blue')

        xu_ref_anp_scaled, xw_ref_anp_scaled = self.displ_grids_scaled

        if self.show_xu:
            ax_x.scatter(*xu_ref_anp_scaled[:, -1, :], s=15, marker='o', color='silver')
            ax_x.plot(*xu_ref_anp_scaled, color='silver', linewidth=0.5);

        if self.show_xw:
            ax_x.plot(*xw_ref_anp_scaled, color='green', linewidth=0.5);

        N0a = self.x_ref_MNa_scaled[self.M0, :self.N0_max]
        ax_x.scatter(*N0a.T, s=20, color='green')
        self.dic_crack.plot_x_Na(ax_x)
        self.plot_COR(ax_x)
        ax_x.axis('equal');

