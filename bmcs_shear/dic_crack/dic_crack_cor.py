
from hashlib import algorithms_guaranteed
from .i_dic_crack import IDICCrack
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

def rotate_around_ref(X_MNa, X_ref_a, T_ab):
    """Rotate the points around X_ref_a pull rotate and push back
    """
    X0_MNa = X_MNa - X_ref_a[np.newaxis, np.newaxis, :] # TODO - this can be done inplace
    # Rotate all points by the inclination of the vertical axis alpha
    x0_MNa = np.einsum('ba,...a->...b', T_ab, X0_MNa)
    # Return to the global coordinate system
    x_ref_mNa = x0_MNa + X_ref_a[np.newaxis, np.newaxis, :]
    return x_ref_mNa

class DICCrackCOR(bu.Model):
    '''Determination of the center of rotation.
    '''
    name = tr.Property
    @tr.cached_property
    def _get_name(self):
        return self.dic_crack.name

    dic_crack = bu.Instance(IDICCrack)

    cl = tr.DelegatesTo('dic_crack')

    dsf = tr.Property
    @tr.cached_property
    def _get_dsf(self):
        return self.dic_crack.cl.dsf

    a_grid = tr.Property
    @tr.cached_property
    def _get_a_grid(self):
        return self.dic_crack.cl.a_grid

    dic_grid = tr.Property()
    @tr.cached_property
    def _get_dic_grid(self):
        return self.dic_crack.dic_grid

    depends_on = ['dic_crack']

    delta_M = bu.Int(2, ALG=True)

    M0 = tr.Property(bu.Int, depends_on='state_changed')
    '''Horizontal index defining the start position fixed line.
    '''
    @tr.cached_property
    def _get_M0(self):
        if self.frame_position == 'vertical':
            return self.dic_crack.M_N[self.dic_crack.N_tip] - self.delta_M
        elif self.frame_position == 'inclined':
            return self.dic_crack.M_N[0] - self.delta_M

    N0 = bu.Int(1, ALG=True)
    '''Vertical index defining the start position of the fixed line.
    '''

    M1 = tr.Property(bu.Int, depends_on='state_changed')
    '''Horizontal index defining the end position the fixed line.
    '''
    @tr.cached_property
    def _get_M1(self):
        if self.frame_position == 'vertical':
            return self.dic_crack.M_N[self.dic_crack.N_tip] - self.delta_M
        elif self.frame_position == 'inclined':
            return self.dic_crack.M_N[self.dic_crack.N_tip] - self.delta_M

    N1 = tr.Property(bu.Int, depends_on='state_changed')
    '''Vertical index defining the end position of the fixed line.
    '''
    @tr.cached_property
    def _get_N1(self):
        return self.dic_crack.N_tip

    M_N = tr.Property(bu.Int, depends_on='state_changed')
    '''Horizontal indexes of the crack segments.
    '''
    @tr.cached_property
    def _get_M_N(self):
        return self.dic_crack.M_N

    frame_position = bu.Enum(options=[
        'vertical','inclined'
    ], ALG=True)

    ipw_view = bu.View(
        bu.Item('delta_M'),
        bu.Item('M0', readonly=True),
        bu.Item('N0', readonly=True),
        bu.Item('M1', readonly=True),
        bu.Item('N1', readonly=True),
        bu.Item('step_N_COR'),
        bu.Item('frame_position'),
        time_editor=bu.HistoryEditor(var='dic_crack.dic_grid.t')
    )

    step_N_COR = bu.Int(2, ALG=True)
    '''Vertical index distance between the markers included in the COR calculation
    '''

    MN_selection = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_MN_selection(self):
        upper_N = self.dic_crack.N_tip
        slice_N = slice(None, upper_N, self.step_N_COR)
        n_N = len(self.M_N)
        m = self.M_N[slice_N,np.newaxis] + np.arange(1,6)
        n = np.zeros_like(m) + np.arange(n_N)[slice_N, np.newaxis]
        return m.flatten(), n.flatten()

    VW_rot_t_pa = tr.Property(depends_on='state_changed')
    '''Displacement vector and the normal to its midpoint 
    '''
    @tr.cached_property
    def _get_VW_rot_t_pa(self):
        self.a_grid.trait_set(
            M0=self.M0, N0=self.N0, M1=self.M1, N1=self.N1,
            MN_selection=self.MN_selection
        )
        V_rot_t_pa, W_rot_t_pa = self.a_grid.VW_rot_t_MNa
        return V_rot_t_pa, W_rot_t_pa

    X_cor_rot_t_pa_sol = tr.Property(depends_on='state_changed')
    '''Center of rotation determined for each patch point separately
    '''
    @tr.cached_property
    def _get_X_cor_rot_t_pa_sol(self):

        V_rot_pa, W_rot_pa = self.VW_rot_t_pa

        def get_X_cor_pa(eta_p):
            '''Get the points on the perpendicular lines with the sliders eta_p'''
            return V_rot_pa + np.einsum('p,pa->pa', eta_p, W_rot_pa)

        def get_R(eta_p):
            '''Residuum of the closest distance condition.
            '''
            x_cor_pa = get_X_cor_pa(eta_p)
            delta_x_cor_pqa = x_cor_pa[:, np.newaxis, :] - x_cor_pa[np.newaxis, :, :]
            R2 = np.einsum('pqa,pqa->', delta_x_cor_pqa, delta_x_cor_pqa)
            return np.sqrt(R2)

        eta0_p = np.zeros((V_rot_pa.shape[0],))
        min_eta_p_sol = minimize(get_R, eta0_p, method='BFGS')
        eta_p_sol = min_eta_p_sol.x
        X_cor_pa_sol = get_X_cor_pa(eta_p_sol)
        return X_cor_pa_sol

    X_cor_rot_t_a = tr.Property(depends_on='state_changed')
    '''Center of rotation of the patch related to the local 
    patch reference system.
    '''
    @tr.cached_property
    def _get_X_cor_rot_t_a(self):
        return np.average(self.X_cor_rot_t_pa_sol, axis=0)

    X_cor_t_pa = tr.Property(depends_on='state_changed')
    '''Center of rotation of the patch related to the local 
    patch reference system.
    '''
    @tr.cached_property
    def _get_X_cor_t_pa(self):
        X_cor_pull_t_pa = np.einsum('ba,...b->...a', 
            self.a_grid.T_t_ab, self.X_cor_rot_t_pa_sol
        )
        X_cor_t_pa = X_cor_pull_t_pa + self.a_grid.X0_t_a
        return X_cor_t_pa

    X_cor_t_a = tr.Property(depends_on='state_changed')
    '''Center of rotation of the patch related to the local 
    patch reference system.
    '''
    @tr.cached_property
    def _get_X_cor_t_a(self):
        X_cor_pull_t_a = np.einsum('ba,...b->...a', 
            self.a_grid.T_t_ab, self.X_cor_rot_t_a
        )
        X_cor_t_a = X_cor_pull_t_a + self.a_grid.X0_t_a
        return X_cor_t_a

    def plot_X_cor_rot_t(self, ax):
        ax.plot(*self.X_cor_rot_t_pa_sol.T, 'o', color = 'blue')
        ax.plot([self.X_cor_rot_t_a[0]], [self.X_cor_rot_t_a[1]], 'o', color='red')
        ax.axis('equal');

    def plot_X_cor_t(self, ax):
        ax.plot(*self.X_cor_t_pa.T, 'o', color = 'blue')
        ax.plot([self.X_cor_t_a[0]], [self.X_cor_t_a[1]], 'o', color='red')
        ax.axis('equal');

    def plot_VW_rot_t(self, ax_x):
        V_rot_t_pa, W_rot_pa = self.VW_rot_t_pa
        ax_x.scatter(*V_rot_t_pa.T, s=20, color='orange')

    def subplots(self, fig):
        return fig.subplots(1,1)

    def update_plot(self, axes):
        ax_x = axes
        self.a_grid.trait_set(
            M0=self.M0, N0=self.N0, M1=self.M1, N1=self.N1,
            MN_selection=self.MN_selection
        )
        #self.a_grid.plot_rot(ax_x)
        self.dic_grid.plot_bounding_box(ax_x)
        self.dic_grid.plot_box_annotate(ax_x)

        self.a_grid.plot_selection_init(ax_x)
        self.plot_X_cor_t(ax_x)

        self.dic_crack.plot_x_1_Ka(ax_x)
        self.dic_crack.plot_x_t_Ka(ax_x)

        ax_x.axis('equal');
