
import traits.api as tr
import numpy as np
from scipy.optimize import root

import bmcs_utils.api as bu
from bmcs_shear.shear_crack.crack_tip_orientation import SZCrackTipOrientation

class CrackExtension(bu.InteractiveModel):
    """Find the parameters of the crack extension

    TODO: Check if there are redundant computations involved upon updates of psi and x_rot_1k

    TODO: Interaction - How to insert a trait DelegatedTo a model subcomponent
     into a current ipw_view? Widgets that are transient are not visible within the
     model component. Check if the trait object is necessary in the current model.

    TODO: Would it be possible to insert an instance as a group into a widget as well?

    TODO: Crack orientation - check the stress at crack normal to crack - should be f_t

    """
    name = "Crack extension"

    sz_cto = tr.Instance(SZCrackTipOrientation, ())

    sz_ctss = tr.DelegatesTo('sz_cto')
    sz_sp = tr.DelegatesTo('sz_ctss')
    sz_cp = tr.DelegatesTo('sz_ctss')
    sz_ctr = tr.DelegatesTo('sz_cp')
    sz_bd = tr.DelegatesTo('sz_cp')

    tree = [
        'sz_cto',
        'sz_bd'
    ]

    psi = tr.DelegatesTo('sz_ctr')
    w_cr = tr.DelegatesTo('sz_ctr','w')
    x_rot_1k = tr.DelegatesTo('sz_ctr')

    U_n = tr.Array(np.float_,
                   value=[0.0, 0.0], auto_set=False, enter_set=True)
    '''Current fundamental value of the primary variable.
    '''
    U_k = tr.Array(np.float_,
                   value=[0.0, 0.0], auto_set=False, enter_set=True)
    '''Primary unknown variables subject to the iteration process.
    - center of rotation
    - inclination angle of a new crack segment
    '''

    xtol = bu.Float(1e-3)
    '''Algorithmic parameter - tolerance
    '''

    psi_tol = bu.Float(0.02)
    '''Crack inclination criterion tolerance
    '''

    maxfev = tr.Int(1000)
    '''Algorithmic parameter maximum number of iterations
    '''

    def init(self):
        '''Initialize state.
        '''
        self.U_n[:] = 0.0
        self.U_k = [self.psi, self.x_rot_1k]
        self.X_iter = self.U_k

    X = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_X(self):
        self.init()
        self.make_iter()
        return self.X_iter

    ############### Implementation ################
    def make_iter(self):
        '''Perform one iteration
        '''
        X0 = np.copy(self.X_iter[:])
        def get_R_X(X):
            self.X_iter = X
            R = self.get_R()
            return R
        res = root(get_R_X, X0, method='hybr',
                   options={'xtol': self.xtol,})
        self.X_iter[:] = res.x
        self.psi = self.X_iter[0]
        # update w_cr based on the \sigma_2 value
        # set the w_cr = 1 / E_c * sig_2 * L_c
        self.x_rot_1k = self.X_iter[1]

        self.U_n[:] = self.U_k[:]
        R_k = self.get_R()
        psi_bar = self.sz_cto.get_psi()
        if np.fabs(self.psi - psi_bar) > np.pi/2 * self.psi_tol:
            print('non-matching crack direction')
            raise StopIteration('non-matching crack direction')
        if res.success == False:
            print('no convergence')
            raise StopIteration('no solution found')
        return res.x

    X_iter = tr.Property()

    def _get_X_iter(self):
        return self.U_k

    def _set_X_iter(self, value):
        self.U_k[:] = value
        self.psi = value[0]
        self.x_rot_1k = value[1]

    def get_R(self):
        '''Residuum checking the lack-of-fit
        - of the normal force equilibrium in the cross section
        - of the orientation of the principal stress and of the fracture
          process segment (FPS)
        '''
        N, _ = self.sz_sp.F_a
        M = self.sz_sp.M
        psi_bar = self.sz_cto.get_psi()
        # work of unbalanced moment devided by lever arm to obtain the right order
        N_M = M*(self.psi - psi_bar) / (self.sz_bd.H / 2)
        #N_M = N*(self.psi - psi_bar)
        R = np.array([N_M, N], dtype=np.float_)
        # print('psi', self.psi, psi_bar, R)
        return R

    def plot_geo(self, ax):
        sz_ds = self.sz_sp.sz_ds
        sz_ds.update_plot(ax)
        self.sz_cto.plot_crack_extension(ax)
        ax.axis('equal')

    def update_plot(self, ax):
        self.X
        self.plot_geo(ax)

