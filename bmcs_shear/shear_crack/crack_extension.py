
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

    crack_tip_orientation = tr.Instance(SZCrackTipOrientation, ())

    crack_tip_shear_stress = tr.DelegatesTo('crack_tip_orientation')
    sz_stress_profile = tr.DelegatesTo('crack_tip_shear_stress')
    sz_cp = tr.DelegatesTo('crack_tip_shear_stress')
    sz_ctr = tr.DelegatesTo('sz_cp')
    sz_bd = tr.DelegatesTo('sz_cp')

    tree = [
        'crack_tip_orientation'
    ]

    psi = tr.DelegatesTo('sz_ctr')
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

    xtol = tr.Float(1e-3, auto_set=False, enter_set=True)
    '''Algorithmic parameter - tolerance
    '''
    maxfev = tr.Int(1000, auto_set=False, enter_set=True)
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
        res = root(get_R_X, X0, method='lm',
                   options={'xtol': self.xtol,})
        self.X_iter[:] = res.x
        self.psi = self.X_iter[0]
        self.x_rot_1k = self.X_iter[1]

        self.U_n[:] = self.U_k[:]
        R_k = self.get_R()
        nR_k = np.linalg.norm(R_k)
        if res.success == False:
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
        N, _ = self.sz_stress_profile.F_a
        psi_bar = self.crack_tip_orientation.get_psi()
        R_M = (self.psi - psi_bar)
        R = np.array([R_M, N], dtype=np.float_)
        return R

    def plot_geo(self, ax):
        sz_cto = self.crack_tip_orientation
        ds = self.sz_stress_profile.ds
        ds.update_plot(ax)
        sz_cto.plot_crack_extension(ax)
        ax.axis('equal')

    def update_plot(self, ax):
        self.X
        self.plot_geo(ax)


