
import traits.api as tr
import numpy as np
from scipy.optimize import root

from bmcs_shear_zone.shear_crack.crack_tip_orientation import \
    SZCrackTipOrientation
import bmcs_utils.api as bu
from bmcs_shear_zone.shear_crack.crack_tip_shear_stress import SZCrackTipShearStress
from bmcs_shear_zone.shear_crack.crack_tip_orientation import SZCrackTipOrientation


class CrackExtension(bu.InteractiveModel):
    """Find the parameters of the crack extension

    TODO: Check if there are redundant computations involved upon updates of psi and x_rot_1k
    """
    name = "Crack extension"
    crack_tip_shear_stress = tr.Instance(SZCrackTipShearStress, ())
    sz_stress_profile = tr.DelegatesTo('crack_tip_shear_stress')
    sz_cp = tr.DelegatesTo('crack_tip_shear_stress')
    sz_ctr = tr.DelegatesTo('sz_cp')
    beam_design = tr.DelegatesTo('sz_cp')
    crack_tip_orientation = tr.Property(depends_on='crack_tip_shear_stress')
    @tr.cached_property
    def _get_crack_tip_orientation(self):
        return SZCrackTipOrientation(
            crack_tip_shear_stress=self.crack_tip_shear_stress
        )

    def update_plot(self, ax):
        self.crack_tip_orientation.plot(ax)

    psi = tr.DelegatesTo('sz_ctr')
    x_rot_1k = tr.DelegatesTo('sz_ctr')

    t_n = tr.Float(0.0, auto_set=False, enter_set=True)
    '''Fundamental state time.
    '''
    t_n1 = tr.Float(0.0, auto_set=False, enter_set=True)
    '''Target value of the control variable.
    '''
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

    def init_state(self):
        '''Initialize state.
        '''
        self.t_n1 = 0.0
        self.U_n[:] = 0.0
        self.x_rot_1 = self.U_k[0]
        self.U_k = [self.psi,
                    self.x_rot_1k]
        self.X = self.U_k

    xtol = tr.Float(1e-3, auto_set=False, enter_set=True)
    maxfev = tr.Int(1000, auto_set=False, enter_set=True)

    def make_iter(self):
        '''Perform one iteration
        '''
        X0 = np.copy(self.X[:])

        def get_R_X(X):
            self.X = X
            R = self.get_R()
            return R
        res = root(get_R_X, X0, method='lm',
                   options={'xtol': self.xtol,
                            })
        self.X[:] = res.x
        self.psi = self.X[0]
        self.x_rot_1k = self.X[1]

        self.U_n[:] = self.U_k[:]
        R_k = self.get_R()
        nR_k = np.linalg.norm(R_k)
        print('R_k', nR_k, self.psi, self.x_rot_1k, 'Success', res.success)
        if res.success == False:
            raise StopIteration('no solution found')
        return res.x

    def make_incr(self):
        '''Update the control, primary and state variabrles..
        '''
        R_k = self.get_R()
        self.hist.record_timestep(self.t_n1, self.U_k, R_k)
        self.t_n = self.t_n1
        self.xd.x_t_Ia = np.vstack(
            [self.xd.x_t_Ia, self.xd.x1_fps_a[np.newaxis, :]])
        self.xd.state_changed = True
        self.state_changed = True

    X = tr.Property()

    def _get_X(self):
        return self.U_k

    def _set_X(self, value):
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




    ipw_view = bu.View(
    )

    def update_plot(self, ax):
        self.crack_tip_orientation.update_plot(ax)