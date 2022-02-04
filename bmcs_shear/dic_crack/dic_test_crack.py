
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
from bmcs_shear.beam_design import RCBeamDesign
from .dic_crack import IDICCrack, get_T_Lab
import matplotlib.gridspec as gridspec
from .dic_stress_profile import DICStressProfile

@tr.provides(IDICCrack)
class DICTestCrack(bu.Model):
    '''
    Test crack model to be used for verification
    of stress profile evaluation for elementary crack geometries
    '''
    name = 'test crack'

    bd = bu.Instance(RCBeamDesign, ())

    sp = bu.Instance(DICStressProfile)

    def _sp_default(self):
        return DICStressProfile(dic_crack=self)

    @tr.on_trait_change('+ALG')
    def _sp_state_change(self, event):
        self.sp.state_changed = True

    tree = ['sp']

    u_x_bot = bu.Float(0.03, ALG=True)
    u_x_top = bu.Float(-0.03, ALG=True)
    u_y_bot = bu.Float(0.0, ALG=True)
    u_y_top = bu.Float(0.0, ALG=True)

    U_K = bu.Int(4, ALG=True)

    ipw_view = bu.View(
        bu.Item('u_x_bot'),
        bu.Item('u_x_top'),
        bu.Item('u_y_bot'),
        bu.Item('u_y_top'),
        bu.Item('U_K')
    )

    X_tip_a = tr.Array(value=[0, 50])

    X_Ka = tr.Property(tr.Array, depends_on='state_changed')
    '''All ligament points.
    '''
    @tr.cached_property
    def _get_X_Ka(self):
        x_0 = np.zeros((self.U_K,), dtype=np.float_)
        x_1 = np.linspace(0, self.bd.H, self.U_K)
        return np.array([x_0, x_1], dtype=np.float_).T

    U1_Ka = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_U1_Ka(self):
        u_x_range = np.linspace(self.u_x_bot, self.u_x_top, self.U_K)
        u_y_range = np.linspace(self.u_y_bot, self.u_y_top, self.U_K)
        return np.array([u_x_range,u_y_range], dtype=np.float_).T

    T1_Kab = tr.Property(depends_on='state_changed')
    '''Smoothed crack profile
    '''
    @tr.cached_property
    def _get_T1_Kab(self):
        line_vec_La = self.X_Ka[1:,:] - self.X_Ka[:-1,:]
        line_vec_La = np.vstack([line_vec_La, line_vec_La[-1:]])
        return get_T_Lab(line_vec_La)

    U1_Kb = tr.Property(depends_on='state_changed')
    '''Local relative displacement of points along the crack'''
    @tr.cached_property
    def _get_U1_Kb(self):
        U1_Kb = np.einsum('...ab,...b->...a', self.T1_Kab, self.U1_Ka)
        return U1_Kb

    X_neutral_a = tr.Property(depends_on='state_changed')
    '''Vertical position of the neutral axis
    '''
    @tr.cached_property
    def _get_X_neutral_a(self):
        idx = np.argmax(self.U1_Ka[:,0] < 0) - 1
        x_1, x_2 = self.X_Ka[(idx, idx + 1), 1]
        u_1, u_2 = self.U1_Ka[(idx, idx + 1), 0]
        d_x = -(x_2 - x_1) / (u_2 - u_1) * u_1
        y_neutral = x_1 + d_x
        x_neutral = self.X_Ka[idx + 1, 0]
        return np.array([x_neutral, y_neutral])

    def _plot_u(self, ax, U1_Ka, idx=0, color='black', label=r'$w$ [mm]'):
        X_Ka = self.X_Ka
        ax.plot(U1_Ka[:, idx], X_Ka[:, 1], 'o-', color=color, label=label)
        ax.fill_betweenx(X_Ka[:, 1], U1_Ka[:, idx], 0, color=color, alpha=0.1)
        ax.set_xlabel(label)
        ax.legend(loc='lower left')

    def plot_U1_Ka(self, ax_w):
        '''Plot the displacement along the crack (w and s) in global coordinates
        '''
        self._plot_u(ax_w, self.U1_Ka, 0, label=r'$u_x$ [mm]', color='blue')
        ax_w.set_xlabel(r'$u_x, u_y$ [mm]', fontsize=10)
        ax_w.plot([0], [self.X_neutral_a[1]], 'o', color='black')
        ax_w.plot([0], [self.X_tip_a[1]], 'o', color='red')
        self._plot_u(ax_w, self.U1_Ka, 1, label=r'$u_y$ [mm]', color='green')
        ax_w.set_title(r'displacement jump')
        ax_w.legend()

    def plot_U1_Kb(self, ax_w):
        '''Plot the displacement (u_x, u_y) in local crack coordinates
        '''
        self._plot_u(ax_w, self.U1_Kb, 0, label=r'$w$ [mm]', color='blue')
        ax_w.plot([0], [self.X_neutral_a[1]], 'o', color='black')
        ax_w.plot([0], [self.X_tip_a[1]], 'o', color='red')
        ax_w.set_xlabel(r'$w, s$ [mm]', fontsize=10)
        self._plot_u(ax_w, self.U1_Kb, 1, label=r'$s$ [mm]', color='green')
        ax_w.set_title(r'opening and sliding')
        ax_w.legend()

    def subplots(self, fig):
        gs = gridspec.GridSpec(1, 2)
        ax_u_0 = fig.add_subplot(gs[0, 0])
        ax_w_0 = fig.add_subplot(gs[0, 1])
        return ax_u_0, ax_w_0

    def update_plot(self, axes):
        ax_u_0, ax_w_0 = axes
        self.plot_U1_Ka(ax_u_0)
        self.plot_U1_Kb(ax_w_0)
