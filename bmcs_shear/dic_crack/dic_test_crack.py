
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
from bmcs_shear.beam_design import RCBeamDesign
from .dic_crack import IDICCrack, get_T_Lab
import matplotlib.gridspec as gridspec

@tr.provides(IDICCrack)
class DICTestCrack(bu.Model):
    '''
    Test crack model to be used for verification
    of stress profile evaluation for elementary crack geometries
    '''
    name = 'test crack'

    bd = bu.Instance(RCBeamDesign, ())

    ipw_view = bu.View(
    )

    u_x_bot = bu.Float(0.03, ALG=True)
    u_x_top = bu.Float(-0.03, ALG=True)
    u_y_bot = bu.Float(0.0, ALG=True)
    u_y_top = bu.Float(0.0, ALG=True)

    ipw_view = bu.View(
        bu.Item('u_x_bot'),
        bu.Item('u_x_top'),
        bu.Item('u_y_bot'),
        bu.Item('u_y_top')
    )

    X_tip_a = tr.Array(value=[0, 50])

    x_Na = tr.Property(tr.Array, depends_on='state_changed')
    '''All ligament points.
    '''
    @tr.cached_property
    def _get_x_Na(self):
        return np.array([[0, 0],[0, self.bd.H]], dtype=np.float_)

    u_Na = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_u_Na(self):
        return np.array([[self.u_x_bot, self.u_y_bot],[self.u_x_top, self.u_y_top]], dtype=np.float_)

    T_Nab = tr.Property(depends_on='state_changed')
    '''Smoothed crack profile
    '''
    @tr.cached_property
    def _get_T_Nab(self):
        line_vec_La = self.x_Na[1:,:] - self.x_Na[:-1,:]
        return get_T_Lab(line_vec_La)

    u_Nb = tr.Property(depends_on='state_changed')
    '''Local relative displacement of points along the crack'''
    @tr.cached_property
    def _get_u_Nb(self):
        u_Nb = np.einsum('...ab,...b->...a', self.T_Nab, self.u_Na)
        return u_Nb

    X_neutral_a = tr.Property(depends_on='state_changed')
    '''Vertical position of the neutral axis
    '''
    @tr.cached_property
    def _get_X_neutral_a(self):
        idx = np.argmax(self.u_Na[:,0] < 0) - 1
        x_1, x_2 = self.x_Na[(idx, idx + 1), 1]
        u_1, u_2 = self.u_Na[(idx, idx + 1), 0]
        d_x = -(x_2 - x_1) / (u_2 - u_1) * u_1
        y_neutral = x_1 + d_x
        x_neutral = self.x_Na[idx + 1, 0]
        return np.array([x_neutral, y_neutral])

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
        ax_w.plot([0], [self.X_neutral_a[1]], 'o', color='black')
        ax_w.plot([0], [self.X_tip_a[1]], 'o', color='red')
        self._plot_u(ax_w, self.u_Na, 1, label=r'$u_y$ [mm]', color='green')
        ax_w.set_title(r'displacement jump')
        ax_w.legend()

    def plot_u_Nb(self, ax_w):
        '''Plot the displacement (u_x, u_y) in local crack coordinates
        '''
        self._plot_u(ax_w, self.u_Nb, 0, label=r'$w$ [mm]', color='blue')
        ax_w.plot([0], [self.X_neutral_a[1]], 'o', color='black')
        ax_w.plot([0], [self.X_tip_a[1]], 'o', color='red')
        ax_w.set_xlabel(r'$w, s$ [mm]', fontsize=10)
        self._plot_u(ax_w, self.u_Nb, 1, label=r'$s$ [mm]', color='green')
        ax_w.set_title(r'opening and sliding')
        ax_w.legend()

    def subplots(self, fig):
        gs = gridspec.GridSpec(1, 2)
        ax_u_0 = fig.add_subplot(gs[0, 0])
        ax_w_0 = fig.add_subplot(gs[0, 1])
        return ax_u_0, ax_w_0

    def update_plot(self, axes):
        ax_u_0, ax_w_0 = axes
        self.plot_u_Na(ax_u_0)
        self.plot_u_Nb(ax_w_0)
