import traits.api as tr
import numpy as np

import bmcs_utils.api as bu
from bmcs_shear.shear_crack.crack_extension import CrackExtension


class CrackPropagation(CrackExtension):
    """Control a loop simulating the crack propagation
    """
    name = "Crack Propagation"

    t_n = bu.Float(0.0, auto_set=False, enter_set=True)
    '''Fundamental state time.
    '''
    t_n1 = bu.Float(0.0, auto_set=False, enter_set=True)
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

    xtol = bu.Float(1e-3, auto_set=False, enter_set=True)
    maxfev = bu.Int(1000, auto_set=False, enter_set=True)

    R_n = tr.List
    M_n = tr.List([0])
    F_beam = tr.List([0])
    v_n = tr.List([0])
    CMOD_n = tr.List([0])

    def record_timestep(self):
        R_k = self.get_R()
        v_k = self.sz_stress_profile.u_Ca[1, 1]
        self.R_n.append(R_k)
        self.F_beam.append(self.crack_tip_shear_stress.F_beam)
        self.v_n.append(v_k)

    def make_incr(self):
        '''Update the control, primary and state variabrles..
        '''
        self.X
        R_k = self.get_R()
        self.t_n = self.t_n1
        self.record_timestep()

    n_seg = bu.Int(5, TIME=True)

    # simulated_crack = tr.Property(depends_on='+TIME, _GEO,_MAT')
    #
    # @tr.cached_property
    # def _get_simulated_crack(self):
    #     self.run()

    seg = bu.Int(0)

    interrupt = tr.Bool(False)

    def run(self, update_progress=lambda t: t):
        crack_seg = np.arange(1, self.n_seg + 1)
        self.sz_cp.reset_crack()
        self.R_n = [0]
        self.F_beam = [0]
        self.v_n = [0]
        while self.seg <= self.n_seg:
            if self.interrupt:
                break
            self.make_incr()
            if self.seg < self.n_seg:
                self.sz_cp.add_x_tip_an(self.sz_cp.sz_ctr.x_tip_ak[:, 0])
            self.seg += 1

    def reset(self):
        self.seg=0
        self.sz_cp.reset_crack()
        self.init()
        self.R_n = [0]
        self.F_beam = [0]
        self.v_n = [0]

    def subplots(self, fig):
        return fig.subplots(1, 2)

    def update_plot(self, ax):
        ax1, ax2 = ax
        self.plot_geo(ax1)
        F_beam_kN = np.array(self.F_beam).flatten() / 1000
        v_n = np.array(self.v_n)
        ax2.plot(v_n, F_beam_kN)
        ax2.set_xlabel(r'Deflection $v$ [mm]')
        ax2.set_ylabel(r'Load $F$ [kN]')

    ipw_view = bu.View(
        bu.Item('c', editor=bu.ProgressEditor(
            run_method='run',
            reset_method='reset',
            interrupt_var='interrupt',
            time_var='seg',
            time_max='n_seg'
        )),
        bu.Item('n_seg', latex=r'n_\mathrm{seg}', minmax=(1, 100)),
    )
