import traits.api as tr
import numpy as np

import bmcs_utils.api as bu
#from bmcs_shear.shear_crack.crack_extension import CrackExtension
from bmcs_shear.shear_crack.crack_extension import CrackExtension
from bmcs_shear.shear_crack.crack_propagation_hist import CrackPropagationHist


from matplotlib import animation, rc

class CrackPropagation(CrackExtension):
    """Control a loop simulating the crack propagation
    """
    name = "Crack Propagation"

    hist = tr.Instance(CrackPropagationHist)
    '''Viewer to the inelastic state evolution'''
    def _hist_default(self):
        return CrackPropagationHist(crack_prop_model=self)

    tree = [
        # 'sz_bd',
        'sz_cto',
        'hist'
    ]

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
    #Q = tr.List([0])
    #Q_red = tr.List([0])
    F_s = tr.List([0])
    x_tip_1n = tr.List([0])
    x_tip_0n = tr.List([0])
    x_tip_1k = tr.List([0])
    F_a = tr.List([0])
    F_Na = tr.List([0])
    s_steel = tr.List([0])
    M = tr.List([0])
    w = tr.List([0])
    w_steel = tr.List([0])
    slip = tr.List([0])
    shear_agg = tr.List([0])
    v_n = tr.List([0])
    CMOD_n = tr.List([0])
    sig_x_tip_0 = tr.List([0])
    sig_z1 = tr.List([0])
    tau_x_tip_1k = tr.List([0])
    #F_N_delta = tr.List([0])

    def record_timestep(self):
        R_k = self.get_R()
        v_k = self.sz_sp.u_Ca[1, 1]
        x_tip_1k = self.sz_cp.sz_ctr.x_tip_ak[:, 0]
        self.R_n.append(R_k)
        self.F_beam.append(self.sz_ctss.F_beam)
        #self.Q.append(self.sz_ctss.Q)
        #self.Q_red.append(self.sz_ctss.Q_reduced)
        self.F_a.append(self.sz_sp.F_a[:])
        self.x_tip_1n.append(self.sz_ctr.x_tip_an[1])
        self.x_tip_0n.append(self.sz_ctr.x_tip_an[0])
        self.x_tip_1k.append(self.sz_ctr.x_tip_ak[1])
        self.M.append(self.sz_sp.M)
        self.s_steel.append(self.sz_sp.u_Na[:, 1])
        self.w_steel.append(self.sz_sp.u_Na[:, 0])
        self.shear_agg.append(self.sz_sp.S_Lb[:, 1])
        self.slip.append(self.sz_sp.u_Lb[:, 1])
        self.w.append(self.sz_sp.u_Lb[:, 0])
        self.F_s.append(self.sz_sp.F_Na[:, 0])
        self.F_Na.append(self.sz_sp.F_Na[:, 1])
        self.v_n.append(v_k)
        self.sig_x_tip_0.append(self.sz_ctss.sig_x_tip_0)
        self.sig_z1.append(self.sz_ctss.sig_z_tip_1)
        self.tau_x_tip_1k.append(self.sz_ctss.tau_x_tip_1k)
        #self.F_N_delta.append(self.sz_ctss.F_N_delta)

    def make_incr(self):
        '''Update the control, primary and state variabrles..
        '''
        self.X
        R_k = self.get_R()
        self.t_n = self.t_n1
        self.record_timestep()

    n_seg = bu.Int(5, TIME=True)

    seg = bu.Int(0)

    interrupt = tr.Bool(False)

    def run(self, update_progress=lambda t: t):
        crack_seg = np.arange(1, self.n_seg + 1)
        self.sz_cp.reset_crack()
        self.R_n = [0]
        self.F_beam = [0]
        #self.Q = [0]
        #self.Q_red = [0]
        self.F_a = [0]
        self.F_Na = [0]
        self.F_s = [0]
        self.M = [0]
        self.s_steel = [0]
        self.w_steel = [0]
        self.slip = [0]
        self.w = [0]
        self.shear_agg = [0]
        self.x_tip_1n = [0]
        self.x_tip_0n = [0]
        self.x_tip_1k = [0]
        self.v_n = [0]
        self.sig_x_tip_0 = [0]
        self.sig_z1 = [0]
        self.tau_x_tip_1k = [0]
        #self.F_N_delta = [0]


        while self.seg <= self.n_seg:
            print('seg', self.seg)
            if self.interrupt:
                break
            try:
                self.make_incr()
            except StopIteration:
                print('stopped after', self.seg)
                break
            if self.seg < self.n_seg:
                self.sz_cp.add_x_tip_an(self.sz_cp.sz_ctr.x_tip_ak[:, 0])
            self.seg += 1

    def reset(self):
        self.seg = 0
        self.sz_cp.reset_crack()
        self.init()
        self.R_n = [0]
        self.F_beam = [0]
        #self.Q = [0]
        #self.Q_red = [0]
        self.x_tip_1n = [0]
        self.x_tip_0n = [0]
        self.x_tip_1k = [0]
        self.F_a = [0]
        self.F_s = [0]
        self.M = [0]
        self.s_steel = [0]
        self.shear_agg = [0]
        self.slip = [0]
        self.w = [0]
        self.w_steel = [0]
        self.F_Na = [0]
        self.v_n = [0]
        self.sig_x_tip_0 = [0]
        self.sig_z1 = [0]
        self.tau_x_tip_1k = [0]
        #self.F_N_delta = [0]

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
        bu.Item('n_seg', latex=r'n_\mathrm{seg}'),
        bu.Item('xtol', latex=r'x_\mathrm{tol}^\mathrm{err}'),
        bu.Item('psi_tol', latex=r'\Delta \psi_\mathrm{tol}^\mathrm{err}'),
        time_editor = bu.ProgressEditor(
            run_method='run',
            reset_method='reset',
            interrupt_var='interrupt',
            time_var='seg',
            time_max='n_seg'
        )
    )


