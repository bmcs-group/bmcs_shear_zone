from bmcs_shear.matmod.i_matmod import IMaterialModel
from bmcs_utils.api import View, Item, Float
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
from bmcs_shear.matmod.sz_crack_bridge.cb_advanced.sz_pull_out_fib import PullOut
from bmcs_shear.matmod.sz_crack_bridge.cb_advanced.sz_dowel_action import DowelAction

@tr.provides(IMaterialModel)
class CrackBridgeAdv(bu.InteractiveModel):

    name = 'Crack Bridge Adv'
    node_name = 'crack bridge model'

    pullout = tr.Instance(PullOut, ())
    dowelaction = tr.Instance(DowelAction, ())

    B = Float(75)  ##mm (width of the beam)
    n = Float(4)  ##number of bars
    d_s = Float(16)  ##dia of steel mm
    f_c = Float(37.9)  ## compressive strength of Concrete in MPa
    s_1 = Float(1)
    s_2 = Float(2)
    s_3 = Float(4)
    alpha = Float(0.4)

    ipw_view = View(
        Item('B'),
        Item('n'),
        Item('d_s'),
        Item('f_c'),
        Item('s_1'),
        Item('s_2'),
        Item('s_3'),
        Item('alpha')
    )

    def get_tau_b(self, s):
        '''Calculating bond stresses '''
        tau_b = self.pullout.get_tau_b(s)
        return tau_b

    def get_df(self, s):
        '''Calculating dowel action force '''
        V_df = self.dowelaction.get_sig_s_f(s)
        return V_df

    def subplots(self,fig):
        return fig.subplots(1,2)

    def update_plot(self,axes):
        '''Plotting function '''
        ax_s, ax_f = axes
        s_ = np.linspace(0, 1, 100)
        tau_b_ = self.get_tau_b(s_)
        V_df_ = self.get_df(s_)
        ax_s.plot(s_, tau_b_)
        ax_f.plot(s_, V_df_)
        ax_s.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax_s.set_ylabel(r'$\tau_b\;\;\mathrm{[MPa]}$')
        ax_f.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax_f.set_ylabel(r'$V_{df}\;\;\mathrm{[N]}$')