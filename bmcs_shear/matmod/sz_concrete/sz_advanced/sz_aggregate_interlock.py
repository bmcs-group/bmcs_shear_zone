import numpy as np
import sympy as sp
import traits.api as tr
from bmcs_utils.api import InteractiveModel, View, Item, Float, SymbExpr, InjectSymbExpr
from bmcs_shear.matmod.i_matmod import IMaterialModel

class AggregateInterlockSymb(SymbExpr):
    d_g, f_c = sp.symbols(r'd_g, f_c', nonnegative=True)
    w = sp.symbols(r'w', nonnegative=True)
    s = sp.symbols(r's', nonnegative=True)

    r = s / w

    tau_0 = 0.25 * f_c

    a_3 = 2.45 / tau_0

    a_4 = 2.44 * (1 - (4 / tau_0))

    tau_ag = tau_0 * (1 - sp.sqrt((2 * w)/d_g)) * r * (a_3 + (a_4 * sp.Abs(r)**3)) / (1 + (a_4 *r**4))

    sigma_ag = -0.62 * sp.sqrt(w) * (r) / ((1 + r ** 2) ** 0.25) * tau_ag

    tau_ag_diff = tau_ag.diff(s)

    symb_model_params = ['d_g', 'f_c']

    symb_expressions = [('tau_ag', ('w','s',)),
                        ('sigma_ag', ('w','s',))]

@tr.provides(IMaterialModel)
class AggregateInterlock(InteractiveModel, InjectSymbExpr):
    name = 'Aggregate Interlock'

    symb_class = AggregateInterlockSymb


    #delta = Float(0.1) ##mm (vertical displacement)
    d_g = Float(22) ##mm (size of aggregate)
    f_c = Float(37.9) ## (compressive strength of Concrete in MPa)

    ipw_view = View(
        Item('d_g', latex=r'd_g'),
        Item('f_c', latex=r'f_c')
    )

    def get_tau_ag(self, w, s):
        # calculating the shear stress due to aggregate interlocking
        return self.symb.get_tau_ag(w, s)

    # def get_tau_ag_diff(self,w , s):
    #     return self.symb.tau_ag_diff(w, s)

    def get_sigma_ag(self, w, s):
        # calculating the normal stress due to aggregate interlocking
        return self.symb.get_sigma_ag(w, s)


    def subplots(self,fig):
        return fig.subplots(1,2)

    def update_plot(self,axes):
        ax_w, ax_s = axes
        w_range = np.linspace(0.1, 1, 3)
        # s_range = np.linspace(0.001, 1, 100)
        # tau_ag = np.zeros((100, 100))
        # sigma_ag = np.zeros((100, 100))
        tau_ag = np.zeros((100, 3))
        sigma_ag = np.zeros((100, 3))

        # for i, w in enumerate(w_range):
        #     tau_ag[i] = self.get_tau_ag(w, s_range)
        #     sigma_ag[i] = self.get_sigma_ag(w, s_range)

        for i, w in enumerate(w_range):
            s_range = np.linspace(0, 1, 100)
            for j, s in enumerate(s_range):
                tau_ag[j, i] = self.get_tau_ag(w, s)
                sigma_ag[j, i] = self.get_sigma_ag(w, s)
        #V = self.get_sig_s_f(delta_range)
        ax_w.plot(s_range, tau_ag[:,:], label = '$w$')
        ax_s.plot(s_range, sigma_ag[:,:], label = '$w$')
        #ax_w.plot(s_range, tau_ag[:])
        #ax_s.plot(s_range, sigma_ag[:])
        ax_w.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax_w.set_ylabel(r'$\tau_{\mathrm{ag}}\;\;\mathrm{[MPa]}$')
        ax_s.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax_s.set_ylabel(r'$\sigma_{\mathrm{ag}}\;\;\mathrm{[MPa]}$')

