import numpy as np
import sympy as sp
import traits.api as tr
from bmcs_utils.api import InteractiveModel, View, Item, Float, SymbExpr, InjectSymbExpr
from .i_matmod import IMaterialModel

class AggregateInterlockSymb(SymbExpr):
    d_g, f_c = sp.symbols(r'd_g, f_c', nonnegative=True)
    w = sp.symbols(r'w', nonnegative=True)
    delta = sp.symbols(r'\delta', nonnegative=True)

    r = delta / w

    tau_0 = 0.25 * f_c

    a_3 = 2.45 / tau_0

    a_4 = 2.44 * (1 - (4 / tau_0))

    tau_ag = tau_0 * (1 - sp.sqrt((2 * w)/d_g)) * r * (a_3 + (a_4 * sp.Abs(r)**3)) / (1 + (a_4 *r**4))

    sigma_ag = -0.62 * sp.sqrt(w) * (r) / ((1 + r ** 2) ** 0.25) * tau_ag


    symb_model_params = ['d_g', 'f_c']

    symb_expressions = [('tau_ag', ('w','delta',)),
                        ('sigma_ag', ('w','delta',))]

@tr.provides(IMaterialModel)
class AggregateInterlock(InteractiveModel, InjectSymbExpr):
    name = 'Aggregate Interlock'

    symb_class = AggregateInterlockSymb


    #delta = Float(0.1) ##mm (vertical displacement)
    d_g = Float(22) ##mm (size of aggregate)
    f_c = Float(37.9) ## (compressive strength of Concrete in MPa)

    ipw_view = View(
        Item('d_g'),
        Item('f_c')
    )

    def get_tau_ag(self, w, delta):
        # calculating the shear stress due to aggregate interlocking
        return self.symb.get_tau_ag(w, delta)

    def get_sigma_ag(self, w, delta):
        # calculating the normal stress due to aggregate interlocking
        return self.symb.get_sigma_ag(w, delta)


    def subplots(self,fig):
        return fig.subplots(1,2)

    def update_plot(self,axes):
        ax_w, ax_s = axes
        w_range = np.linspace(0.1, 1, 3)
        tau_ag = np.zeros((100, 3))
        sigma_ag = np.zeros((100, 3))
        for i, w in enumerate(w_range):
            delta_range = np.linspace(0.001, 1, 100)
            for j, delta in enumerate(delta_range):
                tau_ag[j, i] = self.get_tau_ag(w, delta)
                sigma_ag[j, i] = self.get_sigma_ag(w, delta)
        #V = self.get_sig_s_f(delta_range)
        ax_w.plot(delta_range, tau_ag[:,:])
        ax_s.plot(delta_range, sigma_ag[:,:])
        ax_w.set_xlabel(r'$\delta\;\;\mathrm{[mm]}$')
        ax_w.set_ylabel(r'$\tau_{\mathrm{ag}}\;\;\mathrm{[MPa]}$')
        ax_s.set_xlabel(r'$\delta\;\;\mathrm{[mm]}$')
        ax_s.set_ylabel(r'$\sigma_{\mathrm{ag}}\;\;\mathrm{[MPa]}$')

