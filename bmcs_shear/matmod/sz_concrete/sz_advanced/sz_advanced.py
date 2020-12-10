
from .sz_softening_tensile_behavior import TensileSofteningBehavior
from .sz_compressive_hardening import CompressiveHardeningBehavior
from .sz_aggregate_interlock import AggregateInterlock
from bmcs_shear.matmod.i_matmod import IMaterialModel
from bmcs_utils.api import InteractiveModel, View, Item, Float, SymbExpr, InjectSymbExpr
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np

class ConcreteMaterialModelAdvExpr(bu.SymbExpr):
    # continue here
    pass

@tr.provides(IMaterialModel)
class ConcreteMaterialModelAdv(bu.InteractiveModel):

    name = 'Concrete behavior'
    node_name = 'material model'

    tension = tr.Instance(TensileSofteningBehavior, ())
    compression = tr.Instance(CompressiveHardeningBehavior, ())
    interlock = tr.Instance(AggregateInterlock, ())

    f_t = Float(3.0,
                   MAT=True,
                   unit=r'$\mathrm{MPa}$',
                   symbol=r'f_\mathrm{t}',
                   auto_set=False, enter_set=True,
                   desc='concrete tensile strength'
                   )

    d_ag = Float(16.0,
                   MAT=True,
                   unit=r'$\mathrm{mm}$',
                   symbol=r'd_\mathrm{ag}',
                   auto_set=False, enter_set=True,
                   desc='maximum size of aggregate'
                   )

    f_c = Float(30,
                   MAT=True,
                   unit=r'$\mathrm{MPa}$',
                   symbol=r'f_\mathrm{c}',
                   auto_set=False, enter_set=True,
                   desc='concrete compressive strength'
                   )

    E_c = Float(28000,
                MAT=True,
                unit=r'$\mathrm{MPa}$',
                symbol=r'E_\mathrm{c}',
                auto_set=False, enter_set=True,
                desc='concrete material stiffness')

    L = Float(2000,
                MAT=True,
                unit=r'$\mathrm{mm}$',
                symbol=r'L',
                auto_set=False, enter_set=True,
                desc='Length of section')
    # s_1 = Float(1)
    # s_2 = Float(2)
    # s_3 = Float(4)
    # alpha = Float(0.4)

    G_f = tr.Property

    # G_f = bu.Float(0.5, MAT=True)
    def _get_G_f(self):
        return 0.028 * (self.f_c) ** 0.18 * (self.d_ag) ** 0.32

    L_c = tr.Property

    def _get_L_c(self):
        return self.E_c * self._get_G_f() / self.f_t ** 2


    ipw_view = View(
        Item('f_t'),
        Item('d_ag'),
        Item('f_c'),
        Item('E_c'),
        Item('L'),
        # Item('s_1'),
        # Item('s_2'),
        # Item('s_3'),
        # Item('alpha')
    )

    w_cr = tr.Property
    def _get_w_cr(self):
        return self.f_t / self.E_c * self.L_c

    w_c = tr.Property
    def _get_w_c(self):
        return self.f_c / self.E_c * self.L_c

    eps_cp = tr.Property
    def _get_eps_cp(self):
        return self._get_w_cr() / self.L_c

    eps = tr.Property
    def _get_eps(self, w):
        return w / self.L_c

    co_law_data = tr.Property(depends_on='+MAT')

    @tr.cached_property
    def _get_co_law_data(self):
        return dict(f_t=float(self.f_t),
                    G_f=float(self.G_f),
                    f_c=self.f_c,
                    E_c=self.E_c,
                    L_c=self.L_c,
                    d_a=self.d_ag,
                    L=self.L
                    )

    def _get_f_ce(self):
        return self.f_c * (1 / (0.8))

    def get_sig_a(self, u_a):
        w, s = u_a[...,0], u_a[...,1]


    sigma_w = tr.Property(depends_on='+MAT')

    def get_sigma_w(self,w):
        sigma_w = np.zeros_like(w)
        for i,w_ in enumerate(w):
            if (self.E_c * w_)/self.L_c < - self.f_c:
                sigma_w[i] = -self.f_c
            elif w_<= self._get_w_cr():
                sigma_w[i] =  - self._get_f_ce() * (2*(self._get_eps(w_) / -0.002) - (self._get_eps(w_) / -0.002)**2 )
                # -(self.compression.get_sigma_c(self._get_eps(w_),self._get_eps_cp()))
            # elif  w_<= self._get_w_cr():
            #     sigma_w[i]  = self.E_c * w_ / self.L_c
            elif w_ > self._get_w_cr():
                # if w_ < self._get_w_cr() / 5:
                sigma_w[i] = self.tension.get_sigma_w_t(w_)
            #lif w_ > self._get_w_cr() / 5:
             #        sigma_w[i] = self.tension.get_sigma_w_t(w_) # + self.interlock.get_sigma_ag(w_, s)
                #     print('result 2 ', sigma_w[i])
        return sigma_w

    d_sigma_w = tr.Property(depends_on='+MAT')

    # def get_d_sigma_w(self, w):
    #     d_sigma_w = np.zeros_like(w)
    #     for i, w_ in enumerate(w):
    #         if (self.E_c * w_)/self.L_c < - self.f_c:
    #             d_sigma_w[i] = 0
    #         elif w_<= self._get_w_cr():
    #             d_sigma_w[i] =  self._get_f_ce() *( - 1000 / self.L_c - (500000 * w_) / self.L_c**2)
    #         elif w_ > self._get_w_cr():
    #             d_sigma_w[i] = self.tension.get_sigma_w_t_diff(w_)
    #     return d_sigma_w

    tau_s = tr.Property(depends_on='+MAT')

    def get_tau_s(self, s):
        tau_s = self.pullout.get_tau_b(s)
        return tau_s

    d_tau_s = tr.Property(depends_on = '+MAT')

    def d_tau_s(self, s):
        d_tau_s = self.pullout.get_d_tau_b(s)
        return d_tau_s

    # def get_tau_s(self, s):
    #     w = self._get_w_cr()
    #     tau_s = self.interlock.get_tau_ag(w,s)
    #     return tau_s

    def subplots(self,fig):
        return fig.subplots(1,2)

    def sigma_w_plot(self,ax_w):
        w_ = np.linspace(-1, self.tension.get_w_tc(), 100)
        sigma_w_ = self.get_sigma_w(w_)
        ax_w.plot(w_, sigma_w_, color='red')
        ax_w.fill_between(
            w_, self.get_sigma_w(w_), color='red', alpha=0.2
        )
        ax_w.set_xlabel(r'$w\;\;\mathrm{[mm]}$')
        ax_w.set_ylabel(r'$\sigma_w\;\;\mathrm{[MPa]}$')
        ax_w.set_title('crack opening law')

    def tau_s_plot(self, ax_s):
        s_ = np.linspace(0, 2, 100)
        tau_s_ = self.get_tau_s(s_)
        ax_s.plot(s_, tau_s_)
        ax_s.fill_between(
            s_, self.get_tau_s(s_), color='blue', alpha=0.2
        )
        ax_s.set_xlabel(r'$s\;\;\mathrm{[mm]}$')
        ax_s.set_ylabel(r'$\tau_s\;\;\mathrm{[MPa]}$')
        ax_s.set_title('crack interface law')

    def d_tau_s_plot(self, ax_s):
        s_ = np.linspace(0, 2, 100)
        d_tau_s_ = self.d_tau_s(s_)
        ax_s.plot(s_, d_tau_s_, color='orange')

    # def d_sigma_w_plot(self, ax_w):
    #     w_ = np.linspace(-1, self.tension.get_w_tc(), 100)
    #     d_sigma_w = self.get_d_sigma_w(w_)
    #     ax_w.plot(w_, d_sigma_w, color='orange')

    def update_plot(self, axes):
        ax_w, ax_s = axes
        self.sigma_w_plot(ax_w)
        # self.d_sigma_w_plot(ax_w)
        self.tau_s_plot(ax_s)
        self.d_tau_s_plot(ax_s)


