

import traits.api as tr
import numpy as np
import sympy as sp
from bmcs_utils.api import InteractiveModel, View, Item
from bmcs_shear_zone.shear_crack.crack_tip_rotation import \
    SZCrackTipRotation
from bmcs_shear_zone.shear_crack.beam_design import \
    RCBeamDesign
from bmcs_shear_zone.shear_crack.crack_path import \
    SZCrackPath
from bmcs_shear_zone.shear_crack.stress_profile import \
    SZStressProfile
import ipywidgets as ipw
from bmcs_shear_zone.api import SteelMaterialModel, ConcreteMaterialModel
from bmcs_utils.api import SymbExpr, InjectSymbExpr

# # Shear stress profile

# Since
# \begin{align}
#  Q(x) = \frac{\mathrm{d}M(x)}{\mathrm{d}{x}}
# \end{align}

# we require
# \begin{align}
# \left. Q \right|_{x^{\mathrm{rot}}_0} =  \left. \frac{\mathrm{d}M}{\mathrm{d}{x}} \right|_{x^\mathrm{rot}_0}
# \end{align}

# Thus, for a point load in the middle section we obtain
# \begin{align}
# Q_M = \frac{M}{x_\mathrm{right} - x^{\mathrm{rot}}_0}
# \end{align}

# Global equations are defined as
# \begin{align}
#  M(\theta, x_{ak}^\mathrm{rot}) + Q(\theta, x_{ak}^\mathrm{rot}) L &= 0 \\
#  N(\theta, x_{ak}^\mathrm{rot}) &= 0
# \end{align}

# ## Shear stress in FPS

# In[75]:


Q_, H_, B_ = sp.symbols('Q, H, B', nonnegative=True)
z_ = sp.Symbol('z', positive=True)
a_, b_, c_ = sp.symbols('a, b, c')
z_fps_ = sp.Symbol(r'z^{\mathrm{fps}}', nonnegative=True)

# In[76]:


tau_z = a_ * z_ ** 2 + b_ * z_ + c_

# In[77]:


eqs = {
    tau_z.subs(z_, 0),
    tau_z.subs(z_, H_)
}

# In[78]:


ac_subs = sp.solve(eqs, {a_, c_})
ac_subs

# In[79]:


Tau_z_fn = tau_z.subs(ac_subs)
Tau_z_fn

# In[80]:


Q_int = sp.integrate(Tau_z_fn, (z_, z_fps_, H_))
Q_int

# In[81]:


b_subs = sp.solve({sp.Eq(Q_, B_ * Q_int)}, {b_})
b_subs

# In[82]:


Tau_z_ff = Tau_z_fn.subs(b_subs)
Tau_z_ff

# In[83]:


Tau_z_fps = Tau_z_ff.subs(z_, z_fps_)
sp.simplify(Tau_z_fps)

# In[84]:


tau_z_ = sp.Piecewise(
    (0, z_ < z_fps_),
    (Tau_z_ff, z_ >= z_fps_),
    (0, z_ > H_),
)
get_tau_z = sp.lambdify(
    (z_, z_fps_, Q_, H_, B_), tau_z_, 'numpy'
)

# In[85]:


get_tau_z_fps = sp.lambdify(
    (z_fps_, Q_, H_, B_), Tau_z_fps, 'numpy'
)


# In[86]:


class SZCrackTipShearStress(InteractiveModel):
    name = 'Shear profile'

    Q = tr.Property

    def _get_Q(self):
        M = self.sz_stress_profile.M
        L = self.beam_design.L
        x_tip_0k = self.sz_cp.sz_ctr.x_tip_ak[0]
        Q = M / (L - x_tip_0k)[0]
        return Q

    sz_cp = tr.Instance(SZCrackPath, ())

    sz_stress_profile = tr.Property(depends_on='sz_cp')

    @tr.cached_property
    def _get_sz_stress_profile(self):
        return SZStressProfile(sz_cp=self.sz_cp)

    beam_design = tr.DelegatesTo('sz_cp', 'sz_geo')

    x_tip_1k = tr.Property

    def _get_x_tip_1k(self):
        return self.sz_cp.sz_ctr.x_tip_ak[1]

    tau_x_tip_1k = tr.Property

    def _get_tau_x_tip_1k(self):
        H = self.beam_design.H
        B = self.beam_design.B
        return get_tau_z_fps(self.x_tip_1k, self.Q, H, B)[0]

    ipw_view = View(
        #        Item('Q', latex='Q', minmax=(0,100000)),
    )

    def subplots(self, fig):
        return fig.subplots(1, 2)

    def update_plot(self, axes):
        ax1, ax2 = axes
        H = self.beam_design.H
        B = self.beam_design.B

        Q_val = np.linspace(0, self.Q, 100)
        z_fps_arr = np.linspace(0, H * 0.98, 100)
        tau_z_fps_arr = get_tau_z_fps(z_fps_arr, Q_val, H, B)

        z_arr = np.linspace(0, H, 100)
        tau_z_arr = get_tau_z(z_arr, self.x_tip_1k, self.Q, H, B)

        ax1.plot(tau_z_fps_arr, z_fps_arr, color='green');
        ax1.fill_betweenx(z_fps_arr, tau_z_fps_arr, 0, color='green', alpha=0.1)

        ax2.plot(tau_z_arr, z_arr, color='blue');
        ax2.fill_betweenx(z_arr, tau_z_arr, 0, color='blue', alpha=0.1)

