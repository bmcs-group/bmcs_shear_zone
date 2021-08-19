

import traits.api as tr
import numpy as np
import sympy as sp
from bmcs_shear.shear_crack.crack_tip_shear_stress import SZCrackTipShearStress

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

class SZCrackTipShearStressGlobal(SZCrackTipShearStress):
    name = 'crack tip stress state'

    # sig_x_tip_0 = tr.Property
    #
    # def _get_sig_x_tip_0(self):
    #     x_tip_1 = self.sz_cp.sz_ctr.x_tip_ak[1,0]
    #     idx_tip0 = np.argmax(self.sz_cp.x_Ka[:, 1] >= x_tip_1)
    #     H_fpz = 100
    #     idx_tip1 = np.argmax(self.sz_cp.x_Ka[:, 1] >= x_tip_1 + H_fpz)
    #     N_c = np.sum(self.sz_stress_profile.F_La[idx_tip0:idx_tip1, 0])
    #     B = self.sz_bd.B
    #     H = self.sz_bd.H
    #     sigma_c = N_c / B / H_fpz
    #     return sigma_c

    # Q = tr.Property
    # """Assuming the parabolic profile of the shear stress within the uncracked
    # zone, calculate the value of shear stress corresponding to the height of the
    # crack tip
    # """
    # def _get_Q(self):
    #     M = self.sz_stress_profile.M
    #     L = self.sz_bd.L
    #     x_tip_0k = self.sz_cp.sz_ctr.x_tip_ak[0]
    #     Q = M / (L - x_tip_0k)[0]
    #     return Q

    Q_reduced = tr.Property
    """Amount of shear force calculated from the global moment equilibrium
    reduced by the shear stress transfer due to interlock and dowel action.
    """
    def _get_Q_reduced(self):
        Q_reduced = self.Q - self.sz_stress_profile.F_a[1]
        return Q_reduced


    # F_beam = tr.Property
    # '''Use the reference to MQProfileand BoundaryConditions
    # to calculate the global load. Its interpretation depends on the
    # nature of the load - single mid point, four-point, distributed.
    # '''
    # # TODO: Currently there is just a single midpoint load of a 3pt bending beam assumed.
    # #       then, the load is equal to the shear force
    # def _get_F_beam(self):
    #     return  2 * self.Q

    x_tip_1k = tr.Property

    def _get_x_tip_1k(self):
        return self.sz_cp.sz_ctr.x_tip_ak[1]

    tau_x_tip_1k = tr.Property

    def _get_tau_x_tip_1k(self):  #Shear stress distribution in uncracked region?
        H = self.sz_bd.H
        B = self.sz_bd.B
        Q_reduced = self.Q - self.sz_stress_profile.F_a[1]
        return get_tau_z_fps(self.x_tip_1k, Q_reduced, H, B)[0]

    tau_z = tr.Property

    def _get_tau_z(self): #shear stress at crack tip
        H = self.sz_bd.H
        B = self.sz_bd.B
        z_arr = np.linspace(0, H, 100)
        return get_tau_z(z_arr, self.x_tip_1k, self.Q, H, B)

    def subplots(self, fig):
        return fig.subplots(1, 2)

    def update_plot(self, axes):
        ax1, ax2 = axes
        H = self.sz_bd.H
        B = self.sz_bd.B

        Q_val = np.linspace(0, self.Q, 100)
        z_fps_arr = np.linspace(0, H * 0.98, 100)
        tau_z_fps_arr = get_tau_z_fps(z_fps_arr, Q_val, H, B)

        z_arr = np.linspace(0, H, 100)
        tau_z_arr = get_tau_z(z_arr, self.x_tip_1k, self.Q, H, B)

        ax1.plot(tau_z_fps_arr, z_fps_arr, color='green');
        ax1.fill_betweenx(z_fps_arr, tau_z_fps_arr, 0, color='green', alpha=0.1)

        ax2.plot(tau_z_arr, z_arr, color='blue');
        ax2.fill_betweenx(z_arr, tau_z_arr, 0, color='blue', alpha=0.1)

