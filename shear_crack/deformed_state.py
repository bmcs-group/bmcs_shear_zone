
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
import ipywidgets as ipw
from bmcs_shear_zone.api import SteelMaterialModel, ConcreteMaterialModel
from bmcs_utils.api import SymbExpr, InjectSymbExpr


# **Remark:** Upon a crack extension, the state parameters
# of the crack tip object, namely $\beta$ and $x^\mathrm{rot}_{1k}$ are reset
# to initial values of the next iterative solution step. The value of $\beta$
# is calculated using the last segment of the crack path. The crack tip
# accepts $\beta$ as a constant and sets the value of $\psi = \beta$ and $\theta = 0$

# # Deformed state

# Let us now consider a rotated configuration of the right plate around $x_{ak}^{\mathrm{rot}}$ by the angle $\theta$ inducing the critical crack opening $w_\mathrm{cr}$ at the point $x^{\mathrm{tip}}_{ak}$

# ## Displacement and stress along the crack

# The displacement at the ligament
# \begin{align}
# u_{Lb} &= x^1_{Lb} - x^0_{Lb}
# \end{align}

# By substituting for $x_{Lb}^1$ we obtain the displaced configuration of any point on the right plate.


# By transforming the $u_{Lb}$ to in-line and out-of-line components using the orthonormal basis (\ref{eq:T_Lab})
# \begin{align}
#  w_{Lr} = T_{Lar} u_{La}
# \end{align}

# By applying the constitutive relation
# \begin{align}
# s_{Ls} = \mathcal{S}_{Ls}(w_{Lr})
# \end{align}

# Transformation to the global coordinate system
# \begin{align}
# \sigma_{La} = T_{Las} s_{Ls}
# \end{align}

# In[70]:


class SZDeformedState(InteractiveModel):

    name = 'Deformed state'

    ipw_view = View()

    sz_cp = tr.Instance(SZCrackPath ,())

    sz_ctr = tr.DelegatesTo('sz_cp')

    sz_geo = tr.DelegatesTo('sz_cp')

    crack_extended = tr.Event
    @tr.on_trait_change('sz_cp.crack_extended')
    def _trigger_event(self):
        self.crack_extended = True

    x1_Ia = tr.Property(depends_on='crack_extended')
    '''Displaced segment nodes'''
    @tr.cached_property
    def _get_x1_Ia(self):
        return self.sz_ctr.get_x1_La(self.sz_cp.x_Ia)

    x1_Ja = tr.Property(depends_on='crack_extended')
    '''Displaced segment nodes'''
    @tr.cached_property
    def _get_x1_Ja(self):
        P_ab, _ = self.sz_ctr.get_T_ab_dT_dI_abI()
        x_rot_a = self.sz_ctr.x_rot_ak[: ,0]
        return self.sz_ctr.get_x1_La(self.sz_cp.x_Ja)

    x1_Ka = tr.Property(depends_on='crack_extended')
    '''Displaced integration points'''
    @tr.cached_property
    def _get_x1_Ka(self):
        P_ab, _ = self.sz_ctr.get_T_ab_dT_dI_abI()
        x_rot_a = self.sz_ctr.x_rot_ak[: ,0]
        return self.sz_ctr.get_x1_La(self.sz_cp.x_Ka)

    x1_Ca = tr.Property(depends_on='crack_extended')
    '''Diplaced corner nodes'''
    @tr.cached_property
    def _get_x1_Ca(self):
        P_ab, _ = self.sz_ctr.get_T_ab_dT_dI_abI()
        x_rot_a = self.sz_ctr.x_rot_ak[: ,0]
        return self.sz_ctr.get_x1_La(self.sz_geo.x_Ca)

    def plot_sz1(self, ax):
        x_Ia = self.x1_Ia
        x_Ca = self.x1_Ca
        x_aI = x_Ia.T
        x_LL = self.x1_Ka[0]
        x_LU = self.x1_Ka[-1]
        x_RL = x_Ca[1]
        x_RU = x_Ca[2]
        x_Da = np.array([x_LL, x_RL, x_RU, x_LU])
        D_Li = np.array([[0, 1], [1, 2], [2, 3], ], dtype=np.int_)
        x_aiD = np.einsum('Dia->aiD', x_Da[D_Li])
        ax.plot(*x_aiD, color='black')
        ax.set_title(r'Simulated crack path')
        ax.set_xlabel(r'Horizontal position $x$ [mm]')
        ax.set_ylabel(r'Vertical position $z$ [mm]')
        ax.plot(*x_aI, lw=2, color='black')

    def plot_sz_fill(self, ax):
        x0_Ca = self.sz_geo.x_Ca
        x1_Ca = self.x1_Ca
        x0_Ja = self.sz_cp.x_Ja
        x1_Ja = self.x1_Ja
        x01_Ja = (x0_Ja[-1 ,:] + x1_Ja[-1 ,:]) / 2
        x_Da = np.vstack([
            x0_Ca[:1],
            self.sz_cp.x_Ia,
            self.x1_Ia[::-1],
            x1_Ca[1:3, :],
            x01_Ja[np.newaxis ,:],
            x0_Ca[3:]
        ])
        ax.fill(*x_Da.T, color='gray', alpha=0.2)

    def plot_sz_x1_La(self, ax):
        x1_Ca = self.x1_Ca
        x1_iCa = x1_Ca[self.sz_geo.C_Li]
        x1_aiM = np.einsum('iMa->aiM', x1_iCa)
        ax.plot(*x1_aiM ,color='red')

    def update_plot(self, ax):
        ax.set_ylim(ymin=0 ,ymax=self.sz_geo.H)
        ax.set_xlim(xmin=0 ,xmax=self.sz_geo.L)
        ax.axis('equal');
        self.sz_ctr.plot_crack_tip_rotation(ax)
        #        self.sz_geo.plot_sz_geo(ax)
        #        self.sz_cp.plot_x_Ka(ax)
        #        self.plot_sz_x1_La(ax)
        self.sz_cp.plot_sz0(ax)
        self.plot_sz1(ax)
        self.plot_sz_fill(ax)
