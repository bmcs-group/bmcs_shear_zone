'''
**Remark:** Upon a crack extension, the state parameters
of the crack tip object, namely $\beta$ and $x^\mathrm{rot}_{1k}$ are reset
to initial values of the next iterative solution step. The value of $\beta$
is calculated using the last segment of the crack path. The crack tip
accepts $\beta$ as a constant and sets the value of $\psi = \beta$ and $\theta = 0$

# Deformed state

Let us now consider a rotated configuration of the right plate around $x_{ak}^{\mathrm{rot}}$ by the angle $\theta$ inducing the critical crack opening $w_\mathrm{cr}$ at the point $x^{\mathrm{tip}}_{ak}$

## Displacement and stress along the crack

The displacement at the ligament
\begin{align}
u_{Lb} &= x^1_{Lb} - x^0_{Lb}
\end{align}

By substituting for $x_{Lb}^1$ we obtain the displaced configuration of any point on the right plate.


By transforming the $u_{Lb}$ to in-line and out-of-line components using the orthonormal basis (\ref{eq:T_Lab})
\begin{align}
 w_{Lr} = T_{Lar} u_{La}
\end{align}

By applying the constitutive relation
\begin{align}
s_{Ls} = \mathcal{S}_{Ls}(w_{Lr})
\end{align}

Transformation to the global coordinate system
\begin{align}
\sigma_{La} = T_{Las} s_{Ls}
\end{align}
'''

import traits.api as tr
import numpy as np
from bmcs_utils.api import Model, View, Item
from bmcs_shear.shear_crack.crack_path import \
    SZCrackPath

class SZDeformedState(Model):

    name = 'Deformed state'

    ipw_view = View()

    sz_cp = tr.Instance(SZCrackPath)
    def _sz_cp_default(self):
        return SZCrackPath()

    sz_ctr = tr.DelegatesTo('sz_cp')
    sz_bd = tr.DelegatesTo('sz_cp')

    tree = ['sz_cp']

    x1_Ia = tr.Property(depends_on='state_changed')
    '''Displaced segment nodes'''
    @tr.cached_property
    def _get_x1_Ia(self):
        return self.sz_ctr.get_x1_La(self.sz_cp.x_Ia)

    x1_Ja = tr.Property(depends_on='state_changed')
    '''Displaced segment nodes'''
    @tr.cached_property
    def _get_x1_Ja(self):
        P_ab, _ = self.sz_ctr.get_T_ab_dT_dI_abI()
        x_rot_a = self.sz_ctr.x_rot_ak[: ,0]
        return self.sz_ctr.get_x1_La(self.sz_cp.x_Ja)

    x1_Ka = tr.Property(depends_on='state_changed')
    '''Displaced integration points'''
    @tr.cached_property
    def _get_x1_Ka(self):
        P_ab, _ = self.sz_ctr.get_T_ab_dT_dI_abI()
        x_rot_a = self.sz_ctr.x_rot_ak[: ,0]
        return self.sz_ctr.get_x1_La(self.sz_cp.x_Ka)

    x1_Ca = tr.Property(depends_on='state_changed')
    '''Diplaced corner nodes'''
    @tr.cached_property
    def _get_x1_Ca(self):
        P_ab, _ = self.sz_ctr.get_T_ab_dT_dI_abI()
        x_rot_a = self.sz_ctr.x_rot_ak[: ,0]
        return self.sz_ctr.get_x1_La(self.sz_bd.x_Ca)

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
        ax.set_title(r'Simulated crack path', fontsize=14)
        ax.set_xlabel(r'Horizontal position $x$ [mm]', fontsize=14)
        ax.set_ylabel(r'Vertical position $z$ [mm]', fontsize=14)
        ax.plot(*x_aI, lw=2, color='black')
        ax.plot([x_Ia[-1,0], x_LU[0]], [x_Ia[-1,1], x_LU[1]], color='gray')

    def plot_sz_fill(self, ax):
        x0_Ca = self.sz_bd.x_Ca
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
        x1_iCa = x1_Ca[self.sz_bd.C_Li]
        x1_aiM = np.einsum('iMa->aiM', x1_iCa)
        ax.plot(*x1_aiM ,color='red')

    def plot_reinf1(self, ax):
        """Plot the reinforcement
        """
        x_N = self.sz_cp.x_00
        # loop over the reinforcement layers and plot them.
        for z in self.sz_bd.z_N:
            L = self.sz_bd.L
            x_reinf0 = np.array([[0,z], [x_N, z]], dtype=np.float_)
            x_reinf1_ = np.array([[x_N, z], [L, z]], dtype=np.float_)
            x_reinf1 = self.sz_ctr.get_x1_La(x_reinf1_)
            x_reinf_Ia = np.vstack([x_reinf0, x_reinf1])
            ax.plot(*(x_reinf_Ia.T), color='brown')

    def update_plot(self, ax):
        ax.set_ylim(ymin=0 ,ymax=self.sz_bd.H)
        ax.set_xlim(xmin=0 ,xmax=self.sz_bd.L)
        ax.axis('equal');
        self.sz_ctr.plot_crack_tip_rotation(ax)
        #        self.sz_bd.plot_sz_bd(ax)
        #        self.sz_cp.plot_x_Ka(ax)
        #        self.plot_sz_x1_La(ax)
        self.sz_cp.plot_sz0(ax)
        self.plot_sz1(ax)
        self.plot_sz_fill(ax)
        self.plot_reinf1(ax)
