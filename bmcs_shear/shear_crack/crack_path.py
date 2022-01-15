'''
# Crack path

## Level set crack path representation

Consider a crack path defined by a level set function
\begin{align}
\gamma( x_a ) = 0
\end{align}
This function consists of three branches. The existing crack $\gamma_0( x_a ) = 0$ which ends at a point $x_a^{\mathrm{tip}}$

On the other hand, assuming

Localization length is used to transform the crack opening at the crack tip
to the tensile strain within that zone.

## Discrete crack path representation
The crack path is defined along the nodes $x_{aI}$ with $a \in (0,1)$ representing the dimension index and $I \in 1 \ldots n_I$ defining the global node index.

A starting example of a crack geometry is defined as follows
'''

import numpy as np
import traits.api as tr
from bmcs_utils.api import Model, InteractiveWindow, View, Item, Float, Int
from bmcs_shear.shear_crack.crack_tip_rotation import \
    SZCrackTipRotation
from bmcs_shear.beam_design import \
    RCBeamDesign


def get_I_Li(x_Ia):
    ''' The nodal coordinates rearranged into an array accessible via a line
    segment index $L$ and the local segment node $i \in (0,1)$ is defined as.
    '''
    N_I = np.arange(len(x_Ia))
    I_Li = np.array([N_I[:-1], N_I[1:]], dtype=np.int_).T
    return I_Li

def get_x_Ja(x_Ia, x_Ca, n_J):
    ''' The nodal coordinates rearranged into an array accessible via a line
    segment index $L$ and the local segment node $i \in (0,1)$ is defined as.
    '''
    x_J_1 = np.linspace(x_Ia[-1 ,1], x_Ca[-1 ,1], n_J)
    return np.c_[x_Ia[-1 ,0 ] *np.ones_like(x_J_1), x_J_1]

def get_n_vec_La(x_Ia):
    '''The line vector $v_{La}$ is obtained by subtracting the first node $i=0$
    from the second node $i=1$
    \begin{align}
    v_{La} = x_{L1a} - x_{L0a}
    \end{align}
    '''
    x_Lia = x_Ia[get_I_Li(x_Ia)]
    n_vec_La = x_Lia[: ,1 ,:] - x_Lia[: ,0 ,:]
    return n_vec_La

def get_norm_n_vec_L(x_Ia):
    n_vec_La = get_n_vec_La(x_Ia)
    return np.sqrt(np.einsum('...a,...a->...', n_vec_La, n_vec_La))

def get_normed_n_vec_La(x_Ia):
    '''normalize the vector to a unit length
    \begin{align}
     \hat{v}_{La} = \frac{v_{La}}{| v |_L}
    \end{align}
    '''
    return np.einsum('...a,...->...a',
                     get_n_vec_La(x_Ia), 1. / get_norm_n_vec_L(x_Ia))

EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1
# Using the Levi-Civita symbol
# \begin{align}
# \epsilon_{abc}
# \end{align}
# and an out-of-plane vector $z_a = [0,0,1]$
Z = np.array([0, 0, 1], dtype=np.float_)

def get_T_Lab(x_Ia):
    '''
    Given a sequence of nodes, I with the coordinates a return
    the transformation matrices into the local coordinates of the lines.
    '''
    I_Li = get_I_Li(x_Ia)
    #print('I_Li', I_Li)
    x_Lia = x_Ia[I_Li]
    #print('x_Lia', x_Lia)
    line_vec_La = x_Lia[: ,1 ,:] - x_Lia[: ,0 ,:]
    #print('line_vec_La', line_vec_La)
    norm_line_vec_L = np.sqrt(np.einsum('...a,...a->...',
                                        line_vec_La, line_vec_La))
    #print('norm_line_vec_L', norm_line_vec_L)
    normed_line_vec_La = np.einsum('...a,...->...a',
                                   line_vec_La, 1. / norm_line_vec_L)
    #print('normed_line_vec_La', normed_line_vec_La)
    t_vec_La = np.einsum('ijk,...j,k->...i',
                         EPS[:-1 ,:-1 ,:], normed_line_vec_La, Z);
    #print('t_vec_La', t_vec_La)
    T_bLa = np.array([t_vec_La, normed_line_vec_La])
    #print('T_bLa', T_bLa)
    T_Lab = np.einsum('bLa->Lab', T_bLa)
    #print('T_Lab', T_Lab)
    return T_Lab

class SZCrackPath(Model):
    '''Crack path representation

    Defines the incrementally extensible crack path through the shear zone.

    Crack segments are added by setting the property `x_tip_an`
    Upon setting the property, the last crack tip is appended to the array `x_t_Ia`
    and the `sz_ctr` object representing the crack tip is updated to a new
    crack tip position.
    '''

    name = 'Crack path'

    n_m = Int(40, DSC=True)
    n_J = Int(10, DSC=True)

    ipw_view = View(
        Item('n_m', latex='n_m'),
        Item('n_J', latex='n_J'),
        Item('x_00', latex=r'x_{00}')
    )

    tree = ['sz_bd', 'sz_ctr']

    sz_bd = tr.Instance(RCBeamDesign ,())
    '''Beam design object provides geometrical data and material data.
    '''

    sz_ctr = tr.Instance(SZCrackTipRotation)
    '''Center of tip rotation - private model component 
       representing the crack tip kinematics.
    '''
    def _sz_ctr_default(self):
        # Initialize the crack tip at the bottom of a beam with beta=0
        cmm = self.sz_bd.matrix_
        return SZCrackTipRotation(x_tip_0n=self.x_00, x_tip_1n=0, psi=0,
                                  L_fps=cmm.L_fps, w=cmm.w_cr)

    @tr.on_trait_change('sz_bd.state_changed')
    def _reset_sz_ctr(self):
        cmm = self.sz_bd.matrix_
        self.sz_ctr.trait_set(x_tip_0n=self.x_00, x_tip_1n=0,psi=0,
                              L_fps=cmm.L_fps, w=cmm.w_cr)

    @tr.on_trait_change('sz_bd.state_changed, +GEO')
    def reset_crack(self):
        self.x_t_Ia = np.zeros((0 ,2), dtype=np.float_)
        self.add_x_tip_an(np.array([self.x_00, 0], dtype=np.float_))
        self.sz_ctr.x_rot_1k = self.sz_bd.H / 2
        self.sz_ctr.psi = 0

    x_00 = Float(300, GEO=True)
    '''Initial crack position'''

    x_t_Ia = tr.Array
    '''Crack nodes up to a crack tip'''
    def _x_t_Ia_default(self):
        return np.array([[self.x_00 ,0]], dtype=np.float_)

    def add_x_tip_an(self, value):
        '''Set a current crack tip coordinates.'''
        value = np.array(value ,dtype=np.float_)
        self.x_t_Ia = np.vstack([self.x_t_Ia, value[np.newaxis, :]])
        self.sz_ctr.x_tip_0n, self.sz_ctr.x_tip_1n = value
        # set the inclination of the new segment equal to the inclination of the last segment.
        if len(self.x_t_Ia) > 1:
            T_n_ab = get_T_Lab(self.x_t_Ia[-2:, :])[-1]
            s_psi, _ = T_n_ab[-1,:]
            psi_n = np.arcsin(s_psi)
        else:
            psi_n = 0
        self.sz_ctr.psi = psi_n
        self.crack_extended = True
        self.state_changed = True

    def get_x_tip_an(self):
        return self.x_t_Ia[-1 ,:]

    _ITR = tr.DelegatesTo('sz_ctr', '_ITR')

    _INC = tr.Event
    @tr.on_trait_change('sz_ctr, +INC, sz_ctr._INC')
    def _reset_INC(self):
        self._INC = True

    _GEO = tr.Event
    @tr.on_trait_change('sz_bd, +GEO, sz_bd._GEO')
    def _reset_GEO(self):
        self._GEO = True

    _MAT = tr.Event
    @tr.on_trait_change('sz_bd, +MAT, sz_bd._MAT')
    def _reset_MAT(self):
        self._MAT = True

    _DSC = tr.Event
    @tr.on_trait_change('+DSC')
    def _reset_DSC(self):
        self._DSC = True

    x_Ia = tr.Property(depends_on='state_changed')
    '''Nodes along the crack path including the fps segment'''
    @tr.cached_property
    def _get_x_Ia(self):
        x_fps_ak = self.sz_ctr.x_fps_ak
        x_tip_ak = self.sz_ctr.x_tip_ak
        return np.vstack([self.x_t_Ia, x_tip_ak.T, x_fps_ak.T])

    I_Li = tr.Property(depends_on='state_changed')
    '''Index map of crack segments corresponding to the explicitly added crack extensions
    Index L is the segment index
    Index i is the node index (0,1)
    '''
    @tr.cached_property
    def _get_I_Li(self):
        N_I = np.arange(len(self.x_Ia))
        I_Li = np.array([N_I[:-1], N_I[1:]], dtype=np.int_).T
        return I_Li

    x_Ja = tr.Property(depends_on='state_changed')
    '''Uncracked vertical section
    '''
    @tr.cached_property
    def _get_x_Ja(self):
        x_J_1 = np.linspace(self.x_Ia[-1, 1], self.sz_bd.H, self.n_J)
        return np.c_[self.x_Ia[-1, 0] * np.ones_like(x_J_1), x_J_1]

    xx_Ka = tr.Property(depends_on='state_changed')
    '''Integrated section'''
    @tr.cached_property
    def _get_xx_Ka(self):
        return np.concatenate([self.x_Ia, self.x_Ja[1:]], axis=0)

    x_Ka = tr.Property(depends_on='state_changed')
    '''Integration points'''
    @tr.cached_property
    def _get_x_Ka(self):
        eta_m = np.linspace(0, 1, self.n_m)
        d_La = self.xx_Ka[1:] - self.xx_Ka[:-1]
        d_Kma = np.einsum('Ka,m->Kma', d_La, eta_m)
        x_Kma = self.xx_Ka[:-1, np.newaxis, :] + d_Kma
        return np.vstack([x_Kma[:, :-1, :].reshape(-1, 2), self.xx_Ka[[-1], :]])

    K_Li = tr.Property(depends_on='state_changed')
    '''Crack segments'''
    @tr.cached_property
    def _get_K_Li(self):
        N_K = np.arange(len(self.x_Ka))
        K_Li = np.array([N_K[:-1], N_K[1:]], dtype=np.int_).T
        return K_Li

    x_Lb = tr.Property(depends_on='state_changed')
    '''Midpoints'''
    @tr.cached_property
    def _get_x_Lb(self):
        return np.sum(self.x_Ka[self.K_Li], axis=1) / 2

    norm_n_vec_L = tr.Property(depends_on='state_changed')
    '''Length of a discretization line segment. 
    '''
    @tr.cached_property
    def _get_norm_n_vec_L(self):
        K_Li = self.K_Li
        x_Lia = self.x_Ka[K_Li]
        n_vec_La = x_Lia[:, 1, :] - x_Lia[:, 0, :]
        return np.sqrt(np.einsum('...a,...a->...', n_vec_La, n_vec_La))

    T_Lab = tr.Property(depends_on='state_changed')
    '''Orthonormal bases of the crack segments, first vector is the line vector.
    '''
    @tr.cached_property
    def _get_T_Lab(self):
        return get_T_Lab(self.x_t_Ia)

    T_tip_k_ab = tr.Property(depends_on='state_changed')
    '''Orthonormal base of the crack tip segment'''
    @tr.cached_property
    def _get_T_tip_k_ab(self):
        T_tip_ab = get_T_Lab(self.x_Ia[-2:, :])[0]
        return T_tip_ab

    T_Mab = tr.Property(depends_on='state_changed')
    '''Orthonormal bases of the integration segments, first vector is the line vector.
    '''
    @tr.cached_property
    def _get_T_Mab(self):
        return get_T_Lab(self.x_Ka)

    def plot_x_Ka(self ,ax):
        ax.plot(*self.x_Ka.T, color='green', alpha=0.8)

    def plot_sz0(self, ax):
        x_Ia = self.x_Ia
        x_Ca = self.sz_bd.x_Ca
        x_aI = x_Ia.T
        x_LL = x_Ca[0]
        x_LU = x_Ca[3]
        x_RL = self.x_Ka[0]
        x_RU = self.x_Ka[-1]
        x_Da = np.array([x_LL, x_RL, x_RU, x_LU])
        D_Li = np.array([[0, 1], [2, 3], [3, 0]], dtype=np.int_)
        x_aiD = np.einsum('Dia->aiD', x_Da[D_Li])
        ax.plot(*x_aiD, color='black')
        ax.plot(*x_aI, lw=2, color='black')
        ax.plot([x_Ia[-1,0], x_RU[0]], [x_Ia[-1,1], x_RU[1]], color='gray')

    def update_plot(self, ax):
        ax.set_ylim(ymin=0 ,ymax=self.sz_bd.H)
        ax.set_xlim(xmin=0 ,xmax=self.sz_bd.L)
        ax.axis('equal');
        self.sz_ctr.plot_crack_tip_rotation(ax)
        self.sz_bd.plot_sz_bd(ax)
        self.plot_sz0(ax)
#        self.plot_x_Ka(ax)
