import numpy as np
import traits.api as tr
from bmcs_utils.api import InteractiveModel, InteractiveWindow, View, Item, Float, Int
from bmcs_shear.shear_crack.crack_tip_rotation import \
    SZCrackTipRotation
from bmcs_shear.shear_crack.beam_design import \
    RCBeamDesign

# # Crack path

# ## Level set crack path representation

# Consider a crack path defined by a level set function
# \begin{align}
# \gamma( x_a ) = 0
# \end{align}
# This function consists of three branches. The existing crack $\gamma_0( x_a ) = 0$ which ends at a point $x_a^{\mathrm{tip}}$

# On the other hand, assuming

# Localization length is used to transform the crack opening at the crack tip
# to the tensile strain within that zone.

# ## Discrete crack path representation
# The crack path is defined along the nodes $x_{aI}$ with $a \in (0,1)$ representing the dimension index and $I \in 1 \ldots n_I$ defining the global node index.

# A starting example of a crack geometry is defined as follows


def get_I_Li(x_Ia):
    N_I = np.arange(len(x_Ia))
    I_Li = np.array([N_I[:-1], N_I[1:]], dtype=np.int_).T
    return I_Li


# The nodal coordinates rearranged into an array accessible via a line segment index $L$ and the local segment node $i \in (0,1)$ is defined as.


def get_x_Ja(x_Ia, x_Ca, n_J):
    x_J_1 = np.linspace(x_Ia[-1 ,1], x_Ca[-1 ,1], n_J)
    return np.c_[x_Ia[-1 ,0 ] *np.ones_like(x_J_1), x_J_1]


# The line vector $v_{La}$ is obtained by subtracting the first node $i=0$ from the second node $i=1$
# \begin{align}
# v_{La} = x_{L1a} - x_{L0a}
# \end{align}

# In[60]:


def get_n_vec_La(x_Ia):
    x_Lia = x_Ia[get_I_Li(x_Ia)]
    n_vec_La = x_Lia[: ,1 ,:] - x_Lia[: ,0 ,:]
    return n_vec_La


# In[61]:


def get_norm_n_vec_L(x_Ia):
    n_vec_La = get_n_vec_La(x_Ia)
    return np.sqrt(np.einsum('...a,...a->...', n_vec_La, n_vec_La))


# normalize the vector to a unit length
# \begin{align}
#  \hat{v}_{La} = \frac{v_{La}}{| v |_L}
# \end{align}

# In[62]:


def get_normed_n_vec_La(x_Ia):
    return np.einsum('...a,...->...a',
                     get_n_vec_La(x_Ia), 1. / get_norm_n_vec_L(x_Ia))


# Using the Levi-Civita symbol
# \begin{align}
# \epsilon_{abc}
# \end{align}
# and an out-of-plane vector $z_a = [0,0,1]$

# In[63]:


EPS = np.zeros((3, 3, 3), dtype='f')
EPS[(0, 1, 2), (1, 2, 0), (2, 0, 1)] = 1
EPS[(2, 1, 0), (1, 0, 2), (0, 2, 1)] = -1
Z = np.array([0, 0, 1], dtype=np.float_)


# we obtain the normal vector to the line as
# \begin{align}
# \hat{t}_{La} = \hat{n}_{Lb} z_c \epsilon_{abc}
# \end{align}

# \begin{align}
# \nonumber
# \hat{T}_{bLa} = [\hat{n}_{La}, \hat{t}_{La}]
# \end{align}
# and after reordering indexes the transformation matrix for each line segment along the crack path is obtained as
# \begin{align}
# \label{eq:T_Lab}
# \hat{T}_{Lab} = \hat{T}_{bLa}
# \end{align}

# The function with the parameters defining the current crack path is defined as follows

# In[64]:


def get_T_Lab(x_Ia):
    I_Li = get_I_Li(x_Ia)
    x_Lia = x_Ia[I_Li]
    line_vec_La = x_Lia[: ,1 ,:] - x_Lia[: ,0 ,:]
    norm_line_vec_L = np.sqrt(np.einsum('...a,...a->...',
                                        line_vec_La, line_vec_La))
    normed_line_vec_La = np.einsum('...a,...->...a',
                                   line_vec_La, 1. / norm_line_vec_L)
    t_vec_La = np.einsum('ijk,...j,k->...i',
                         EPS[:-1 ,:-1 ,:], normed_line_vec_La, Z);
    T_bLa = np.array([t_vec_La, normed_line_vec_La])
    T_Lab = np.einsum('bLa->Lab', T_bLa)
    return T_Lab


# **Treatment of the crack propagation:** The crack tip is updated during the
# crack extension by setting the values of `x_tip_0n` and `x_tip_1n`.

# In[65]:


class SZCrackPath(InteractiveModel):
    '''Crack path representation

    Defines the incrementally extensible crack path through the shear zone.

    Crack segments are added by setting the property `x_tip_an`
    Upon setting the property, the last crack tip is appended to the array `x_t_Ia`
    and the `sz_ctr` object representing the crack tip is updated to a new
    crack tip position.
    '''

    name = 'Crack path'

    n_m = Int(4, DSC=True)
    n_J = Int(10, DSC=True)

    ipw_view = View(
        Item('n_m', latex='n_m'),
        Item('n_J', latex='n_J'),
        Item('x_00', latex=r'x_{00}')
    )

    sz_bd = tr.Instance(RCBeamDesign ,())
    '''Beam design object provides geometrical data and maerial data.
    '''

    sz_ctr = tr.Instance(SZCrackTipRotation)
    '''Center of tip rotation - private model component 
       representing the crck tip kinematics.
    '''
    def _sz_ctr_default(self):
        # Initializa the crack tip at the bottom of a beam with beta=0
        cmm = self.sz_bd.cmm
        return SZCrackTipRotation(x_tip_0n=self.x_00, x_tip_1n=0, psi=0,
                                  L_fps=cmm.L_fps, w=cmm.w_cr)

    @tr.on_trait_change('sz_bd, sz_bd._GEO, sz_bd._MAT')
    def _reset_sz_ctr(self):
        cmm = self.sz_bd.cmm
        self.sz_ctr.trait_set(x_tip_0n=self.x_00, x_tip_1n=0,psi=0,
                              L_fps=cmm.L_fps, w=cmm.w_cr)

    @tr.on_trait_change('_MAT, _GEO')
    def reset_crack(self):
        self.x_t_Ia = np.zeros((0 ,2), dtype=np.float_)
        self.add_x_tip_an(np.array([self.x_00, 0], dtype=np.float_))
        self.sz_ctr.x_rot_1k = self.sz_bd.H / 2
        self.sz_ctr.psi = 0

    x_00 = Float(300, GEO=True)
    '''Initial crack position'''

    # TODO: adjust the range of parameters that support sliding.
    #       this requires the definition of the traits using the tr.Range type
    #       direct specification of the range from within the Range editor
    #       provides an alternative solution - then, the min-max attribute
    #       must be defined as a trait and named in low / high Range trait parameter.
        # self.sz_ctr.traits(trantient=False)['x_rot_1k'].minmax = (0, self.sz_bd.H)
        # self.sz_ctr.traits(trantient=False)['x_tip_0n'].minmax = (0, self.sz_bd.L)
        # self.sz_ctr.traits(trantient=False)['x_tip_1n'].minmax = (0, self.sz_bd.H)
        # self.traits(param=True)['x_00'].minmax = (0, self.sz_bd.L)

    x_t_Ia = tr.Array
    '''Crack nodes up to a crack tip'''
    def _x_t_Ia_default(self):
        return np.array([[self.x_00 ,0]], dtype=np.float_)

    def add_x_tip_an(self, value):
        '''Set a current crack tip coordinates.'''
        value = np.array(value ,dtype=np.float_)
        self.x_t_Ia = np.vstack([self.x_t_Ia, value[np.newaxis, :]])
        self.sz_ctr.x_tip_0n, self.sz_ctr.x_tip_1n = value
        self.crack_extended = True

    def get_x_tip_an(self):
        return self.x_t_Ia[-1 ,:]

    # TODO: distinguish the changes in ITER, INCR and PARAM
    #       Following state changes can occur
    #       ctr: +ITR - the control parameters changed by the
    #            the iterative solver - currently psi and x_ctr_1k
    #       ctr: +INC - x_tip_an - crack tip positions
    #       ctr: +MAT - material parameters only changed by user
    #       self: +GEO - parameters describing the geometry
    #                    that means initial crack position / length, / width
    #       dependency - GEO|MAT -> INCR -> ITER - there must be a handling
    #                    of the change at the higher level (reset)

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

    x_Ia = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT')
    '''Nodes along the crack path including the fps segment'''
    @tr.cached_property
    def _get_x_Ia(self):
        x_fps_ak = self.sz_ctr.x_fps_ak
        return np.vstack([self.x_t_Ia, x_fps_ak.T])

    I_Li = tr.Property(depends_on='_INC, _GEO, _MAT')
    '''Crack segments'''
    @tr.cached_property
    def _get_I_Li(self):
        N_I = np.arange(len(self.x_Ia))
        I_Li = np.array([N_I[:-1], N_I[1:]], dtype=np.int_).T
        return I_Li

    x_Ja = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT')
    '''Uncracked vertical section'''
    @tr.cached_property
    def _get_x_Ja(self):
        x_J_1 = np.linspace(self.x_Ia[-1, 1], self.sz_bd.H, self.n_J)
        return np.c_[self.x_Ia[-1, 0] * np.ones_like(x_J_1), x_J_1]

    xx_Ka = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Integrated section'''
    @tr.cached_property
    def _get_xx_Ka(self):
        return np.concatenate([self.x_Ia, self.x_Ja[1:]], axis=0)

    x_Ka = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Integration points'''
    @tr.cached_property
    def _get_x_Ka(self):
        eta_m = np.linspace(0, 1, self.n_m)
        d_La = self.xx_Ka[1:] - self.xx_Ka[:-1]
        d_Kma = np.einsum('Ka,m->Kma', d_La, eta_m)
        x_Kma = self.xx_Ka[:-1, np.newaxis, :] + d_Kma
        return np.vstack([x_Kma[:, :-1, :].reshape(-1, 2), self.xx_Ka[[-1], :]])

    K_Li = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Crack segments'''
    @tr.cached_property
    def _get_K_Li(self):
        N_K = np.arange(len(self.x_Ka))
        K_Li = np.array([N_K[:-1], N_K[1:]], dtype=np.int_).T
        return K_Li

    x_Lb = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Midpoints'''
    @tr.cached_property
    def _get_x_Lb(self):
        return np.sum(self.x_Ka[self.K_Li], axis=1) / 2

    beta = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT')
    '''Inclination of the last crack segment with respect to vertical axis'''
    @tr.cached_property
    def _get_beta(self):
        if len(self.x_t_Ia) <= 2:
            return 0
        else:
            T_tip_Lab = self.T_Lab[-1]
            s_beta, _ = T_tip_Lab[1]
            beta = np.arcsin(s_beta)
            return beta

    norm_n_vec_L = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DSC')
    '''Length of a discretization line segment. 
    '''
    @tr.cached_property
    def _get_norm_n_vec_L(self):
        K_Li = self.K_Li
        x_Lia = self.x_Ka[K_Li]
        n_vec_La = x_Lia[:, 1, :] - x_Lia[:, 0, :]
        return np.sqrt(np.einsum('...a,...a->...', n_vec_La, n_vec_La))

    T_Lab = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT')
    '''Orthonormal bases of the crack segments, first vector is the line vector.
    '''
    @tr.cached_property
    def _get_T_Lab(self):
        return get_T_Lab(self.x_t_Ia)

    T_Mab = tr.Property(depends_on='_ITR, _INC, _GEO, _MAT, _DCS')
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

    def update_plot(self, ax):
        ax.set_ylim(ymin=0 ,ymax=self.sz_bd.H)
        ax.set_xlim(xmin=0 ,xmax=self.sz_bd.L)
        ax.axis('equal');
        self.sz_ctr.plot_crack_tip_rotation(ax)
        self.sz_bd.plot_sz_bd(ax)
        self.plot_sz0(ax)
#        self.plot_x_Ka(ax)
