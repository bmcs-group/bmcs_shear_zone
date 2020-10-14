import numpy as np
import traits.api as tr
from bmcs_utils.api import InteractiveModel, InteractiveWindow, View, Item
from bmcs_shear_zone.shear_crack.crack_tip_rotation import \
    SZCrackTipRotation
from bmcs_shear_zone.shear_crack.beam_design import \
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

    n_m = tr.Int(4, param=True, latex='n_m', minmax=(1 ,10))
    n_J = tr.Int(10, param=True, latex='n_J', minmax=(1 ,20))

    ipw_view = View(
        Item('n_m', latex='n_m', minmax=(1 ,10)),
        Item('n_J', latex='n_J', minmax=(1 ,20))
    )

    sz_ctr = tr.Instance(SZCrackTipRotation ,)
    def _sz_ctr_default(self):
        # Initializa the crack tip at the bottom of a beam with beta=0
        return SZCrackTipRotation(
            x_tip_0n=self.x_00, x_tip_1n=0, beta=0
        )

    sz_geo = tr.Instance(RCBeamDesign ,())

    x_00 = tr.Float(300, param=True, latex='x_{00}', minmax=(0 ,1000))
    '''Initial crack position'''
    def _x_00_changed(self):
        self.x_t_Ia = np.zeros((0 ,2), dtype=np.float_)
        self.add_x_tip_an(np.array([self.x_00, 0], dtype=np.float_))
        self.sz_ctr.traits(param=True)['x_rot_1k'].minmax = (0, self.sz_geo.H)
        self.sz_ctr.traits(param=True)['x_tip_0n'].minmax = (0, self.sz_geo.L)
        self.sz_ctr.traits(param=True)['x_tip_1n'].minmax = (0, self.sz_geo.H)
        self.traits(param=True)['x_00'].minmax = (0, self.sz_geo.L)

    param_names = ['x_00'] # ,'n_m','n_J']

    x_t_Ia = tr.Array
    '''Crack nodes up to a crack tip'''
    def _x_t_Ia_default(self):
        return np.array([[self.x_00 ,0]], dtype=np.float_)

    crack_extended = tr.Event
    '''Event controling an update of crack data after a crack extension.'''

    def add_x_tip_an(self, value):
        '''Set a current crack tip coordinates.'''
        value = np.array(value ,dtype=np.float_)
        self.x_t_Ia = np.vstack([self.x_t_Ia, value[np.newaxis, :]])
        self.sz_ctr.x_tip_0n, self.sz_ctr.x_tip_1n = value
        self.sz_ctr.x_rot_1k = value[1] + (self.sz_geo.H - value[1]) / 2
        self.crack_extended = True
        self.sz_ctr.beta = self.beta
    def get_x_tip_an(self):
        return self.x_t_Ia[-1 ,:]

    x_Ia = tr.Property(depends_on='crack_extended')
    '''Nodes along the crack path including the fps segment'''
    @tr.cached_property
    def _get_x_Ia(self):
        x_fps_ak = self.sz_ctr.x_fps_ak
        return np.vstack([self.x_t_Ia, x_fps_ak.T])

    I_Li = tr.Property(depends_on='crack_extended')
    '''Crack segments'''
    @tr.cached_property
    def _get_I_Li(self):
        N_I = np.arange(len(self.x_Ia))
        I_Li = np.array([N_I[:-1], N_I[1:]], dtype=np.int_).T
        return I_Li

    x_Ja = tr.Property(depends_on='crack_extended')
    '''Uncracked vertical section'''
    @tr.cached_property
    def _get_x_Ja(self):
        x_J_1 = np.linspace(self.x_Ia[-1, 1], self.sz_geo.H, self.n_J)
        return np.c_[self.x_Ia[-1, 0] * np.ones_like(x_J_1), x_J_1]

    xx_Ka = tr.Property(depends_on='crack_extended')
    '''Integrated section'''
    @tr.cached_property
    def _get_xx_Ka(self):
        return np.concatenate([self.x_Ia, self.x_Ja[1:]], axis=0)

    x_Ka = tr.Property(depends_on='crack_extended')
    '''Integration points'''
    @tr.cached_property
    def _get_x_Ka(self):
        eta_m = np.linspace(0, 1, self.n_m)
        d_La = self.xx_Ka[1:] - self.xx_Ka[:-1]
        d_Kma = np.einsum('Ka,m->Kma', d_La, eta_m)
        x_Kma = self.xx_Ka[:-1, np.newaxis, :] + d_Kma
        return np.vstack([x_Kma[:, :-1, :].reshape(-1, 2), self.xx_Ka[[-1], :]])

    K_Li = tr.Property(depends_on='crack_extended')
    '''Crack segments'''
    @tr.cached_property
    def _get_K_Li(self):
        N_K = np.arange(len(self.x_Ka))
        K_Li = np.array([N_K[:-1], N_K[1:]], dtype=np.int_).T
        return K_Li

    x_Lb = tr.Property(depends_on='crack_extended')
    '''Midpoints'''
    @tr.cached_property
    def _get_x_Lb(self):
        return np.sum(self.x_Ka[self.K_Li], axis=1) / 2

    beta = tr.Property(depends_on='crack_extended')
    '''Inclination of the last crack segment with respect to vertical axic'''
    @tr.cached_property
    def _get_beta(self):
        if len(self.x_t_Ia) <= 2:
            return 0
        else:
            T_tip_Lab = self.T_Lab[-1]
            s_beta, _ = T_tip_Lab[1]
            beta = np.arcsin(s_beta)
            return beta

    norm_n_vec_L = tr.Property(depends_on='crack_extended')
    '''Length of a discretization line segment. 
    '''
    @tr.cached_property
    def _get_norm_n_vec_L(self):
        K_Li = self.K_Li
        x_Lia = self.x_Ka[K_Li]
        n_vec_La = x_Lia[:, 1, :] - x_Lia[:, 0, :]
        return np.sqrt(np.einsum('...a,...a->...', n_vec_La, n_vec_La))

    T_Lab = tr.Property(depends_on='crack_extended')
    '''Orthonormal bases of the crack segments, first vector is the line vector.
    '''
    @tr.cached_property
    def _get_T_Lab(self):
        return get_T_Lab(self.x_t_Ia)

    T_Mab = tr.Property(depends_on='crack_extended')
    '''Orthonormal bases of the integration segments, first vector is the line vector.
    '''
    @tr.cached_property
    def _get_T_Mab(self):
        return get_T_Lab(self.x_Ka)

    def plot_x_Ka(self ,ax):
        ax.plot(*self.x_Ka.T, color='green', alpha=0.8)

    def plot_sz0(self, ax):
        x_Ia = self.x_Ia
        x_Ca = self.sz_geo.x_Ca
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
        ax.set_ylim(ymin=0 ,ymax=self.sz_geo.H)
        ax.set_xlim(xmin=0 ,xmax=self.sz_geo.L)
        ax.axis('equal');
        self.sz_ctr.plot_crack_tip_rotation(ax)
        self.sz_geo.plot_sz_geo(ax)
        self.plot_sz0(ax)
#        self.plot_x_Ka(ax)

