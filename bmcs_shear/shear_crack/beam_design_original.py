
import numpy as np
from bmcs_shear.matmod import CrackBridgeSteel, ConcreteMaterialModel, SteelMaterialModel
from bmcs_shear.matmod import DowelAction, AggregateInterlock
import traits.api as tr
from bmcs_utils.api import \
    View, Item, Float
from bmcs_beam.beam_config.beam_design import BeamDesign
from bmcs_cross_section.cs_design.cs_design import CrossSectionDesign


# # Material constitutive laws

# material laws for slip and for crack opening are defined independently as
# \begin{align}
# \mathcal{T} := (w,s) \rightarrow (\sigma, \tau)
# \end{align}

# # Beam design
# Define the initial configuration of the reinforced beam including the geometry and reinforcement layout with associated crack bridge laws.

# ## Beam geometry
# The geometry of the shear zone is defined as length $L$, height $H$ and width $B$. They are defined numerically for the running example as

# In[50]:


L_, H_, B_ = 100, 60, 20

# The corner coordinates are stored in an array $x_{aC}$ where $a \in (0,1)$ is the dimension index and $C \in (0,1,2,3)$ is the global corner index

# In[51]:


x_aC = np.array([[0, L_, L_, 0],
                 [0, 0, H_, H_]], dtype=np.float_)
x_Ca = x_aC.T

# The lines delimiting the shear zone are defined by an index map $C_{Li}$ with $L \in (0,1,2,3)$ denoting the line and $i \in (0,1)$ the local node index defining a line.

# In[52]:


C_iL = np.array([[0, 1, 2, 3],
                 [1, 2, 3, 0]], dtype=np.int_)

# **Example:** With this zone definition, the lines can be ordered to an array $x_{iLa}$ using the index operator as follows.

# In[53]:


x_iLa = x_Ca[C_iL, :]
np.einsum('iLa->aiL', x_iLa)

# ## Reinforcement layout

# In[54]:


class RCBeamDesign(BeamDesign):
    name = 'Beam design'

    cmm = tr.Instance(ConcreteMaterialModel, ())
    smm = tr.Instance(CrackBridgeSteel, ())
    da = tr.Instance(DowelAction, ())
    ag_in = tr.Instance(AggregateInterlock,())

    # Only for visualization to delimit the plotted area
    H = Float(200, GEO=True)
    L = Float(800, GEO=True)
    B = Float(100, GEO=True)

    _GEO = tr.Event
    @tr.on_trait_change('+GEO') #, cs_layout, cs_layout.+GEO')
    def _reset_GEO(self):
        self._GEO = True

    _MAT = tr.Event
    @tr.on_trait_change('+MAT, cmm, cmm.+MAT, smm, smm.+MAT')
    def _reset_MAT(self):
        self._MAT = True

    ipw_view = View(
        Item('H', minmax=(1, 400), latex=r'H'),
        Item('L', minmax=(1, 1000), latex=r'L'),
        Item('B', minmax=(1, 100), latex=r'B')
    )

    C_Li = tr.Array(value=[[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int_)

    x_Ca = tr.Property(depends_on='+GEO')
    '''Array of corner nodes [C-node, a-dimension]'''

    @tr.cached_property
    def _get_x_Ca(self):
        x_aC = np.array([[0, self.L, self.L, 0],
                         [0, 0, self.H, self.H]], dtype=np.float_)
        return x_aC.T

    x_aiM = tr.Property(depends_on='+GEO')
    '''Array of boundary lines [a-dimension,i-line node,M-line]'''

    @tr.cached_property
    def _get_x_aiM(self):
        x_iCa = self.x_Ca[self.C_Li]
        return np.einsum('iMa->aiM', x_iCa)

    def plot_sz_bd(self, ax):
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.H)
        ax.plot(*self.x_aiM, color='black')

    def plot_sz_cross_section(self, ax):
        ax.plot([0, self.B, self.B, 0, 0],
                [0, 0, self.H, self.H, 0], color='black')
        ax.set_xlim(0, self.B)
        ax.set_ylim(0, self.H)

    def update_plot(self, ax1):
        ax1.axis('equal');
        self.plot_sz_bd(ax1)
#        self.plot_sz_cross_section(ax2)
