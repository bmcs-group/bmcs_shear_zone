
import numpy as np
from bmcs_shear.matmod import CrackBridgeSteel, ConcreteMaterialModel, SteelMaterialModel
from bmcs_shear.matmod.sz_concrete.sz_advanced.sz_advanced import ConcreteMaterialModelAdv
from bmcs_shear.matmod.sz_crack_bridge.cb_advanced import CrackBridgeAdv
#from bmcs_shear.sz_matmod import DowelAction, AggregateInterlock
from bmcs_cross_section.cs_design import CrossSectionDesign, Rectangle
import traits.api as tr
from bmcs_utils.api import \
    View, Item, Float, EitherType
from bmcs_beam.beam_config.beam_design import BeamDesign
#from bmcs_cross_section.cs_design.cs_design import CrossSectionDesign


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

    matrix = EitherType(options=[
        ('advanced', ConcreteMaterialModelAdv),
        ('simple', ConcreteMaterialModel)
        ], MAT=True)

    cross_section_shape = EitherType(
                          options=[('rectangle', Rectangle)],
                          CS=True )

    tree = [
        'matrix',
        'cross_section_shape',
        'cross_section_layout',
        'system'
    ]


    B = tr.Property(Float)
    def _get_B(self):
        return self.cross_section_shape_.B
    def _set_B(self,value):
        self.cross_section_shape_.B = value

    ipw_view = View(
        Item('matrix'),
        Item('H', latex=r'H'),
        Item('B', latex=r'B'),
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
        L, H = self.L, self.H
        X_00 = [0, 0]
        X_01 = [L, 0]
        X_11 = [L, H]
        X_10 = [0, H]
        X_Lia = np.array([[X_00, X_01],
                          [X_01, X_11],
                          [X_11, X_10],
                          [X_10, X_00],
                          ])
        X_aiL = np.einsum('Lia->aiL', X_Lia)
        ax.plot(*X_aiL, color='black', lw=0.5)
        X_La = np.sum(X_Lia, axis=1) / 2
        x, y = X_La[0, :]
        ax.annotate(f'{L} mm',
                    xy=(x, y), xytext=(0, -1), xycoords='data',
                    textcoords='offset pixels',
                    horizontalalignment='center',
                    verticalalignment='top',
                    rotation=0
                    )
        x, y = X_La[3, :]
        ax.annotate(f'{H} mm',
                    xy=(x, y), xytext=(-5, 0),
                    xycoords='data',
                    textcoords='offset pixels',
                    horizontalalignment='right',
                    verticalalignment='center',
                    rotation=90
                    )
        ax.axis('off')
        ax.axis('equal')

    def plot_sz_cross_section(self, ax):
        ax.plot([0, self.B, self.B, 0, 0],
                [0, 0, self.H, self.H, 0], color='black')
        ax.set_xlim(0, self.B)
        ax.set_ylim(0, self.H)

    def subplots(self, fig):
        return fig.subplots(1,2)

    def update_plot(self, axes):
        ax1, ax2 = axes
        ax1.axis('equal');
        self.plot_sz_bd(ax1)
        self.csl.plot_csl(ax2)
