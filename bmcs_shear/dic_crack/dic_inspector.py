
import bmcs_utils.api as bu
from .dic_cor import DICCOR
from .dic_grid import DICGrid
from .dic_strain_grid import DICStrainGrid
from .dic_crack_list import DICCrackList
import traits.api as tr

class DICInspector(bu.Model):
    '''
    Top level class of the DIC Shear Zone Inspector
    - Input data grid
    - Evaluation of strains using bilinear shape functions
    - Evaluation of principle strain directions
    - Contains a list of the cracks, each of which evaluates the crack kinematics
    '''
    name = 'crack inspector'
    dic_grid = bu.Instance(DICGrid, ())

    dic_strain_grid = tr.Property(bu.Instance(DICStrainGrid), depends_on='state_changed')
    @tr.cached_property
    def _get_dic_strain_grid(self):
        return DICStrainGrid(dic_grid=self.dic_grid)

    dic_cracks = bu.Instance(DICCrackList, ())

    tree = ['dic_grid', 'dic_strain_grid', 'dic_cracks']
