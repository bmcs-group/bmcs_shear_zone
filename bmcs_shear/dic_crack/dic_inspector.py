
import bmcs_utils.api as bu
from .dic_cor import DICCOR
from .dic_grid import DICGrid
import traits.api as tr

class DICInspector(bu.Model):
    '''
    Top level class of the DIC Shear Zone Inspector
    - Input data grid
    - Evaluation of strains using bilinear shape functions
    - Evaluation of principle strain directions
    - Contains a list of the cracks, each of which evaluates the crack kinematics
    '''
    dic_grid = bu.Instance(DICGrid)
    dic_cracks = bu.List(DICCOR)

