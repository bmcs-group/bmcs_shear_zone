
import bmcs_utils.api as bu
from .dic_crack import DICCrack
from .dic_state_fields import DICStateFields

class DICCrackList(bu.ModelList):
    name = 'crack list'

    dic_state_fields = bu.Instance(DICStateFields)

    items = bu.List(DICCrack, [])

    def _get_cracks(self):
        self.dic_state_fields.cracks