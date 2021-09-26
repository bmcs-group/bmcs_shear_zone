
import bmcs_utils.api as bu
from .dic_crack import DICCrack

class DICCrackList(bu.ModelList):
    name = 'crack list'
    items = bu.List(DICCrack, [])

