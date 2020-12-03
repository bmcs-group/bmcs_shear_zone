
from .sz_softening_tensile_behavior import TensileSofteningBehavior
from .sz_compressive_hardening import CompressiveHardeningBehavior
from .sz_aggregate_interlock import AggregateInterlock
from bmcs_shear.matmod.i_matmod import IMaterialModel
import bmcs_utils.api as bu
import traits.api as tr

@tr.provides(IMaterialModel)
class ConcreteMaterialModel(bu.InteractiveModel):

    name = 'Concrete behavior'
    node_name = 'material model'

    tension = tr.Instance(TensileSofteningBehavior, ())
    copression = tr.Instance(CompressiveHardeningBehavior, ())
    interlock = tr.Instance(AggregateInterlock, ())

