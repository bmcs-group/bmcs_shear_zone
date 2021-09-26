
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

    t = bu.Float(1, ALG=True)
    def _t_changed(self):
        n_t = self.dic_grid.n_t
        d_t = (1 / n_t)
        self.dic_grid.end_t = int((n_t - 1) * (self.t + d_t / 2))

    ipw_view = bu.View(
        time_editor=bu.HistoryEditor(
            var='t'
        )
    )
    def update_plot(self, axes):
        self.dic_strain_grid.update_plot(axes)
        for dic_crack in self.dic_cracks.items:
            dic_crack.dic_cor.plot_cor(axes)