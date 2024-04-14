
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
from .dic_inp_unstructured_points import DICInpUnstructuredPoints
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from .cached_array import cached_array

class DICGrid(bu.Model):
    """
    History of displacment grids imported from the DIC measurement.
    """
    name = 'DIC grid history'
    depends_on = ['dic_inp']
    tree = ['dic_inp']

    bd = tr.DelegatesTo('dic_inp', 'sz_bd')

    t = tr.DelegatesTo('dic_inp')

    ipw_view = bu.View(
        bu.Item('d_x'),
        bu.Item('d_y'),
        bu.Item('n_I', readonly=True),
        bu.Item('n_J', readonly=True),
        time_editor=bu.HistoryEditor(
            var='t', max_value = 1.1
        )
    )

    dic_inp = bu.Instance(DICInpUnstructuredPoints, ())

    data_dir = tr.DelegatesTo('dic_inp')
    beam_param_file = tr.DelegatesTo('dic_inp')
    sz_bd = tr.DelegatesTo('dic_inp')
    F_T_t = tr.DelegatesTo('dic_inp')
    Q_t = tr.DelegatesTo('dic_inp')
    Q_1 = tr.DelegatesTo('dic_inp')
    M_t = tr.DelegatesTo('dic_inp')
    M_1 = tr.DelegatesTo('dic_inp')

    n_I = tr.Property(bu.Int, depends_on='state_changed')
    """Number of horizontal nodes of the DIC input displacement grid.
    """
    @tr.cached_property
    def _get_n_I(self):
        return int(self.dic_inp.L_x / self.d_x)

    n_J = tr.Property(bu.Int, depends_on='state_changed')
    """Number of vertical nodes of the DIC input displacement grid
    """
    @tr.cached_property
    def _get_n_J(self):
        return int(self.dic_inp.L_y / self.d_y)

    d_x = bu.Float(5 , ALG=True)
    """Horizontal spacing between nodes of the DIC input displacement grid.
    """

    d_y = bu.Float(5 , ALG=True)
    """Vertical spacing between nodes of the DIC input displacement grid.
    """

    X0_IJa = tr.Property(depends_on='state_changed')
    """Coordinates of the DIC markers in the grid"""
    @tr.cached_property
    def _get_X0_IJa(self):
        n_I, n_J = self.n_I, self.n_J
        X_min_a, X_max_a = self.dic_inp.X_outer_frame
        pad_l, pad_r, pad_b, pad_t = (
            self.dic_inp.pad_l, self.dic_inp.pad_r, self.dic_inp.pad_b, self.dic_inp.pad_t
        )
        min_x, min_y, _ = X_min_a
        max_x, max_y, _ = X_max_a
        X_aIJ = np.mgrid[
                min_x + pad_l:max_x - pad_r:complex(n_I),
                min_y + pad_b:max_y - pad_t:complex(n_J)]
        x_IJ, y_IJ = X_aIJ
        return np.einsum('aIJ->IJa', np.array([x_IJ, y_IJ]))

    delaunay = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_delaunay(self):
        points = self.dic_inp.X_Qa[:, :-1]
        return Delaunay(points)

    n_T = tr.DelegatesTo('dic_inp')
    t_T = tr.DelegatesTo('dic_inp')
    U_factor = tr.DelegatesTo('dic_inp')

    U_TIJa = tr.Property(depends_on='state_changed')
    """Read the displacement data from the individual csv files"""
    @cached_array(source_name="beam_param_file", names='U_TIJa',
                  data_dir_trait='data_dir')
    def _get_U_TIJa(self):
        x0_IJ, y0_IJ = np.einsum('IJa->aIJ', self.X0_IJa)
        U_IJa_list = []
        for T in range(self.n_T):
            values = self.dic_inp.U_TQa[T, :, :]
            get_U = LinearNDInterpolator(self.delaunay, values)
            U_IJa = get_U(x0_IJ, y0_IJ)
            U_IJa_list.append(U_IJa)
        U_TIJa = np.array(U_IJa_list)
        return U_TIJa[...,:-1]


    X_IJa = tr.Property(depends_on='state_changed')
    """Coordinates of the DIC markers in the grid"""
    @tr.cached_property
    def _get_X_IJa(self):
        X0_IJa = self.X0_IJa
        x_min, y_min = X0_IJa[0,0,(0, 1)]
        x0_IJ, y0_IJ = np.einsum('...a->a...', X0_IJa)
        x_offset, y_offset = (
            self.dic_inp.x_offset, self.dic_inp.y_offset
        )
        X_aIJ = np.array([x0_IJ-x_min+x_offset, y0_IJ-y_min+y_offset])
        return np.einsum('a...->...a', X_aIJ)


    U_IJa = tr.Property(depends_on='state_changed')
    """Total displacement at step T_t w.r.t. T0
    """
    @tr.cached_property
    def _get_U_IJa(self):
        return self.U_TIJa[self.dic_inp.T_t] - self.U_TIJa[self.dic_inp.T0]

    X_Ca = tr.Property
    def _geT_X_Ca(self):
        X_Ca = self.X_IJa[(0, 0, -1, -1, 0), (0, -1, -1, 0, 0), :]
        return X_Ca

    L_x = tr.DelegatesTo('dic_inp')
    L_y = tr.DelegatesTo('dic_inp')
    x_offset = tr.DelegatesTo('dic_inp')
    y_offset = tr.DelegatesTo('dic_inp')
    T_t = tr.DelegatesTo('dic_inp')

    X_frame = tr.Property
    """Define the bottom left and top right corners"""
    def _get_X_frame(self):
        L_x, L_y = self.L_x, self.L_y
        x_offset, y_offset = self.x_offset, self.y_offset
        x_min, y_min = self.X_IJa[0,0,(0,1)] #x_offset + self.pad_l
        #x_max = x_min + L_x - self.pad_r
        x_max, y_max = self.X_IJa[-1,-1,(0,1)]
        # y_max = y_min + L_y - self.pad_t
        return x_min, y_min, x_max, y_max


    def plot_grid(self, ax_u):
        XU_aIJ = np.einsum('IJa->aIJ', self.X_IJa + self.U_IJa * self.U_factor)
        ax_u.scatter(*XU_aIJ.reshape(2, -1), s=15, marker='o', color='darkgray')
        ax_u.axis('equal')

    def plot_load_deflection(self, ax_load):
        self.dic_inp.plot_load_deflection(ax_load)

    X_Ca = tr.Property
    def _get_X_Ca(self):
        X_Ca = self.X_IJa[(0, 0, -1, -1, 0), (0, -1, -1, 0, 0), :]
        return X_Ca

    def plot_bounding_box(self, ax):
        X_Ca = self.X_Ca
        X_iLa = np.array([X_Ca[:-1], X_Ca[1:]], dtype=np.float_)
        X_aiL = np.einsum('iLa->aiL', X_iLa)
        ax.plot(*X_aiL, color='black', lw=0.5)

    def plot_box_annotate(self, ax):
        self.dic_inp.plot_box_annotate(ax)

    def subplots(self, fig):
        return fig.subplots(2,1)

    def update_plot(self, axes):
        ax_u, ax_load = axes
        self.plot_grid(ax_u)
        self.plot_bounding_box(ax_u)
        self.plot_box_annotate(ax_u)
        self.plot_load_deflection(ax_load)

    def get_latex_grid_params(self):
        return f'''
    \\begin{{center}}
    \\begin{{tabular}}{{|c|c|c|c|}}
    \\hline
    Name & Symbol & Unit & Value \\\\
    \\hline
    d\\_x & $\Delta x$ & mm & {self.d_x:.1f} \\\\
    \\hline
    d\\_y & $\Delta y$ & mm & {self.d_y:.1f} \\\\
    \\hline
    \\end{{tabular}}
    \\end{{center}}
    '''
