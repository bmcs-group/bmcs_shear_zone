
from .dic_grid import DICGrid
import bmcs_utils.api as bu
import traits.api as tr
import numpy as np
import ibvpy.api as ib
import numpy as np

class DICStrainGrid(bu.Model):

    name = 'principal tensile strain'

    dic_grid = bu.Instance(DICGrid)

    ball_size = bu.Float(1000, ALG=True)

    tree = ['dic_grid']

    t = bu.Float(1, ALG=True)
    def _t_changed(self):
        n_t = self.dic_grid.n_t
        d_t = (1 / n_t)
        self.dic_grid.end_t = int((n_t - 1) * (self.t + d_t / 2))

    ipw_view = bu.View(
        bu.Item('ball_size'),
        time_editor=bu.HistoryEditor(
            var='t'
        )
    )

    #grid_number_vertical = bu.Bool(True)

    fe_grid = tr.Property(bu.Instance(ib.XDomainFEGrid), depends_on='state_changed')
    @tr.cached_property
    def _get_fe_grid(self):
        n_x, n_y = self.dic_grid.n_x, self.dic_grid.n_y
        L_x, L_y = self.dic_grid.L_x, self.dic_grid.L_y

        if self.dic_grid.grid_number_vertical:
            grid = ib.XDomainFEGrid(coord_min=(L_x, L_y),
                             coord_max=(0, 0),
                             integ_factor=1,
                             shape=(n_x - 1, n_y - 1),  # number of elements!
                             fets=ib.FETS2D4Q());
        else:
            grid = ib.XDomainFEGrid(coord_min=(L_x, 0),
                                    coord_max=(0, L_y),
                                    integ_factor=1,
                                    shape=(n_x - 1, n_y - 1),  # number of elements!
                                    fets=ib.FETS2D4Q());

        return grid

    #coord_min = (L_x, 0),
    #coord_max = (0, L_y),
    U_o = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_U_o(self):
        end_t = self.dic_grid.end_t
        u_ija = self.dic_grid.U_tija[end_t]
        U_Ia = u_ija.reshape(-1, 2)
        U_o = U_Ia.flatten()  # array of displacements corresponding to the DOF enumeration
        return U_o

    u_Ea = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_u_Ea(self):
        U_o = self.U_o
        U_Eia = U_o[self.fe_grid.o_Eia]
        u_Ema = np.einsum('im,Eia->Ema', self.fe_grid.fets.N_im, U_Eia)
        u_Ea = np.average(u_Ema, axis=1)
        return u_Ea

    pos_max_eps_E = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_pos_max_eps_E(self):
        # strain evaluated using bilinear shape functions in the gauss points
        eps_Emab = self.fe_grid.map_U_to_field(self.U_o)
        eps_abp = np.einsum('Emab->abEm', eps_Emab).reshape(2, 2, -1)
        # principal strain values and directions
        eps_Emc, eps_Emcd = np.linalg.eig(eps_Emab)
        # average strain directions within an alement
        eps_Ec = np.average(eps_Emc, axis=1)
        # maximum positive principal strain
        max_eps_E = np.max(eps_Ec, axis=-1)
        pos_max_eps_E = ((max_eps_E + np.fabs(max_eps_E)) / 2)
        # positions of element midpoints
        return pos_max_eps_E

    def plot_eps_a(self, ax, x_aE):
        # plot
        pos_max_eps_E = self.pos_max_eps_E
        ax.scatter(*x_aE, s=self.ball_size * pos_max_eps_E, color='red')
        # x_aE_ = np.round(x_aE,1)
        # import joblib
        #
        # xVal = x_aE_[0,:] #size of 812
        # yVal = x_aE_[1,:] #size of 812
        # fracS = self.ball_size * pos_max_eps_E #size of 812
        # data = [xVal, yVal, fracS]
        # file = 'D:\Shear zones\drawings\data.pkl.lz4'
        # joblib.dump(data, file)

        #ax.axis('equal')

    def subplots(self, fig):
        return fig.subplots(1,2)

    def update_plot(self, axes):
        ax_u, _ = axes
        u_Ea = self.u_Ea
        x_Ea = np.average(self.fe_grid.x_Ema, axis=1)
        U_factor = self.dic_grid.U_factor
        x_aE = np.einsum('Ea->aE', x_Ea + u_Ea * U_factor)
        self.plot_eps_a(ax_u, x_aE)