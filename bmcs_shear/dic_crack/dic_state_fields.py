'''
@author: rch
'''

import bmcs_utils.api as bu
from .dic_grid import DICGrid

import traits.api as tr

from ibvpy.sim.tstep import \
    TStep

from matplotlib import cm

import ibvpy.api as ib
from mayavi import mlab
from tvtk.api import tvtk
import numpy as np
import copy
from scipy.interpolate import interp2d
from scipy.signal import argrelextrema

class DICStateFields(ib.TStepBC):
    '''State analysis of the field simulated by DIC.
    '''

    dic_grid = bu.Instance(DICGrid)

    tmodel = bu.EitherType(options=[('miproplane_mdm', ib.MATS2DMplDamageEEQ),
                                    ('scalar_damage', ib.MATS2DScalarDamage)])

    # tmodel = tr.Property(depends_on='state_changed')
    # '''Material model
    # '''
    # @tr.cached_property
    # def _get_tmodel(self):
    #     m_mic = ib.MATS2DMplDamageEEQ()
    #     return m_mic
    #     m_scalar_damage = ib.MATS2DScalarDamage()
    #     m_scalar_damage.strain_norm = 'Masars'
    #     m_scalar_damage.omega_fn = 'exp-slope'
    #     return m_scalar_damage

    xmodel = tr.Property(depends_on='dic_grid')
    '''Finite element discretization of the monotored grid field
    '''
    @tr.cached_property
    def _get_xmodel(self):
        n_x, n_y = self.dic_grid.n_x, self.dic_grid.n_y
        L_x, L_y = self.dic_grid.L_x, self.dic_grid.L_y
        return ib.XDomainFEGrid(coord_min=(L_x, 0), coord_max=(0, L_y),
                               integ_factor=1, shape=(n_x - 1, n_y - 1),  # number of elements!
                               fets=ib.FETS2D4Q());

    domains = tr.Property(depends_on='state_changed')
    '''Finite element discretization of the monotored grid field
    '''
    @tr.cached_property
    def _get_domains(self):
        return [(self.xmodel, self.tmodel_)]


    tree = ['dic_grid', 'tmodel', 'xmodel']

    def eval(self):
        self.hist.init_state()
        self.fe_domain[0].state_k = copy.deepcopy(self.fe_domain[0].state_n)
        for n in range(0, self.dic_grid.n_t):
            self.t_n1 = n
            U_ija = self.dic_grid.U_tija[n]
            U_Ia = U_ija.reshape(-1, 2)
            U_o = U_Ia.flatten()  # array of displacements corresponding to the DOF enumeration
            eps_Emab = self.xmodel.map_U_to_field(U_o)
            self.tmodel_.get_corr_pred(eps_Emab, 1, **self.fe_domain[0].state_k)
            self.U_k[:] = U_o[:]
            self.U_n[:] = self.U_k[:]
#            self.fe_domain[0].record_state()
            states = [self.fe_domain[0].state_k]
            self.hist.record_timestep(self.t_n1, self.U_k, self.F_k, states)
            self.t_n = self.t_n1

    t = bu.Float(1, ALG=True)
    def _t_changed(self):
        n_t = self.dic_grid.n_t
        d_t = (1 / n_t)
        self.dic_grid.end_t = int((n_t - 1) * (self.t + d_t / 2))
        self.t_idx = self.dic_grid.end_t

    t_idx = bu.Int(1)

    R = bu.Float(8, ALG=True)
    '''Averaging radius'''

    n_x = bu.Int(116, ALG=True)
    '''Number of interpolation points in x direction'''

    n_y = bu.Int(28, ALG=True)
    '''Number of interpolation points in y direction'''

    ipw_view = bu.View(
        bu.Item('R'),
        bu.Item('n_x'),
        bu.Item('n_y'),
        bu.Item('t_idx', read_only=True),
        time_editor=bu.HistoryEditor( var='t' )
    )

    n_EF = tr.Property
    def _get_n_EF(self):
        return self.dic_grid.n_x - 1, self.dic_grid.n_y - 1

    def transform_mesh_to_grid(self, field_Em):
        '''Map the field from a mesh to a regular grid
        '''
        n_E, n_F = self.n_EF
        field_Em_shape = field_Em.shape
        # reshape into EFmn and preserve the dimensionality of the input field
        field_EFmn_shape = (n_E, n_F, 2, 2) + field_Em_shape[2:]
        # reorder the Gauss points to comply with the grid point order
        # this reordering might be parameterized by the finite-element formulation
        field_EFmn = field_Em[:, (0, 3, 1, 2)].reshape(*field_EFmn_shape)
        # swap the dimensions of elements and gauss points
        field_EmFn = np.einsum('EFmn...->EmFn...', field_EFmn)
        # merge the element index and gauss point subgrid into globarl point indexes
        field_MN_shape = (2 * n_E, 2 * n_F) + field_Em_shape[2:]
        # reshape the field
        field_MN = field_EmFn.reshape(*field_MN_shape)
        return field_MN

    def get_z_MN_ironed(self, x_JK, y_JK, z_JK):
        RR = self.R
        n_J, n_K = x_JK.shape
        delta_x_JK = x_JK[None, None, ...] - x_JK[..., None, None]
        delta_y_JK = y_JK[None, None, ...] - y_JK[..., None, None]
        r2_n = (delta_x_JK ** 2 + delta_y_JK ** 2) / (2 * RR ** 2)
        alpha_r_MNJK = np.exp(-r2_n)
        a_MN = np.trapz(np.trapz(alpha_r_MNJK, x_JK[:, 0], axis=-2), y_JK[0, :], axis=-1)
        normed_a_MNJK = np.einsum('MNJK,MN->MNJK', alpha_r_MNJK, 1 / a_MN)
        z_MNJK = np.einsum('MNJK,JK...->MNJK...', normed_a_MNJK, z_JK)
        # note that the inner integral cancels the dimension J on the axis with
        # index 2. Therefore, the outer integral integrates over K - again on
        # the axis with index 2
        z_MN = np.trapz(np.trapz(z_MNJK, x_JK[:, 0], axis=2), y_JK[0, :], axis=2)
        return z_MN

    def interp_omega(self, x_MN, y_MN, omega_MN):
        n_x, n_y = self.n_x, self.n_y
        x_M, x_N = x_MN[:, 0], y_MN[0, :]
        f_omega = interp2d(x_M, x_N, omega_MN.T, kind='cubic')
        xx_M = np.linspace(x_M[-1], x_M[0], n_x)
        yy_N = np.linspace(x_N[0], x_N[-1], n_y)
        xx_NM, yy_NM = np.meshgrid(xx_M, yy_N)
        omega_ipl_NM = f_omega(xx_M, yy_N)
        omega_ipl_NM[omega_ipl_NM < 0] = 0
        return xx_NM, yy_NM, omega_ipl_NM

    def crack_detection(self, xx_MN, yy_MN, omega_MN):
        N_range = np.arange(yy_MN.shape[1])
        omega_NM = omega_MN.T
        n_N, n_M = omega_NM.shape
        omega_NM[omega_NM < 0.15] = 0  # cutoff small damage values
        # smooth the landscape
        # initial crack positions at the bottom of the zone
        arg_C = argrelextrema(omega_NM[0, :], np.greater)[0]
        if len(arg_C) == 0:
            return np.zeros((n_N, 0)), np.zeros((n_N, 0))
        # list of intervals decomposing the crack
        intervals_Cp_ = []
        # distance between right interval boundary and crack position
        arg_C_shift_ = []
        # list of crack horizontal indexes for each horizontal slice
        arg_x_NC_ = [np.copy(arg_C)]
        # crack tip N
        crack_tip_y = np.zeros_like(arg_C, dtype=np.int_)
        C_fixed = np.zeros_like(arg_C, dtype=np.bool_)
        for N1 in N_range[1:]:
            # horizontal indexes of midpoints between cracks
            #             arg_C_left_ = np.hstack([[int(arg_C[0] / 2)],
            #                                      np.array((0.25 * arg_C[:-1] + 0.75 * arg_C[1:]),
            #                                               dtype=np.int_)])
            arg_C_left_ = arg_C - 5
            arg_C_left_[arg_C_left_ < 0] = 0
            arg_C_right_ = arg_C + 3
            # array of intervals - first index - crack, second index (left, right)
            intervals_Cp = np.vstack([arg_C_left_, arg_C_right_]).T
            # index distance from the right boundary of the crack interval
            arg_C_shift = np.array([
                np.argmax(omega_NM[N1, interval_p[-1]:interval_p[0]:-1])
                for interval_p in intervals_Cp
            ])
            # cracks, for which the next point could be identified
            C_shift = ((arg_C_shift > 0) & np.logical_not(C_fixed))
            C_fixed = np.logical_not(C_shift)
            # crack tips
            crack_tip_y[C_shift] = N1
            # next index position of the crack
            arg_C[C_shift] = intervals_Cp[C_shift, -1] - arg_C_shift[C_shift]
            arg_x_NC_.append(np.copy(arg_C))
            # for debugging
            intervals_Cp_.append(intervals_Cp)
            arg_C_shift_.append(arg_C_shift)
        arg_x_NC = np.array(arg_x_NC_)
        n_C = arg_x_NC.shape[1]
        arg_y_C = np.arange(n_N)
        arg_y_NC = np.repeat(arg_y_C, n_C).reshape(n_N, -1)
        xx_NC = xx_MN[arg_x_NC, arg_y_NC]
        yy_NC = yy_MN[arg_x_NC, arg_y_NC]
        return xx_NC, yy_NC, crack_tip_y

    def mlab_tensor(self, x_NM, y_NM, omega_NM, tensor_MNab, factor=100, label='damage'):
        mlab.figure()
        scene = mlab.get_engine().scenes[-1]
        scene.name = label
        scene.scene.background = (1.0, 1.0, 1.0)
        scene.scene.foreground = (0.0, 0.0, 0.0)
        scene.scene.z_plus_view()
        scene.scene.parallel_projection = True
        pts_shape = x_NM.shape + (1,)
        pts = np.empty(pts_shape + (3,), dtype=float)
        pts[..., 0] = x_NM[..., np.newaxis]
        pts[..., 1] = y_NM[..., np.newaxis]
        pts[..., 2] = omega_NM[..., np.newaxis] * factor
        pts = pts.transpose(2, 1, 0, 3).copy()
        pts.shape = int(pts.size / 3), 3
        sg = tvtk.StructuredGrid(dimensions=pts_shape, points=pts)
        sg.point_data.scalars = omega_NM.ravel()
        sg.point_data.scalars.name = 'damage'
        delta_23 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float_)
        tensor_MNab_3D = np.einsum('...ab,ac,bd->...cd', tensor_MNab, delta_23, delta_23)
        sg.point_data.tensors = tensor_MNab_3D.reshape(-1, 9)
        sg.point_data.tensors.name = label
        # Now visualize the data.
        d = mlab.pipeline.add_dataset(sg)
        mlab.pipeline.iso_surface(d)
        mlab.pipeline.surface(d)
        mlab.show()

    def mlab_scalar(self, x_NM, y_NM, z_NM, factor=100, label='damage'):
        mlab.figure()
        scene = mlab.get_engine().scenes[-1]
        scene.name = label
        scene.scene.background = (1.0, 1.0, 1.0)
        scene.scene.foreground = (0.0, 0.0, 0.0)
        scene.scene.z_plus_view()
        scene.scene.parallel_projection = True
        pts_shape = x_NM.shape + (1,)
        pts = np.empty(pts_shape + (3,), dtype=float)
        pts[..., 0] = x_NM[..., np.newaxis]
        pts[..., 1] = y_NM[..., np.newaxis]
        pts[..., 2] = z_NM[..., np.newaxis] * factor
        pts = pts.transpose(2, 1, 0, 3).copy()
        pts.shape = int(pts.size / 3), 3
        sg = tvtk.StructuredGrid(dimensions=pts_shape, points=pts)
        sg.point_data.scalars = z_NM.T.ravel()
        sg.point_data.scalars.name = label
        d = mlab.pipeline.add_dataset(sg)
        mlab.pipeline.iso_surface(d)
        mlab.pipeline.surface(d)
        mlab.show()

    def subplots(self, fig):
        self.fig = fig
        return fig.subplots(3, 2)

    def update_plot(self, axes):
        ((ax_eps, ax_FU), (ax_sig, ax_sig_eps), (ax_omega, ax_cracks)) = axes
        fig = self.fig
        # spatial coordinates
        x_MNa = self.transform_mesh_to_grid(self.xmodel.x_Ema)
        x_aMN = np.einsum('MNa->aMN', x_MNa)
        x_MN, y_MN = x_aMN
        # evaluate the state variables
        t_idx = self.t_idx
        # state variables
        kappa_Emr = self.hist.state_vars[t_idx][0]['kappa']
        omega_Emr = self.hist.state_vars[t_idx][0]['omega']
        phi_Emab = self.tmodel_._get_phi_Emab(kappa_Emr)
        # plot parameters - ge them from the state evaluation
        max_sig = 5
        max_eps = 0.02
        # strain fields
        U_o = self.hist.U_t[t_idx]
        eps_Emab = self.xmodel.map_U_to_field(U_o)
        eps_MNab = self.transform_mesh_to_grid(eps_Emab)
        eps_MNa, _ = np.linalg.eig(eps_MNab)
        max_eps_MN = np.max(eps_MNa, axis=-1)
        max_eps_MN[max_eps_MN < 0] = 0
        # stress fields
        sig_Emab, _ = self.tmodel_.get_corr_pred(eps_Emab, 1, kappa_Emr, omega_Emr)
        sig_MNab = self.transform_mesh_to_grid(sig_Emab)
        sig_MNa, _ = np.linalg.eig(sig_MNab)
        max_sig_MN = np.max(sig_MNa, axis=-1)
        max_sig_MN[max_sig_MN < 0] = 0
        # damage tensor field
        phi_MNab = self.transform_mesh_to_grid(phi_Emab)
        omega_MNab = np.identity(2) - phi_MNab
        phi_MNa, _ = np.linalg.eig(phi_MNab)
        min_phi_MN = np.min(phi_MNa, axis=-1)
        omega_MN = 1 - min_phi_MN
        omega_MN[omega_MN < 0.2] = 0
        # cumulative stress strain analysis
        kappa_zero = np.zeros_like(kappa_Emr[0, 0, :])
        omega_zero = np.zeros_like(kappa_zero)
        eps_test = np.zeros((2, 2), dtype=np.float_)
        eps_range = np.linspace(0, 0.5, 1000)
        sig_range = []
        for eps_i in eps_range:
            eps_test[0, 0] = eps_i
            arg_sig, _ = self.tmodel_.get_corr_pred(eps_test, 1, kappa_zero, omega_zero)
            sig_range.append(arg_sig)
        # max_eps = np.max(max_eps_MN)
        arg_max_eps = np.argwhere(eps_range > max_eps)[0][0]
        sig_range = np.array(sig_range, dtype=np.float_)
        G_f = np.trapz(sig_range[:, 0, 0], eps_range)
        # crack detection
        xx_NM, yy_NM, omega_ipl_NM = self.interp_omega(x_MN, y_MN, omega_MN)
        xx_MN, yy_MN, omega_ipl_MN = xx_NM.T, yy_NM.T, omega_ipl_NM.T
        omega_irn_MN = self.get_z_MN_ironed(xx_MN, yy_MN, omega_ipl_MN)
        xx_NC, yy_NC, crack_tip_y = self.crack_detection(xx_MN, yy_MN, omega_irn_MN)
        # plot
        cs_eps = ax_eps.contourf(x_aMN[0], x_aMN[1], max_eps_MN, cmap='BuPu',
                                 vmin=0, vmax=max_eps)
        cbar_eps = fig.colorbar(cm.ScalarMappable(norm=cs_eps.norm, cmap=cs_eps.cmap),
                                ax=ax_eps, ticks=np.arange(0, max_eps * 1.01, 0.005),
                                orientation='horizontal')
        cbar_eps.set_label(r'$\max(\varepsilon_I) > 0$')
        ax_eps.axis('equal')
        ax_eps.axis('off')
        cs_sig = ax_sig.contourf(x_aMN[0], x_aMN[1], max_sig_MN, cmap='Reds',
                                 vmin=0, vmax=max_sig)
        cbar_sig = fig.colorbar(cm.ScalarMappable(norm=cs_sig.norm, cmap=cs_sig.cmap),
                                ax=ax_sig, ticks=np.arange(0, max_sig * 1.01, 0.5),
                                orientation='horizontal')
        cbar_sig.set_label(r'$\max(\sigma_I) > 0$')
        ax_sig.axis('equal')
        ax_sig.axis('off')
        cs = ax_omega.contourf(x_aMN[0], x_aMN[1], omega_MN, cmap='BuPu', vmin=0, vmax=1)
        cbar_omega = fig.colorbar(cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap),
                                  ax=ax_omega, ticks=np.arange(0, 1.1, 0.2),
                                  orientation='horizontal')
        cbar_omega.set_label(r'$\omega = 1 - \min(\phi_I)$')
        ax_omega.axis('equal');
        ax_omega.axis('off')

        self.dic_grid.plot_load_deflection(ax_FU)

        ax_sig_eps.plot(eps_MNa[..., 0].flatten(), sig_MNa[..., 0].flatten(), 'o', color='green')
        ax_sig_eps.plot(eps_range[:arg_max_eps], sig_range[:arg_max_eps, 0, 0],
                        color='white', lw=2, label='$G_f$ = %g [N/mm]' % G_f)
        ax_sig_eps.set_xlabel(r'$\varepsilon$ [-]')
        ax_sig_eps.set_ylabel(r'$\sigma$ [MPa]')
        ax_sig_eps.legend()

        # check if an array can be used in a slice to obtain variable length arrays?
        for C, y_tip in enumerate(crack_tip_y):
            ax_cracks.plot(xx_NC[:y_tip, C], yy_NC[:y_tip, C], color='black', linewidth=1);

        cs = ax_cracks.contour(xx_MN, yy_MN, omega_irn_MN, cmap=cm.coolwarm, antialiased=False)
        cbar_cracks = fig.colorbar(cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap),
                                   ax=ax_cracks, ticks=np.linspace(0, 1, 6),
                                   orientation='horizontal')
        cbar_cracks.set_label(r'$\omega = 1 - \min(\phi_I)$')
        ax_cracks.axis('equal')
        ax_cracks.axis('off');