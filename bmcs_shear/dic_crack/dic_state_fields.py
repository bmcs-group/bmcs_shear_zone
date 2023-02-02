'''
@author: rch
'''

import bmcs_utils.api as bu
from .dic_grid import DICGrid
import traits.api as tr
from matplotlib import cm
import ibvpy.api as ib
import numpy as np
import copy
from scipy.interpolate import interp2d, LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from .cached_array import cached_array


class DICStateFields(ib.TStepBC):
    name = 'state fields'

    '''State analysis of the field simulated by DIC.
    '''

    dic_grid = bu.Instance(DICGrid)

    bd = tr.DelegatesTo('dic_grid', 'bd')

    tmodel = bu.EitherType(options=[('miproplane_mdm', ib.MATS2DMplDamageEEQ),
                                    ('scalar_damage', ib.MATS2DScalarDamage)])

    depends_on = ['dic_grid', 'tmodel']
    tree = ['dic_grid', 'tmodel', 'xmodel']

    xmodel = tr.Property(bu.Instance(ib.XDomainFEGrid), depends_on='dic_grid')
    '''Finite element discretization of the monotored grid field
    '''
    @tr.cached_property
    def _get_xmodel(self):
        n_E, n_F = self.n_EF
        X_min, Y_min, X_max, Y_max = self.dic_grid.X_frame
        return ib.XDomainFEGrid(coord_min=(X_min, Y_min), coord_max=(X_max, Y_max),
                                integ_factor=1, shape=(n_E, n_F),  # number of elements!
                                fets=ib.FETS2D4Q());

    domains = tr.Property(depends_on='state_changed')
    '''Finite element discretization of the monitored grid field
    '''

    @tr.cached_property
    def _get_domains(self):
        return [(self.xmodel, self.tmodel_)]

    data_dir = tr.DelegatesTo('dic_grid')
    beam_param_file = tr.DelegatesTo('dic_grid')

    kappa_omega_sig_U = tr.Property(depends_on='state_changed')
    @cached_array(source_name="beam_param_file",
                  names=['kappa_TEmr', 'omega_TEmr', 'eps_TEmab', 'sig_TEmab', 'U_To'],
                  data_dir_trait='data_dir')
    def _get_kappa_omega_sig_U(self):
        '''Run the FE analysis for the dic load levels
        '''
        # self.hist.init_state()
        kappa_TEmr = []
        omega_TEmr = []
        eps_TEmab = []
        sig_TEmab = []
        U_To = []
        self.t_n = 0
        self.fe_domain[0].state_k = copy.deepcopy(self.fe_domain[0].state_n)
        for T in range(0, self.dic_grid.n_T):
            if self.verbose_eval:
                print('T:', T)
            self.t_n1 = T
            U_IJa = self.dic_grid.U_TIJa[T]
            U_Ia = U_IJa.reshape(-1, 2)
            U_o = U_Ia.flatten()  # array of displacements corresponding to the DOF enumeration
            eps_Emab = self.xmodel.map_U_to_field(U_o)
            sig_Emab, _ = self.tmodel_.get_corr_pred(eps_Emab, 1, **self.fe_domain[0].state_k)
            self.U_k[:] = U_o[:]
            self.U_n[:] = self.U_k[:]
            kappa_TEmr.append(copy.deepcopy(self.fe_domain[0].state_k['kappa']))
            omega_TEmr.append(copy.deepcopy(self.fe_domain[0].state_k['omega']))
            eps_TEmab.append(eps_Emab)
            sig_TEmab.append(sig_Emab)
            U_To.append(np.copy(U_o))
            # domain_states = [self.fe_domain[0].state_k]
            # self.hist.record_timestep(self.t_n1, self.U_k, self.F_k, domain_states)
            self.t_n = self.t_n1

        return (np.array(kappa_TEmr),
                np.array(omega_TEmr),
                np.array(eps_TEmab),
                np.array(sig_TEmab),
                np.array(U_To))

    kappa_TEmr = tr.Property
    def _get_kappa_TEmr(self):
        return self.kappa_omega_sig_U[0]

    omega_TEmr = tr.Property
    def _get_omega_TEmr(self):
        return self.kappa_omega_sig_U[1]

    eps_TEmab = tr.Property
    def _get_eps_TEmab(self):
        return self.kappa_omega_sig_U[2]

    sig_TEmab = tr.Property
    def _get_sig_TEmab(self):
        return self.kappa_omega_sig_U[3]

    U_To = tr.Property
    def _get_U_To(self):
        return self.kappa_omega_sig_U[4]

    verbose_eval = bu.Bool(False)
    '''Report time step in simulation'''

    R = bu.Float(8, ALG=True)
    '''Averaging radius'''

    n_ipl_M = bu.Int(116, ALG=True)
    #n_ipl_M = tr.DelegatesTo('dic_grid', 'n_I') # .Int(116, ALG=True)
    '''Number of interpolation points in x direction'''

    n_ipl_N = bu.Int(28, ALG=True)
    #n_ipl_N = tr.DelegatesTo('dic_grid', 'n_J')
    '''Number of interpolation points in y direction'''

    T_t = tr.Property(bu.Int, depends_on='state_changed')
    @tr.cached_property
    def _get_T_t(self):
        return self.dic_grid.T_t

    omega_t_on = bu.Bool(True, ALG=True)

    ipw_view = bu.View(
        bu.Item('verbose_eval'),
        bu.Item('omega_t_on'),
        bu.Item('R'),
        bu.Item('omega_threshold'),
        bu.Item('n_ipl_M'),
        bu.Item('n_ipl_N'),
        bu.Item('T_t', readonly=True),
        time_editor=bu.HistoryEditor(var='dic_grid.t')
    )

    n_EF = tr.Property

    def _get_n_EF(self):
        return self.dic_grid.n_I - 1, self.dic_grid.n_J - 1

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
        print('ironing with', RR)
        delta_x_JK = x_JK[None, None, ...] - x_JK[..., None, None]
        delta_y_JK = y_JK[None, None, ...] - y_JK[..., None, None]
        r2_n = (delta_x_JK ** 2 + delta_y_JK ** 2) / (2 * RR ** 2)
        alpha_r_MNJK = np.exp(-r2_n)
        a_MN = np.trapz(np.trapz(alpha_r_MNJK, x_JK[:, 0], axis=-2), y_JK[0, :], axis=-1)
        normed_a_MNJK = np.einsum('MNKL,MN->MNKL', alpha_r_MNJK, 1 / a_MN)
        z_MNJK = np.einsum('MNKL,KL...->MNKL...', normed_a_MNJK, z_JK)
        # note that the inner integral cancels the dimension J on the axis with
        # index 2. Therefore, the outer integral integrates over K - again on
        # the axis with index 2
        z_MN = np.trapz(np.trapz(z_MNJK, x_JK[:, 0], axis=2), y_JK[0, :], axis=2)
        return z_MN

    ####################################################################################

    f_dic_U_xy = tr.Property(depends_on='state_changed')
    '''Construct an interpolator over the domain
    '''
    @tr.cached_property
    def _get_f_dic_U_xy(self):
        xy = self.dic_grid.X_IJa.reshape(-1, 2)
        u = self.dic_grid.U_IJa.reshape(-1, 2)
        return LinearNDInterpolator(xy, u)

    X_ipl_MNa = tr.Property(depends_on='state_changed')
    '''Interpolation grid
    '''
    @tr.cached_property
    def _get_X_ipl_MNa(self):
        n_ipl_M, n_ipl_N = self.n_ipl_M, self.n_ipl_N
        x_MN, y_MN = np.einsum('MNa->aMN', self.X_fe_KLa)
        x_M, x_N = x_MN[:, 0], y_MN[0, :]
        xx_M = np.linspace(x_M[0], x_M[-1], n_ipl_M)
        yy_N = np.linspace(x_N[0], x_N[-1], n_ipl_N)
        xx_NM, yy_NM = np.meshgrid(xx_M, yy_N)
        X_aNM = np.array([xx_NM, yy_NM])
        X_ipl_KLa = np.einsum('aNM->MNa', X_aNM)
        return X_ipl_KLa

    X_ipl_bb_Ca = tr.Property(depends_on='state_changed')
    """Bounding box of the interpolated field"""
    @tr.cached_property
    def _get_X_ipl_bb_Ca(self):
        return self.X_ipl_MNa[(0,-1),(0,-1)]

    U_ipl_MNa = tr.Property(depends_on='state_changed')
    '''Displacement on the ipl_MN grid
    '''
    @tr.cached_property
    def _get_U_ipl_MNa(self):
        return self.f_dic_U_xy(self.X_ipl_MNa)

    X_fe_KLa = tr.Property(depends_on='state_changed')
    """Regular grid of the FE quadrature points
    """
    @tr.cached_property
    def _get_X_fe_KLa(self):
        return self.transform_mesh_to_grid(self.xmodel.x_Ema)

    #######################################################################

    f_U_ipl_txy = tr.Property(depends_on='state_changed')
    '''Interpolator of displacements over the time and spatial domains.
    This method is used to extract the displacements along the crack path
    '''
    @tr.cached_property
    def _get_f_U_ipl_txy(self):
        n_T = self.dic_grid.n_T
        dic_T = np.arange(n_T)
        X_IJa = self.dic_grid.X_IJa
        U_TIJa = self.dic_grid.U_TIJa[:n_T, ...]
        x_IJ, y_IJ = np.einsum('IJa->aIJ', X_IJa)
        dic_t = dic_T / (self.dic_grid.n_T - 1)
        txy = (dic_t, x_IJ[:, 0], y_IJ[0, :])
        return RegularGridInterpolator(txy, U_TIJa[:, :, :, :])

    #######################################################################

    eps_fe_fields = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_eps_fe_fields(self):
        # U_o = self.U_To[self.dic_grid.T_t]
        # eps_Emab = self.xmodel.map_U_to_field(U_o)
        eps_Emab = self.eps_TEmab[self.dic_grid.T_t]
        eps_KLab = self.transform_mesh_to_grid(eps_Emab)
        eps_KLa, _ = np.linalg.eig(eps_KLab)
        max_eps_KL = np.max(eps_KLa, axis=-1)
        max_eps_KL[max_eps_KL < 0] = 0
        return eps_Emab, eps_KLab, eps_KLa, max_eps_KL

    eps_fe_TKLab = tr.Property(depends_on='state_changed')
    """History of strains in the quadrature points
    """
    @cached_array(source_name="beam_param_file",
                  names='eps_KLab',
                  data_dir_trait='data_dir')
    def _get_eps_fe_TKLab(self):
        # state variables
        eps_KLab_list = []
        for T in range(self.dic_grid.n_T):
            # U_o = self.U_To[T]
            # eps_Emab = self.xmodel.map_U_to_field(U_o)
            eps_Emab = self.eps_TEmab[self.dic_grid.T_t]
            eps_KLab = self.transform_mesh_to_grid(eps_Emab)
            eps_KLab_list.append(np.copy(eps_KLab))
        return np.array(eps_KLab_list, dtype=np.float_)

    f_eps_fe_txy = tr.Property(depends_on='state_changed')
    """Time-space interpolator for strains
    """
    @tr.cached_property
    def _get_f_eps_fe_txy(self):
        dic_T = np.arange(self.dic_grid.n_T)
        x_fe_KL, y_fe_KL = np.einsum('KLa->aKL', self.X_fe_KLa)
        dic_t = dic_T / (self.dic_grid.n_T - 1)
        args = (dic_t, x_fe_KL[:, 0], y_fe_KL[0, :])
        return RegularGridInterpolator(args, self.eps_fe_TKLab[:, :, :])

    kappa_fe_TKLab = tr.Property(depends_on='state_changed')
    """History of strains in the quadrature points
    """
    @cached_array(source_name="beam_param_file",
                  names='kappa_KLab',
                  data_dir_trait='data_dir')
    def _get_kappa_fe_TKLab(self):
        # state variables
        kappa_KLab_list = []
        for T in range(self.dic_grid.n_T):
            # U_o = self.U_To[T]
            # eps_Emab = self.xmodel.map_U_to_field(U_o)
            kappa_Emab = self.kappa_TEmr[self.dic_grid.T_t]
            kappa_KLab = self.transform_mesh_to_grid(kappa_Emab)
            kappa_KLab_list.append(np.copy(kappa_KLab))
        return np.array(kappa_KLab_list, dtype=np.float_)

    f_kappa_fe_txy = tr.Property(depends_on='state_changed')
    """Time-space interpolator for strains
    """
    @tr.cached_property
    def _get_f_kappa_fe_txy(self):
        dic_T = np.arange(self.dic_grid.n_T)
        x_fe_KL, y_fe_KL = np.einsum('KLa->aKL', self.X_fe_KLa)
        dic_t = dic_T / (self.dic_grid.n_T - 1)
        args = (dic_t, x_fe_KL[:, 0], y_fe_KL[0, :])
        return RegularGridInterpolator(args, self.kappa_fe_TKLab[:, :, :])


    sig_fe_fields = tr.Property(depends_on='state_changed')
    """Stress fields on fe_KL grid
    """
    @tr.cached_property
    def _get_sig_fe_fields(self):
        # kappa_TEmr = self.kappa_TEmr[self.dic_grid.T_t]
        # omega_TEmr = self.omega_TEmr[self.dic_grid.T_t]
        # eps_Emab, _, _, _ = self.eps_fe_fields
        # sig_Emab, _ = self.tmodel_.get_corr_pred(eps_Emab, 1, kappa=kappa_TEmr, omega=omega_TEmr)
        sig_Emab = self.sig_TEmab[self.dic_grid.T_t]
        sig_KLab = self.transform_mesh_to_grid(sig_Emab)
        sig_KLa, _ = np.linalg.eig(sig_KLab)
        max_sig_KL = np.max(sig_KLa, axis=-1)
        max_sig_KL[max_sig_KL < 0] = 0
        return sig_Emab, sig_KLab, sig_KLa, max_sig_KL

    #######################################################################
    omega_threshold = bu.Float(0.2, ALG=True)

    omega_fe_TKL = tr.Property(depends_on='state_changed')
    """Maximum damage value in each material point of the fe_KL grid
    """
    @cached_array(source_name="beam_param_file",
                  names='omega_fe_TKL',
                  data_dir_trait='data_dir')
    def _get_omega_fe_TKL(self):
        # state variables
        omega_fe_KL_list = []
        x_fe_KL, y_fe_KL = np.einsum('KLa->aKL', self.X_fe_KLa)
        for T in range(self.dic_grid.n_T):
            print('omega_fe_TKL', 2)
            kappa_Emr = self.kappa_TEmr[T] # self.hist.state_vars[T][0]['kappa']
            omega_Emr = self.tmodel_.omega_fn_(kappa_Emr)
            omega_fe_KL = self.transform_mesh_to_grid(omega_Emr)

            # phi_Emab = self.tmodel_._get_phi_Emab(kappa_Emr)
            # phi_KLab = self.transform_mesh_to_grid(phi_Emab)
            # phi_KLa, phi_ev_KLab = np.linalg.eig(phi_KLab)
            # min_phi_KL = np.min(phi_KLa, axis=-1)
            # omega_fe_KL = 1 - min_phi_KL
            #omega_fe_KL = self.get_z_MN_ironed(x_fe_KL, y_fe_KL, omega_fe_KL)
            omega_fe_KL[omega_fe_KL < self.omega_threshold] = 0
            omega_fe_KL_list.append(np.copy(omega_fe_KL))
        return np.array(omega_fe_KL_list, dtype=np.float_)# , None # phi_ev_KLab

    f_omega_fe_txy = tr.Property(depends_on='state_changed')
    """Interpolator of maximum damage value in time-space domain"""
    @tr.cached_property
    def _get_f_omega_fe_txy(self):
        dic_T = np.arange(self.dic_grid.n_T)
        x_MN, y_MN = np.einsum('MNa->aMN', self.X_fe_KLa)
        dic_t = dic_T / (self.dic_grid.n_T - 1)
        args = (dic_t, x_MN[:, 0], y_MN[0, :])
        omega_fe_TKL = self.omega_fe_TKL
        return RegularGridInterpolator(args, omega_fe_TKL)

    # f_phi_fe_txy = tr.Property(depends_on='state_changed')
    # """Interpolator of maximum damage value in time-space domain"""
    # @tr.cached_property
    # def _get_f_phi_fe_txy(self):
    #     x_MN, y_MN = np.einsum('MNa->aMN', self.X_fe_KLa)
    #     args = (x_MN[:, 0], y_MN[0, :])
    #     _, phi_fe_TKL = self.omega_fe_TKL
    #     return RegularGridInterpolator(args, phi_fe_TKL)

    n_ipl_T = tr.Property
    def _get_n_ipl_T(self):
        return self.dic_grid.n_T

    mgrid_ipl_TMN = tr.Property(depends_on='state_changed')
    """Time-space grid for interpolated fields
    """
    @tr.cached_property
    def _get_mgrid_ipl_TMN(self):
        # state variables
        x_min, x_max, y_min, y_max = self.X_fe_KLa[(0, -1, 0, 0), (0, 0, 0, -1), (0, 0, 1, 1)]
        return np.mgrid[
                      0:1:complex(0,self.n_ipl_T),
                      x_min:x_max:complex(0, self.n_ipl_M),
                      y_min:y_max:complex(0, self.n_ipl_N)
                      ]

    omega_ipl_TMN = tr.Property(depends_on='+ALG')
    """Maximum principal damage field in interpolated time-domain
    """
    @cached_array(source_name="beam_param_file",
                  names='omega_ipl_TMN',
                  data_dir_trait='data_dir')
    def _get_omega_ipl_TMN(self):
        # state variables
        t_TMN, X_TMN, Y_TMN = self.mgrid_ipl_TMN
        txy = np.c_[t_TMN.flatten(), X_TMN.flatten(), Y_TMN.flatten()]
        omega_txy = self.f_omega_fe_txy(txy)
        return omega_txy.reshape(self.n_ipl_T, self.n_ipl_M, self.n_ipl_N)

    ###########################################################################
    # Crack detection related services
    #
    omega_fe_KL = tr.Property(depends_on='+ALG')
    """Maximum principal damage field in at the ultimate load 
    """
    @tr.cached_property
    def _get_omega_fe_KL(self):
        # state variables
        return self.omega_fe_TKL[0][-1]

    ###########################################################################
    # Crack detection related services
    #
    omega_ipl_MN = tr.Property(depends_on='+ALG')
    """Maximum principal damage field in at the ultimate load 
    """
    @tr.cached_property
    def _get_omega_ipl_MN(self):
        # state variables
        return self.omega_ipl_TMN[-1]

    omega_irn_t_MN = tr.Property(depends_on='state_changed')
    """Crack detection field on the ipl grid
    """
    @tr.cached_property
    def _get_omega_irn_t_MN(self):
        _, X_TMN, Y_TMN = self.mgrid_ipl_TMN
        T_t = self.dic_grid.T_t
        x_MN, y_MN = X_TMN[T_t,...], Y_TMN[T_t,...]
        return x_MN, y_MN, self.omega_irn_TMN[T_t]

    omega_irn_1_MN = tr.Property(depends_on='+ALG')
    """Crack detection field on the ipl grid
    """
    @tr.cached_property
    def _get_omega_irn_1_MN(self):
        _, X_TMN, Y_TMN = self.mgrid_ipl_TMN
        x_MN, y_MN = X_TMN[-1,...], Y_TMN[-1,...]
        return x_MN, y_MN, self.omega_irn_TMN[-1]

    omega_irn_TMN = tr.Property(depends_on='+ALG')
    """Maximum damage value in each material point of the ipl_MN grid in each step
    """
    @cached_array(source_name="beam_param_file",
                  names='omega_irn_TMN',
                  data_dir_trait='data_dir')
    def _get_omega_irn_TMN(self):
        # state variables
        omega_irn_MN_list = []
        _, X_TMN, Y_TMN = self.mgrid_ipl_TMN
        x_MN, y_MN = X_TMN[0, ...], Y_TMN[0, ...]
        for T in range(self.dic_grid.n_T):
            omega_ipl_MN = self.omega_ipl_TMN[T, ...]
            omega_irn_MN = self.get_z_MN_ironed(x_MN, y_MN, omega_ipl_MN)
            omega_irn_MN[omega_irn_MN < self.omega_threshold] = 0
            omega_irn_MN_list.append(np.copy(omega_irn_MN))
        return np.array(omega_irn_MN_list, dtype=np.float_)

    f_omega_irn_txy = tr.Property(depends_on='+ALG')
    """Interpolator of maximum damage value in time-space domain"""
    @tr.cached_property
    def _get_f_omega_irn_txy(self):
        dic_T = np.arange(self.dic_grid.n_T)
        _, X_TMN, Y_TMN = self.mgrid_ipl_TMN
        x_MN, y_MN = X_TMN[0,...], Y_TMN[0,...]
        dic_t = dic_T / (self.dic_grid.n_T - 1)
        args = (dic_t, x_MN[:, 0], y_MN[0, :])
        return RegularGridInterpolator(args, self.omega_irn_TMN)


    #######################################################################

    # plot parameters - get them from the state evaluation
    # max_sig = bu.Float(5)
    max_eps = bu.Float(0.02)

    def plot_sig_eps(self, ax_sig_eps, color='white'):
        # plot the stress strain curve
        state_var_shape = (1,) + self.tmodel_.state_var_shapes['kappa']
        kappa_zero = np.zeros(state_var_shape)
        omega_zero = np.zeros_like(kappa_zero)
        eps_test = np.zeros((1, 2, 2), dtype=np.float_)
        eps_range = np.linspace(0, 0.5, 1000)
        sig_range = []
        for eps_i in eps_range:
            eps_test[0, 0, 0] = eps_i
            arg_sig, _ = self.tmodel_.get_corr_pred(eps_test, 1, kappa_zero, omega_zero)
            sig_range.append(arg_sig)
        arg_max_eps = np.argwhere(eps_range > self.max_eps)[0][0]
        sig_range = np.array(sig_range, dtype=np.float_)
        G_f = np.trapz(sig_range[:, 0, 0, 0], eps_range)

        ax_sig_eps.plot(eps_range[:arg_max_eps], sig_range[:arg_max_eps, 0, 0, 0],
                        color=color, lw=2, label='$G_f$ = %g [N/mm]' % G_f)
        ax_sig_eps.set_xlabel(r'$\varepsilon$ [-]')
        ax_sig_eps.set_ylabel(r'$\sigma$ [MPa]')

    def plot_eps_field(self, ax_eps, fig):
        eps_fe_Emab, eps_fe_KLab, eps_fe_KLa, max_eps_fe_KL = self.eps_fe_fields
        X_fe_KLa = self.X_fe_KLa
        X_fe_aKL = np.einsum('MNa->aMN', X_fe_KLa)
        max_fe_eps = np.max(max_eps_fe_KL)
        cs_eps = ax_eps.contourf(X_fe_aKL[0], X_fe_aKL[1], max_eps_fe_KL, cmap='BuPu',
                                 vmin=0, vmax=max_fe_eps)
        cbar_eps = fig.colorbar(cm.ScalarMappable(norm=cs_eps.norm, cmap=cs_eps.cmap),
                                ax=ax_eps, ticks=np.arange(0, max_fe_eps * 1.01, 0.005),
                                orientation='horizontal')
        cbar_eps.set_label(r'$\max(\varepsilon_I) > 0$')
        ax_eps.axis('equal')
        ax_eps.axis('off')

    def plot_sig_field(self, ax_sig, fig):
        sig_fe_Emab, sig_fe_KLab, sig_fe_KLa, max_sig_fe_KL = self.sig_fe_fields
        X_fe_KLa = self.X_fe_KLa
        X_fe_aKL = np.einsum('MNa->aMN', X_fe_KLa)
        max_fe_sig = np.max(max_sig_fe_KL)
        cs_sig = ax_sig.contourf(X_fe_aKL[0], X_fe_aKL[1], max_sig_fe_KL, cmap='Reds',
                                 vmin=0, vmax=max_fe_sig)
        cbar_sig = fig.colorbar(cm.ScalarMappable(norm=cs_sig.norm, cmap=cs_sig.cmap),
                                ax=ax_sig, ticks=np.arange(0, max_fe_sig * 1.01, 0.5),
                                orientation='horizontal')
        cbar_sig.set_label(r'$\max(\sigma_I) > 0$')
        ax_sig.axis('equal')
        ax_sig.axis('off')

    def subplots(self, fig):
        self.fig = fig
        return fig.subplots(3, 2)

    show_color_bar = bu.Bool(False, ALG=True)
    def plot_crack_detection_field(self, ax_cracks, fig):
        if self.omega_t_on:
            xx_MN, yy_MN, cd_field_irn_MN = self.omega_irn_t_MN
        else:
            xx_MN, yy_MN, cd_field_irn_MN = self.omega_irn_1_MN
        if np.sum(cd_field_irn_MN) == 0:
            # return without warning if there is no damage or strain
            return
        contour_levels = np.array([0.15, 0.35, 0.65, 0.85, 1.05], dtype=np.float_)
        cs = ax_cracks.contourf(xx_MN, yy_MN, cd_field_irn_MN, contour_levels,
                                cmap=cm.GnBu,
                               #cmap=cm.coolwarm,
                               antialiased=False)
        if self.show_color_bar:
            cbar_cracks = fig.colorbar(cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap),
                                       ax=ax_cracks, ticks=np.linspace(0, 1, 6),
                                       orientation='horizontal')
            cbar_cracks.set_label(r'$\omega = 1 - \min(\phi_I)$')

    def update_plot(self, axes):
        print('update plot')
        ((ax_eps, ax_FU), (ax_sig, ax_sig_eps), (ax_omega, ax_cracks)) = axes
        fig = self.fig
        # spatial coordinates
        print('update plot 2')
        X_fe_KLa = self.X_fe_KLa
#        X_fe_aKL = np.einsum('MNa->aMN', X_fe_KLa)
        # strain fields
        print('update plot 3')
        eps_Emab, eps_MNab, eps_MNa, max_eps_MN = self.eps_fe_fields
        # stress fields
        print('update plot 4')
        sig_Emab, sig_MNab, sig_MNa, max_sig_MN = self.sig_fe_fields
        # damage field
        omega_fe_KL = self.omega_fe_KL
        # plot
        print('update plot 5')
        self.plot_eps_field(ax_eps, fig)

        print('update plot 6')
        self.plot_sig_field(ax_sig, fig)

        print('update plot 7')
        self.plot_crack_detection_field(ax_omega, fig)
        ax_omega.axis('equal');
        ax_omega.axis('off')

        self.dic_grid.plot_load_deflection(ax_FU)

        ax_sig_eps.plot(eps_MNa[..., 0].flatten(), sig_MNa[..., 0].flatten(), 'o', color='green')
        self.plot_sig_eps(ax_sig_eps)
        ax_sig_eps.legend()

        self.dic_grid.plot_bounding_box(ax_cracks)
        self.dic_grid.plot_box_annotate(ax_cracks)

        ax_cracks.axis('equal')
        ax_cracks.axis('off');

    def mlab_tensor(self, x_NM, y_NM, tensor_MNab, factor=100, label='damage'):
        from mayavi import mlab
        from tvtk.api import tvtk
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
        # pts[..., 2] = omega_NM[..., np.newaxis] * factor
        tensor_MNa, _ = np.linalg.eig(tensor_MNab)
        max_tensor_MN = np.max(tensor_MNa, axis=-1)
        max_tensor_NM = max_tensor_MN.T
        max_tensor_NM[max_tensor_NM < 0] = 0
        pts[..., 2] = max_tensor_NM[..., np.newaxis] * factor
        pts = pts.transpose(2, 1, 0, 3).copy()
        pts.shape = int(pts.size / 3), 3
        sg = tvtk.StructuredGrid(dimensions=pts_shape, points=pts)
        sg.point_data.scalars = max_tensor_NM.ravel()
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
        from mayavi import mlab
        from tvtk.api import tvtk
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
