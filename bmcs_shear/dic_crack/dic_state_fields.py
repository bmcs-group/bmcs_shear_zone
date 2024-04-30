"""
@author: rch
"""

import bmcs_utils.api as bu
from .dic_grid import DICGrid
import traits.api as tr
from matplotlib import cm
from matplotlib.colors import SymLogNorm
import ibvpy.api as ib
import numpy as np
import numpy.ma as ma
from scipy.integrate import cumtrapz
from scipy.interpolate import RegularGridInterpolator
from .cached_array import cached_array
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import matplotlib.cm as cm

class DICStateFields(bu.Model):
    """
    State analysis of the field simulated by DIC.

    DICStateFields class for state analysis of the field simulated by DIC.

    Attributes:
        dic_grid: Instance of DICGrid.
        ct_tmodel: Either 'microplane_mdm' or 'scalar_damage'.
        R: Averaging radius.
        n_irn_M: Number of interpolation points in x direction.
        n_irn_N: Number of interpolation points in y direction.
        R_i: New Averaging radius.
        T_t: Temperature field.
        omega_t_on: Boolean indicating if omega is on.
        ipw_view: View configuration.
        get_z_MN_ironed: Static method to calculate the ironed z-coordinate field with radial averaging.

    Properties:

    Methods:
        plot_sig_eps: Plot stress-strain curve.
        plot_eps_field: Plot strain field.
        plot_sig_field: Plot stress field.
        plot_crack_detection_field: Plot crack detection field.
        update_plot: Update the plot.
        mlab_tensor: Visualize tensor data using Mayavi.
        mlab_scalar: Visualize scalar data using Mayavi.
    """

    name = 'state fields'

    """State analysis of the field simulated by DIC.
    """

    dic_grid = bu.Instance(DICGrid)

    X_IJa = tr.DelegatesTo('dic_grid')
    n_I = tr.DelegatesTo('dic_grid')
    n_J = tr.DelegatesTo('dic_grid')
    n_T = tr.DelegatesTo('dic_grid')

    ct_tmodel = bu.EitherType(options=[('microplane_mdm', ib.MATS2DMplDamageEEQ),
                                    ('scalar_damage', ib.MATS2DScalarDamage)])

    cc_tmodel = bu.Instance(ib.MATS2DMplDamageEEQ, MAT=True)
    def _cc_tmodel_default(self):
        return ib.MATS2DMplDamageEEQ(E=28000, nu=0.2, 
                                     epsilon_0=0.00008, epsilon_f=0.001)

    depends_on = ['dic_grid', 'ct_tmodel', 'cc_tmodel']
    tree = ['dic_grid', 'ct_tmodel', 'cc_tmodel']

    force_array_refresh = bu.Bool(False)

    data_dir = tr.DelegatesTo('dic_grid')

    beam_param_file = tr.DelegatesTo('dic_grid')

    verbose_eval = bu.Bool(False)
    """Report time step in simulation"""

    R = bu.Float(8, ALG=True)
    """Averaging radius"""

    n_irn_M = bu.Int(116, ALG=True)

    n_irn_N = bu.Int(28, ALG=True)

    L_corr = bu.Float(13, ALG=True)
    """Autocorrelation length"""

    R_MN = tr.Property()
    def _get_R_MN(self):
        """Averaging radius"""
        return self.L_corr * 1.4

    n_M = tr.Property()
    def _get_n_M(self):
        """Number of macroscale points in x direction"""
        return int(self.dic_grid.L_x / (self.R_MN))
    
    n_N = tr.Property()
    def _get_n_N(self):
        """Number of macroscale points in y direction"""
        return int(self.dic_grid.L_y / (self.R_MN))
    

    T_t = tr.Property(bu.Int, depends_on='state_changed')
    @tr.cached_property
    def _get_T_t(self):
        return self.dic_grid.T_t

    omega_t_on = bu.Bool(True, ALG=True)

    ipw_view = bu.View(
        bu.Item('verbose_eval'),
        bu.Item('omega_t_on'),
        bu.Item('R'),
        bu.Item('n_irn_M'),
        bu.Item('n_irn_N'),
        bu.Item('T_t', readonly=True),
        time_editor=bu.HistoryEditor(var='dic_grid.t')
    )

    @staticmethod
    def get_z_MN_ironed_masked(x_JK, y_JK, z_JK, RR, x_MN=None, y_MN=None):
        x_MN = x_JK if x_MN is None else x_MN
        y_MN = y_JK if y_MN is None else y_MN

        distances_sq = (x_JK[None, None, ...] - x_MN[..., None, None]) ** 2 + (y_JK[None, None, ...] - y_MN[..., None, None]) ** 2
        mask = distances_sq > (2 * RR) ** 2

        alpha_r_MNJK = np.exp(-distances_sq / (2 * RR ** 2))

        # Apply mask to alpha_r_MNJK
        alpha_r_MNJK = ma.array(alpha_r_MNJK, mask=mask)
        
        # Normalize alpha_r_MNJK
        # a_MN = np.trapz(np.trapz(alpha_r_MNJK, x_JK, axis=-2), y_JK, axis=-1)
        a_MN = np.trapz(np.trapz(alpha_r_MNJK, x_JK[:, 0], axis=-2), 
                        y_JK[0, :], axis=-1)

        alpha_r_MNJK /= a_MN[..., None, None]

        z_MNJK = np.einsum('MNKL,KL...->MNKL...', alpha_r_MNJK, z_JK)

        return np.trapz(np.trapz(z_MNJK, x_JK[:, 0], axis=2), y_JK[0, :], axis=2)

    @staticmethod
    def get_z_MN_ironed(x_JK, y_JK, z_JK, RR, x_MN=None, y_MN=None):
        """
        Calculates the ironed z-coordinate field 
        with radial averaging.

        Args:
            x_JK: Array of x-coordinates.
            y_JK: Array of y-coordinates.
            z_JK: Array of z-coordinates.
            RR: Radial averaging parameter.

        Returns:
            Ironed z-coordinate field 
            with radial averaging.
        """
        x_MN = x_JK if x_MN is None else x_MN
        y_MN = x_JK if y_MN is None else y_MN
        alpha_r_MNJK = np.exp(-((x_JK[None, None, ...] - 
                                 x_MN[..., None, None]) ** 2 + 
                                (y_JK[None, None, ...] - 
                                 y_MN[..., None, None]) ** 2) 
                                / (2 * RR ** 2))
        a_MN = np.trapz(np.trapz(alpha_r_MNJK, x_JK[:, 0], axis=-2), 
                        y_JK[0, :], axis=-1)
        alpha_r_MNJK /= a_MN[..., None, None]
        z_MNJK = np.einsum('MNKL,KL...->MNKL...', alpha_r_MNJK, z_JK)
        return np.trapz(
            np.trapz(z_MNJK, x_JK[:, 0], axis=2), 
            y_JK[0, :], axis=2)
    
    @staticmethod
    def _get_eps(X_IJa, U_TIJa):
        """
        Calculates the mesoscale strain tensor field.

        Returns the mesoscale strain tensor field based on the displacement field.

        Returns:
            Mesoscale strain tensor field.
        """
        x_IJ, y_IJ = np.einsum('IJa->aIJ', X_IJa)
        n_T, n_I, n_J, _ = U_TIJa.shape

        # Set spacing in x and y directions
        dx = x_IJ[1,0]-x_IJ[0,0]
        dy = y_IJ[0,1]-y_IJ[0,0]

        # Compute the gradients
        gradient_x = np.gradient(U_TIJa, dx, edge_order=2, axis=1)  # Gradient with respect to x
        gradient_y = np.gradient(U_TIJa, dy, edge_order=2, axis=2)  # Gradient with respect to y

        # Now we have the gradients of the displacement field with respect to x and y.
        # We can compute the tensor F as follows:
        F_TIJab = np.empty((n_T, n_I, n_J, 2, 2))

        F_TIJab[..., 0, 0] = gradient_x[..., 0]
        F_TIJab[..., 0, 1] = gradient_y[..., 0]
        F_TIJab[..., 1, 0] = gradient_x[..., 1]
        F_TIJab[..., 1, 1] = gradient_y[..., 1]

        # Compute the strain tensor using numpy.einsum
        return np.array(0.5 * (F_TIJab + np.einsum('...ij->...ji', F_TIJab)), dtype=np.float_)

    f_U_TIJ_txy = tr.DelegatesTo('dic_grid')
    
    eps_TIJab = tr.Property(depends_on='state_changed')
    @cached_array(names='eps_TIJab')
    def _get_eps_TIJab(self):
        return self._get_eps(self.X_IJa, self.dic_grid.U_TIJa)
    
    f_eps_TIJab_txy = tr.Property(depends_on='state_changed')
    """Interpolator of strains over the time and spatial domains.
    This method is used to provide an interpolator for a fine scale resolution 
    of strains.
    """
    @tr.cached_property
    def _get_f_eps_TIJab_txy(self):
        x_IJ, y_IJ = self.xy_IJ
        t_T = self.dic_grid.t_T
        txy = (t_T, x_IJ[:, 0], y_IJ[0, :])
        return RegularGridInterpolator(txy, self.eps_TIJab)

    state_fields_TIJ = tr.Property(depends_on='state_changed')
    @cached_array(names=['kappa_TIJr', 'omega_TIJr', 'sig_TIJab', 'dY_TIJ'])
    def _get_state_fields_TIJ(self):
        """Run the stress analysis for all load levels
        """
        # self.hist.init_state()
        kappa_TIJ = np.zeros((self.n_T, self.n_I, self.n_J))
        omega_TIJ = np.zeros((self.n_T, self.n_I, self.n_J))
        sig_TIJab = np.zeros_like(self.eps_TIJab)
        kappa_IJ = np.copy(kappa_TIJ[0])
        omega_IJ = np.copy(omega_TIJ[0])
        for T in range(self.n_T):
            sig_TIJab[T], _ = self.ct_tmodel_.get_corr_pred(
                self.eps_TIJab[T], 1, kappa_IJ, omega_IJ)
            kappa_TIJ[T, ...] = kappa_IJ            
            omega_TIJ[T, ...] = omega_IJ

        t_T = self.dic_grid.t_T
        domega_TIJ = np.gradient(omega_TIJ, t_T, axis=0)
        D_abcd = self.ct_tmodel_.D_abcd
        dY_TIJ = np.einsum('TIJab, abcd, TIJcd, TIJ->TIJ', 
                           self.eps_TIJab, D_abcd, self.eps_TIJab, domega_TIJ)

        return sig_TIJab, kappa_TIJ, omega_TIJ, dY_TIJ

    sig_TIJab = tr.Property
    def _get_sig_TIJab(self):
        return self.state_fields_TIJ[0]

    kappa_TIJ = tr.Property
    def _get_kappa_TIJ(self):
        return self.state_fields_TIJ[1]

    omega_TIJ = tr.Property
    def _get_omega_TIJ(self):
        return self.state_fields_TIJ[2]

    f_omega_TIJ_txy = tr.Property(depends_on='state_changed')
    """Interpolator of maximum damage value in time-space domain"""
    @tr.cached_property
    def _get_f_omega_TIJ_txy(self):
        x_IJ, y_IJ = self.xy_IJ
        txy = (self.dic_grid.t_T, x_IJ[:, 0], y_IJ[0, :])
        return RegularGridInterpolator(txy, self.omega_TIJ, bounds_error=False, fill_value=0)

    dY_TIJ = tr.Property
    def _get_dY_TIJ(self):
        return self.state_fields_TIJ[3]

    Y_TIJ = tr.Property
    def _get_Y_TIJ(self):
        return cumtrapz(self.dY_TIJ, self.dic_grid.t_T, axis=0, initial=0)

    #========================================================================
    # Fields evaluated on an on a macro grid 
    #========================================================================

    X_MNa = tr.Property(depends_on='state_changed')
    """Interpolation grid
    """
    @tr.cached_property
    def _get_X_MNa(self):
        n_M, n_N = self.n_M, self.n_N
        x_0, y_0, x_1, y_1 = self.dic_grid.X_frame
        xx_M = np.linspace(x_0, x_1, n_M)
        yy_N = np.linspace(y_0, y_1, n_N)
        xx_NM, yy_NM = np.meshgrid(xx_M, yy_N)
        X_aNM = np.array([xx_NM, yy_NM])
        X_MNa = np.einsum('aNM->MNa', X_aNM)
        return X_MNa

    xy_MN = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_xy_MN(self):
        return np.einsum('MNa->aMN', self.X_MNa)

    sig_TMNab = tr.Property(depends_on='+ALG')
    @cached_array(names='sig_TMNab')
    def _get_sig_TMNab(self):
        """Interpolation grid
        """
        return np.einsum('abcd, TMNcd->TMNab', self.cc_tmodel.D_abcd, self.eps_eff_TMNab)
    
    def plot_sig_compression_field(self, ax, fig=None, f_c=40, len_V = 70):
        x_MN, y_MN = self.xy_MN
        T = self.dic_grid.T_t
        sig_MNa, V_sig_MNab = np.linalg.eig(self.sig_TMNab[T])
        neg_sig_MNa = np.where(sig_MNa < 0)
        pos_sig_MNa = np.where(sig_MNa >= 0)

        sig_indices = np.argmin(sig_MNa, axis=-1)
        sig_c_MN = np.take_along_axis(sig_MNa, sig_indices[..., np.newaxis], axis=-1).squeeze(axis=-1)
        sig_c_MN[sig_c_MN > 0] = 0

        levels = np.arange(1,7) * 5
        levels = np.append(levels, 40)
        levels = [0, 3, 4, 8, 16, 32, 40]
        cnt = ax.contourf(x_MN, y_MN, -sig_c_MN, levels, cmap=cm.Blues)
        if fig:
            fig.colorbar(cnt, ax=ax)
        
        min_sig_MN = np.min(sig_MNa)
        max_sig_MN = np.max(sig_MNa)

        V_sig_MNab = np.einsum('MNa, MNab->MNab', sig_MNa, V_sig_MNab) / f_c * len_V
        for i in range(2):
            V_sig_MNa = V_sig_MNab[:,:,i,:]
            u_sig_MNa, v_sig_MNa = np.einsum('MNa->aMN', V_sig_MNa)
            sig_MN = sig_MNa[:,:,i]
            pos = sig_MN >= 0
            neg = sig_MN < 0
            # fields = zip((pos, neg),('firebrick', 'midnightblue'))
            fields = zip((neg,),('midnightblue',))
            for mask, c in fields:
                ax.quiver(x_MN[mask], y_MN[mask], u_sig_MNa[mask], v_sig_MNa[mask], color=c,
                        linewidth=0.1, angles='xy', scale_units='xy', scale=1, 
                        headwidth=1, headlength=2, headaxislength=1)
                ax.quiver(x_MN[mask], y_MN[mask], -u_sig_MNa[mask], -v_sig_MNa[mask], color=c,
                        linewidth=0.1, angles='xy', scale_units='xy', scale=1, 
                        headwidth=1, headlength=2, headaxislength=1)


        ax.axis('equal')
        ax.axis('off')

    f_sig_TMNab_txy = tr.Property(depends_on='state_changed')
    """Interpolator of strains over the time and spatial domains.
    This method is used to provide an interpolator for a fine scale resolution 
    of strains.
    """
    @tr.cached_property
    def _get_f_sig_TMNab_txy(self):
        x_MN, y_MN = np.einsum('MNa->aMN', self.X_MNa)
        txy = (self.dic_grid.t_T, x_MN[:, 0], y_MN[0, :])
        return RegularGridInterpolator(txy, self.sig_TMNab)
    
    eps_eff_TMNab = tr.Property(depends_on='+ALG')
    """Interpolation grid
    """
    @cached_array(names='eps_eff_TMNab')
    def _get_eps_eff_TMNab(self):
        print('averaging')
        x_IJ, y_IJ = self.xy_IJ
        x_MN, y_MN = self.xy_MN
        eps_eff_TIJab = np.einsum('TIJab,TIJ->TIJab', self.eps_TIJab, (1-self.omega_TIJ))
        return np.array([
            self.get_z_MN_ironed(x_IJ, y_IJ, eps_eff_IJab, self.L_corr, x_MN, y_MN)
            for eps_eff_IJab in eps_eff_TIJab
            ])
    
    f_eps_TMNab_txy = tr.Property(depends_on='state_changed')
    """Interpolator of strains over the time and spatial domains.
    This method is used to provide an interpolator for a fine scale resolution 
    of strains.
    """
    @tr.cached_property
    def _get_f_eps_TMNab_txy(self):
        x_MN, y_MN = np.einsum('MNa->aMN', self.X_MNa)
        txy = (self.dic_grid.t_T, x_MN[:, 0], y_MN[0, :])
        return RegularGridInterpolator(txy, self.eps_eff_TMNab)

    def plot_eps_MNab(self, ax, cax_neg, cax_pos):
        x_MN, y_MN = self.xy_MN
        T = self.dic_grid.T_t
        eps_MNa, _ = np.linalg.eig(self.eps_eff_TMNab[T])
        eps_indices = np.argmax(np.fabs(eps_MNa), axis=-1)
        minmax_eps_MN = np.take_along_axis(eps_MNa, eps_indices[..., np.newaxis], axis=-1).squeeze(axis=-1)
        self._plot_eps_field(x_MN, y_MN, minmax_eps_MN, ax, cax_neg, cax_pos)
        ax.axis('equal')
        ax.axis('off')

    #========================================================================
    # Fields evaluated on averaged grid 
    #========================================================================

    X_irn_MNa = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_X_irn_MNa(self):
        """Interpolation grid
        """
        n_M, n_N = self.n_irn_M, self.n_irn_N
        x_0, y_0, x_1, y_1 = self.dic_grid.X_frame
        xx_M = np.linspace(x_0, x_1, n_M)
        yy_N = np.linspace(y_0, y_1, n_N)
        xx_NM, yy_NM = np.meshgrid(xx_M, yy_N)
        X_irn_aNM = np.array([xx_NM, yy_NM])
        X_irn_MNa = np.einsum('aNM->MNa', X_irn_aNM)
        return X_irn_MNa
    
    xy_irn_MN = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_xy_irn_MN(self):
        return np.einsum('IJa->aIJ', self.X_irn_MNa)

    X_irn_bb_Ca = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_X_irn_bb_Ca(self):
        """Bounding box of the interpolated field"""
        return self.X_irn_MNa[(0,-1),(0,-1)]

    omega_irn_TMN = tr.Property(depends_on='+ALG')
    @cached_array(names='omega_irn_TMN')
    def _get_omega_irn_TMN(self):
        """Interpolation grid
        """
        print('averaging')
        x_IJ, y_IJ = np.einsum('IJa->aIJ', self.X_IJa)
        x_irn_MN, y_irn_MN = np.einsum('MNa->aMN', self.X_irn_MNa)
        return np.array([
            self.get_z_MN_ironed(x_IJ, y_IJ, omega_IJ, self.R, x_irn_MN, y_irn_MN)
            for omega_IJ in self.omega_TIJ
            ])

    f_omega_irn_txy = tr.Property(depends_on='+ALG')
    @tr.cached_property
    def _get_f_omega_irn_txy(self):
        """Interpolator of maximum damage value in time-space domain"""
        x_irn_MN, y_irn_MN = self.xy_irn_MN
        txy = (self.dic_grid.t_T, x_irn_MN[:, 0], y_irn_MN[0, :])
        return RegularGridInterpolator(txy, self.omega_irn_TMN, 
                                       bounds_error=False, fill_value=0.0)


    #========================================================================
    # Plot functions 
    #========================================================================

    # plot parameters - get them from the state evaluation
    # max_sig = bu.Float(5)
    max_eps = bu.Float(0.02)

    def plot_sig_eps(self, ax_sig_eps, color='white'):
        # plot the stress strain curve
        state_var_shape = (1,) + self.ct_tmodel_.state_var_shapes['kappa']
        kappa_zero = np.zeros(state_var_shape)
        omega_zero = np.zeros_like(kappa_zero)
        eps_test = np.zeros((1, 2, 2), dtype=np.float_)
        eps_range = np.linspace(0, 0.5, 1000)
        sig_range = []
        for eps_i in eps_range:
            eps_test[0, 0, 0] = eps_i
            arg_sig, _ = self.ct_tmodel_.get_corr_pred(eps_test, 1, kappa_zero, omega_zero)
            sig_range.append(arg_sig)
        arg_max_eps = np.argwhere(eps_range > self.max_eps)[0][0]
        sig_range = np.array(sig_range, dtype=np.float_)
        G_f = np.trapz(sig_range[:, 0, 0, 0], eps_range)

        ax_sig_eps.plot(eps_range[:arg_max_eps], sig_range[:arg_max_eps, 0, 0, 0],
                        color=color, lw=2, label='$G_f$ = %g [N/mm]' % G_f)
        ax_sig_eps.set_xlabel(r'$\varepsilon$ [-]')
        ax_sig_eps.set_ylabel(r'$\sigma$ [MPa]')

    @staticmethod
    def _plot_eps_field(x_, y_, eps_, ax, cax_neg=None, cax_pos=None):

        pos = np.ma.masked_less(eps_, 0)
        neg = np.ma.masked_greater(eps_, 0)

        neg_min = -0.004  # macroscopic critical strain
        pos_max = 0.04  # macroscopic critical strain

        pos[pos > pos_max] = pos_max
        neg[neg < neg_min] = neg_min

        cmap_pos = cm.Reds
        cmap_neg = cm.Blues_r

        levels_pos = np.linspace(0, pos_max, 10)
        levels_neg = np.linspace(neg_min, 0, 10)

        ax.contourf(x_, y_, pos, levels_pos, cmap=cmap_pos)
        ax.contourf(x_, y_, neg, levels_neg, cmap=cmap_neg)

        if cax_pos:
            cbar_pos = ColorbarBase(cax_pos, cmap=cmap_pos, norm=Normalize(vmin=0, vmax=pos_max), orientation='horizontal')
            cbar_pos.set_ticks(np.linspace(0, pos_max, 5))  # Specify custom ticks for the positive regime
        if cax_neg:
            cbar_neg = ColorbarBase(cax_neg, cmap=cmap_neg, norm=Normalize(vmin=neg_min, vmax=0), orientation='horizontal')
            cbar_neg.set_ticks(np.linspace(neg_min, -0.001, 4))  # Specify custom ticks for the negative regime

        levels = [0]
        cs = ax.contour(x_, y_, eps_, levels, linewidths=0.2, colors='k')
        ax.set_aspect('equal')


    def plot_eps_IJab(self, ax, cax_neg=None, cax_pos=None):
        x_IJ, y_IJ = np.einsum('IJa->aIJ', self.X_IJa)
        T = self.dic_grid.T_t
        eps_IJa, _ = np.linalg.eig(self.eps_TIJab[T])
        eps_indices = np.argmax(np.fabs(eps_IJa), axis=-1)
        minmax_eps_IJ = np.take_along_axis(eps_IJa, eps_indices[..., np.newaxis], axis=-1).squeeze(axis=-1)
        self._plot_eps_field(x_IJ, y_IJ, minmax_eps_IJ, ax, cax_neg, cax_pos)
        ax.axis('equal')
        ax.axis('off')

    # def plot_eps_field(self, ax_eps, fig):
    #     eps_fe_Emab, eps_fe_KLab, eps_fe_KLa, max_eps_fe_KL = self.eps_fe_fields
    #     X_fe_KLa = self.X_fe_KLa
    #     X_fe_aKL = np.einsum('MNa->aMN', X_fe_KLa)
    #     max_fe_eps = np.max(max_eps_fe_KL)
    #     cs_eps = ax_eps.contourf(X_fe_aKL[0], X_fe_aKL[1], max_eps_fe_KL, cmap='BuPu',
    #                              vmin=0, vmax=max_fe_eps)
    #     cbar_eps = fig.colorbar(cm.ScalarMappable(norm=cs_eps.norm, cmap=cs_eps.cmap),
    #                             ax=ax_eps, ticks=np.arange(0, max_fe_eps * 1.01, 0.005),
    #                             orientation='horizontal')
    #     cbar_eps.set_label(r'$\max(\varepsilon_I) > 0$')
    #     ax_eps.axis('equal')
    #     ax_eps.axis('off')

    # def plot_sig_field(self, ax_sig, fig):
    #     sig_fe_Emab, sig_fe_KLab, sig_fe_KLa, max_sig_fe_KL = self.sig_fe_fields
    #     X_fe_KLa = self.X_fe_KLa
    #     X_fe_aKL = np.einsum('MNa->aMN', X_fe_KLa)
    #     max_fe_sig = np.max(max_sig_fe_KL)
    #     cs_sig = ax_sig.contourf(X_fe_aKL[0], X_fe_aKL[1], max_sig_fe_KL, cmap='Reds',
    #                              vmin=0, vmax=max_fe_sig)
    #     cbar_sig = fig.colorbar(cm.ScalarMappable(norm=cs_sig.norm, cmap=cs_sig.cmap),
    #                             ax=ax_sig, ticks=np.arange(0, max_fe_sig * 1.01, 0.5),
    #                             orientation='horizontal')
    #     cbar_sig.set_label(r'$\max(\sigma_I) > 0$')
    #     ax_sig.axis('equal')
    #     ax_sig.axis('off')

    def plot_dY_t_IJ(self, ax, t):
        x_IJ, y_IJ = np.einsum('IJa->aIJ', self.X_IJa)
        self.dic_grid.t = t
        T = self.dic_grid.T_t
        dY_IJ = self.dY_TIJ[T]
        vmax, vmid = np.max(dY_IJ), np.average(dY_IJ) * 20
        if vmax > 0:
            cnt = ax.contourf(x_IJ, y_IJ, dY_IJ, 
                        levels=np.geomspace(vmid, vmax*1.1, 10), 
                        norm=SymLogNorm(linthresh=vmid, linscale=0.1, vmin=0, vmax=vmax, base=10),
                        cmap=cm.hot_r
                        )
        ax.axis('equal')
        ax.axis('off')

    def plot_Y_t_IJ(self, ax, t):
        x_IJ, y_IJ = self.xy_IJ
        # Y_IJ = self.Y_TIJ[T]
        # Y_IJ = np.trapz(self.dY_TIJ[:-4], self.dic_grid.t_T[:-4], axis=0)
        # Y_TIJ = np.copy(self.Y_TIJ)
        self.dic_grid.t = t
        T = self.dic_grid.T_t
        Y_TIJ = np.copy(self.Y_TIJ)
        Y_TIJ[Y_TIJ > 3] = 3
        # vmax, vmid = np.max(Y_IJ), np.average(Y_IJ) * 20
        ax.contourf(x_IJ, y_IJ, self.Y_TIJ[T],
                    # levels=np.geomspace(vmid, vmax*1.1, 10), 
                    # norm=SymLogNorm(linthresh=vmid, linscale=0.1, vmin=0, vmax=vmax, base=10),
                    # cmap=cm.hot_r
                    )
        ax.axis('equal')
        ax.axis('off')

    def subplots(self, fig):
        self.fig = fig
        return fig.subplots(3, 2)

    show_color_bar = bu.Bool(False, ALG=True)
    def plot_crack_detection_field(self, ax_cracks, fig=None):
        T = self.dic_grid.T_t if self.omega_t_on else -1
        x_irn_MN, y_irn_MN = self.xy_irn_MN
        cd_field_irn_MN = self.omega_irn_TMN[T]
        sum_cd_field = np.sum(cd_field_irn_MN)
        if sum_cd_field == 0:
            # return without warning if there is no damage or strain
            return
        contour_levels = np.array([-1, 0.35, 0.4, 0.45, 0.6, 0.8], dtype=np.float_)
        cs = ax_cracks.contourf(x_irn_MN, y_irn_MN, cd_field_irn_MN, 
                                contour_levels,
                                cmap=cm.Greys_r,
#                                cmap=cm.GnBu_r,
                               antialiased=False)
        if self.show_color_bar and fig:
            cbar_cracks = fig.colorbar(cm.ScalarMappable(norm=cs.norm, cmap=cs.cmap),
                                       ax=ax_cracks, ticks=np.linspace(0, 1, 6),
                                       orientation='horizontal')
            cbar_cracks.set_label(r'$\omega = 1 - \min(\phi_I)$')

    def update_plot(self, axes):
        ((ax_eps, ax_FU), (ax_sig, ax_sig_eps), (ax_omega, ax_cracks)) = axes
        fig = self.fig
        # spatial coordinates
        # X_fe_KLa = self.X_fe_KLa
#        X_fe_aKL = np.einsum('MNa->aMN', X_fe_KLa)
        # strain fields
        # eps_Emab, eps_MNab, eps_MNa, max_eps_MN = self.eps_fe_fields
        # eps_MNa = self.eps_TIJab[self.dic_grid.T_t]
        # # stress fields
        # sig_Emab, sig_MNab, sig_MNa, max_sig_MN = self.sig_fe_fields
        
        # damage field
        # omega_fe_KL = self.omega_fe_KL
        # # plot
        self.plot_eps_IJab(ax_eps)

        self.plot_sig_compression_field(ax_sig, fig)

        self.plot_crack_detection_field(ax_omega, fig)
        ax_omega.axis('equal');
        ax_omega.axis('off')

        self.dic_grid.plot_load_deflection(ax_FU)

        # ax_sig_eps.plot(eps_MNa[..., 0].flatten(), sig_MNa[..., 0].flatten(), 'o', color='green')
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
