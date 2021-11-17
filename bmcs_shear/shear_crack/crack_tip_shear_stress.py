

import traits.api as tr
import numpy as np
import sympy as sp
from bmcs_utils.api import Model, View, Item, Float, SymbExpr, InjectSymbExpr
from bmcs_shear.shear_crack.stress_profile import \
    SZStressProfile

class SZCrackTipShearStress(Model):
    name = 'Crack tip stress state'

    sz_sp = tr.Instance(SZStressProfile, ())
    sz_cp = tr.DelegatesTo('sz_sp')
    sz_bd = tr.DelegatesTo('sz_cp', 'sz_bd')

    tree = ['sz_sp']

    L_cs = Float(200, MAT=True)  ##distance between cracks [mm]

    ipw_view = View(
        Item('L_cs', latex=r'L_{cs}'),
    )

    Q = tr.Property
    """Assuming the parabolic profile of the shear stress within the uncracked
    zone, calculate the value of shear stress corresponding to the height of the
    crack tip
    """

    def _get_Q(self):
        M = self.sz_sp.M
        L = self.sz_bd.L
        x_tip_0k = self.sz_cp.sz_ctr.x_tip_ak[0]
        Q = M / (L - x_tip_0k)[0]
        #print(Q)
        return Q

    F_beam = tr.Property
    '''Use the reference to MQProfileand BoundaryConditions
    to calculate the global load. Its interpretation depends on the   
    nature of the load - single mid point, four-point, distributed.
    '''

    # TODO: Currently there is just a single midpoint load of a 3pt bending beam assumed.
    #       then, the load is equal to the shear force
    def _get_F_beam(self):
        return 2 * self.Q

    sig_x_tip_0 = tr.Property

    def _get_sig_x_tip_0(self):
        return self.sz_sp.sig_x_tip_ak[0]

    sig_z_tip_1 = tr.Property(depends_on='state_changed')
    '''Crack parallel stress from cantilever action'''

    @tr.cached_property
    def _get_sig_z_tip_1(self):
        M_cantilever = self.M_cantilever
        B = self.sz_bd.B
        L_cs = self.L_cs
        S = (B * L_cs ** 2) / 6
        sigma_tip_1 = -(M_cantilever / S)
        if sigma_tip_1 < 0:
            sigma_tip_1 = 0
        #print('sig_tip_1', sigma_tip_1)
        return sigma_tip_1

    F_N_delta = tr.Property(depends_on='state_changed')
    '''Force at steel'''

    @tr.cached_property
    def _get_F_N_delta(self):
        sp = self.sz_sp
        x_tip_1k = sp.sz_cp.sz_ctr.x_tip_ak[1, 0]
        H = self.sz_bd.H
        F_N_delta = self.Q * self.L_cs / H
        return F_N_delta

    M_cantilever = tr.Property(depends_on='state_changed')
    '''Clamping moment'''

    @tr.cached_property
    def _get_M_cantilever(self):
        # crack paths of two neighboring cracks to calculate the cantilever action
        sp = self.sz_sp
        x_tip_0 = sp.sz_ds.sz_ctr.x_tip_ak[0,0] #[:,0]
        x_tip_1n = sp.sz_ds.sz_ctr.x_tip_1n
        x_mid_0 = np.copy(x_tip_0)
        x_mid_0 -= self.L_cs / 2

        x_Ka = sp.sz_ds.sz_cp.x_Ka
        K_Li = sp.sz_ds.sz_cp.K_Li
        x_Lia = x_Ka[K_Li]
        x_La = np.sum(x_Lia, axis=1) / 2
        F_La = sp.F_La
        x_right_La = x_La[...]
        M_right_agg = np.einsum('L,L', F_La[:, 1], (x_right_La[:, 0] - x_mid_0))
        x_left_La = x_La[...]
        x_left_La[..., 0] -= self.L_cs
        M_left_agg = np.einsum('L,L', -F_La[:, 1], (x_left_La[:, 0] - x_mid_0))
        # print('M_cantilever')
        # print('x_mid_a', x_mid_a)
        # print('x_mid_a - x_left_La', x_mid_a - x_left_La[:, 0])
        if len(sp.z_N > 0):
            F_Na = sp.F_Na
            x_00 = np.ones_like(sp.z_N) * sp.sz_cp.x_00
            M_right_da = np.einsum('L,L', F_Na[:, 1], x_00 - x_mid_0)
            x_00_L = x_00 - self.L_cs
            M_left_da = np.einsum('L,L', -F_Na[:, 1], x_00_L - x_mid_0)
            # print('x_mid_a + x_00_L', np.abs(x_mid_a - x_00_L))
            x_tip_1k = sp.sz_cp.sz_ctr.x_tip_ak[1, 0]
            H = self.sz_bd.H
            # print('x_tip_1n', x_tip_1n, end=', ')
            if x_tip_1n > sp.z_N:
                delta_z_N = x_tip_1k - sp.z_N
                # print('d_z_N', delta_z_N, end=', ')
                # Get the current lever arm
                L = self.sz_bd.L
                F_N_delta_ = F_Na[:,0] / (L-self.sz_cp.x_00) * self.L_cs
                f_cm = self.f_c
                p_N = self.sz_bd.csl.p_j
                F_N_delta_max = p_N * self.L_cs * 1.26 * np.sqrt(f_cm/20) #p_N
                F_N_delta = np.min(np.c_[F_N_delta_, F_N_delta_max], axis=1)
                #print(F_N_delta_, F_N_delta_max)
                # print('F_N_delta', F_N_delta, end=', ')
                # print('F_Na', F_Na[:,0], end=', ')
                # print('d_z_N', delta_z_N, end=', ')
                M_delta_F = (-F_N_delta * delta_z_N)[0]
            else:
                M_delta_F = 0
        else:
            M_right_da = 0
            M_left_da = 0
            M_delta_F = 0

        # M_left_da=0
        # M_right_da=0
        # M_left_agg=0
        # M_right_agg=0

        # print('M_delta_F', M_delta_F, end=', ')
        # print('M_left_agg', M_left_agg, end=', ')
        # print('M_right_agg', M_right_agg, end=', ')
        # print('M_left_da', M_left_da, end=', ')
        # print('M_right_da', M_right_da, end=', ')
        # print()
        return (M_delta_F + M_left_agg + M_right_agg + M_right_da + M_left_da)
