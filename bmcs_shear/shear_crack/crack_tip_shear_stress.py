

import traits.api as tr
import numpy as np
import sympy as sp
from bmcs_utils.api import Model, View, Item, Float, SymbExpr, InjectSymbExpr
from bmcs_shear.shear_crack.stress_profile import \
    SZStressProfile

class SZCrackTipShearStress(Model):
    name = 'Crack tip stress state'

    sz_stress_profile = tr.Instance(SZStressProfile, ())
    sz_cp = tr.DelegatesTo('sz_stress_profile')
    sz_bd = tr.DelegatesTo('sz_cp', 'sz_bd')

    tree = ['sz_stress_profile']

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
        M = self.sz_stress_profile.M
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
        return self.sz_stress_profile.sig_x_tip_ak[0]
        # B = self.sz_bd.B
        # x_tip_1 = self.sz_cp.sz_ctr.x_tip_ak[1,0]
        # idx_tip0 = np.argmax(self.sz_cp.x_Ka[:, 1] >= x_tip_1)
        # S_La = (self.sz_stress_profile.S_La[idx_tip0, 0])
        # S_Lb = (self.sz_stress_profile.S_Lb[idx_tip0, 0])
        # # print('S_La',S_La)
        # # print('S_Lb', S_Lb)
        # S_tip_0 = S_La / B
        # # print('S_tip_0', S_tip_0)
        # # print('B', B)
        # #S_tip_0 = 0 #self.sz_bd.matrix_.f_t - 0.000001
        # #print('sig_x_tip_0', sigma_c)
        # return S_tip_0

    sig_z_tip_1 = tr.Property
    '''Crack parallel stress from cantilever action'''

    def _get_sig_z_tip_1(self):
        M_cantilever = self.M_cantilever
        B = self.sz_bd.B

        L_cs = self.L_cs
        S = (B * L_cs ** 2) / 6
        #sigma_z_tip_1 = (M_cantilever / S)
        sigma_z_tip_1 = 0#self.sz_bd.matrix_.f_t - 0.1
        #print('sigma_z_tip_1', sigma_z_tip_1)
        return sigma_z_tip_1

    F_N_delta = tr.Property(depends_on='state_changed')
    '''Force at steel'''

    @tr.cached_property
    def _get_F_N_delta(self):
        sp = self.sz_stress_profile
        x_tip_1k = sp.sz_cp.sz_ctr.x_tip_ak[1, 0]
        H = self.sz_bd.H
        F_N_delta = self.Q * self.L_cs / H
        return F_N_delta

    M_cantilever = tr.Property(depends_on='state_changed')
    '''Clamping moment'''

    @tr.cached_property
    def _get_M_cantilever(self):
        sp = self.sz_stress_profile
        x_Ka = sp.ds.sz_cp.x_Ka
        K_Li = sp.ds.sz_cp.K_Li
        x_Lia = x_Ka[K_Li]
        x_La = np.sum(x_Lia, axis=1) / 2
        F_La = sp.F_La
        F_Na = sp.F_Na

        # crack paths of two neighboring cracks to calculate the cantilever action
        x_tip_a = sp.ds.sz_ctr.x_tip_ak[0,0] #[:,0]
        x_mid_a = x_tip_a
        x_mid_a -= self.L_cs / 2
        x_00 = np.ones_like(sp.z_N) * sp.sz_cp.x_00
        M_right_da = np.einsum('L,L', F_Na[:, 1], x_00 - x_mid_a)
        x_00_L = x_00 - self.L_cs
        M_left_da = np.einsum('L,L', - F_Na[:, 1], x_mid_a - x_00_L)
        # print(M_left_da + M_right_da)
        # print(np.abs(x_mid_a - x_00_L))
        x_right_La = x_La[...]
        M_right_agg = np.einsum('L,L', F_La[:, 1], (x_right_La[:, 0] - x_mid_a))
        x_left_La = x_La[...]
        x_left_La[..., 0] -= self.L_cs
        M_left_agg = np.einsum('L,L', - F_La[:, 1], (x_mid_a - x_left_La[:, 0]))
        # print(x_mid_a)
        # print(x_mid_a - x_left_La[:, 0])
        x_tip_1k = sp.sz_cp.sz_ctr.x_tip_ak[1,0]
        H = self.sz_bd.H
        delta_z_N = x_tip_1k - sp.z_N
        F_N_delta = self.Q * self.L_cs / H
        # print(F_N_delta)
        M_delta_F = (- F_N_delta) * delta_z_N
        # print(M_delta_F)
        #print(-(M_delta_F + M_left_agg + M_right_agg + M_right_da + M_left_da)[0])
        return (- M_delta_F + M_left_agg + M_right_agg + M_right_da + M_left_da)[0] #-
