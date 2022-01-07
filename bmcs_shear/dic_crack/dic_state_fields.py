'''
@author: rch
'''

import bmcs_utils.api as bu
from .dic_grid import DICGrid

import traits.api as tr

from ibvpy.sim.tstep import \
    TStep

import ibvpy.api as ib
from mayavi import mlab
import numpy as np
import copy

class DICStateFields(ib.TStepBC):
    '''State analysis of the field simulated by DIC.
    '''

    dic_grid = bu.Instance(DICGrid)

    tmodel = tr.Property(depends_on='state_changed')
    '''Material model
    '''
    @tr.cached_property
    def _get_tmodel(self):
        m_mic = ib.MATS2DMplDamageEEQ()
        return m_mic
        m_scalar_damage = ib.MATS2DScalarDamage()
        m_scalar_damage.strain_norm = 'Masars'
        m_scalar_damage.omega_fn = 'exp-slope'
        return m_scalar_damage

    xmodel = tr.Property(depends_on='state_changed')
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
        return [(self.xmodel, self.tmodel)]


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
            self.tmodel.get_corr_pred(eps_Emab, 1, **self.fe_domain[0].state_k)
            self.U_k[:] = U_o[:]
            self.U_n[:] = self.U_k[:]
#            self.fe_domain[0].record_state()
            states = [self.fe_domain[0].state_k]
            self.hist.record_timestep(self.t_n1, self.U_k, self.F_k, states)
            self.t_n = self.t_n1
