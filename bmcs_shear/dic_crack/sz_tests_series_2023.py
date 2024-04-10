
import numpy as np

B1_TV1 = dict(
    dir_name = 'B1_TV1',
    U_factor=10,
    n_t_max = 40,
    T_stepping = 'delta_T',
    R=8, omega_threshold=0.15, verbose_eval=True, tmodel='scalar_damage',
    E=1600, nu=0.0, omega_fn='exp-slope', strain_norm='Rankine', eps_max=0.01,
    t_detect = 0.95,
    delta_alpha_min = -0.1*np.pi,
    delta_alpha_max = 0.5*np.pi,
    delta_s = 20,
    x_boundary = 30
)

B1_TV2 = dict(
    dir_name = 'B1_TV2',
    U_factor=10,
    n_t_max = 40,
    T_stepping = 'delta_T',
    R=5, omega_threshold=0.15, verbose_eval=True, tmodel='scalar_damage',
    E=1600, nu=0.18, omega_fn='exp-slope', strain_norm='Rankine', eps_max=0.01,
    t_detect = 0.72,
    delta_alpha_min = -0.1*np.pi,
    delta_alpha_max = 0.5*np.pi,
    delta_s = 20,
    x_boundary = 30
)

B5_TV1 = dict(
    dir_name = 'B5_TV1',
    U_factor=10,
    n_t_max = 40,
    T_stepping = 'delta_T',
    R=8, omega_threshold=0.1, verbose_eval=True, tmodel='scalar_damage',
    E=1600, nu=0.18, omega_fn='exp-slope', strain_norm='Rankine', eps_max=0.01,
    t_detect = 0.80,
    delta_alpha_min = -np.pi/6,
    delta_alpha_max = np.pi/3,
    delta_s = 20,
    x_boundary = 30
)

B6_TV1 = dict(
    dir_name = 'B6_TV1',
    U_factor=10,
    n_t_max = 40,
    T_stepping = 'delta_T',
    R=8, omega_threshold=0.1, verbose_eval=True, tmodel='scalar_damage',
    E=1600, nu=0, omega_fn='exp-slope', strain_norm='Rankine', eps_max=0.01,
    t_detect = 0.9,
    delta_alpha_min =  -0.2*np.pi,
    delta_alpha_max =  0.2*np.pi,
    delta_s = 20,
    x_boundary = 30
)

B6_TV2 = dict(
    dir_name = 'B6_TV2',
    U_factor=10,
    n_t_max = 40,
    T_stepping = 'delta_T',
    R=8, 
    omega_threshold=0.1, 
    verbose_eval=True, tmodel='scalar_damage',
    E=1600, nu=0.18, omega_fn='exp-slope', strain_norm='Rankine', eps_max=0.01,
    t_detect = 0.98,
    delta_alpha_min =  -0.1*np.pi,
    delta_alpha_max =  0.2*np.pi,
    delta_s = 20,
    x_boundary = 30
)

B7_TV1 = dict(
    dir_name = 'B7_TV1',
    U_factor=10,
    n_t_max = 40,
    T_stepping = 'delta_T',
    R=8, 
    omega_threshold=0.1, 
    verbose_eval=True, tmodel='scalar_damage',
    E=1600, nu=0.18, omega_fn='exp-slope', strain_norm='Rankine', eps_max=0.01,
    t_detect = 0.93,
    delta_alpha_min =  -0.1*np.pi,
    delta_alpha_max =  0.2*np.pi,
    delta_s = 20,
    x_boundary = 30
)


B7_TV2 = dict(
    dir_name = 'B7_TV2',
    U_factor=10,
    n_t_max = 40,
    T_stepping = 'delta_T',
    R=8, 
    omega_threshold=0.1, verbose_eval=True, tmodel='scalar_damage',
    E=1600, nu=0.18, omega_fn='exp-slope', strain_norm='Rankine', eps_max=0.01,
    t_detect = 0.98,
    delta_alpha_min =  -0.1*np.pi,
    delta_alpha_max =  0.2*np.pi,
    delta_s = 20,
    x_boundary = 40
)

B8_TV1 = dict(
    dir_name = 'B8_TV1',
    U_factor=10,
    n_t_max = 40,
    T_stepping = 'delta_T',
    R=8, 
    omega_threshold=0.1, verbose_eval=True, tmodel='scalar_damage',
    E=1600, nu=0.18, omega_fn='exp-slope', strain_norm='Rankine', eps_max=0.01,
    t_detect = 0.98,
    delta_alpha_min =  -0.1*np.pi,
    delta_alpha_max =  0.2*np.pi,
    delta_s = 20,
    x_boundary = 30
)


B8_TV2 = dict(
    dir_name = 'B8_TV2',
    U_factor=10,
    n_t_max = 40,
    T_stepping = 'delta_T',
    R=8, 
    omega_threshold=0.1, 
    verbose_eval=True, tmodel='scalar_damage',
    E=1600, nu=0.18, omega_fn='exp-slope', strain_norm='Rankine', eps_max=0.01,
    t_detect = 0.98,
    delta_alpha_min =  -0.3*np.pi,
    delta_alpha_max =  0.2*np.pi,
    delta_s = 20,
    x_boundary = 30
)

B9_TV1 = dict(
    dir_name = 'B9_TV1',
    U_factor=10,
    n_t_max = 40,
    T_stepping = 'delta_T',
    R=8, 
    omega_threshold=0.1, verbose_eval=True, tmodel='scalar_damage',
    E=1600, nu=0.18, omega_fn='exp-slope', strain_norm='Rankine', eps_max=0.01,
    t_detect = 0.98,
    delta_alpha_min =  -0.25*np.pi,
    delta_alpha_max =  0.3*np.pi,
    delta_s = 20,
    x_boundary = 30
)

B9_TV2 = dict(
    dir_name = 'B9_TV2',
    U_factor=10,
    n_t_max = 40,
    T_stepping = 'delta_T',
    R=8, 
    omega_threshold=0.1, 
    verbose_eval=True, tmodel='scalar_damage',
    E=1600, nu=0.18, omega_fn='exp-slope', strain_norm='Rankine', eps_max=0.01,
    t_detect = 0.98,
    delta_alpha_min =  -0.1*np.pi,
    delta_alpha_max =  0.2*np.pi,
    delta_s = 20,
    x_boundary = 30
)

B10_TV1 = dict(
    dir_name = 'B10_TV1',
    U_factor=10,
    n_t_max = 40,
    T_stepping = 'delta_T',
    R=8, 
    omega_threshold=0.1, 
    verbose_eval=True, tmodel='scalar_damage',
    E=1600, nu=0.18, omega_fn='exp-slope', strain_norm='Rankine', eps_max=0.01,
    t_detect = 0.98,
    delta_alpha_min =  -0.1*np.pi,
    delta_alpha_max =  0.2*np.pi,
    delta_s = 20,
    x_boundary = 30
)


B10_TV2 = dict(
    dir_name = 'B10_TV2',
    U_factor=10,
    n_t_max = 40,
    T_stepping = 'delta_T',
    R=8, 
    omega_threshold=0.1, 
    verbose_eval=True, tmodel='scalar_damage',
    E=1600, nu=0.18, omega_fn='exp-slope', strain_norm='Rankine', eps_max=0.01,
    t_detect = 0.98,
    delta_alpha_min =  -0.1*np.pi,
    delta_alpha_max =  0.2*np.pi,
    delta_s = 20,
    x_boundary = 30
)

from bmcs_shear.dic_crack import\
    DICInpUnstructuredPoints, DICStateFields, \
    DICGrid
from bmcs_shear.dic_crack.dic_crack_list import DICCrackList
import numpy as np
np.seterr(divide ='ignore', invalid='ignore');
def new_dcl(test):
    dic_points = DICInpUnstructuredPoints(**test)
    dic_points.read_beam_design()
    dic_grid = DICGrid(dic_inp=dic_points, **test)
    dsf = DICStateFields(dic_grid=dic_grid, **test)
    dsf.tmodel_.trait_set(**test)
    dsf.tmodel_.omega_fn_.trait_set(kappa_0=0.002, kappa_f=0.0028);
    return DICCrackList(dsf=dsf, **test)