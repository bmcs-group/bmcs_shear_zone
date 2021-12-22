'''
# Fracture process segment

![image.png](attachment:image.png)

The crack path consists of three segments. The lower segment $x_\mathrm{L}$ represents the localized crack. It can be represented by a continuous curve, i.e. spline or by piecewise linear geometry. The upper segment $x_\mathrm{U}$ represents the uncracked zone. The segments in-between represents the fracture process zone FZP which has a fixed length $L^\mathrm{fps}$ and orientation $\theta$ related to the vertical direction $z$. The coordinates of the segment start $x^{\mathrm{fps}}_{ak}$ is implicitly considered to be in the state of peak tensile stress. The position of the tip of the segment is then given as
\begin{align}
x^{\mathrm{fps}}_{ak} = x^{\mathrm{tip}}_{an} + L^{\mathrm{fps}}
\left[
\begin{array}{c}
-\sin(\psi) \\ \cos(\psi)
\end{array}
\right]
\end{align}
'''

import numpy as np
import sympy as sp
from bmcs_utils.api import SymbExpr, InjectSymbExpr
import traits.api as tr
from bmcs_utils.api import \
    InteractiveModel, InteractiveWindow, Item, View, Float

psi = sp.symbols(r'\psi', nonnegative=True)
theta = sp.symbols(r'\theta', nonnegative=True)
psi = sp.symbols(r'\psi', nonnegative=True)
phi = sp.symbols(r'\phi', nonnegative=True)
L_fps = sp.symbols(r'L_\mathrm{fps}', nonnegative=True)
ell = sp.symbols(r'\ell', nonnegative=True)

theta, x_rot_1k = sp.symbols(
    r'theta, x^\mathrm{rot}_{1k}', nonegative=True
)
phi, psi = sp.symbols(
    r'phi, psi', nonegative=True
)
x_tip_0n = sp.symbols(r'x^{\mathrm{tip}}_{0n}')
x_tip_1n = sp.symbols(r'x^{\mathrm{tip}}_{1n}')
xi = sp.symbols('xi')
w = sp.Symbol(r'w_\mathrm{cr}', nonnegative=True)

c_psi = sp.Symbol('c_psi')
s_psi = sp.Symbol('s_psi')

B = sp.Matrix([c_psi, s_psi])
B_ = sp.Matrix([sp.cos(psi), sp.sin(psi)])

t_vec = sp.Matrix([-s_psi, c_psi])
n_vec = sp.Matrix([c_psi, s_psi])
x_tip_an = sp.Matrix([x_tip_0n, x_tip_1n])
x_fps_ak = x_tip_an + L_fps * t_vec
x_rot_0k = x_fps_ak[0]
x_tip_ak = x_tip_an + ell * t_vec
x_rot_ak = sp.Matrix([x_rot_0k, x_rot_1k])
x_theta_xi_ak = x_tip_ak + w * n_vec + xi * t_vec
p_a = (x_tip_ak - x_rot_ak)
q_xi_a = (x_theta_xi_ak - x_rot_ak)
pp = p_a.T * p_a
qq_xi = q_xi_a.T * q_xi_a
Eq_xi_ = sp.Eq(pp[0], qq_xi[0])
xi_solved = sp.solve(Eq_xi_, xi)[0]
xi_solved_c = xi_solved.subs(c_psi ** 2 + s_psi ** 2, 1)

py_vars = ('psi', 'x_rot_1k', 'x_tip_0n', 'x_tip_1n', 'L_fps', 'ell', 'w')
get_params = lambda py_vars, **kw: tuple(kw[name] for name in py_vars)
sp_vars = tuple(globals()[py_var] for py_var in py_vars)

get_B = sp.lambdify(sp_vars, B_)
get_xi_B = sp.lambdify(sp_vars + (c_psi, s_psi), xi_solved)

def get_xi(*params):
    B_ = get_B(*params)
    xi_ = get_xi_B(*(params + tuple(B_)))
    return xi_[0]

class CrackTipExpr(SymbExpr):
    psi = psi
    x_rot_1k = x_rot_1k
    x_tip_0n = x_tip_0n
    x_tip_1n = x_tip_1n
    x_tip_ak = x_tip_ak
    c_psi = c_psi
    s_psi = s_psi

    L_fps = L_fps
    ell = ell
    w = w

    B = B_
    xi_solved = xi_solved
    symb_model_params = ['L_fps', 'ell', 'w']
    symb_expressions = [
        ('B', ('psi', 'x_rot_1k', 'x_tip_0n', 'x_tip_1n')),
        ('xi_solved', ('psi', 'x_rot_1k', 'x_tip_0n', 'x_tip_1n', 'c_psi', 's_psi')),
        ('x_tip_ak', ('x_tip_0n', 'x_tip_1n', 'c_psi', 's_psi'))
    ]

get_x_theta_B_ak = sp.lambdify(sp_vars + (c_psi, s_psi, xi), x_theta_xi_ak)

def get_x_theta_ak(*params):
    xi = get_xi_B(*params)
    return get_x_theta_B_ak(*(params + (xi,)))

get_x_tip_an = sp.lambdify(sp_vars + (c_psi, s_psi), x_tip_an)
get_x_tip_ak = sp.lambdify(sp_vars + (c_psi, s_psi), x_tip_ak)
get_x_rot_ak = sp.lambdify(sp_vars + (c_psi, s_psi), x_rot_ak)
get_x_fps_ak = sp.lambdify(sp_vars + (c_psi, s_psi), x_fps_ak)
get_p_a = sp.lambdify(sp_vars + (c_psi, s_psi), p_a)
get_q_xi_a = sp.lambdify(sp_vars + (c_psi, s_psi, xi), q_xi_a)


def get_q_a(*params):
    xi = get_xi_cs(*params)
    return get_q_xi_a(*(params + (xi,)))

dxi_dB = xi_solved_c.diff(B)
dB_dpsi = B_.diff(psi)
dB_dpsi_ = dB_dpsi.subs(
    {sp.cos(psi): c_psi, sp.sin(psi): s_psi})
dxi_psi = (dxi_dB.T * dB_dpsi)[0, 0]
dxi_x_rot_1k = xi_solved_c.diff(x_rot_1k)
get_dxi_psi = sp.lambdify(sp_vars + (c_psi, s_psi), dxi_psi)
get_dxi_x_rot_1k = sp.lambdify(sp_vars + (c_psi, s_psi), dxi_x_rot_1k)

dp_a_psi = sp.Matrix([p_a.T.diff(c) for c in B]).T * B_
dp_a_x_rot_1k = p_a.diff(x_rot_1k)
dp_aI_ = sp.Matrix([dp_a_psi.T, dp_a_x_rot_1k.T]).T

get_dp_a_psi = sp.lambdify(sp_vars + (c_psi, s_psi), dp_a_psi)
get_dp_a_x_rot_1k = sp.lambdify(sp_vars + (c_psi, s_psi), dp_a_x_rot_1k)

def get_p_a_dI(*params):
    p_a = get_p_a(*params)
    dp_a_psi = get_dp_a_psi(*params)
    dp_a_x_rot_1k = get_dp_a_x_rot_1k(*params)
    return p_a[:, 0], np.c_[dp_a_psi, dp_a_x_rot_1k]

dq_a_psi_dir = sp.Matrix([q_xi_a.T.diff(c) for c in B]).T * dB_dpsi_
dq_a_xi = q_xi_a.diff(xi)
dq_a_x_rot_1k_dir = q_xi_a.diff(x_rot_1k)
dq_aI_dir_ = sp.Matrix([dq_a_psi_dir.T, dq_a_x_rot_1k_dir.T]).T

get_dq_a_psi_dir = sp.lambdify(sp_vars + (c_psi, s_psi, xi), dq_a_psi_dir)
get_dq_a_x_rot_1k_dir = sp.lambdify(sp_vars + (c_psi, s_psi, xi), dq_a_x_rot_1k_dir)
get_dq_a_xi = sp.lambdify(sp_vars + (c_psi, s_psi), dq_a_xi)

def get_q_a_dI(*params):
    xi = get_xi_B(*params)
    q_a = get_q_xi_a(*(params + (xi,)))
    dq_a_psi_dir = get_dq_a_psi_dir(*(params + (xi,)))
    dq_a_xi = get_dq_a_xi(*params)
    dxi_psi = get_dxi_psi(*params)
    dq_a_psi = dq_a_psi_dir + dq_a_xi * dxi_psi
    dq_a_x_rot_1k_dir = get_dq_a_x_rot_1k_dir(*(params + (xi,)))
    dxi_x_rot_1k = get_dxi_x_rot_1k(*params)
    dq_a_x_rot_1k = dq_a_x_rot_1k_dir + dq_a_xi * dxi_x_rot_1k
    return q_a[:, 0], np.c_[dq_a_psi, dq_a_x_rot_1k]

def get_cos_theta_dI(p_a, q_a, dp_aI, dq_aI):
    pq = np.einsum('a,a', p_a, q_a)
    norm_p = np.sqrt(np.einsum('a,a', p_a, p_a))
    norm_q = np.sqrt(np.einsum('a,a', q_a, q_a))
    norm_pq = norm_p * norm_q
    d_pq_I = (
            np.einsum('a,aI->I', p_a, dq_aI) + np.einsum('aI,a->I', dp_aI, q_a)
    )
    d_norm_p_I = np.einsum('a,aI->I', p_a, dp_aI) / norm_p
    d_norm_q_I = np.einsum('a,aI->I', q_a, dq_aI) / norm_q
    d_norm_pq_I = d_norm_p_I * norm_q + norm_p * d_norm_q_I
    cos_theta = pq / norm_pq
    d_cos_theta_I = (d_pq_I - cos_theta * d_norm_pq_I) / norm_pq
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    d_sin_theta_I = -cos_theta / sin_theta * d_cos_theta_I
    return cos_theta, sin_theta, d_cos_theta_I, d_sin_theta_I

def get_cos_theta_dI2(*params):
    c_psi, s_psi = get_B(*params)
    params_B = params + (c_psi[0], s_psi[0])
    p_a, dp_aI = get_p_a_dI(*params_B)
    q_a, dq_aI = get_q_a_dI(*params_B)
    return get_cos_theta_dI(p_a, q_a, dp_aI, dq_aI)

c_theta, s_theta = sp.symbols(r'\cos(\theta), \sin(\theta)')
c_theta = sp.Function(r'cos')(theta)
s_theta = sp.Function(r'sin')(theta)

T = sp.Matrix(
    [[c_theta, -s_theta],
     [s_theta, c_theta]], dtype=np.float_)

dT_dcs_theta = T.diff(c_theta), T.diff(s_theta)
get_T = sp.lambdify((c_theta, s_theta), T, 'numpy')
get_dT_dcs = sp.lambdify((c_theta, s_theta), dT_dcs_theta, 'numpy')

def get_T_ab_dT_dI_abI(*params):
    cos_theta, sin_theta, d_cos_theta_I, d_sin_theta_I = get_cos_theta_dI2(
        *params
    )
    cs_c = np.array([cos_theta, sin_theta], dtype=np.float_)
    dcs_cI = np.array([d_cos_theta_I, d_sin_theta_I], dtype=np.float_)
    T_ab = get_T(cos_theta, sin_theta)
    dT_dcs_cab = np.array(get_dT_dcs(cos_theta, sin_theta))
    dT_dI_abI = np.einsum('cab,cI->abI', dT_dcs_cab, dcs_cI)
    return T_ab, dT_dI_abI

def get_x1_La(T_ab, x0_La, x_rot_a):
    x_rot_La = x_rot_a[np.newaxis, ...]
    return np.einsum('ab,Lb->La', T_ab, x0_La - x_rot_La) + x_rot_La

class SZCrackTipRotation(InteractiveModel, InjectSymbExpr):
    symb_class = CrackTipExpr

    name = 'Crack tip'

    # Define the free parameters as traits with default, min and max values
    # Classification to handle update of dependent components
    psi = Float(0.8, ITR=True, MAT=True)
    x_rot_1k = Float(100,ITR=True, MAT=True)
    x_tip_0n = Float(200, INC=True, MAT=True)
    x_tip_1n = Float(50, INC=True, MAT=True)
    L_fps = Float(20, MAT=True)
    ell = Float(5, MAT=True)
    w = Float(0.3, MAT=True)

    ipw_view = View(
        Item('psi', latex=r'\psi', minmax=(0, np.pi / 2)),
        Item('x_rot_1k', latex=r'x^\mathrm{rot}_{1k}', minmax=(0, 200)),
        Item('x_tip_0n', latex=r'x^\mathrm{tip}_{0n}', minmax=(0, 500)),
        Item('x_tip_1n', latex=r'x^\mathrm{tip}_{1n}', minmax=(0, 200)),
        Item('L_fps', latex=r'L_\mathrm{fps}', minmax=(0, 100)),
        Item('ell', latex=r'\ell', minmax=(0, 10)),
        Item('w', latex=r'w', minmax=(0, 20))
    )

    all_points = tr.Property(depends_on='state_changed')

    @tr.cached_property
    def _get_all_points(self):
        params = self.psi, self.x_rot_1k, self.x_tip_0n, self.x_tip_1n
        c_theta, s_theta = self.symb.get_B(*params)
        model_params = self.symb.get_model_params()
        params = params + model_params + (c_theta[0], s_theta[0])
        x_rot_ak = get_x_rot_ak(*params)
        x_tip_an = get_x_tip_an(*params)
        x_tip_ak = get_x_tip_ak(*params)
        x_theta_ak = get_x_theta_ak(*params)
        x_fps_ak = get_x_fps_ak(*params)
        return x_rot_ak, x_tip_an, x_tip_ak, x_theta_ak, x_fps_ak

    x_rot_ak = tr.Property

    def _get_x_rot_ak(self):
        x_rot_ak, x_tip_an, x_tip_ak, x_theta_ak, x_fps_ak = self.all_points
        return x_rot_ak

    x_tip_an = tr.Property

    def _get_x_tip_an(self):
        x_rot_ak, x_tip_an, x_tip_ak, x_theta_ak, x_fps_ak = self.all_points
        return x_tip_an

    x_tip_ak = tr.Property

    def _get_x_tip_ak(self):
        x_rot_ak, x_tip_an, x_tip_ak, x_theta_ak, x_fps_ak = self.all_points
        return x_tip_ak

    x_theta_ak = tr.Property

    def _get_x_theta_ak(self):
        x_rot_ak, x_tip_an, x_tip_ak, x_theta_ak, x_fps_ak = self.all_points
        return x_theta_ak

    x_fps_ak = tr.Property

    def _get_x_fps_ak(self):
        x_rot_ak, x_tip_an, x_tip_ak, x_theta_ak, x_fps_ak = self.all_points
        return x_fps_ak

    def get_T_ab_dT_dI_abI(self):
        '''Rotation matrix for the right plate'''
        variables = self.psi, self.x_rot_1k, self.x_tip_0n, self.x_tip_1n
        model_params = self.symb.get_model_params()
        params = variables + model_params
        T_ab, dT_dI_abI = get_T_ab_dT_dI_abI(*params)
        return T_ab, dT_dI_abI

    def get_x1_La(self, x0_La):
        P_ab, _ = self.get_T_ab_dT_dI_abI()
        x_rot_a = self.x_rot_ak[: ,0]
        return get_x1_La(P_ab, x0_La, x_rot_a)

    def plot_crack_tip_rotation(self, ax):
        ax.plot(*self.x_rot_ak, marker='o', color='blue')
        ax.plot(*self.x_tip_an, marker='o', color='green')
        ax.plot(*self.x_tip_ak, marker='o', color='orange')
        ax.plot(*self.x_theta_ak, marker='o', color='orange')
        ax.plot(*self.x_fps_ak, marker='o', color='red')
        ax.plot(*np.c_[self.x_tip_an, self.x_fps_ak], color='red')
        ax.plot(*np.c_[self.x_fps_ak, self.x_rot_ak], color='blue')

    def update_plot(self, ax):
        ax.axis('equal')
        self.plot_crack_tip_rotation(ax)
