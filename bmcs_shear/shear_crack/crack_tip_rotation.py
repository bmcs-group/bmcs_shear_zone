
import numpy as np
import sympy as sp
from bmcs_utils.api import SymbExpr, InjectSymbExpr

# # Fracture process segment

# ![image.png](attachment:image.png)

# The crack path consists of three segments. The lower segment $x_\mathrm{L}$ represents the localized crack. It can be represented by a continuous curve, i.e. spline or by piecewise linear geometry. The upper segment $x_\mathrm{U}$ represents the uncracked zone. The segments in-between represents the fracture process zone FZP which has a fixed length $L^\mathrm{fps}$ and orientation $\theta$ related to the vertical direction $z$. The coordinates of the segment start $x^{\mathrm{fps}}_{ak}$ is implicitly considered to be in the state of peak tensile stress. The position of the tip of the segment is then given as
# \begin{align}
# x^{\mathrm{fps}}_{ak} = x^{\mathrm{tip}}_{an} + L^{\mathrm{fps}}
# \left[
# \begin{array}{c}
# -\sin(\psi) \\ \cos(\psi)
# \end{array}
# \right]
# \end{align}

psi = sp.symbols(r'\psi', nonnegative=True)
theta = sp.symbols(r'\theta', nonnegative=True)
psi = sp.symbols(r'\psi', nonnegative=True)
phi = sp.symbols(r'\phi', nonnegative=True)
L_fps = sp.symbols(r'L_\mathrm{fps}', nonnegative=True)
ell = sp.symbols(r'\ell', nonnegative=True)

# ## Identify the rotation for given crack opening

# Kinematics of the section based on the assumption
# that the right side of the shear zone rotates relatively
# to the fixed left side around an unknown position $x^{\mathrm{rot}}_{ak}$. The horizontal position of the center of rotation is assumed identical with the tip of the process zone segment $x^{\mathrm{tip}}_{ak}$, i.e.
# \begin{align}
# x^{\mathrm{rot}}_{0k} = x^{\mathrm{tip}}_{0k}
# \end{align}

# The criterion of a crack reaching the strength is defined in strains. This brings about the possibility to circumvent the discussion on the bi-axial stress criterion. This can be seen in analogy to a compression test failure simulated as micro-cracking with cracks oriented in the loading direction.
# For this purpose, it is important to include the

# In[3]:


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

# Require that at the crack tip, the crack opening $w$ is equal to critical crack opening
# \begin{align}
# w_\mathrm{cr} = \frac{ f_\mathrm{t} } { E_\mathrm{c} } L_{c}
# \end{align}
# where $L_\mathrm{c}$ represents the characteristic length.

# ![image.png](attachment:image.png)

# In[4]:


c_psi = sp.Symbol('c_psi')
s_psi = sp.Symbol('s_psi')

# Assembling the unit vector components into the vector
# \begin{align}
# B_c = [ \cos(\psi), \sin(\psi) ]
# \end{align}

# In[5]:


B = sp.Matrix([c_psi, s_psi])
B_ = sp.Matrix([sp.cos(psi), sp.sin(psi)])

# In[6]:


t_vec = sp.Matrix([-s_psi, c_psi])
n_vec = sp.Matrix([c_psi, s_psi])
t_vec, n_vec

# In[7]:


x_tip_an = sp.Matrix([x_tip_0n, x_tip_1n])
x_fps_ak = x_tip_an + L_fps * t_vec
x_rot_0k = x_fps_ak[0]
x_rot_0k

#  * $t$ is the line vector of the fracture propagation segment
#  * $n$ is a vector normal to the fracture propagation segment

# In[8]:


x_tip_ak = x_tip_an + ell * t_vec
x_rot_ak = sp.Matrix([x_rot_0k, x_rot_1k])
x_tip_an, x_tip_ak, x_rot_ak, x_fps_ak

# The final position must be on a line parallel to $t$ in
# the distance $w$
# \begin{align}
# x^{\mathrm{tip},\varphi}_{ak} = x^{\mathrm{tip}}_{ak} + w \, n_{ak} + \xi \, t_{ak}
# \end{align}

# In[9]:


x_theta_xi_ak = x_tip_ak + w * n_vec + xi * t_vec
x_theta_xi_ak

# Vector connecting rotation center with the current crack tip point on the fixed plate $k$
# \begin{align}
# p_{a} = x^{\mathrm{tip}}_{ak} - x^{\mathrm{rot}}_{ak}
# \end{align}

# Vector connecting the rotation center with the current crack tip point on the rotated plate
# \begin{align}
# q_{a}(\xi)
# = x^{\mathrm{tip},\varphi}_{ak}(\xi) - x^{\mathrm{rot}}_{ak}
# \end{align}

# In[10]:


p_a = (x_tip_ak - x_rot_ak)
p_a

# In[11]:


q_xi_a = (x_theta_xi_ak - x_rot_ak)
q_xi_a

# The lengths of vectors $p_a$ and $q_a$ must be equal
# \begin{align}
#  \| p \|
# &=
#  \| q^\xi \|
#  \; \implies
#  \sqrt{p_{a}  p_{a}} =
#  \sqrt{q_{a}^\xi  q_a^\xi} \; \implies
#  p_a p_a = q_a^\xi q_a^\xi
# \end{align}

# In[12]:


pp = p_a.T * p_a
qq_xi = q_xi_a.T * q_xi_a

# The equal distance can be used to resolve the unknown slip parameter $\xi$

# In[13]:


Eq_xi_ = sp.Eq(pp[0], qq_xi[0])
Eq_xi_

# In[14]:


xi_solved = sp.solve(Eq_xi_, xi)[0]
xi_solved_c = xi_solved.subs(c_psi ** 2 + s_psi ** 2, 1)

# In[15]:


sp.simplify(xi_solved_c)

# In[16]:


py_vars = ('psi', 'x_rot_1k', 'x_tip_0n', 'x_tip_1n', 'L_fps', 'ell', 'w')
get_params = lambda py_vars, **kw: tuple(kw[name] for name in py_vars)
sp_vars = tuple(globals()[py_var] for py_var in py_vars)
sp_vars

# In[17]:


get_params(py_vars, psi=0, x_rot_1k=1, x_tip_0n=0, x_tip_1n=0, L_fps=0.1, ell=0.1, w=0.1)

# In[18]:


get_B = sp.lambdify(sp_vars, B_)
get_xi_B = sp.lambdify(sp_vars + (c_psi, s_psi), xi_solved)


def get_xi(*params):
    B_ = get_B(*params)
    xi_ = get_xi_B(*(params + tuple(B_)))
    return xi_[0]


# In[19]:


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


# In[20]:


params = 0, 1, 0, 0, 0.1, 0.1, 0.1
param_dict = dict(
    psi=0, x_rot_1k=1, x_tip_0n=0, x_tip_1n=0, L_fps=0.1, ell=0.1, w=0.1
)
get_xi(*params), get_xi(*get_params(py_vars, **param_dict))

# In[21]:


x_theta_xi_ak

# In[22]:


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


# In[23]:


get_x_fps_ak(*(params + (0, 1)))

# ## Derivatives of the side rotation vectors $p, q$

# To derive an efficient Newton-Raphson algorithm, let us directly prepare the derivatives of $\xi$ with respect to the angle primary unknowns - crack inclination $\theta$ and to the vertical position of the center of rotation $x^\mathrm{rot}_{1k}$ assembled in a vector
# \begin{align}
# \zeta_I = [ \psi, x_{1k}^\mathrm{rot} ]
# \end{align}
# As the solution of $(\xi, x^\mathrm{rot}_{1k})$ has been obtained in terms of $\cos(\psi)$ and $\sin(\psi)$ the derivatives
# \begin{align}
# \frac{\partial \xi}{\partial \zeta_I} &= \xi_{,I} =
# \left[\frac{\partial \xi}{\partial \psi}, \frac{\partial \xi}{\partial x^\mathrm{rot}_{1k}}\right]
# \end{align}
# where
# \begin{align}
# \frac{\partial \xi}{\partial \psi} &=
# \frac{\partial \xi}{\partial \cos(\psi)}
# \frac{\partial \cos(\theta)}{\partial \psi} +
# \frac{\partial \xi}{\partial \sin(\psi)}
# \frac{\partial \sin(\theta)}{\partial \psi} \\
# &= \frac{\partial \xi}{\partial B_c} \frac{\partial B_c}{\partial \psi}
# \end{align}

# In[24]:


dxi_dB = xi_solved_c.diff(B)
dB_dpsi = B_.diff(psi)
dB_dpsi_ = dB_dpsi.subs(
    {sp.cos(psi): c_psi, sp.sin(psi): s_psi})
dxi_psi = (dxi_dB.T * dB_dpsi)[0, 0]
dxi_x_rot_1k = xi_solved_c.diff(x_rot_1k)
get_dxi_psi = sp.lambdify(sp_vars + (c_psi, s_psi), dxi_psi)
get_dxi_x_rot_1k = sp.lambdify(sp_vars + (c_psi, s_psi), dxi_x_rot_1k)

# Derivatives of the left side vector $p_{,I}$ with respect to $\psi$
# \begin{align}
# p_{a,\psi} = p_{a,B_c} B_{c,\psi}
# \end{align}
# and with respect to $x^\mathrm{rot}_{1k}$
# \begin{align}
# p_{a,x_{1k}^\mathrm{rot}} =
# \left[
# \begin{array}{c}0 \\ -1 \end{array}
# \right]
# \end{align}

# In[25]:


dp_a_psi = sp.Matrix([p_a.T.diff(c) for c in B]).T * B_
dp_a_x_rot_1k = p_a.diff(x_rot_1k)
dp_aI_ = sp.Matrix([dp_a_psi.T, dp_a_x_rot_1k.T]).T

# In[26]:


get_dp_a_psi = sp.lambdify(sp_vars + (c_psi, s_psi), dp_a_psi)
get_dp_a_x_rot_1k = sp.lambdify(sp_vars + (c_psi, s_psi), dp_a_x_rot_1k)


def get_p_a_dI(*params):
    p_a = get_p_a(*params)
    dp_a_psi = get_dp_a_psi(*params)
    dp_a_x_rot_1k = get_dp_a_x_rot_1k(*params)
    return p_a[:, 0], np.c_[dp_a_psi, dp_a_x_rot_1k]


get_p_a_dI(*(params + (0, 1)))

# Derivatives of the right plate vector $q_{,I}$ with respect to $\theta$
# \begin{align}
# q_{a,psi} = q_{a,B_c} B_{c,psi} + q_{a,\xi} \xi_{,\psi}
# \end{align}
# and with respect to $x^\mathrm{rot}_{1k}$
# \begin{align}
# q_{a,x_{1k}^\mathrm{rot}} =
# \left[\begin{array}{c}0 \\ -1\end{array}\right] +
# q_{a,\xi} \xi_{,x^\mathrm{rot}_{1k}}
# \end{align}

# In[27]:

dq_a_psi_dir = sp.Matrix([q_xi_a.T.diff(c) for c in B]).T * dB_dpsi_
dq_a_xi = q_xi_a.diff(xi)
dq_a_x_rot_1k_dir = q_xi_a.diff(x_rot_1k)
dq_aI_dir_ = sp.Matrix([dq_a_psi_dir.T, dq_a_x_rot_1k_dir.T]).T

# In[28]:

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


get_q_a_dI(*(params + (0, 1)))

# ## Rotation angle and rotation matrix
#
# To abbreviate notation, let us denote vector connecting the center of rotation
# $x^\mathrm{rot}_{ak}$ and $x^\mathrm{tip}_{ak}$ as $p_{a}$ and the
# rotated
# vector $x^\mathrm{rot}_{ak}, v_{ak}^\mathrm{\varphi}$ as $q_{ak}$.

# the $\cos$ and $\sin$
# needed for the rotation matrix is then expressed as
# \begin{align}
# \| p \| &= \sqrt{p_{a} p_{a}} = \left( p_{a} p_{a} \right)^{\frac{1}{2}}
# \end{align}
# \begin{align}
# \| q \| &= \sqrt{q_{a} q_{a}} = \left( q_{a} q_{a} \right)^{\frac{1}{2}}
# \end{align}

# \begin{align}
# \cos(\theta) &=
# \frac
# {p_{a} q_{a}}
# {\| p \| \| q \| }
# \end{align}

# In[29]:


zeta = sp.Symbol('\zeta')

# Assuming that the vectors $p_a$ and $q_a$ both depend on a vector $\zeta_I$,
# let us denote the derivatives of both vectors with respect to these variables as
# \begin{align}
# \frac{\partial p_a}{\partial \zeta_I} &= p_{a,I}  \\
# \frac{\partial q_a}{\partial \zeta_I} &= q_{a,I}  \\
# \end{align}

# Derivative of their scalar product

# In[30]:


p_zeta = sp.Function('p')(zeta)
q_zeta = sp.Function('q')(zeta)
(p_zeta * q_zeta).diff(zeta)

# \begin{align}
# (pq)_{,I} = p_{a,I} q_a + p_a q_{a,I}
# \end{align}

# In[31]:


(sp.sqrt(p_zeta * p_zeta)).diff(zeta)

# Derivative of the vector norms
# \begin{align}
# \| p \|_{,I} &= \left[ \left( p_a p_a \right)^{\frac{1}{2}} \right]_{,I}
# \\
# &= \frac{1}{2}\left( p_a p_a \right)^{-\frac{1}{2}}   2 p_{a,I} p_a \\
# &= \frac{p_a}{\|p\|} p_{a,I} \\
# \| q \|_{,I} &= \frac{q_a}{\|q\|} q_{a,I}
# \end{align}

# Derivative of the norm product
# \begin{align}
# (\| p \| \| q \|)_{,I} &= \| p \|_{,I} \| q \| + \| p \| \| q \|_{,I}
# \end{align}

# Derivative of division of two functions of $\zeta_I$

# In[32]:


pq_zeta = sp.Function(r'pq')(zeta)
norm_pq_zeta = sp.Function(r'\|pq\|')(zeta)
(pq_zeta / norm_pq_zeta).diff(zeta)


# Thus, the derivative of the cosine can be obtained as
# \begin{align}
# \cos_{,I}\theta &=  \frac{1}{\| p \| \| q \| }
# \left[
# (pq)_{,I} -
# \frac{pq }
# {\| p \| \| q \|}
# (\| p \| \| q \|)_{,I}
# \right] \\
# &=  \frac{1}{\| p \| \| q \| }
# \left[
# (pq)_{,I} -
# \cos(\theta) \, (\| p \| \| q \|)_{,I}
# \right]
# \end{align}

# Since
# \begin{align}
# \sin \theta = \sqrt{ 1 - \cos^2{\theta}}
# \end{align}
# we can obtain
# \begin{align}
# \sin_{,I}\theta &=
# \frac{1}{2\sqrt{ 1 - \cos^2\theta }}
# \cdot (-2) \cos \theta \cdot \cos_{,I}\theta \\
# &= - \frac{\cos \theta}{\sin{\theta}}
#  \cdot \cos_{,I}\theta \\
# \end{align}

# In[33]:


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


# In[34]:


def get_cos_theta_dI2(*params):
    c_psi, s_psi = get_B(*params)
    params_B = params + (c_psi[0], s_psi[0])
    p_a, dp_aI = get_p_a_dI(*params_B)
    q_a, dq_aI = get_q_a_dI(*params_B)
    return get_cos_theta_dI(p_a, q_a, dp_aI, dq_aI)


# In[35]:


params = 0, 1, 0, 0, 0.1, 0.1, 0.1
get_cos_theta_dI2(*params)

# The primary kinematic variables are introduced as the vertical displacement on the left hand side $v$ and the rotation of the left section $\varphi$.

# In[36]:


c_theta, s_theta = sp.symbols(r'\cos(\theta), \sin(\theta)')
c_theta = sp.Function(r'cos')(theta)
s_theta = sp.Function(r'sin')(theta)

# ## Rotation matrix

# The rotation matrix is then defined as
# \begin{align}
# T_{ab}
# &=
# \left[
# \begin{array}{cc}
# \cos{\theta} & - \sin{\theta} \\
# \sin{\theta} & \cos{\theta}
# \end{array}
# \right]
# \end{align}

# In[37]:


T = sp.Matrix(
    [[c_theta, -s_theta],
     [s_theta, c_theta]], dtype=np.float_)
T

# \begin{align}
# T_{ab,I} =
# T_{ab,\cos(\theta)} \cos_{,I}(\theta) +
# T_{ab,\sin(\theta)} \sin_{,I}(\theta)
# \end{align}

# In[38]:


dT_dcs_theta = T.diff(c_theta), T.diff(s_theta)
dT_dcs_theta

# In[39]:


get_T = sp.lambdify((c_theta, s_theta), T, 'numpy')
get_dT_dcs = sp.lambdify((c_theta, s_theta), dT_dcs_theta, 'numpy')


# In[40]:


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


T_ab, dT_dI_abI = get_T_ab_dT_dI_abI(*params)

# ## Rotation operator

# Let $x^0_{Lb}$ represent the nodes on the line of the ligament running through the cross section with $L$ segments. This line is fixed on the left-hand side. The only deformation assumed to happen within the studied shear zone is the rotation around the center of rotation $x^\mathrm{rot}_a$.
# The rotated position of a point $x^1_{Lb}$ on the ligmanet $L$ is obtained
# \begin{align}
# x^1_{Lb} = T_{ab} (x^0_{La} - x^{\mathrm{rot}}_a ) + x^{\mathrm{rot}}_a
# \end{align}
# and the derivatives
# \begin{align}
# x^1_{Lb,I} =
# T_{ab,I} (x^0_{La} - x^{\mathrm{rot}}_a ) +
# T_{ab} (x^0_{La,I} - x^{\mathrm{rot}}_{a,I} )
# +
# x^{\mathrm{rot}}_{a,I}
# \end{align}

# In[41]:


x_rot_0, x_rot_1 = sp.symbols(r'x^{\mathrm{rot}}_0, x^{\mathrm{rot}}_1')
x_rot = sp.Matrix([[x_rot_0], [x_rot_1]])
x_L_0, x_L_1 = sp.symbols(r'x^{\mathrm{L}}_0, x^{\mathrm{L}}_1')
x_L = sp.Matrix([[x_L_0], [x_L_1]])

# In[42]:


x1_a = sp.simplify(T * (x_L - x_rot) + x_rot)
x1_a


# In[43]:

def get_x1_La(T_ab, x0_La, x_rot_a):
    x_rot_La = x_rot_a[np.newaxis, ...]
    return np.einsum('ab,Lb->La', T_ab, x0_La - x_rot_La) + x_rot_La

# In[44]:

_x0_La = np.array([[1, 0]])
_x_rot_a = np.array([0, 1])
params = 0, 1, 1, 0, 0.1, 0.1, 0.1
py_vars, params

# In[45]:


_T_ab, _ = get_T_ab_dT_dI_abI(*params)
_T_ab, get_x1_La(T_ab, _x0_La, _x_rot_a)

# In[46]:


_x0_La - _x_rot_a[np.newaxis, ...]

# In[47]:


np.einsum('ab,b->a', _T_ab, [1, -1])

# **TODO:** Define the chained derivatives for the rotated coordinate
# \begin{align}
# x^{1}_{a,\psi \zeta} = x^{1}_{,\zeta} + x^{1}_{a, c} c_{,\psi \zeta}
# \end{align}

# In[48]:


import traits.api as tr
from bmcs_utils.api import \
    InteractiveModel, InteractiveWindow, Item, View, Float


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



