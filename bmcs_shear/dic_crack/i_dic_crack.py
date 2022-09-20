import traits.api as tr
import numpy as np

from matplotlib.animation import FuncAnimation

class IDICCrack(tr.Interface):
    """DIC Crack interface
    """
    x_Na = tr.Array(np.float)
    u_Na = tr.Array(np.float)
    u_Nb = tr.Array(np.float)

