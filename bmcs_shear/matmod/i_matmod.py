

import numpy as np
import sympy as sp
import traits.api as tr
from bmcs_utils.api import InteractiveModel, View, Item, Float


class IMaterialModel(tr.Interface):

    get_sig_w = tr.Property(depends_on='+param')
    get_tau_s = tr.Property(depends_on='+param')
