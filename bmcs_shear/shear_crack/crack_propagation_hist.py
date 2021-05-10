import bmcs_utils.api as bu
import traits.api as tr
from bmcs_utils.api import mpl_align_yaxis
import numpy as np


class CrackPropagationHist(bu.InteractiveModel):
    name = 'Crack propagation history'

    crack_prop_model = tr.WeakRef(bu.Model)

    t_slider = bu.Float(0)
    t_max = tr.DelegatesTo(crack_prop_model)

    ipw_view = bu.View(
        time_editor=bu.HistoryEditor(var='t_slider', low=0, high_name='t_max', n_steps=50),
    )

