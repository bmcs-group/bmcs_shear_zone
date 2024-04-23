

from matplotlib import animation, rc
from IPython.display import HTML
import traits.api as tr
import matplotlib.gridspec as gridspec
import numpy as np
import bmcs_utils.api as bu

class DICAnimator(bu.Model):
    """
    Animate the history
    """

    model = bu.Instance(bu.Model, ALG=True)

    n_T = bu.Int(20, ALG=True)

    def init(self):
        n_T = self.n_T
        self.t_dic_T = np.hstack([
            np.linspace(0,1.5,n_T),
        ])


    def subplots(self, fig):
        gs = gridspec.GridSpec(ncols=2, nrows=1,
                               width_ratios=[5, 1],
                               # wspace=0.5,
                               # hspace=0.5,
                               # height_ratios=[2, 1]
                               )
        ax_dcl = fig.add_subplot(gs[0, 0])
        ax_FU = fig.add_subplot(gs[0, 1])
        self.fig = fig
        return ax_dcl, ax_FU

    def plot(self, i):
        self.fig.clear()
        t = self.t_dic_T[i]
        print('t', t)
        axes = self.subplots(self.fig)
        dcl = self.model
        dcl.dsf.dic_grid.t = t

        ax_dcl, ax_FU = axes

        dcl.dsf.plot_sig_compression_field(ax_dcl)
        dcl.dsf.plot_dY_t_IJ(ax_dcl, t)
        dcl.plot_primary_cracks(ax_cracks=ax_dcl)
        dcl.plot_crack_roots(ax_dcl)

        ax_dcl.axis('equal')
        ax_dcl.axis('off');
        dcl.dsf.dic_grid.plot_load_deflection(ax_FU)

    def mp4_video(self):
        # call the animator. blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(self.fig, self.plot, init_func=self.init,
                                       frames=self.n_T, interval=500, blit=True)
        return anim.save("cracking_animation.gif")

    def html5_video(self):
        # call the animator. blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(self.fig, self.plot, init_func=self.init,
                                       frames=self.n_T, interval=300, blit=True)
        return anim.to_html5_video()

