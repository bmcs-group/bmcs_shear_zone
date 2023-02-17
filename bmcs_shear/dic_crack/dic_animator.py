

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

    n_T = bu.Int(10, ALG=True)

    T_range = tr.Property(depends_on='state_changed')
    @tr.cached_property
    def _get_T_range(self):
        return np.hstack([
            np.linspace(0, 1, self.n_T),
            np.ones((int(0.5 * self.n_T),))
        ])

    def subplots(self, fig):
        gs = gridspec.GridSpec(ncols=2, nrows=1,
                               width_ratios=[3, 1],
                               # wspace=0.5,
                               hspace=0.5,
                               # height_ratios=[2, 1]
                               )
        ax_dcl = fig.add_subplot(gs[0, 0])
        ax_FU = fig.add_subplot(gs[0, 1])
        self.fig = fig
        #        return fig.subplots(1,1)
        #        return ax_dsf#, ax_FU
        return ax_dcl, ax_FU

    def plot(self, i):
        self.fig.clear()
        t = self.t_dic_T[i]
        print('t', t)
        axes = self.subplots(self.fig)
        self.model.dsf.dic_grid.t = t

        ax_dcl, ax_FU = axes

        self.model.bd.plot_sz_bd(ax_dcl)
        self.model.dsf.plot_crack_detection_field(ax_dcl, self.fig)
        self.model.plot_primary_cracks(ax_dcl)
        self.model.critical_crack.plot_X_crc_t_Ka(ax_dcl, line_width=2, line_color='red', tip_color='red')
        for crack in self.model.cracks:
            crack.cor.trait_set(cor_marker_size=8, cor_marker_color='brown')
            crack.cor.plot_X_cor_t(ax_dcl)
        ax_dcl.axis('equal')
        ax_dcl.axis('off');
        self.model.dsf.dic_grid.plot_load_deflection(ax_FU)

    def mp4_video(self):
        # call the animator. blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(self.fig, self.plot, init_func=self.init,
                                       frames=self.n_T, interval=300, blit=True)
        return anim.save("cracking_animation.gif")

    def html5_video(self):
        # call the animator. blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(self.fig, self.plot, init_func=self.init,
                                       frames=self.n_T, interval=300, blit=True)
        return anim.to_html5_video()

