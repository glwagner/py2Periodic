import numpy as np
import matplotlib.pyplot as plt

class plotter(object):
    def __init__(self, qg, name='Plot'):

        self.qg = qg
        self.name = name

        self.q1 = qg.q1
        self.q2 = qg.q2
        self.c1 = qg.c1
        self.c2 = qg.c2

        self.initialize_plot()

    def initialize_plot(self):

        qg = self.qg

        # Initialize plot
        self.fig, self.axArr = plt.subplots(nrows=2, ncols=2,
            sharex=True, sharey=True, figsize=(8, 8));

        self.fig.canvas.set_window_title(self.name)
        
        self.fig.subplots_adjust(hspace=0.1, wspace=0.1, 
            right=0.80, top=0.92, bottom=0.15)

        self.axArr[0, 0].set_ylabel("$y$ (km)", labelpad=12.0)
        self.axArr[1, 0].set_ylabel("$y$ (km)", labelpad=12.0)

        self.axArr[1, 0].set_xlabel("$x$ (km)", labelpad=5.0)
        self.axArr[1, 1].set_xlabel("$x$ (km)", labelpad=5.0)


    def make_plot(self):

        axArr = self.axArr
        qg = self.qg

        Ro = 2e-1
        qmin, qmax = -Ro, Ro
        cmin, cmax = 1e-2*qg.c1.max(), qg.c1.max()
        xkm, ykm = qg.x*1e-3, qg.y*1e-3

        for ax in axArr.flat: ax.cla()

        axArr[0, 1].set_xticklabels([], visible=False)
        axArr[0, 1].set_yticklabels([], visible=False)

        self.axArr[0, 0].set_ylabel("$y$ (km)", labelpad=12.0)
        self.axArr[1, 0].set_ylabel("$y$ (km)", labelpad=12.0)

        self.axArr[1, 0].set_xlabel("$x$ (km)", labelpad=5.0)
        self.axArr[1, 1].set_xlabel("$x$ (km)", labelpad=5.0)

        quadMesh = \
        axArr[0, 0].pcolormesh(xkm, ykm, qg.q1/qg.f0, vmin=qmin, vmax=qmax, 
            cmap='RdBu_r')
        axArr[0, 1].pcolormesh(xkm, ykm, qg.q2/(qg.f0*qg.delta), vmin=qmin, vmax=qmax,
            cmap='RdBu_r')

        axArr[1, 0].pcolormesh(xkm, ykm, qg.c1, vmin=cmin, vmax=cmax)
        axArr[1, 1].pcolormesh(xkm, ykm, qg.c2/qg.delta, vmin=cmin, vmax=cmax)

        box = axArr[0, 1].get_position()
        pad, width = 0.02, 0.03

        cax = self.fig.add_axes([box.xmax + pad, box.ymin, width, box.height]) 
        self.fig.colorbar(quadMesh, cax=cax)
