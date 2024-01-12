import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns;sns.set()
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize,BoundaryNorm
import matplotlib

class SeabornFig2Grid():
    """Class from https://stackoverflow.com/questions/47535866/how-to-iteratively-populate-matplotlib-gridspec-with-a-multipart-seaborn-plot/47624348#47624348"""

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))

        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())



iris = sns.load_dataset("iris")
tips = sns.load_dataset("tips")

# A JointGrid
g0 = sns.jointplot(x="sepal_width", y="petal_length", data=iris,
                   kind="kde",
                   #space=0, color="g"
                   )

g1 = sns.FacetGrid(tips,  hue="smoker",subplot_kws = {"fc":"white"},margin_titles=True)
g1.set(yticks=[])
g1.set(xticks=[])
g1_axes = g1.axes.flatten()
g1_axes[0].set_title("Internal")
g1.map(plt.scatter, "total_bill", "tip", edgecolor="w")
g1_axes[0].set_xlabel("X")
g1_axes[0].set_ylabel("Y")

g2 = sns.FacetGrid(tips,  hue="smoker")
g2_axes = g2.axes.flatten()
#divider2 = make_axes_locatable(g2_axes[0])
g2.map(plt.scatter, "total_bill", "tip", edgecolor="w")

g3 = sns.FacetGrid(tips,  hue="smoker")
g3.map(plt.scatter, "total_bill", "tip", edgecolor="w")


g4 = sns.FacetGrid(tips,  hue="smoker")
g4.map(plt.scatter, "total_bill", "tip", edgecolor="w")


fig = plt.figure(figsize=(17,8))
#fig.set_size_inches(8.27, 11.69, forward=True)
gs = gridspec.GridSpec(2, 6,width_ratios=[2,1,0.1,0.07,1,0.1])

mg0 = SeabornFig2Grid(g0, fig, gs[0:2,0])
mg1 = SeabornFig2Grid(g1, fig, gs[0,1])
#mg2 = SeabornFig2Grid(g2, fig, gs[0,4])
mg3 = SeabornFig2Grid(g3, fig, gs[1,1])
mg4 = SeabornFig2Grid(g4, fig, gs[1,4])

gs.update(top=0.9)
#gs.update(right=0.4)

#Following: https://www.sc.eso.org/~bdias/pycoffee/codes/20160407/gridspec_demo.html
cbax2 = plt.subplot(gs[0,5]) # Place it where it should be.
cbax3 = plt.subplot(gs[1,2]) # Place it where it should be.
cbax4 = plt.subplot(gs[1,5]) # Place it where it should be.


#cb2 = Colorbar(ax = cbax2, mappable = plt.cm.ScalarMappable(cmap="viridis"))
#cb2.set_ticks([0,1],labels=alleles_ticks)

alleles_ticks = ["One","Two","Three","Four","Five"]
unique_alleles = [0,1,2,3,4] #this is a list, that is why it works latter
colormap_unique = matplotlib.cm.get_cmap("viridis", len(unique_alleles))
cb2 = Colorbar(ax=cbax2,
               mappable=plt.cm.ScalarMappable(norm=BoundaryNorm(unique_alleles + [max(unique_alleles) +1], len(unique_alleles)),cmap=colormap_unique),
               boundaries=unique_alleles)
# cb2.ax.xaxis.set_ticks_position('none')
#cb2.ax.set_xticks([])
#cb2.ax.tick_params(size=0) #set size of the ticks to 0
ticks_locations = np.arange(0.5,len(unique_alleles),1)
#n_clusters = len(unique_alleles)
#ticks_locations = (np.arange(n_clusters) + 0.5)*(n_clusters-1)/n_clusters
cb2.set_ticks(ticks_locations)


cb2.set_ticklabels(alleles_ticks)
cb2.ax.tick_params(size=0) #set size of the ticks to 0



cb3 = Colorbar(ax = cbax3, mappable = plt.cm.ScalarMappable(cmap="viridis"))
cb4 = Colorbar(ax = cbax4, mappable = plt.cm.ScalarMappable(cmap="viridis"))
fig.suptitle("UMAP latent space (z) projections")
plt.savefig("plot_join")