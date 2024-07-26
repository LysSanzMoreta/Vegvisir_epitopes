import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colorbar import Colorbar

# Create a custom LinearSegmentedColormap
#colors = ['#FF0000', '#00FF00', '#0000FF']  # Replace with your desired colors
colors = ["#fafa6e","#ffd151","#ffbc49","#ffa745","#62006d"]
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
matplotlib.cm.register_cmap("custom_cmap", custom_cmap)
cpalette = sns.color_palette("custom_cmap", n_colors=len(colors), desat=0)


# Generate some example data
np.random.seed(42)
data = sns.load_dataset("iris")

# Create a FacetGrid using the custom colormap
g = sns.FacetGrid(data, col="species", hue="species", palette=cpalette, height=4)

# Map the data onto the grid
scatter_plot = g.map(plt.scatter, "sepal_length", "sepal_width", alpha=0.7)

# Add a legend
g.add_legend()

# Add a colorbar
cbar_ax = g.fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position as needed
cbar = Colorbar(ax=cbar_ax,cmap=custom_cmap)

# Show the plot
plt.show()