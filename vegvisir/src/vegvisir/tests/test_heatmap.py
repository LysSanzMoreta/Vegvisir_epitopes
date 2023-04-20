import matplotlib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def NonLinCdict(steps, hexcol_array):
    cdict = {'red': (), 'green': (), 'blue': ()}
    for s, hexcol in zip(steps, hexcol_array):
        rgb =matplotlib.colors.hex2color(hexcol)
        cdict['red'] = cdict['red'] + ((s, rgb[0], rgb[0]),)
        cdict['green'] = cdict['green'] + ((s, rgb[1], rgb[1]),)
        cdict['blue'] = cdict['blue'] + ((s, rgb[2], rgb[2]),)
    return cdict

hexcol_array = ['#e5e5ff', '#acacdf', '#7272bf', '#39399f', '#000080']
steps = [0, 0.1, 0.5, 0.9, 1]

cdict = NonLinCdict(steps, hexcol_array)
#cm = matplotlib.colors.LinearSegmentedColormap('test', cdict)

cm = matplotlib.colors.LinearSegmentedColormap.from_list("test",["black","red","green","blue"])


data= np.array([[0,2,0],[1,0,0],[3,0,1]])

plt.figure()
sns.heatmap(
        vmin=0.0,
        vmax=3.0,
        data=data,
        cmap=cm,
        linewidths=0.75)

plt.show()