import imageio as iio
import igraph as ig
import networkx as nx
import numpy as np
from sys import platform
import matplotlib
if platform == "linux":
    matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
from pprint import pprint

# https://stackoverflow.com/questions/76752021/producing-a-gif-of-plot-over-time-python
def visualizeExample( g, layout, times, file=None, cmap=None):
    """Creates a gif from igraph, t0, T, stepsize. weights are a function of time"""

    frames = []
    for t in times:
        fig = createFrame(g, layout, t, cmap=cmap)
        fig.canvas.draw()
        frames.append(np.array(fig.canvas.renderer._renderer))
        # Save GIF
        if file == None:
            file = "visualizeExample"
        iio.mimsave(f'figures/{file}.gif',
                        frames,
                        fps=5)
        

def createFrame(g, layout, t, cmap=None):
    """Creates t-th frame of graph g and returns figure."""
    # Convert to networkx for visualizing
    a = np.array(g.get_adjacency())
    g = nx.from_numpy_array(a, create_using=nx.DiGraph())

    # Default cmap
    if cmap == None:
        cmap = plt.cm.plasma

    # create time dependent weights
    M = g.number_of_edges()
    weights = [(w + t) % M for w in range(2,M+2) ]
    norm = plt.Normalize(vmin=min(weights),vmax=max(weights))

    edge_colors = [cmap(norm(w)) for w in weights]


    pprint(layout)
    pprint(min(edge_colors))
    fig, ax = plt.subplots()
    # subcatchments
    nx.draw_networkx_nodes(g, layout, nodelist=[0,1,2], node_color="black", node_shape='s')
    nx.draw_networkx_edges(g, layout, edgelist=[(0,3),(1,4),(2,5)],style='dashed',arrows=False)
    # junctions
    nx.draw_networkx_nodes(g, layout, nodelist=[3,4,5,6], node_color="indigo", node_shape='o')
    edges = nx.draw_networkx_edges(g, layout, 
                           edgelist=[(3,6),(4,5),(5,6),(6,7)],
                           arrowstyle="->",
                           arrowsize=10,
                           edge_color=edge_colors,
                           edge_cmap=cmap,
                           width=2)
    # outflow
    nx.draw_networkx_nodes(g, layout, nodelist=[7], node_color="black", node_shape='^')

    # create legend for color
    pc = matplotlib.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)
    plt.colorbar(pc, ax=ax)
    ax.set_axis_off()
    # plt.savefig(f"test.png")
    plt.show()
    return fig

if __name__ == "__main__":
    g = ig.Graph(n=8,edges=[[0,3],[1,4],[2,5],[3,6],[4,5],[5,6],[6,7]],directed=True)
    t0 = 1
    T = 20
    stepsize = 1
    times = [t for t in range(t0, T, stepsize)]

    # Layout for example
    layout = {
            0: [0.5,2.5],
            1: [1.5,2.5],
            2: [1.5,0.5],
            3: [1.0,2.0],
            4: [2.0,2.0],
            5: [2.0,1.0],
            6: [1.0,1.0],
            7: [0.0,1.0]
        }

    visualizeExample(g, layout, times, cmap=plt.cm.plasma )
