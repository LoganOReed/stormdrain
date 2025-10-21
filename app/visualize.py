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
from .network import SubcatchmentGraph, SewerGraph, StreetGraph

# https://stackoverflow.com/questions/76752021/producing-a-gif-of-plot-over-time-python
def visualizeExample( subcatchments, street, sewer, times, file="visualizeExample", cmap=plt.cm.plasma, fps=5):
    """Creates a gif from igraph, t0, T, stepsize. weights are a function of time"""

    frames = []
    for t in times:
        fig = createFrame(subcatchments, street, sewer, t, cmap=cmap)
        fig.canvas.draw()
        frames.append(np.array(fig.canvas.renderer._renderer))
        # Save GIF
        iio.mimsave(f'figures/{file}.gif',
                        frames,
                        fps=fps)
        

def createFrame(subcatchments, street, sewer, t, cmap=plt.cm.plasma):
    """Creates t-th frame of graph g and returns figure."""
    # Convert to networkx for visualizing
    aSubcatchments = np.array(subcatchments.G.get_adjacency())
    aStreet = np.array(street.G.get_adjacency())
    aSewer = np.array(sewer.G.get_adjacency())
    nxSubcatchments = nx.from_numpy_array(aSubcatchments, create_using=nx.DiGraph())
    nxStreet = nx.from_numpy_array(aStreet, create_using=nx.DiGraph())
    nxSewer = nx.from_numpy_array(aSewer, create_using=nx.DiGraph())

    # Create graph from multiple graphs, where they are labeled 0, size(subcatch) -1 then size(subcatchments), size(subcatch)+size(street)-1
    # g = nx.disjoint_union(nxSubcatchments, nxStreet)
    g = nxStreet

    # TODO: Add subcatchment and shift ids by size(subcatch)
    ids = [i for i in range(subcatchments.G.vcount() + street.G.vcount())]
    coordsSubcatchment = [list(coord) for coord in zip(subcatchments.G.vs["x"],subcatchments.G.vs["y"])]
    coordsStreet = [list(coord) for coord in zip(street.G.vs["x"],street.G.vs["y"])]
    coords = coordsSubcatchment + coordsStreet
    pprint(coords)
    layout = dict(zip(ids, coords))
    pprint(layout)



    # Create edges from networks
    # This is already in nxName

    # TODO: Create edges from coupling


    # Create layout from edges

    # Create weights from igraph attrs

    # create time dependent weights
    M = g.number_of_edges()
    weights = [(w + t) % M for w in range(2,M+2) ]
    # TODO: Make cmap stuff normalize over all 3 graphs
    norm = plt.Normalize(vmin=min(weights),vmax=max(weights))

    edge_colors = [cmap(norm(w)) for w in weights]

    pprint(f"street edgelist: {street.G.get_edgelist()}")
    newStreetEdgeList = [tuple(map(lambda x: x + subcatchments.G.vcount(), t)) for t in street.G.get_edgelist()]
    pprint(f"Adjusted Edgelist: {newStreetEdgeList}")

    pprint(min(edge_colors))
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
    # fig, ax = plt.subplots()
    # subcatchments
    nx.draw_networkx_nodes(g, layout, nodelist=[i for i in range(subcatchments.G.vcount())], node_color="black", node_shape='s', ax=ax[0,0])
    nx.draw_networkx_edges(g, layout, 
                           edgelist=subcatchments.G.get_edgelist(),
                           style='dashed',
                           arrows=False,
                           ax=ax[0,0])
    # junctions
    streetNodeList = [i for i in range(subcatchments.G.vcount(),subcatchments.G.vcount() + street.G.vcount())]
    nx.draw_networkx_nodes(g, layout, nodelist=[item for item, condition in zip(streetNodeList, street.G.vs["type"]) if condition == 0], node_color="indigo", node_shape='o', ax=ax[0,0])
    nx.draw_networkx_nodes(g, layout, nodelist=[item for item, condition in zip(streetNodeList, street.G.vs["type"]) if condition == 1], node_color="black", node_shape='^', ax=ax[0,0])
    edges = nx.draw_networkx_edges(g, layout, 
                           edgelist=newStreetEdgeList,
                           arrowstyle="->",
                           arrowsize=10,
                           edge_color=edge_colors,
                           edge_cmap=cmap,
                           width=2,
                           ax=ax[0,0])
    # outflow
    # nx.draw_networkx_nodes(g, layout, nodelist=[7], node_color="black", node_shape='^')



    # create legend for color
    pc = matplotlib.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)
    plt.colorbar(pc, ax=ax)
    ax[0,0].set_axis_off()
    ax[1,0].set_axis_off()
    # plt.savefig(f"test.png")
    plt.show()
    return fig

if __name__ == "__main__":
    # g = ig.Graph(n=8,edges=[[0,3],[1,4],[2,5],[3,6],[4,5],[5,6],[6,7]],directed=True)
    file = "largerExample"


    subcatchment = SubcatchmentGraph(file)
    # pprint(subcatchment.hydraulicCoupling)
    # pprint(subcatchment.G.vs['coupledID'])
    # pprint(subcatchment.update(2,0.5,rainfall[3]))
    street = StreetGraph(file)
    pprint(street.G.summary())
    sewer = SewerGraph(file)


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

    visualizeExample(subcatchment, street, sewer, times, cmap=plt.cm.plasma )
