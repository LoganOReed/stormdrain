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

import matplotlib.ticker as mticker

def seconds_to_hms(x, pos):
    """
    Converts a time in seconds to a string in HH:MM format.
    """
    hours = int(x // 3600)
    minutes = int((x % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}"

# https://stackoverflow.com/questions/76752021/producing-a-gif-of-plot-over-time-python
def visualize( subcatchment, street, streetYFull, sewer, sewerYFull, subcatchmentDepths, runoffs, streetDepths, streetEdgeAreas, sewerDepths, sewerEdgeAreas, drainOverflows, drainInflows, times, rainfall, peakDischarges, file="visualizeExample", cmap=plt.cm.plasma, fps=5):
    """Creates a gif from igraph, t0, T, stepsize. weights are a function of time"""

    frames = []
    for it in range(len(times)):
        fig = createFrame(subcatchment, street, streetYFull, sewer, sewerYFull, subcatchmentDepths, runoffs, streetDepths, streetEdgeAreas, sewerDepths, sewerEdgeAreas, drainOverflows, drainInflows, it, times, rainfall, peakDischarges, cmap=cmap)
        fig.canvas.draw()
        frames.append(np.array(fig.canvas.renderer._renderer))
        # Save GIF
        iio.mimsave(f'figures/{file}.gif',
                        frames,
                        fps=fps)
        

def createFrame(subcatchment, street, streetYFull, sewer, sewerYFull, subcatchmentDepths, runoffs, streetDepths, streetEdgeAreas, sewerDepths, sewerEdgeAreas, drainOverflows, drainInflows, it, times, rainfall, peakDischarges, cmap=plt.cm.plasma):
    """Creates t-th frame of graph g and returns figure."""
    # Convert to networkx for visualizing
    aSubcatchments = np.array(subcatchment.G.get_adjacency())
    aStreet = np.array(street.G.get_adjacency())
    aSewer = np.array(sewer.G.get_adjacency())
    nxSubcatchments = nx.from_numpy_array(aSubcatchments, create_using=nx.DiGraph())
    nxStreet = nx.from_numpy_array(aStreet, create_using=nx.DiGraph())
    nxSewer = nx.from_numpy_array(aSewer, create_using=nx.DiGraph())

    # Create graph from multiple graphs, where they are labeled 0, size(subcatch) -1 then size(subcatchments), size(subcatch)+size(street)-1
    # g = nx.disjoint_union(nxSubcatchments, nxStreet)
    g = nxStreet

    # TODO: Add subcatchment and shift ids by size(subcatch)
    ids = [i for i in range(subcatchment.G.vcount() + street.G.vcount())]
    coordsSubcatchment = [list(coord) for coord in zip(subcatchment.G.vs["x"],subcatchment.G.vs["y"])]
    coordsStreet = [list(coord) for coord in zip(street.G.vs["x"],street.G.vs["y"])]
    coords = coordsSubcatchment + coordsStreet
    # pprint(coords)
    layout = dict(zip(ids, coords))
    # pprint(layout)

    # sewer layout
    idsSewer = [i for i in range(sewer.G.vcount())]
    coordsSewer = [list(coord) for coord in zip(sewer.G.vs["x"],sewer.G.vs["y"])]
    layoutSewer = dict(zip(idsSewer, coordsSewer))




    # Create edges from networks
    # This is already in nxName

    # Create layout from edges

    # Create weights from igraph attrs

    # create time dependent weights
    # weights = [(w + times[it]) % M for w in range(2,M+2) ]
    # TODO: Make cmap stuff normalize over all 3 graphs
    norm = plt.Normalize(vmin=0,vmax=max(streetYFull,0))
    normRunoff = plt.Normalize(vmin=0,vmax=max(0.05,0))
    # TODO: Choose this in a better way
    # norm = plt.Normalize(vmin=0,vmax=max(streetYFull,sewerYFull))

    # edge_colors = [cmap(norm(w)) for w in weights]
    # NOTE: This is very hacky but I dont want to chase down the bug
    if it >= len(subcatchmentDepths):
        it = it - 1
    subcatchmentNodesColors = [cmap(norm(w)) for w in subcatchmentDepths[it]]
    runoffsColors = [cmap(normRunoff(w)) for w in runoffs[it]]
    streetNodesColors = [cmap(norm(w)) for w in streetDepths[it]]
    streetEdgeColors = [cmap(norm(w)) for w in streetEdgeAreas[it]]
    sewerNodesColors = [cmap(norm(w)) for w in sewerDepths[it]]
    sewerEdgeColors = [cmap(norm(w)) for w in sewerEdgeAreas[it]]

    # pprint(f"street edgelist: {street.G.get_edgelist()}")
    newStreetEdgeList = [tuple(map(lambda x: x + subcatchment.G.vcount(), t)) for t in street.G.get_edgelist()]
    # pprint(f"Adjusted Edgelist: {newStreetEdgeList}")

    # pprint(min(edge_colors))
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
    # fig, ax = plt.subplots()
    # subcatchments
    nx.draw_networkx_nodes(g, layout, nodelist=[i for i in range(subcatchment.G.vcount())], 
                           node_color=subcatchmentNodesColors, 
                           # node_cmap=cmap,
                           node_shape='s', 
                           ax=ax[0,0])
    nx.draw_networkx_edges(g, layout, 
                           edgelist=subcatchment.G.get_edgelist(),
                           arrowstyle="->",
                           arrowsize=10,
                           edge_color=runoffsColors,
                           edge_cmap=cmap,
                           width=2,
                           ax=ax[0,0])
    # junctions
    streetNodeList = [i for i in range(subcatchment.G.vcount(),subcatchment.G.vcount() + street.G.vcount())]
    nx.draw_networkx_nodes(g, layout,
                           nodelist=[item for item, condition in zip(streetNodeList, street.G.vs["type"]) if condition == 0],
                           node_color=[item for item, condition in zip(streetNodesColors, street.G.vs["type"]) if condition == 0], 
                           # node_cmap=cmap,
                           node_shape='o',
                           ax=ax[0,0])
    nx.draw_networkx_nodes(g, layout, nodelist=[item for item, condition in zip(streetNodeList, street.G.vs["type"]) if condition == 1], node_color="black", node_shape='^', ax=ax[0,0])
    edges = nx.draw_networkx_edges(g, layout, 
                           edgelist=newStreetEdgeList,
                           arrowstyle="->",
                           arrowsize=10,
                           edge_color=streetEdgeColors,
                           edge_cmap=cmap,
                           width=2,
                           ax=ax[0,0])

    # Coupled Edges
    # [tuple(map(lambda x: x + subcatchment.G.vcount(), t)) for t in street.G.get_edgelist()]
    subcatchmentCoupling = [i for i in range(subcatchment.G.vcount())]
    streetCoupling = [street.G.vs["coupledID"].index(i) for i in subcatchment.hydraulicCoupling]
    couplingEdges = [coord for coord in zip(subcatchmentCoupling,streetCoupling)]
    couplingEdges = [(item[0], item[1] + subcatchment.G.vcount()) for item in couplingEdges]
    nx.draw_networkx_edges(g, layout, 
                           edgelist=couplingEdges,
                           style='dashed',
                           edge_color=runoffsColors,
                           edge_cmap=cmap,
                           arrows=False,
                           ax=ax[0,0])

    # sewer
    nx.draw_networkx_nodes(nxSewer, 
                           layoutSewer,
                           nodelist=[item for item, condition in zip(idsSewer, sewer.G.vs["type"]) if condition == 0],
                           # TODO: Update these
                           node_color=[item for item, condition in zip(sewerNodesColors, sewer.G.vs["type"]) if condition == 0], 
                           node_shape='o', 
                           ax=ax[1,0])
    nx.draw_networkx_nodes(nxSewer, layoutSewer, nodelist=[item for item, condition in zip(idsSewer, sewer.G.vs["type"]) if condition == 1], node_color="black", node_shape='^', ax=ax[1,0])
    edges = nx.draw_networkx_edges(nxSewer, layoutSewer, 
                           edgelist=sewer.G.get_edgelist(),
                           arrowstyle="->",
                           arrowsize=10,
                           edge_color=sewerEdgeColors,
                           edge_cmap=cmap,
                           width=2,
                           ax=ax[1,0])


    # Plot cumulative rainfall
    cumulative_rainfall = np.cumsum(rainfall[:it+1])
    time_hours = times[:it+1]

    ax[0,1].clear()
    ax[0,1].plot(time_hours, cumulative_rainfall, 
                 marker='o', linewidth=2, markersize=6, color='steelblue')
    ax[0,1].set_xlabel('Time ', fontsize=10)
    ax[0,1].set_ylabel('Cumulative Rainfall (meters)', fontsize=10)
    ax[0,1].set_title('Cumulative Rainfall Over Time', fontsize=12, fontweight='bold')
    ax[0,1].grid(True, alpha=0.3)

    # Set fixed axis limits so the plot doesn't rescale every frame
    ax[0,1].set_xlim(times[0], times[-1])
    ax[0,1].set_ylim(0, np.sum(rainfall) * 1.1)


    # Plot Peak Discharges
    time_hours = times[:it+1]
    current_peak_discharges = peakDischarges[:it+1]

    ax[1,1].clear()
    ax[1,1].plot(time_hours, current_peak_discharges, 
                 marker='o', linewidth=2, markersize=6, color='crimson')
    ax[1,1].set_xlabel('Time ', fontsize=10)
    ax[1,1].set_ylabel('Peak Discharge (m^3/s)', fontsize=10)
    ax[1,1].set_title('Peak Discharge Over Time', fontsize=12, fontweight='bold')
    ax[1,1].grid(True, alpha=0.3)

    # Set fixed axis limits so the plot doesn't rescale every frame
    ax[1,1].set_xlim(times[0], times[-1])
    ax[1,1].set_ylim(0, max(abs(x) for x in peakDischarges) * 1.1)

    formatter = mticker.FuncFormatter(seconds_to_hms)
    ax[0,1].xaxis.set_major_formatter(formatter)
    ax[1,1].xaxis.set_major_formatter(formatter)

    # create legend for color
    pc = matplotlib.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(streetNodesColors)
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
    street = StreetGraph(file)
    # pprint(street.G.summary())
    sewer = SewerGraph(file)


    t0 = 1
    T = 20
    stepsize = 1
    times = [t for t in range(t0, T, stepsize)]
    runoff = []
    drainOverflow = []
    drainInflow = []

    # Layout for example
    visualizeExample(subcatchment, street, sewer, runoff, drainOverflow, drainInflow, times, rainfall, peakDischarges, cmap=plt.cm.plasma )

