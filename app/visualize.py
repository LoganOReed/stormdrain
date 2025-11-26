import imageio as iio
import igraph as ig
import networkx as nx
import numpy as np
from sys import platform
import matplotlib

if platform == "linux":
    matplotlib.use("module://matplotlib-backend-kitty")
import matplotlib.pyplot as plt
from pprint import pprint

import matplotlib.ticker as mticker


def seconds_to_hms(x, pos):
    """
    Converts a time in seconds to a string in HH:MM format.
    """
    hours = int(x // 3600)
    minutes = int((x % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}"


# https://stackoverflow.com/questions/76752021/producing-a-gif-of-plot-over-time-python
def visualize(
    subcatchment,
    street,
    streetYFull,
    sewer,
    sewerYFull,
    subcatchmentDepths,
    runoffs,
    streetDepths,
    streetEdgeAreas,
    sewerDepths,
    sewerEdgeAreas,
    drainOverflows,
    drainInflows,
    times,
    rainfall,
    peakDischarges,
    dt,
    file="visualizeExample",
    cmap=plt.cm.plasma,
    fps=5,
):
    """Creates a gif from igraph, t0, T, stepsize. weights are a function of time"""

    frames = []
    pprint(f"times length: {len(times)} and times: {times}")
    pprint(f"runoffs length: {len(runoffs)} and times: {len(runoffs)}")
    pprint(
        f"subcatchment length: {len(subcatchmentDepths)} and times: {len(subcatchmentDepths)}"
    )
    if (
        len(runoffs) != len(subcatchmentDepths)
        or len(subcatchmentDepths) != len(streetDepths)
        or len(streetDepths) != len(sewerDepths)
        or len(sewerDepths) != len(peakDischarges)
    ):
        pprint(f"Error: one of the calculated datas are the wrong size!")

    # NOTE: I use runoffs arbitrarily, all of the measurable data should be the same size, otherwise the previous error will be thrown.
    numIters = len(runoffs)
    # resizedTimes = np.linspace(0,max(times),numIters)
    # rainfall = np.interp(resizedTimes, times, rainfall)
    # times = resizedTimes
    pprint(f"times length: {len(times)} and times: {times}")
    pprint(f"rainfall length: {len(rainfall)} and rainfall: {rainfall}")
    for it in range(len(runoffs)):
        fig = createFrame(
            subcatchment,
            street,
            streetYFull,
            sewer,
            sewerYFull,
            subcatchmentDepths,
            runoffs,
            streetDepths,
            streetEdgeAreas,
            sewerDepths,
            sewerEdgeAreas,
            drainOverflows,
            drainInflows,
            it,
            times,
            rainfall,
            peakDischarges,
            dt,
            cmap=cmap,
        )
        fig.canvas.draw()
        frames.append(np.array(fig.canvas.renderer._renderer))
        # Save GIF
        iio.mimsave(f"figures/{file}.gif", frames, fps=fps)


def createFrame(
    subcatchment,
    street,
    streetYFull,
    sewer,
    sewerYFull,
    subcatchmentDepths,
    runoffs,
    streetDepths,
    streetEdgeAreas,
    sewerDepths,
    sewerEdgeAreas,
    drainOverflows,
    drainInflows,
    it,
    times,
    rainfall,
    peakDischarges,
    dt,
    cmap=plt.cm.plasma,
):
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
    coordsSubcatchment = [
        list(coord) for coord in zip(subcatchment.G.vs["x"], subcatchment.G.vs["y"])
    ]
    coordsStreet = [list(coord) for coord in zip(street.G.vs["x"], street.G.vs["y"])]
    coords = coordsSubcatchment + coordsStreet
    # pprint(coords)
    layout = dict(zip(ids, coords))
    # pprint(layout)

    # sewer layout
    idsSewer = [i for i in range(sewer.G.vcount())]
    coordsSewer = [list(coord) for coord in zip(sewer.G.vs["x"], sewer.G.vs["y"])]
    layoutSewer = dict(zip(idsSewer, coordsSewer))

    # Create edges from networks
    # This is already in nxName

    # Create layout from edges

    # Create weights from igraph attrs

    # create time dependent weights
    # weights = [(w + times[it]) % M for w in range(2,M+2) ]
    # TODO: Make cmap stuff normalize over all 3 graphs
    norm = plt.Normalize(vmin=0, vmax=max(streetYFull, 0))
    normRunoff = plt.Normalize(vmin=0, vmax=max(0.05, 0))
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
    newStreetEdgeList = [
        tuple(map(lambda x: x + subcatchment.G.vcount(), t))
        for t in street.G.get_edgelist()
    ]
    # pprint(f"Adjusted Edgelist: {newStreetEdgeList}")

    # pprint(min(edge_colors))
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    # fig, ax = plt.subplots()
    # subcatchments
    nx.draw_networkx_nodes(
        g,
        layout,
        nodelist=[i for i in range(subcatchment.G.vcount())],
        node_color=subcatchmentNodesColors,
        # node_cmap=cmap,
        node_shape="s",
        ax=ax[0, 0],
    )
    nx.draw_networkx_edges(
        g,
        layout,
        edgelist=subcatchment.G.get_edgelist(),
        arrowstyle="->",
        arrowsize=10,
        edge_color=runoffsColors,
        edge_cmap=cmap,
        width=2,
        ax=ax[0, 0],
    )
    # junctions
    streetNodeList = [
        i
        for i in range(
            subcatchment.G.vcount(), subcatchment.G.vcount() + street.G.vcount()
        )
    ]
    nx.draw_networkx_nodes(
        g,
        layout,
        nodelist=[
            item
            for item, condition in zip(streetNodeList, street.G.vs["type"])
            if condition == 0
        ],
        node_color=[
            item
            for item, condition in zip(streetNodesColors, street.G.vs["type"])
            if condition == 0
        ],
        # node_cmap=cmap,
        node_shape="o",
        ax=ax[0, 0],
    )
    nx.draw_networkx_nodes(
        g,
        layout,
        nodelist=[
            item
            for item, condition in zip(streetNodeList, street.G.vs["type"])
            if condition == 1
        ],
        node_color="black",
        node_shape="^",
        ax=ax[0, 0],
    )
    edges = nx.draw_networkx_edges(
        g,
        layout,
        edgelist=newStreetEdgeList,
        arrowstyle="->",
        arrowsize=10,
        edge_color=streetEdgeColors,
        edge_cmap=cmap,
        width=2,
        ax=ax[0, 0],
    )

    # Coupled Edges
    # [tuple(map(lambda x: x + subcatchment.G.vcount(), t)) for t in street.G.get_edgelist()]
    subcatchmentCoupling = [i for i in range(subcatchment.G.vcount())]
    streetCoupling = [
        street.G.vs["coupledID"].index(i) for i in subcatchment.hydraulicCoupling
    ]
    couplingEdges = [coord for coord in zip(subcatchmentCoupling, streetCoupling)]
    couplingEdges = [
        (item[0], item[1] + subcatchment.G.vcount()) for item in couplingEdges
    ]
    nx.draw_networkx_edges(
        g,
        layout,
        edgelist=couplingEdges,
        style="dashed",
        edge_color=runoffsColors,
        edge_cmap=cmap,
        arrows=False,
        ax=ax[0, 0],
    )

    # sewer
    nx.draw_networkx_nodes(
        nxSewer,
        layoutSewer,
        nodelist=[
            item
            for item, condition in zip(idsSewer, sewer.G.vs["type"])
            if condition == 0
        ],
        # TODO: Update these
        node_color=[
            item
            for item, condition in zip(sewerNodesColors, sewer.G.vs["type"])
            if condition == 0
        ],
        node_shape="o",
        ax=ax[1, 0],
    )
    nx.draw_networkx_nodes(
        nxSewer,
        layoutSewer,
        nodelist=[
            item
            for item, condition in zip(idsSewer, sewer.G.vs["type"])
            if condition == 1
        ],
        node_color="black",
        node_shape="^",
        ax=ax[1, 0],
    )
    edges = nx.draw_networkx_edges(
        nxSewer,
        layoutSewer,
        edgelist=sewer.G.get_edgelist(),
        arrowstyle="->",
        arrowsize=10,
        edge_color=sewerEdgeColors,
        edge_cmap=cmap,
        width=2,
        ax=ax[1, 0],
    )

    # Plot cumulative rainfall

    # TODO: Clean this up.
    # mPerHrainfall = [r * 3600 for r in rainfall]
    # pprint(f"mperhrainfall: {mPerHrainfall}")
    # cumulative_rainfall = np.cumsum(mPerHrainfall[:it+1])
    # pprint(f"cumsum: {cumulative_rainfall}")
    # time_hours = times[:it+1]

    # TODO: fix this plot scaling
    # 1 generate cumsum array
    # 2 create updated time list
    # 3 interpolate the cumulative rainfall graph for totalIts points
    # 4 chop this to match current iteration
    totalIts = len(runoffs)
    # TODO: Make this scaling dependent on normalizeRainfall
    mPerHrainfall = [r * 3600 for r in rainfall]
    cumulative_rainfall = np.cumsum(mPerHrainfall)
    # scaledTimes = [t * (dt/3600) for t in times]
    fixedTimes = np.linspace(0, times[-1], totalIts)
    fixedCumulativeRainfall = np.interp(fixedTimes, times, cumulative_rainfall)
    # Reformat to just be upto iteration
    cumulative_rainfall = fixedCumulativeRainfall[: it + 1]
    time_hours = fixedTimes[: it + 1]

    ax[0, 1].clear()
    ax[0, 1].plot(
        time_hours,
        cumulative_rainfall,
        marker="",
        linewidth=2,
        markersize=6,
        color="steelblue",
    )
    ax[0, 1].plot(
        time_hours,
        cumulative_rainfall * (1 - subcatchment.oldwaterRatio),
        marker="",
        linestyle="--",
        linewidth=2,
        markersize=6,
        color="steelblue",
    )
    ax[0, 1].set_xlabel("Time (hr:min)", fontsize=10)
    ax[0, 1].set_ylabel("Cumulative Rainfall (meters)", fontsize=10)
    ax[0, 1].set_title("Cumulative Rainfall Over Time", fontsize=12, fontweight="bold")
    ax[0, 1].grid(True, alpha=0.3)

    # Set fixed axis limits so the plot doesn't rescale every frame
    ax[0, 1].set_xlim(times[0], times[-1])
    ax[0, 1].set_ylim(0, np.sum(mPerHrainfall) * 1.1)

    # Plot Peak Discharges
    time_hours = fixedTimes[: it + 1]
    # time_hours = [t * (dt/3600) for t in time_hours]
    current_peak_discharges = peakDischarges[: it + 1]

    ax[1, 1].clear()
    ax[1, 1].plot(
        time_hours,
        current_peak_discharges,
        marker="",
        linewidth=2,
        markersize=6,
        color="crimson",
    )
    ax[1, 1].set_xlabel("Time (hr:min)", fontsize=10)
    ax[1, 1].set_ylabel("Peak Discharge (m^3/s)", fontsize=10)
    ax[1, 1].set_title("Peak Discharge Over Time", fontsize=12, fontweight="bold")
    ax[1, 1].grid(True, alpha=0.3)

    # Set fixed axis limits so the plot doesn't rescale every frame
    ax[1, 1].set_xlim(times[0], times[-1])
    ax[1, 1].set_ylim(0, max(abs(x) for x in peakDischarges) * 1.1)

    formatter = mticker.FuncFormatter(seconds_to_hms)
    ax[0, 1].xaxis.set_major_formatter(formatter)
    ax[1, 1].xaxis.set_major_formatter(formatter)

    # create legend for color
    pc = matplotlib.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(streetNodesColors)
    plt.colorbar(pc, ax=ax)
    ax[0, 0].set_axis_off()
    ax[1, 0].set_axis_off()
    # plt.savefig(f"test.png")
    plt.show()
    return fig


def visualize_compare_networks(
    model1,
    model2,
    model1_name="Model 1",
    model2_name="Model 2",
    file="compare_networks",
    cmap=plt.cm.plasma,
    fps=5,
):
    """
    Creates a GIF comparing two networks side-by-side.
    
    Layout:
    - Top left: Model 1 street + subcatchment network
    - Top right: Model 2 street + subcatchment network
    - Bottom left: Model 1 sewer network
    - Bottom right: Model 2 sewer network
    
    Parameters:
    -----------
    model1 : Model
        First model to compare
    model2 : Model
        Second model to compare
    model1_name : str
        Label for first model
    model2_name : str
        Label for second model
    file : str
        Output filename (without extension)
    cmap : colormap
        Matplotlib colormap
    fps : int
        Frames per second for GIF
    """
    frames = []
    
    # Ensure both models have same number of timesteps for comparison
    numIters = min(len(model1.streetDepths), len(model2.streetDepths))
    
    if numIters == 0:
        pprint("Error: Models must be run before visualization")
        return
    
    # Get max values for normalization across both models
    streetYFull = max(
        model1.street.G.es[0]["yFull"] if model1.street.G.ecount() > 0 else 0.1,
        model2.street.G.es[0]["yFull"] if model2.street.G.ecount() > 0 else 0.1
    )
    sewerYFull = max(
        model1.sewer.G.es[0]["yFull"] if model1.sewer.G.ecount() > 0 else 0.1,
        model2.sewer.G.es[0]["yFull"] if model2.sewer.G.ecount() > 0 else 0.1
    )
    
    for it in range(numIters):
        fig = _create_comparison_frame(
            model1, model2,
            model1_name, model2_name,
            it, streetYFull, sewerYFull,
            cmap=cmap
        )
        fig.canvas.draw()
        frames.append(np.array(fig.canvas.renderer._renderer))
        plt.close(fig)
    
    # Save GIF
    iio.mimsave(f"figures/{file}.gif", frames, fps=fps)
    pprint(f"Saved comparison GIF to figures/{file}.gif")


def _create_comparison_frame(
    model1, model2,
    model1_name, model2_name,
    it, streetYFull, sewerYFull,
    cmap=plt.cm.plasma
):
    """Creates a single frame for network comparison visualization."""
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    
    # Normalization for colors
    norm = plt.Normalize(vmin=0, vmax=max(streetYFull, 0.01))
    normSewer = plt.Normalize(vmin=0, vmax=max(sewerYFull, 0.01))
    normRunoff = plt.Normalize(vmin=0, vmax=0.05)
    
    # Draw Model 1 networks (left column)
    _draw_street_network(
        ax[0, 0], model1.subcatchment, model1.street,
        model1.subcatchmentDepths[it], model1.runoffs[it],
        model1.streetDepths[it], model1.streetEdgeAreas[it],
        norm, normRunoff, cmap
    )
    ax[0, 0].set_title(f"{model1_name} - Street Network", fontsize=12, fontweight="bold")
    
    _draw_sewer_network(
        ax[1, 0], model1.sewer,
        model1.sewerDepths[it], model1.sewerEdgeAreas[it],
        normSewer, cmap
    )
    ax[1, 0].set_title(f"{model1_name} - Sewer Network", fontsize=12, fontweight="bold")
    
    # Draw Model 2 networks (right column)
    _draw_street_network(
        ax[0, 1], model2.subcatchment, model2.street,
        model2.subcatchmentDepths[it], model2.runoffs[it],
        model2.streetDepths[it], model2.streetEdgeAreas[it],
        norm, normRunoff, cmap
    )
    ax[0, 1].set_title(f"{model2_name} - Street Network", fontsize=12, fontweight="bold")
    
    _draw_sewer_network(
        ax[1, 1], model2.sewer,
        model2.sewerDepths[it], model2.sewerEdgeAreas[it],
        normSewer, cmap
    )
    ax[1, 1].set_title(f"{model2_name} - Sewer Network", fontsize=12, fontweight="bold")
    
    # Add time indicator
    time_seconds = model1.ts[it] if it < len(model1.ts) else 0
    hours = int(time_seconds // 3600)
    minutes = int((time_seconds % 3600) // 60)
    fig.suptitle(f"Time: {hours:02d}:{minutes:02d}", fontsize=14, fontweight="bold")
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Depth (m)', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    return fig


def _draw_street_network(ax, subcatchment, street, subcatchmentDepths, runoffs,
                         streetDepths, streetEdgeAreas, norm, normRunoff, cmap):
    """Helper function to draw street + subcatchment network on an axis."""
    
    # Convert to networkx
    aStreet = np.array(street.G.get_adjacency())
    nxStreet = nx.from_numpy_array(aStreet, create_using=nx.DiGraph())
    g = nxStreet
    
    # Create layout
    ids = [i for i in range(subcatchment.G.vcount() + street.G.vcount())]
    coordsSubcatchment = [
        list(coord) for coord in zip(subcatchment.G.vs["x"], subcatchment.G.vs["y"])
    ]
    coordsStreet = [list(coord) for coord in zip(street.G.vs["x"], street.G.vs["y"])]
    coords = coordsSubcatchment + coordsStreet
    layout = dict(zip(ids, coords))
    
    # Colors
    subcatchmentNodesColors = [cmap(norm(w)) for w in subcatchmentDepths]
    runoffsColors = [cmap(normRunoff(w)) for w in runoffs] if len(runoffs) > 0 else []
    streetNodesColors = [cmap(norm(w)) for w in streetDepths]
    streetEdgeColors = [cmap(norm(w)) for w in streetEdgeAreas]
    
    # Adjusted edge list
    newStreetEdgeList = [
        tuple(map(lambda x: x + subcatchment.G.vcount(), t))
        for t in street.G.get_edgelist()
    ]
    
    # Draw subcatchments
    nx.draw_networkx_nodes(
        g, layout,
        nodelist=[i for i in range(subcatchment.G.vcount())],
        node_color=subcatchmentNodesColors,
        node_shape="s",
        node_size=200,
        ax=ax,
    )
    
    # Draw subcatchment edges if they exist
    if subcatchment.G.ecount() > 0 and len(runoffsColors) > 0:
        nx.draw_networkx_edges(
            g, layout,
            edgelist=subcatchment.G.get_edgelist(),
            arrowstyle="->",
            arrowsize=10,
            edge_color=runoffsColors,
            edge_cmap=cmap,
            width=2,
            ax=ax,
        )
    
    # Draw street junctions
    streetNodeList = [
        i for i in range(subcatchment.G.vcount(), subcatchment.G.vcount() + street.G.vcount())
    ]
    
    # Regular junctions
    junction_nodes = [
        item for item, condition in zip(streetNodeList, street.G.vs["type"])
        if condition == 0
    ]
    junction_colors = [
        item for item, condition in zip(streetNodesColors, street.G.vs["type"])
        if condition == 0
    ]
    if junction_nodes:
        nx.draw_networkx_nodes(
            g, layout,
            nodelist=junction_nodes,
            node_color=junction_colors,
            node_shape="o",
            node_size=200,
            ax=ax,
        )
    
    # Outfall nodes
    outfall_nodes = [
        item for item, condition in zip(streetNodeList, street.G.vs["type"])
        if condition == 1
    ]
    if outfall_nodes:
        nx.draw_networkx_nodes(
            g, layout,
            nodelist=outfall_nodes,
            node_color="black",
            node_shape="^",
            node_size=200,
            ax=ax,
        )
    
    # Draw street edges
    if newStreetEdgeList:
        nx.draw_networkx_edges(
            g, layout,
            edgelist=newStreetEdgeList,
            arrowstyle="->",
            arrowsize=10,
            edge_color=streetEdgeColors,
            edge_cmap=cmap,
            width=2,
            ax=ax,
        )
    
    # Draw coupling edges
    if subcatchment.G.vcount() > 0:
        try:
            subcatchmentCoupling = [i for i in range(subcatchment.G.vcount())]
            streetCoupling = [
                street.G.vs["coupledID"].index(i) for i in subcatchment.hydraulicCoupling
            ]
            couplingEdges = [coord for coord in zip(subcatchmentCoupling, streetCoupling)]
            couplingEdges = [
                (item[0], item[1] + subcatchment.G.vcount()) for item in couplingEdges
            ]
            nx.draw_networkx_edges(
                g, layout,
                edgelist=couplingEdges,
                style="dashed",
                edge_color="gray",
                arrows=False,
                alpha=0.5,
                ax=ax,
            )
        except (ValueError, IndexError):
            pass  # Skip coupling edges if there's an issue
    
    ax.set_axis_off()


def _draw_sewer_network(ax, sewer, sewerDepths, sewerEdgeAreas, norm, cmap):
    """Helper function to draw sewer network on an axis."""
    
    # Convert to networkx
    aSewer = np.array(sewer.G.get_adjacency())
    nxSewer = nx.from_numpy_array(aSewer, create_using=nx.DiGraph())
    
    # Create layout
    idsSewer = [i for i in range(sewer.G.vcount())]
    coordsSewer = [list(coord) for coord in zip(sewer.G.vs["x"], sewer.G.vs["y"])]
    layoutSewer = dict(zip(idsSewer, coordsSewer))
    
    # Colors
    sewerNodesColors = [cmap(norm(w)) for w in sewerDepths]
    sewerEdgeColors = [cmap(norm(w)) for w in sewerEdgeAreas]
    
    # Regular junctions
    junction_nodes = [
        item for item, condition in zip(idsSewer, sewer.G.vs["type"])
        if condition == 0
    ]
    junction_colors = [
        item for item, condition in zip(sewerNodesColors, sewer.G.vs["type"])
        if condition == 0
    ]
    if junction_nodes:
        nx.draw_networkx_nodes(
            nxSewer, layoutSewer,
            nodelist=junction_nodes,
            node_color=junction_colors,
            node_shape="o",
            node_size=200,
            ax=ax,
        )
    
    # Outfall nodes
    outfall_nodes = [
        item for item, condition in zip(idsSewer, sewer.G.vs["type"])
        if condition == 1
    ]
    if outfall_nodes:
        nx.draw_networkx_nodes(
            nxSewer, layoutSewer,
            nodelist=outfall_nodes,
            node_color="black",
            node_shape="^",
            node_size=200,
            ax=ax,
        )
    
    # Draw edges
    if sewer.G.ecount() > 0:
        nx.draw_networkx_edges(
            nxSewer, layoutSewer,
            edgelist=sewer.G.get_edgelist(),
            arrowstyle="->",
            arrowsize=10,
            edge_color=sewerEdgeColors,
            edge_cmap=cmap,
            width=2,
            ax=ax,
        )
    
    ax.set_axis_off()


def visualize_observables(
    model,
    file="observables",
    show_plot=True,
    save_gif=True,
    fps=5,
):
    """
    Creates visualization of model observables (time series plots only, no network).
    
    Layout (2x3 grid):
    - Top left: Cumulative Rainfall
    - Top middle: Peak Discharge (street and combined)
    - Top right: Outfall Flows (street and sewer)
    - Bottom left: Max Street Depth
    - Bottom middle: Street vs Sewer Outfall Comparison
    - Bottom right: Rainfall Rate (instantaneous)
    
    Parameters:
    -----------
    model : Model
        Model with simulation results
    file : str
        Output filename (without extension)
    show_plot : bool
        Whether to display the final plot
    save_gif : bool
        Whether to save as animated GIF
    fps : int
        Frames per second for GIF
    """
    
    if len(model.peakDischarges) == 0:
        pprint("Error: Model must be run before visualization")
        return
    
    numIters = len(model.peakDischarges)
    
    if save_gif:
        frames = []
        for it in range(numIters):
            fig = _create_observables_frame(model, it)
            fig.canvas.draw()
            frames.append(np.array(fig.canvas.renderer._renderer))
            plt.close(fig)
        
        iio.mimsave(f"figures/{file}.gif", frames, fps=fps)
        pprint(f"Saved observables GIF to figures/{file}.gif")
    
    # Create and optionally show final frame
    fig = _create_observables_frame(model, numIters - 1)
    plt.savefig(f"figures/{file}.png", dpi=150, bbox_inches='tight')
    pprint(f"Saved observables plot to figures/{file}.png")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def _create_observables_frame(model, it):
    """Creates a single frame for observables visualization."""
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    
    formatter = mticker.FuncFormatter(seconds_to_hms)
    
    # Get time arrays
    totalIts = len(model.peakDischarges)
    fixedTimes = np.linspace(0, model.T, totalIts)
    time_current = fixedTimes[:it + 1]
    
    # =========================================================================
    # Top Left: Cumulative Rainfall
    # =========================================================================
    mPerHrainfall = [r * 3600 for r in model.rainfall]
    cumulative_rainfall = np.cumsum(mPerHrainfall)
    fixedCumulativeRainfall = np.interp(fixedTimes, model.rainfallTimes, cumulative_rainfall)
    
    ax[0, 0].plot(
        time_current,
        fixedCumulativeRainfall[:it + 1],
        linewidth=2,
        color="steelblue",
        label="Total Rainfall"
    )
    ax[0, 0].plot(
        time_current,
        fixedCumulativeRainfall[:it + 1] * (1 - model.subcatchment.oldwaterRatio),
        linewidth=2,
        linestyle="--",
        color="steelblue",
        label=f"Effective (1-{model.subcatchment.oldwaterRatio:.0%} loss)"
    )
    ax[0, 0].set_xlabel("Time (hr:min)", fontsize=10)
    ax[0, 0].set_ylabel("Cumulative Rainfall (m)", fontsize=10)
    ax[0, 0].set_title("Cumulative Rainfall", fontsize=12, fontweight="bold")
    ax[0, 0].set_xlim(0, model.T)
    ax[0, 0].set_ylim(0, np.sum(mPerHrainfall) * 1.1)
    ax[0, 0].xaxis.set_major_formatter(formatter)
    ax[0, 0].legend(loc='upper left', fontsize=8)
    ax[0, 0].grid(True, alpha=0.3)
    
    # =========================================================================
    # Top Middle: Peak Discharge
    # =========================================================================
    ax[0, 1].plot(
        time_current,
        model.peakDischarges[:it + 1],
        linewidth=2,
        color="crimson",
        label="Combined (Street + Sewer)"
    )
    ax[0, 1].plot(
        time_current,
        model.streetPeakDischarges[:it + 1],
        linewidth=2,
        linestyle="--",
        color="darkorange",
        label="Street Only"
    )
    ax[0, 1].set_xlabel("Time (hr:min)", fontsize=10)
    ax[0, 1].set_ylabel("Peak Discharge (m³/s)", fontsize=10)
    ax[0, 1].set_title("Peak Discharge", fontsize=12, fontweight="bold")
    ax[0, 1].set_xlim(0, model.T)
    max_peak = max(model.peakDischarges) if model.peakDischarges else 1
    ax[0, 1].set_ylim(0, max_peak * 1.1)
    ax[0, 1].xaxis.set_major_formatter(formatter)
    ax[0, 1].legend(loc='upper right', fontsize=8)
    ax[0, 1].grid(True, alpha=0.3)
    
    # =========================================================================
    # Top Right: Outfall Flows
    # =========================================================================
    ax[0, 2].plot(
        time_current,
        model.streetOutfallFlows[:it + 1],
        linewidth=2,
        color="forestgreen",
        label="Street Outfall"
    )
    ax[0, 2].plot(
        time_current,
        model.sewerOutfallFlows[:it + 1],
        linewidth=2,
        linestyle="--",
        color="purple",
        label="Sewer Outfall"
    )
    ax[0, 2].set_xlabel("Time (hr:min)", fontsize=10)
    ax[0, 2].set_ylabel("Outfall Flow (m³/s)", fontsize=10)
    ax[0, 2].set_title("Outfall Flows", fontsize=12, fontweight="bold")
    ax[0, 2].set_xlim(0, model.T)
    max_outfall = max(max(model.streetOutfallFlows), max(model.sewerOutfallFlows)) if model.streetOutfallFlows else 1
    ax[0, 2].set_ylim(0, max(max_outfall * 1.1, 0.001))
    ax[0, 2].xaxis.set_major_formatter(formatter)
    ax[0, 2].legend(loc='upper right', fontsize=8)
    ax[0, 2].grid(True, alpha=0.3)
    
    # =========================================================================
    # Bottom Left: Max Street Depth
    # =========================================================================
    ax[1, 0].plot(
        time_current,
        model.streetMaxDepths[:it + 1],
        linewidth=2,
        color="teal",
        label="Max Street Depth"
    )
    # Add yFull reference line
    if model.street.G.ecount() > 0:
        street_yfull = model.street.G.es[0]["yFull"]
        ax[1, 0].axhline(y=street_yfull, color='red', linestyle=':', linewidth=1.5, 
                         label=f'Street yFull ({street_yfull:.3f} m)')
    ax[1, 0].set_xlabel("Time (hr:min)", fontsize=10)
    ax[1, 0].set_ylabel("Max Depth (m)", fontsize=10)
    ax[1, 0].set_title("Maximum Street Depth", fontsize=12, fontweight="bold")
    ax[1, 0].set_xlim(0, model.T)
    max_depth = max(model.streetMaxDepths) if model.streetMaxDepths else 0.1
    ax[1, 0].set_ylim(0, max(max_depth * 1.2, street_yfull * 1.1 if model.street.G.ecount() > 0 else 0.1))
    ax[1, 0].xaxis.set_major_formatter(formatter)
    ax[1, 0].legend(loc='upper right', fontsize=8)
    ax[1, 0].grid(True, alpha=0.3)
    
    # =========================================================================
    # Bottom Middle: Cumulative Outfall Comparison
    # =========================================================================
    cumulative_street_outfall = np.cumsum(model.streetOutfallFlows[:it + 1]) * model.dt
    cumulative_sewer_outfall = np.cumsum(model.sewerOutfallFlows[:it + 1]) * model.dt
    cumulative_total_outfall = cumulative_street_outfall + cumulative_sewer_outfall
    
    ax[1, 1].plot(
        time_current,
        cumulative_total_outfall,
        linewidth=2,
        color="navy",
        label="Total Outflow"
    )
    ax[1, 1].plot(
        time_current,
        cumulative_street_outfall,
        linewidth=2,
        linestyle="--",
        color="forestgreen",
        label="Street Outflow"
    )
    ax[1, 1].plot(
        time_current,
        cumulative_sewer_outfall,
        linewidth=2,
        linestyle=":",
        color="purple",
        label="Sewer Outflow"
    )
    # Add effective rainfall reference
    total_subcatchment_area = sum(model.subcatchment.G.vs["area"])
    cumulative_effective_rain = fixedCumulativeRainfall[:it + 1] * (1 - model.subcatchment.oldwaterRatio) * total_subcatchment_area
    ax[1, 1].plot(
        time_current,
        cumulative_effective_rain,
        linewidth=1.5,
        linestyle="-.",
        color="steelblue",
        alpha=0.7,
        label="Effective Rainfall Volume"
    )
    
    ax[1, 1].set_xlabel("Time (hr:min)", fontsize=10)
    ax[1, 1].set_ylabel("Cumulative Volume (m³)", fontsize=10)
    ax[1, 1].set_title("Cumulative Outflow Volume", fontsize=12, fontweight="bold")
    ax[1, 1].set_xlim(0, model.T)
    ax[1, 1].xaxis.set_major_formatter(formatter)
    ax[1, 1].legend(loc='upper left', fontsize=8)
    ax[1, 1].grid(True, alpha=0.3)
    
    # =========================================================================
    # Bottom Right: Rainfall Rate (instantaneous)
    # =========================================================================
    rain_interp = np.interp(fixedTimes, model.rainfallTimes, model.rainfall)
    rain_mm_per_hr = [r * 3600 * 1000 for r in rain_interp]  # Convert m/s to mm/hr
    
    ax[1, 2].fill_between(
        time_current,
        rain_mm_per_hr[:it + 1],
        alpha=0.4,
        color="steelblue",
        label="Rainfall Rate"
    )
    ax[1, 2].plot(
        time_current,
        rain_mm_per_hr[:it + 1],
        linewidth=2,
        color="steelblue",
    )
    ax[1, 2].set_xlabel("Time (hr:min)", fontsize=10)
    ax[1, 2].set_ylabel("Rainfall Rate (mm/hr)", fontsize=10)
    ax[1, 2].set_title("Rainfall Intensity", fontsize=12, fontweight="bold")
    ax[1, 2].set_xlim(0, model.T)
    max_rain = max(rain_mm_per_hr) if rain_mm_per_hr else 1
    ax[1, 2].set_ylim(0, max_rain * 1.1)
    ax[1, 2].xaxis.set_major_formatter(formatter)
    ax[1, 2].grid(True, alpha=0.3)
    
    # Add time indicator
    time_seconds = fixedTimes[it] if it < len(fixedTimes) else 0
    hours = int(time_seconds // 3600)
    minutes = int((time_seconds % 3600) // 60)
    fig.suptitle(f"Model Observables - Time: {hours:02d}:{minutes:02d}", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    return fig


def visualize_observables_comparison(
    model1,
    model2,
    model1_name="Model 1",
    model2_name="Model 2",
    file="observables_comparison",
    show_plot=True,
):
    """
    Creates a static comparison plot of observables from two models.
    
    Parameters:
    -----------
    model1 : Model
        First model with simulation results
    model2 : Model
        Second model with simulation results
    model1_name : str
        Label for first model
    model2_name : str
        Label for second model
    file : str
        Output filename (without extension)
    show_plot : bool
        Whether to display the plot
    """
    
    if len(model1.peakDischarges) == 0 or len(model2.peakDischarges) == 0:
        pprint("Error: Both models must be run before visualization")
        return
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    formatter = mticker.FuncFormatter(seconds_to_hms)
    
    # Time arrays
    times1 = np.linspace(0, model1.T, len(model1.peakDischarges))
    times2 = np.linspace(0, model2.T, len(model2.peakDischarges))
    
    # =========================================================================
    # Top Left: Peak Discharge Comparison
    # =========================================================================
    ax[0, 0].plot(times1, model1.peakDischarges, linewidth=2, color="crimson", 
                  label=f"{model1_name}")
    ax[0, 0].plot(times2, model2.peakDischarges, linewidth=2, linestyle="--", 
                  color="darkblue", label=f"{model2_name}")
    ax[0, 0].set_xlabel("Time (hr:min)", fontsize=10)
    ax[0, 0].set_ylabel("Peak Discharge (m³/s)", fontsize=10)
    ax[0, 0].set_title("Peak Discharge Comparison", fontsize=12, fontweight="bold")
    ax[0, 0].xaxis.set_major_formatter(formatter)
    ax[0, 0].legend(loc='upper right', fontsize=8)
    ax[0, 0].grid(True, alpha=0.3)
    
    # =========================================================================
    # Top Middle: Street Outfall Flow Comparison
    # =========================================================================
    ax[0, 1].plot(times1, model1.streetOutfallFlows, linewidth=2, color="forestgreen",
                  label=f"{model1_name}")
    ax[0, 1].plot(times2, model2.streetOutfallFlows, linewidth=2, linestyle="--",
                  color="darkgreen", label=f"{model2_name}")
    ax[0, 1].set_xlabel("Time (hr:min)", fontsize=10)
    ax[0, 1].set_ylabel("Street Outfall Flow (m³/s)", fontsize=10)
    ax[0, 1].set_title("Street Outfall Flow Comparison", fontsize=12, fontweight="bold")
    ax[0, 1].xaxis.set_major_formatter(formatter)
    ax[0, 1].legend(loc='upper right', fontsize=8)
    ax[0, 1].grid(True, alpha=0.3)
    
    # =========================================================================
    # Top Right: Sewer Outfall Flow Comparison
    # =========================================================================
    ax[0, 2].plot(times1, model1.sewerOutfallFlows, linewidth=2, color="purple",
                  label=f"{model1_name}")
    ax[0, 2].plot(times2, model2.sewerOutfallFlows, linewidth=2, linestyle="--",
                  color="darkviolet", label=f"{model2_name}")
    ax[0, 2].set_xlabel("Time (hr:min)", fontsize=10)
    ax[0, 2].set_ylabel("Sewer Outfall Flow (m³/s)", fontsize=10)
    ax[0, 2].set_title("Sewer Outfall Flow Comparison", fontsize=12, fontweight="bold")
    ax[0, 2].xaxis.set_major_formatter(formatter)
    ax[0, 2].legend(loc='upper right', fontsize=8)
    ax[0, 2].grid(True, alpha=0.3)
    
    # =========================================================================
    # Bottom Left: Max Street Depth Comparison
    # =========================================================================
    ax[1, 0].plot(times1, model1.streetMaxDepths, linewidth=2, color="teal",
                  label=f"{model1_name}")
    ax[1, 0].plot(times2, model2.streetMaxDepths, linewidth=2, linestyle="--",
                  color="darkcyan", label=f"{model2_name}")
    ax[1, 0].set_xlabel("Time (hr:min)", fontsize=10)
    ax[1, 0].set_ylabel("Max Street Depth (m)", fontsize=10)
    ax[1, 0].set_title("Max Street Depth Comparison", fontsize=12, fontweight="bold")
    ax[1, 0].xaxis.set_major_formatter(formatter)
    ax[1, 0].legend(loc='upper right', fontsize=8)
    ax[1, 0].grid(True, alpha=0.3)
    
    # =========================================================================
    # Bottom Middle: Cumulative Outflow Comparison
    # =========================================================================
    cum_outflow1 = np.cumsum(np.array(model1.streetOutfallFlows) + np.array(model1.sewerOutfallFlows)) * model1.dt
    cum_outflow2 = np.cumsum(np.array(model2.streetOutfallFlows) + np.array(model2.sewerOutfallFlows)) * model2.dt
    
    ax[1, 1].plot(times1, cum_outflow1, linewidth=2, color="navy",
                  label=f"{model1_name}")
    ax[1, 1].plot(times2, cum_outflow2, linewidth=2, linestyle="--",
                  color="darkblue", label=f"{model2_name}")
    ax[1, 1].set_xlabel("Time (hr:min)", fontsize=10)
    ax[1, 1].set_ylabel("Cumulative Outflow (m³)", fontsize=10)
    ax[1, 1].set_title("Cumulative Outflow Comparison", fontsize=12, fontweight="bold")
    ax[1, 1].xaxis.set_major_formatter(formatter)
    ax[1, 1].legend(loc='upper left', fontsize=8)
    ax[1, 1].grid(True, alpha=0.3)
    
    # =========================================================================
    # Bottom Right: Summary Statistics
    # =========================================================================
    ax[1, 2].axis('off')
    
    # Calculate statistics
    stats_text = f"""
    Summary Statistics
    {'='*40}
    
    {model1_name}:
      Peak Discharge: {max(model1.peakDischarges):.4f} m³/s
      Max Street Depth: {max(model1.streetMaxDepths):.4f} m
      Total Street Outflow: {sum(model1.streetOutfallFlows) * model1.dt:.2f} m³
      Total Sewer Outflow: {sum(model1.sewerOutfallFlows) * model1.dt:.2f} m³
    
    {model2_name}:
      Peak Discharge: {max(model2.peakDischarges):.4f} m³/s
      Max Street Depth: {max(model2.streetMaxDepths):.4f} m
      Total Street Outflow: {sum(model2.streetOutfallFlows) * model2.dt:.2f} m³
      Total Sewer Outflow: {sum(model2.sewerOutfallFlows) * model2.dt:.2f} m³
    
    Ratios ({model2_name}/{model1_name}):
      Peak Discharge: {max(model2.peakDischarges)/max(model1.peakDischarges):.2f}x
      Max Street Depth: {max(model2.streetMaxDepths)/max(model1.streetMaxDepths):.2f}x
      Total Outflow: {(sum(model2.streetOutfallFlows)+sum(model2.sewerOutfallFlows))/(sum(model1.streetOutfallFlows)+sum(model1.sewerOutfallFlows)):.2f}x
    """
    
    ax[1, 2].text(0.1, 0.9, stats_text, transform=ax[1, 2].transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f"Model Comparison: {model1_name} vs {model2_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    plt.savefig(f"figures/{file}.png", dpi=150, bbox_inches='tight')
    pprint(f"Saved comparison plot to figures/{file}.png")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
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
    visualizeExample(
        subcatchment,
        street,
        sewer,
        runoff,
        drainOverflow,
        drainInflow,
        times,
        rainfall,
        peakDischarges,
        cmap=plt.cm.plasma,
    )
