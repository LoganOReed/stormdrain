import imageio as iio
import igraph as ig
import networkx as nx
import numpy as np
from sys import platform
import matplotlib
if platform == "linux":
    matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import scipy as sc
import random
import csv
from pprint import pprint
from .network import SubcatchmentGraph, SewerGraph, StreetGraph
from .newtonBisection import newtonBisection
from .visualize import visualize





if __name__ == "__main__":
    # rainfall = [0.0,0.5,1.0,0.75,0.5,0.25,0.0]
    rainfall = [0.0,0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.75,0.5,0.25,0.0]
    # rainfall = [0.0,0.5,1.0,0.75,0.5,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    rainfall = [e * 0.0254 for e in rainfall]

    T = len(rainfall)

    file = "largerExample"
    subcatchmentDepths = []
    runoffs = []
    streetDepths = []
    streetEdgeAreas = []
    sewerDepths = []
    sewerEdgeAreas = []
    drainOverflows = []
    drainInflows = []
    drainOutflows = []


    subcatchment = SubcatchmentGraph(file)
    street = StreetGraph(file)
    pprint(street.G.summary())
    # street.update(0,0.5, np.zeros(street.G.vcount()), np.zeros(street.G.vcount()))
    sewer = SewerGraph(file)

    # TODO: Have subcatchment coupling happen by passing hydraulicCoupling and runoff to sewer update function

    # NOTE: Actual Update Loop
    dt = 1
    for t in range(len(rainfall)):
        subcatchmentDepth, runoffUnsorted = subcatchment.update(t, dt, rainfall[t])
        subcatchmentDepths.append(subcatchmentDepth)
        # setup runoff
        runoff = np.zeros(street.G.vcount())
        i = 0
        for nid in subcatchment.hydraulicCoupling:
            pprint(f"Search {nid} using find: {street.G.vs.find(coupledID=nid)}")
            runoff[street.G.vs.find(coupledID=nid).index] = runoffUnsorted[i]
            # runoff[street.find(nid)] = runoffUnsorted[i]
            i += 1
        # pprint(f"runoff unsorted: {runoffUnsorted}")
        # pprint(f"runoff: {runoff}")
        runoffs.append(runoff)

        drainOverflow = np.zeros(street.G.vcount())

        streetDepth, streetEdgeArea, drainInflow = street.update(t,dt,runoff,drainOverflow)
        streetDepths.append(streetDepth)
        streetEdgeAreas.append(streetEdgeArea)
        drainInflows.append(drainInflow)

        # update sewer
        sewerDepth, sewerEdgeArea, drainOutflow = sewer.update(t,dt,drainInflow)
        sewerDepths.append(streetDepth)
        sewerEdgeAreas.append(streetEdgeArea)
        drainOutflows.append(drainOutflow)
        # pprint(f"Inflow: {drainInflow}")
        # self.G.vs['depth'], averageArea

        # pprint(f"SubcatchmentDepth: {subcatchmentDepth}")
        # pprint(f"runoff: {runoff}")
        
    # TODO: Actually store these 

    times = [i for i in range(T)]

    visualize(subcatchment, street, street.yFull, sewer, 0.5, subcatchmentDepths, runoffs, streetDepths, streetEdgeAreas, sewerDepths, sewerEdgeAreas, drainOverflows, drainInflows, times, cmap=plt.cm.plasma, fps=24 )

    pprint(f"Runoffs: {runoffs}")
    pprint(f"streetDepths: {streetDepths}")
    pprint(f"streetEdgeAreas: {streetEdgeAreas}")
    pprint(f"drainInflows: {drainInflows}")


