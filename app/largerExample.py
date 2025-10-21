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
    # rainfall = [0.01,0.5,1.0,1.0,1.0,1.5,1.8,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,1.6,1.2,1.2,1.1,0.75,0.75,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.1,0.1]
    # rainfall = [0.0,0.5,1.0,0.75,0.5,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    # rainfall = [rain / 4 for rain in rainfall for _ in range(4)]
    rainfall = [0.01,0.5,0.6,0.8,1.0,1.0,1.0,1.5,1.8,2.0,2.0,2.0,2.0,2.0,2.0]
    rainfall = rainfall + rainfall[::-1]

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
    peakDischarges = []
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
        peakDischarge = np.max(np.abs(runoff))
        runoffs.append(runoff)

        drainOverflow = np.zeros(street.G.vcount())

        streetDepth, streetEdgeArea, drainInflow, tempPeakDischarge = street.update(t,dt,runoff,drainOverflow)
        if peakDischarge < tempPeakDischarge:
            peakDischarge = tempPeakDischarge
        streetDepths.append(streetDepth)
        streetEdgeAreas.append(streetEdgeArea)
        drainInflows.append(drainInflow)

        # update sewer
        sewerDepth, sewerEdgeArea, drainOverflow, tempPeakDischarge = sewer.update(t,dt,drainInflow)
        if peakDischarge < tempPeakDischarge:
            peakDischarge = tempPeakDischarge
        sewerDepths.append(sewerDepth)
        sewerEdgeAreas.append(sewerEdgeArea)
        drainOutflows.append(drainOverflow)
        # pprint(f"Inflow: {drainInflow}")
        # self.G.vs['depth'], averageArea

        # pprint(f"SubcatchmentDepth: {subcatchmentDepth}")
        # pprint(f"runoff: {runoff}")
        peakDischarges.append(peakDischarge)
        
    # TODO: Actually store these 

    times = [i for i in range(T)]

    visualize(subcatchment, street, street.yFull, sewer, 0.5, subcatchmentDepths, runoffs, streetDepths, streetEdgeAreas, sewerDepths, sewerEdgeAreas, drainOverflows, drainInflows, times, rainfall, peakDischarges, cmap=plt.cm.plasma, fps=5 )

    pprint(f"Runoffs: {runoffs}")
    pprint(f"streetDepths: {streetDepths}")
    pprint(f"streetEdgeAreas: {streetEdgeAreas}")
    pprint(f"drainInflows: {drainInflows}")


