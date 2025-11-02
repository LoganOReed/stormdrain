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
from .rain import normalizeRainfall



def example(file, rainfall, rainfallTimes, dt, createVisuals=True):
    """Generic Example function which can be called with different networks and rainfall."""

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
    peakDischarges = []
    # 900 s = 15 min, needs to match rainfall array
    T = max(rainfallTimes)
    pprint(f"0 to {T}, total steps = {T / dt}")
    N = int(T/dt)
    ts = np.linspace(0,T,N)
    rain = np.interp(ts, rainfallTimes, rainfall)
    pprint(f"rainfall: {rain}")
    for n in range(len(ts)):
        subcatchmentDepth, runoffUnsorted = subcatchment.update(ts[n], dt, rain[n])
        maxRunoff = np.max(runoffUnsorted)
        pprint(f"Max Runoff: {maxRunoff}")
        subcatchmentDepths.append(subcatchmentDepth)
        # setup runoff
        runoff = np.zeros(street.G.vcount())
        i = 0
        for nid in subcatchment.hydraulicCoupling:
            # pprint(f"Search {nid} using find: {street.G.vs.find(coupledID=nid)}")
            runoff[street.G.vs.find(coupledID=nid).index] = runoffUnsorted[i]
            # runoff[street.find(nid)] = runoffUnsorted[i]
            i += 1
        # pprint(f"runoff unsorted: {runoffUnsorted}")
        # pprint(f"runoff: {runoff}")
        peakDischarge = np.max(np.abs(runoff))
        runoffs.append(runoff)

        drainOverflow = np.zeros(street.G.vcount())

        streetDepth, streetEdgeArea, drainInflow, tempPeakDischarge = street.update(ts[n],dt,runoff,drainOverflow)
        if peakDischarge < tempPeakDischarge:
            pprint(f"The largest is from street: {peakDischarge} < {tempPeakDischarge}")
            peakDischarge = tempPeakDischarge
        streetDepths.append(streetDepth)
        streetEdgeAreas.append(streetEdgeArea)
        drainInflows.append(drainInflow)

        # update sewer
        sewerDepth, sewerEdgeArea, drainOverflow, tempPeakDischarge = sewer.update(ts[n],dt,drainInflow)
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

    if createVisuals == True:
        visualize(subcatchment, street, street.yFull, sewer, 0.5, subcatchmentDepths, runoffs, streetDepths, streetEdgeAreas, sewerDepths, sewerEdgeAreas, drainOverflows, drainInflows, rainfallTimes, rainfall, peakDischarges, cmap=plt.cm.plasma, fps=5 )
    subcatchment.visualize(ts,subcatchmentDepths,fileName="subcatchmentGraph")

    # pprint(f"Runoffs: {runoffs}")
    # pprint(f"streetDepths: {streetDepths}")
    # pprint(f"streetEdgeAreas: {streetEdgeAreas}")
    # pprint(f"drainInflows: {drainInflows}")

   


if __name__ == "__main__":
    spaceConversion=0.0254
    timeConversion=3600
    dt = 1800
    rainfall = np.array([0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05, 0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0])
    # rainfall = np.array([0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05])
    # rainfall = [0.0,0.5,1.0,0.75,0.5]
    rainfallTimes = [i for i in range(len(rainfall))]
    # rainfall, rainfallTimes = normalizeRainfall(rainfall, rainfallTimes, spaceConversion=0.0254, timeConversion=3600)
    rainfall, rainfallTimes = normalizeRainfall(rainfall, rainfallTimes, spaceConversion, timeConversion)

    file = "doubled_largerExample"
    example(file, rainfall, rainfallTimes, dt, createVisuals=True)

