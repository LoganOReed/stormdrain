import imageio as iio
import pandas as pd
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
from .network import SubcatchmentGraph
from .hydraulicGraph import HydraulicGraph
from .newtonBisection import newtonBisection
from .visualize import visualize
from .rain import normalizeRainfall

class Model:
    """Wraps the coupling and timestepping."""
    def __init__(self, file, dt, rainInfo, oldwaterRatio=0.2):
        self.file = file
        self.data = pd.read_csv(f"data/{file}.csv")




        # unwrap rain info
        self.spaceConversion=rainInfo["spaceConversion"]
        self.timeConversion=rainInfo["timeConversion"]
        self.rainfall, self.rainfallTimes = normalizeRainfall(rainInfo["rainfall"], rainInfo["rainfallTimes"], rainInfo["spaceConversion"], rainInfo["timeConversion"])

        # initialize time
        self.T = max(self.rainfallTimes)
        self.dt = dt
        self.N = int(self.T/self.dt)
        self.ts = np.linspace(0,self.T,self.N)
        self.rain = np.interp(self.ts, self.rainfallTimes, self.rainfall)


        # networks
        self.subcatchment = SubcatchmentGraph(self.file, oldwaterRatio)
        self.street = HydraulicGraph("STREET", self.file)
        self.sewer = HydraulicGraph("SEWER", self.file)

        # Setup coupling
        self.coupling = {"subcatchmentRunoff": np.zeros(self.data.shape[0]), "drainCapture": np.zeros(self.data.shape[0]), "drainOverflow": np.zeros(self.data.shape[0])}

        # setup storage for observables
        self.subcatchmentDepths = []
        self.runoffs = []
        self.streetDepths = []
        self.streetEdgeAreas = []
        self.sewerDepths = []
        self.sewerEdgeAreas = []
        self.drainOverflows = []
        self.drainInflows = []
        self.peakDischarges = []



        # TODO: Add whatever we want to report (e.g. depth, flow, etc)

    def run(self):
        for n in range(len(self.ts)):
            self.step(n)
            pprint(f"peakDischarges: {self.peakDischarges}")
        visualize(self.subcatchment, self.street, self.street.yFull, self.sewer, 0.5, self.subcatchmentDepths, self.runoffs, self.streetDepths, self.streetEdgeAreas, self.sewerDepths, self.sewerEdgeAreas, self.drainOverflows, self.drainInflows, self.rainfallTimes, self.rainfall, self.peakDischarges, self.dt, file=f"{file}{dt}dt", cmap=plt.cm.plasma, fps=5 )
        


    def step(self, n):
        """Does one iteration (time step)."""
        newCoupling = {"subcatchmentRunoff": np.zeros(self.data.shape[0]), "drainCapture": np.zeros(self.data.shape[0]), "drainOverflow": np.zeros(self.data.shape[0])}

        # Subcatchment Update
        subcatchmentDepth, tempRunoff = self.subcatchment.update(self.ts[n], self.dt, self.rain[n])
        # map runoff to correct indexes and add to coupling terms
        for i in range(len(tempRunoff)):
            # pprint(f' street id for subcatchment: {self.subcatchment.G.vs[i]["coupledStreet"]-1}')
            newCoupling["subcatchmentRunoff"][self.subcatchment.G.vs[i]["coupledStreet"]-1] = tempRunoff[i]

        # Street Update
        streetDepth, streetEdgeArea, tempCoupling, streetPeakDischarge = self.street.update(self.ts[n],self.dt,self.coupling)
        # pprint(f"after street: {tempCoupling}")
        newCoupling["drainCapture"] = tempCoupling["drainCapture"]


        # Sewer Update
        sewerDepth, sewerEdgeArea, tempCoupling, sewerPeakDischarge = self.sewer.update(self.ts[n],self.dt,self.coupling)
        newCoupling["drainOverflow"] = tempCoupling["drainOverflow"]

        # pprint(self.sewer.G.es["Q1"])

        # Call observable functions
        self.subcatchmentDepths.append(subcatchmentDepth)
        self.runoffs.append(np.zeros(self.subcatchment.G.vcount()))
        self.streetDepths.append(streetDepth)
        self.streetEdgeAreas.append(streetEdgeArea)
        self.sewerDepths.append(sewerDepth)
        self.sewerEdgeAreas.append(sewerEdgeArea)
        self.drainOverflows.append(np.zeros(self.street.G.vcount()))
        self.drainInflows.append(np.zeros(self.sewer.G.vcount()))
        self.peakDischarges.append(streetPeakDischarge+sewerPeakDischarge)

        
        # pprint(f"New Coupling: {newCoupling}")

        # Update Coupling Terms
        self.coupling["subcatchmentRunoff"] = newCoupling["subcatchmentRunoff"]
        self.coupling["drainCapture"] = newCoupling["drainCapture"]
        self.coupling["drainOverflow"] = newCoupling["drainOverflow"]

        


        



if __name__ == "__main__":
    file = "SimpleExample"
    tempRainfall = np.array([0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05, 0.0,0.0,0.0,0.0])
    rainInfo = {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": tempRainfall,
            "rainfallTimes": np.array([i for i in range(len(tempRainfall))])
            }
    dt = 1800
    model = Model(file, dt, rainInfo, oldwaterRatio=0.2)
    model.run()


