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

        # TODO: Add whatever we want to report (e.g. depth, flow, etc)


    def step(self, n):
        """Does one iteration (time step)."""
        newCoupling = {"subcatchmentRunoff": np.zeros(self.data.shape[0]), "drainCapture": np.zeros(self.data.shape[0]), "drainOverflow": np.zeros(self.data.shape[0])}

        # Subcatchment Update
        _, tempRunoff = self.subcatchment.update(self.ts[n], self.dt, self.rain[n])
        # map runoff to correct indexes and add to coupling terms
        for i in range(len(tempRunoff)):
            # pprint(f' street id for subcatchment: {self.subcatchment.G.vs[i]["coupledStreet"]-1}')
            newCoupling["subcatchmentRunoff"][self.subcatchment.G.vs[i]["coupledStreet"]-1] = tempRunoff[i]

        # Street Update
        _, _, tempCoupling, _ = self.street.update(self.ts[n],self.dt,self.coupling)
        # pprint(f"after street: {tempCoupling}")
        newCoupling["drainCapture"] = tempCoupling["drainCapture"]


        # Sewer Update
        _, _, tempCoupling, _ = self.sewer.update(self.ts[n],self.dt,self.coupling)
        newCoupling["drainOverflow"] = tempCoupling["drainOverflow"]

        pprint(self.sewer.G.es["Q1"])

        # Call observable functions
        # pprint(f"New Coupling: {newCoupling}")

        # Update Coupling Terms
        self.coupling["subcatchmentRunoff"] = newCoupling["subcatchmentRunoff"]
        self.coupling["drainCapture"] = newCoupling["drainCapture"]
        self.coupling["drainOverflow"] = newCoupling["drainOverflow"]
        
        



if __name__ == "__main__":
    file = "doubled_largerExample"
    tempRainfall = np.array([0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05, 0.0,0.0,0.0,0.0])
    rainInfo = {
            "spaceConversion": 0.0254,
            "timeConversion": 3600,
            "rainfall": tempRainfall,
            "rainfallTimes": np.array([i for i in range(len(tempRainfall))])
            }
    dt = 1800
    model = Model(file, dt, rainInfo, oldwaterRatio=0.2)
    for i in range(100):
        model.step(i)

