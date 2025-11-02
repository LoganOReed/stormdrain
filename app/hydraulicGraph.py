import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
import random
from pprint import pprint
from .newtonBisection import newtonBisection
from .drainCapture import capturedFlow
from .streetGeometry import depthFromAreaStreet, psiFromAreaStreet, psiPrimeFromAreaStreet
from .circularGeometry import depthFromAreaCircle, psiFromAreaCircle, psiPrimeFromAreaCircle
from . import A_tbl, R_tbl, STREET_Y_FULL


class HydraulicGraph:
    """Parent Hydraulic Graph for both Street and Sewer Graph."""
    def __init__(self, graphType, file):
        super(HydraulicGraph, self).__init__()
        self.graphType = graphType
        if graphType == "STREET":
            self.depthFromArea = depthFromAreaStreet
            self.psiFromArea = psiFromAreaStreet
            self.psiPrimeFromArea = psiPrimeFromAreaStreet
            self.yFull = STREET_Y_FULL # diff between lowest and highest point from choice of Street Parameters
        else:
            # TODO: Change this to circular geometry
            self.depthFromArea = depthFromAreaCircle
            self.psiFromArea = psiFromAreaCircle
            self.psiPrimeFromArea = psiPrimeFromAreaCircle

        data = pd.read_csv(f"data/{file}.csv")
        data = data[data["type"].str.contains(graphType)]
        n = data.shape[0]
        # pprint(n)
        # pprint(data["type"])
        # pprint(data["x"].astype(float))
        # pprint(data)
        # pprint(data["type"].str.contains("OUTFALL").astype(int))

        # Needed to create edges
        edges = []
        mapToID = []
        i = 0
        for _, row in data.iterrows():
            mapToID.append((row["id"],i))
            i = i+1



        # Creates the edges by translating the node id's in the csv into 0-indexed sewer nodes
        for _, row in data.iterrows():
            # pprint(f"{index}, {row["id"]}")
            if row["outgoing"] != -1:
                id = row["id"]
                outgoing = row["outgoing"]
                for pair in mapToID:
                    if pair[0] == id:
                        id = pair[1]
                    if pair[0] == outgoing:
                        outgoing = pair[1]
                edges.append( (id, outgoing))



        self.G = ig.Graph(n=n,edges=edges,directed=True,
              vertex_attrs={
                  'coupledID': np.array(data["id"].astype(int)),
                  'invert': np.zeros(n),
                  'x': np.array(data["x"].astype(float)),
                  'y': np.array(data["y"].astype(float)),
                  'z': np.array(data["z"].astype(float)),
                  'depth': np.zeros(n),
                  # 0 - junction
                  # 1 - outfall
                  'type': np.array(data["type"].str.contains("OUTFALL").astype(int)),
                  'drain': np.array(data["drain"].astype(int)),
                  'drainType': np.array(data["drainType"]),
                  'drainCoupledID': np.array(data["drainCoupledID"].astype(int)),
                  'drainLength': np.array(data["drainLength"].astype(float)),
                  'drainWidth': np.array(data["drainWidth"].astype(float)),
                  })
        # calculate the lengths of each pipe
        for e in self.G.es:
            s = np.array([self.G.vs[e.source]['x'], self.G.vs[e.source]['y'], self.G.vs[e.source]['z']])
            d = np.array([self.G.vs[e.target]['x'], self.G.vs[e.target]['y'], self.G.vs[e.target]['z']])
            self.G.es[e.index]['length'] = np.linalg.norm(s - d)
        # calculate the slope of each pipe
        for e in self.G.es:
            slope = (self.G.vs[e.source]['z'] - self.G.vs[e.target]['z']) / self.G.es[e.index]['length']
            if slope < 0.0001:
                print(f"WARNING: slope for edge ({e.source}, {e.target}) is too small.")
                print(f"{e.source}: ({self.G.vs[e.source]['x']}, {self.G.vs[e.source]['y']}, {self.G.vs[e.source]['z']})")
                print(f"{e.target}: ({self.G.vs[e.target]['x']}, {self.G.vs[e.target]['y']}, {self.G.vs[e.target]['z']})")
            self.G.es[e.index]['slope'] = slope
        # pprint(f"Slopes: {self.G.es['slope']}")
        # pprint(f"Length: {self.G.es['length']}")
        # TODO: add offset height calculations
        # Needs to be given a priori
        self.G.es['offsetHeight'] = [0.0 for _ in range(self.G.ecount())]
        self.G.es['n'] =  np.full(n, 0.013)

        if self.graphType == "STREET":
            # This is a choice made when creating the street lookup tables
            self.G.es['yFull'] = [0.3197 for _ in self.G.es]
        else:
            # corresponds to around 18in pipe
            self.G.es['yFull'] = [0.5 for _ in self.G.es]

        # 1 is source node and 2 is target node
        self.G.es['Q1'] = np.zeros(self.G.ecount())
        self.G.es['Q2'] = np.zeros(self.G.ecount())
        # NOTE: Cant initialize as zero because first update will fail


        self.G.es['A1'] = np.zeros(self.G.ecount())
        self.G.es['A2'] = np.zeros(self.G.ecount())
        # self.G.es['A1'] = np.full(self.G.ecount(),0.0001)
        # self.G.es['A2'] = np.full(self.G.ecount(),0.0001)

        self.G.es['Q1New'] = np.zeros(self.G.ecount())
        self.G.es['Q2New'] = np.zeros(self.G.ecount())
        # NOTE: Cant initialize as zero because first update will fail
        self.G.es['A1New'] = np.zeros(self.G.ecount())
        self.G.es['A2New'] = np.zeros(self.G.ecount())
        # pprint(self.G.summary())
        
    # TODO: change runoff/DrainOverflows/DrainInflows to be hydraulicGraph agnostic for inputs and outputs
    def update(self, t, dt, coupledInputs):
        """
        Updates the attributes of the network.

        Parameters:
        -----------
        t : float
            initial time
        dt : float
            time between initial time and desired end time
        coupledInputs : list(float)
            List of coupled Flow inputs, subcatchments and overflow for street and drain for Sewer

        Returns:
        --------
        depths : list
            Updated depths ordered by igraph id

        """
        # Parameters
        PHI = 0.6
        THETA = 0.6
        pass



if __name__ == "__main__":
    print("Dont call this directly :(")
