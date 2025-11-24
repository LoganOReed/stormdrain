import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import random
from pprint import pprint
from .newtonBisection import newtonBisection
from .drainCapture import capturedFlow
from .streetGeometry import (
    depthFromAreaStreet,
    psiFromAreaStreet,
    psiPrimeFromAreaStreet,
    # areaFromPsiStreet,
    fullAreaStreet,
    maxPsiStreet,
)
from .circularGeometry import (
    depthFromAreaCircle,
    psiFromAreaCircle,
    psiPrimeFromAreaCircle,
    areaFromPsiCircle,
    getPsiMax,
)
from . import A_tbl, R_tbl, STREET_Y_FULL, STREET_LANE_SLOPE

class HydraulicGraph:
    """Parent Hydraulic Graph for both Street and Sewer Graph."""

    def __init__(self, graphType, file):
        super(HydraulicGraph, self).__init__()
        self.graphType = graphType
        if graphType == "STREET":
            self.depthFromArea = depthFromAreaStreet
            self.psiFromArea = psiFromAreaStreet
            self.psiPrimeFromArea = psiPrimeFromAreaStreet
        else:
            # TODO: Change this to circular geometry
            self.depthFromArea = depthFromAreaCircle
            self.psiFromArea = psiFromAreaCircle
            self.psiPrimeFromArea = psiPrimeFromAreaCircle

        data = pd.read_csv(f"data/{file}.csv")
        data = data[data["type"].str.contains(graphType)]
        n = data.shape[0]

        # Needed to create edges
        edges = []
        mapToID = []
        i = 0
        for _, row in data.iterrows():
            mapToID.append((row["id"], i))
            i = i + 1

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
                edges.append((id, outgoing))

        self.G = ig.Graph(
            n=n,
            edges=edges,
            directed=True,
            vertex_attrs={
                "coupledID": np.array(data["id"].astype(int)),
                "invert": np.zeros(n),
                "x": np.array(data["x"].astype(float)),
                "y": np.array(data["y"].astype(float)),
                "z": np.array(data["z"].astype(float)),
                "depth": np.zeros(n),
                # 0 - junction
                # 1 - outfall
                "type": np.array(data["type"].str.contains("OUTFALL").astype(int)),
                "drain": np.array(data["drain"].astype(int)),
                "drainType": np.array(data["drainType"]),
                "drainCoupledID": np.array(data["drainCoupledID"].astype(int)),
                "drainLength": np.array(data["drainLength"].astype(float)),
                "drainWidth": np.array(data["drainWidth"].astype(float)),
            },
        )
        # calculate the lengths of each pipe
        for e in self.G.es:
            s = np.array(
                [
                    self.G.vs[e.source]["x"],
                    self.G.vs[e.source]["y"],
                    self.G.vs[e.source]["z"],
                ]
            )
            d = np.array(
                [
                    self.G.vs[e.target]["x"],
                    self.G.vs[e.target]["y"],
                    self.G.vs[e.target]["z"],
                ]
            )
            self.G.es[e.index]["length"] = np.linalg.norm(s - d)
        # calculate the slope of each pipe
        for e in self.G.es:
            slope = (self.G.vs[e.source]["z"] - self.G.vs[e.target]["z"]) / self.G.es[
                e.index
            ]["length"]
            if slope < 0.00003048:
                print(f"WARNING: slope for edge ({e.source}, {e.target}) is too small.")
                print(
                    f"{e.source}: ({self.G.vs[e.source]['x']}, {self.G.vs[e.source]['y']}, {self.G.vs[e.source]['z']})"
                )
                print(
                    f"{e.target}: ({self.G.vs[e.target]['x']}, {self.G.vs[e.target]['y']}, {self.G.vs[e.target]['z']})"
                )
            self.G.es[e.index]["slope"] = slope

        # pprint(f"Slopes: {self.G.es['slope']}")
        # pprint(f"Length: {self.G.es['length']}")
        # TODO: add offset height calculations
        # Needs to be given a priori
        self.G.es["offsetHeight"] = [0.0 for _ in range(self.G.ecount())]
        self.G.es["n"] = np.full(n, 0.013)
        # self.G.es["Sx"] = np.full(n, STREET_LANE_SLOPE)
        self.G.es["T_curb"] = 8 * 0.3048
        self.G.es["T_crown"] = 15 * 0.3048
        self.G.es["H_curb"] = 1 * 0.3048
        self.G.es["S_back"] = 0.02 * 0.3048
        self.G.es["Sx"] = 0.02 * 0.304

        if self.graphType == "STREET":
            # This is a choice made when creating the street lookup tables
            self.G.es["yFull"] = np.array([STREET_Y_FULL for _ in self.G.es])
            self.G.es["Afull"] = np.array([fullAreaStreet(e) for e in self.G.es])
            self.G.es["PsiFull"] = np.array([maxPsiStreet(e) for e in self.G.es]) 
        else:
            # corresponds to around 18in pipe
            self.G.es["yFull"] = np.array([0.5 for _ in self.G.es])
            self.G.es["Afull"] = np.array([(np.pi / 4)*e["yFull"]*e["yFull"] for e in self.G.es])
            self.G.es["PsiFull"] = np.array([getPsiMax(e) for e in self.G.es]) 

        # 1 is source node and 2 is target node
        self.G.es["Q1"] = np.zeros(self.G.ecount())
        self.G.es["Q2"] = np.zeros(self.G.ecount())
        self.G.es["A1"] = np.zeros(self.G.ecount())
        self.G.es["A2"] = np.zeros(self.G.ecount())

        self.G.es["Q1Prev"] = np.zeros(self.G.ecount())
        self.G.es["Q2Prev"] = np.zeros(self.G.ecount())
        self.G.es["A1Prev"] = np.zeros(self.G.ecount())
        self.G.es["A2Prev"] = np.zeros(self.G.ecount())
        # pprint(self.G.summary())

    # TODO: change runoff/DrainOverflows/DrainInflows to be hydraulicGraph agnostic for inputs and outputs
    def update(self, t, dt, coupledInputs):
        """
        Updates the A1,A2,Q1,Q2 of the network.

        Parameters:
        -----------
        t : float
            initial time
        dt : float
            time between initial time and desired end time
        coupledInputs : list(float)
            List of coupled Flow inputs, subcatchments and overflow for street and drain for Sewer.
            These are a list the length of the data file for ease of access

        Returns:
        --------
        A1 : list
            Updated area ordered by igraph id
        A2 : list
            Updated area ordered by igraph id
        Q1 : list
            Updated flow ordered by igraph id
        Q2 : list
            Updated flow ordered by igraph id


        """
        def solveContinuity(eid, THETA, PHI):
            """uses normalized flow."""
            # Initial guess at previous timestep
            aNew = eid["A2Prev"]
            #1. get consts
            c1 = (eid["length"] * THETA) / (dt * PHI)
            c2 = (
                c1
                * (
                    (1 - THETA) * (eid["A1"] - eid["A1Prev"])
                    - (THETA * eid["A2Prev"])
                )
                + ((1 - PHI) / PHI)
                * (eid["Q2Prev"] - eid["Q1Prev"])
                - eid["Q1"]
            )
            beta = np.sqrt(eid["slope"]) / eid["n"]
            

            #2. determine bounds on a
            aHi = 1.0;
            fHi = 1.0 + c1 + c2;

            # try lower bound st section factor is max
            aLo = self.psiFromArea(eid["PsiFull"], eid) / eid["Afull"]

            if aLo < aHi:
                fLo = ( beta * eid["PsiFull"] ) + (c1 * aLo) + c2;
            else:
                fLo = fHi

            # if fLo and fHi have same sign, set lo to 0
            if fLo*fHi > 0:
                aHi = aLo
                fHi = fLo
                aLo = 0.0
                fLo = c2
            if fLo*fHi >= 0:
                # do search
                # check that A2Prev is in interval
                if aNew < aLo or aNew > aHi:
                    aNew = (aLo + aHi) / 2
                if fLo > fHi:
                    # check that bounds are the right way around
                    aTmp = aLo
                    aLo = aHi
                    aHi = aTmp
                aNew,n = newtonBisection(aLo,aHi,continuity, p={
                    "c1": c1,
                    "c2": c2,
                    "beta": beta,
                    "eid": eid,
                    }, xinit=aNew)
            elif fLo < 0:
                # use full flow
                aNew = 1.0
            elif fLo > 0:
                # use no flow
                aNew = 0.0
            pprint(f"new aNew: {aNew}")
            return aNew



        def continuity(x,p):
            """uses normalized flow."""
            f = (
                p["beta"] * self.psiFromArea(x, p["eid"])
                + p["c1"] * x
                + p["c2"]
            )
            fp = (
                p["beta"]
                * self.psiPrimeFromArea(x, p["eid"])
                + p["c1"]
            )
            return f, fp


            pass
        def kinematic(dt,THETA,PHI):
            """wrapper for kinematic wave model."""
            for eid in self.G.es:
                solveContinuity(eid,THETA,PHI)
        THETA = 0.6
        PHI = 0.6
        kinematic(dt,THETA,PHI)
        depth = 0
        averageArea = 0
        peakDischarge = 0
        return depth, averageArea, coupledInputs, peakDischarge


if __name__ == "__main__":
    print("Dont call this directly :(")
