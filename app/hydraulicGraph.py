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
)
from .circularGeometry import (
    depthFromAreaCircle,
    psiFromAreaCircle,
    psiPrimeFromAreaCircle,
    areaFromPsiCircle,
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
            self.yFull = STREET_Y_FULL  # diff between lowest and highest point from choice of Street Parameters
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
            self.G.es["yFull"] = [STREET_Y_FULL for _ in self.G.es]
        else:
            # corresponds to around 18in pipe
            self.G.es["yFull"] = [0.5 for _ in self.G.es]

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

    def getCoupledInputs(coupled):
        """Converts coupled matrix for all networks into sum of coupling terms in network id order."""
        pprint(f'Coupled IDs for {self.graphType} network: self.G.vs["coupledID"]')

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
        newCoupledInputs = coupledInputs
        # save previous iteration incase its needed
        self.G.es["Q1Prev"] = self.G.es["Q1"]
        self.G.es["Q2Prev"] = self.G.es["Q2"]
        self.G.es["A1Prev"] = self.G.es["A1"]
        self.G.es["A2Prev"] = self.G.es["A2"]
        # Parameters
        PHI = 0.6
        THETA = 0.6
        # 0. check acyclic
        if not self.G.is_dag():
            raise ValueError(f"{self.graphType} Network must be acyclic.")

        # 1. top sort
        order = self.G.topological_sorting()

        for nid in order:
            # NOTE: This should only be outfall nodes, so it would be a good place to track discharge
            # pprint(f"ordered nodes: {order}")

            if self.G.vs[nid].outdegree() != 0:
                eid = self.G.vs[nid].incident(mode="out")[0].index
            # pprint(f"eid: {eid}")
            # 2. get Q_2^n+1 of prev nodes
            if self.G.vs[nid].indegree() != 0:
                iedges = np.array([a.index for a in self.G.vs[nid].incident(mode="in")])
            else:
                iedges = []
            # pprint(f"indegree of {nid}: {self.G.vs[nid].indegree()} and incident to: {iedges}")
            # pprint(f"Node: {nid} edges: {iedges}")

            # get incoming edges Q2^n+1
            # incomingEdgeFlows = [self.G.get_eid(i, nid) for i in iedges]
            # NOTE: Because of the topological sorting these will be Q2^n+1
            incomingEdgeFlows = np.sum(self.G.es[iedges]["Q2"])
            # pprint(f'incident to {nid}: {iedges}')
            # pprint(f"IncomingEdgeFlows: {incomingEdgeFlows}")
            # pprint(f"incomingEdgeFlows: {incomingEdgeFlows}")

            # If this loop is the outfall node, keepthis as the peak discharge
            peakDischarge = 0.0
            if self.G.vs[nid].outdegree() == 0:
                peakDischarge = incomingEdgeFlows
                # pprint(f"Peak Discharge for {self.graphType} is {peakDischarge}")

            # 3. get any coupled inputs for the node (which are computed elsewhere)
            # subcatchments
            if self.graphType == "STREET":
                subcatchmentIncomingFlow = coupledInputs["subcatchmentRunoff"][
                    self.G.vs[nid]["coupledID"]
                ]

                # pprint(f"incoming runoff {nid}: {subcatchmentIncomingFlow}")
            else:
                subcatchmentIncomingFlow = 0

            # drains
            if self.graphType == "STREET":
                # compute captured flow if node has drain
                if self.G.vs[nid]["drain"] == 1:
                    # pprint(f"Previous Q: {self.G.es[eid]["Q1Prev"]}\nPrevious A: {self.G.es[eid]["A1Prev"]}")
                    coupledInputs["drainCapture"][self.G.vs[nid]["coupledID"] - 1] = (
                        capturedFlow(
                            self.G.es[eid]["Q1Prev"],
                            self.G.es[eid]["A1Prev"],
                            self.G.es[eid]["slope"],
                            self.G.es[eid]["Sx"],
                            self.G.vs[nid]["drainLength"],
                            self.G.vs[nid]["drainWidth"],
                            self.G.es[eid]["n"],
                        )
                    )
                    # pprint(f'Q: {self.G.es[eid]["Q1Prev"]}\n A: {self.G.es[eid]["A1Prev"]}\n slope: {self.G.es[eid]["slope"]}\n Sx: {self.G.es[eid]["Sx"]}\n drain length: {self.G.vs[nid]["drainLength"]}\n drain width: {self.G.vs[nid]["drainWidth"]}\n {self.G.es[eid]["n"]}')
                    # pprint(f"capturedFlow for {nid}: {coupledInputs['drainCapture'][self.G.vs[nid]["coupledID"]-1]}")
                drainCaptureIncomingFlow = 0
                drainCaptureOutgoingFlow = (
                    -1 * coupledInputs["drainCapture"][self.G.vs[nid]["coupledID"] - 1]
                )
                if drainCaptureOutgoingFlow > 0:
                    raise ValueError(
                        f"{self.graphType} node {nid} has positive drain capture, it should be negative: {drainCaptureOutgoingFlow}."
                    )
            else:
                # TODO: Need to actual to overflow
                drainCaptureOutgoingFlow = 0
                if self.G.vs[nid]["drainCoupledID"] != -1:
                    drainCaptureIncomingFlow = coupledInputs["drainCapture"][
                        self.G.vs[nid]["drainCoupledID"] - 1
                    ]
                else:
                    drainCaptureIncomingFlow = 0
                if drainCaptureIncomingFlow < 0:
                    raise ValueError(
                        f"{self.graphType} node {nid} has negative drain capture, it should be positive: {drainCaptureIncomingFlow}."
                    )

            # overflow
            # TODO: Do overflow this eventually
            drainOverflow = 0

            # combine all of the incoming coupling terms
            incomingCoupledFlows = (
                subcatchmentIncomingFlow + drainCaptureIncomingFlow + drainOverflow
            )

            # update coupledInputs

            # coupledIncomingFlows = coupledInputs[self.G.vs[nid]["coupledID"]]
            # pprint(f"coupledIncomingFlows: {coupledIncomingFlows}")

            # 4. Use #2. and #3. to find Q_1^n+1
            self.G.es[eid]["Q1"] = incomingEdgeFlows + incomingCoupledFlows
            self.G.es[eid]["Q1"] = max(
                0, self.G.es[eid]["Q1"] + drainCaptureOutgoingFlow
            )
            # pprint(f"New Q1: {self.G.es[eid]["Q1"]}")

            # 5. Compute A_1^n+1 using Manning and #4.
            p = {
                "Q1": self.G.es[eid]["Q1"],
                "n": self.G.es[eid]["n"],
                "yFull": self.G.es[eid]["yFull"],
                "T_curb": self.G.es[eid]["T_curb"],
                "T_crown": self.G.es[eid]["T_crown"],
                "H_curb": self.G.es[eid]["H_curb"],
                "S_back": self.G.es[eid]["S_back"],
                "Sx": self.G.es[eid]["Sx"],
            }
            if self.graphType == "STREET":

                def phiInverse(x, p):
                    f = psiFromAreaStreet((p["Q1"] / p["n"]), p) - x
                    fp = psiPrimeFromAreaStreet((p["Q1"] / p["n"]), p)
                    return f, fp
            else:

                def phiInverse(x, p):
                    f = psiFromAreaCircle((p["Q1"] / p["n"]), p["yFull"]) - x
                    fp = psiPrimeFromAreaCircle((p["Q1"] / p["n"]), p["yFull"])
                    return f, fp

            if self.G.es[eid]["Q1"] == 0.0:
                self.G.es[eid]["A1"] = 0.0
            elif self.graphType == "STREET":
                self.G.es[eid]["A1"], _ = newtonBisection(
                    0, fullAreaStreet(p), phiInverse, p=p
                )
                pprint(f"New A1 STREET: {self.G.es[eid]['A1']}")
            else:
                self.G.es[eid]["A1"], _ = newtonBisection(
                    0,
                    (np.pi / 4) * self.G.es[eid]["yFull"] * self.G.es[eid]["yFull"],
                    phiInverse,
                    p=p,
                )
                pprint(f"New A1 SEWER: {self.G.es[eid]['A1']}")
            # pprint(self.G.es[eid]["A1"])

            # 6. Setup nonlinear equation to get A_2^n+1
            c1 = (self.G.es[eid]["length"] * THETA) / (dt * PHI)
            # if c1 < 0:
            #     pprint(f"C1 is negative")

            c2 = (
                c1
                * (
                    (1 - THETA) * (self.G.es[eid]["A1"] - self.G.es[eid]["A1Prev"])
                    - (THETA * self.G.es[eid]["A2Prev"])
                )
                + ((1 - PHI) / PHI)
                * (self.G.es[eid]["Q2Prev"] - self.G.es[eid]["Q1Prev"])
                - self.G.es[eid]["Q1"]
            )
            # if c2 < 0:
            #     pprint(f"C2 is negative")
            beta = np.sqrt(self.G.es[eid]["slope"]) / self.G.es[eid]["n"]

            # if beta < 0:
            #     pprint(f"beta is negative")
            def A2Func(x):
                if self.graphType == "STREET":
                    return beta * psiFromAreaStreet(x, p) + c1 * x + c2
                else:
                    return (
                        beta * psiFromAreaCircle(x, self.G.es[eid]["yFull"])
                        + c1 * x
                        + c2
                    )

            # 7. Use newton to find the root aka A_2^n+1 instead of having to use derivative of psi

            # This checks for issues with bounding a root
            Afull = 0.7854 * self.G.es[eid]["yFull"] * self.G.es[eid]["yFull"]
            if self.graphType == "STREET" and A2Func(0.0) > 0 and A2Func(A_tbl[-1]) > 0:
                # pprint(f"A2func is bounded by two positive STREET")
                self.G.es[eid]["A2"] = 0.0
            elif (
                self.graphType == "STREET" and A2Func(0.0) < 0 and A2Func(A_tbl[-1]) < 0
            ):
                # pprint(f"A2func is bounded by two negative STREET")
                self.G.es[eid]["A2"] = A_tbl[-1]
            elif self.graphType == "SEWER" and A2Func(0.0) > 0 and A2Func(Afull) > 0:
                # pprint(f"A2func is bounded by two positive SEWER")
                self.G.es[eid]["A2"] = 0.0
            elif self.graphType == "SEWER" and A2Func(0.0) < 0 and A2Func(Afull) < 0:
                # pprint(f"A2func is bounded by two negative SEWER")
                self.G.es[eid]["A2"] = Afull
            # This is the usual case, without bad arithmetic
            else:
                if self.graphType == "STREET":
                    self.G.es[eid]["A2"] = newtonBisection(
                        0, fullAreaStreet(p), A2Func, p=p, xinit=self.G.es[eid]["A2"]
                    )
                else:
                    Afull = 0.7854 * self.G.es[eid]["yFull"] * self.G.es[eid]["yFull"]
                    sol = sp.optimize.root_scalar(
                        A2Func,
                        method="newton",
                        bracket=(
                            0.0,
                            0.7854 * self.G.es[eid]["yFull"] * self.G.es[eid]["yFull"],
                        ),
                        x0=self.G.es[eid]["A2Prev"],
                    )

                if sol.converged == False:
                    pprint(
                        f"WARNING: Kinematic Update on {self.graphType} failed to converge at edge {eid}. Setting A2 to 0"
                    )
                    self.G.es[eid]["A2"] = 0.0
                else:
                    self.G.es[eid]["A2"] = sol.root

            # pprint(f"A2: {self.G.es[eid]["A2"]}")

            # 8. Use Manning and #7. to get Q_2^n+1
            if self.G.es[eid]["A2"] == 0.0:
                self.G.es[eid]["Q2"] = 0.0
            elif self.graphType == "STREET":
                self.G.es[eid]["Q2"] = beta * psiFromAreaStreet(self.G.es[eid]["A2"], p)
            else:
                self.G.es[eid]["Q2"] = beta * psiFromAreaCircle(
                    self.G.es[eid]["A2"], self.G.es[eid]["yFull"]
                )
            # pprint(self.G.es[eid]["Q2"])

            # calculate depth
            for nid in order:
                maxDepth = 0.0
                for edge in self.G.vs[nid].out_edges():
                    if self.graphType == "STREET":
                        tempDepth = depthFromAreaStreet(edge["A1"], p)
                    else:
                        tempDepth = depthFromAreaCircle(
                            edge["A1"], self.G.es[edge.index]["yFull"]
                        )
                    if tempDepth > maxDepth:
                        maxDepth = tempDepth
                for edge in self.G.vs[nid].in_edges():
                    if self.graphType == "STREET":
                        tempDepth = depthFromAreaStreet(edge["A2"], p)
                    else:
                        tempDepth = depthFromAreaCircle(
                            edge["A2"], self.G.es[edge.index]["yFull"]
                        )
                    if tempDepth > maxDepth:
                        maxDepth = tempDepth
                if self.G.vs[nid]["depth"] > self.G.es[edge.index]["yFull"]:
                    pprint(
                        f"WARNING: Node {nid} lost {self.G.vs[nid]['depth'] - self.G.es[edge.index]['yFull']} due to overflow. Forcing depth to yFull."
                    )
                    self.G.vs[nid]["depth"] = self.G.es[edge.index]["yFull"]
                else:
                    self.G.vs[nid]["depth"] = maxDepth

        # return self.G.es[eid]["A1"], self.G.es[eid]["A2"], self.G.es[eid]["Q1"], self.G.es[eid]["Q2"]
        # return self.G.vs['depth'], averageArea, drainInflow, peakDischarge
        averageArea = np.divide(self.G.es["A1"] + self.G.es["A2"], 2.0)

        # if self.graphType == "STREET":
        #     peakDischarge = np.max(self.G.es['Q2'])
        # else:
        #     peakDischarge = 0
        peakDischarge = np.max(self.G.es["Q2"])
        return self.G.vs["depth"], averageArea, coupledInputs, peakDischarge


if __name__ == "__main__":
    print("Dont call this directly :(")
