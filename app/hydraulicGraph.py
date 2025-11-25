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
    areaFromPsiStreet,
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
        self.PHI = 0.6
        self.THETA = 0.6
        if graphType == "STREET":
            self.depthFromArea = depthFromAreaStreet
            self.psiFromArea = psiFromAreaStreet
            self.psiPrimeFromArea = psiPrimeFromAreaStreet
            self.areaFromPsi = areaFromPsiStreet
        else:
            # TODO: Change this to circular geometry
            self.depthFromArea = depthFromAreaCircle
            self.psiFromArea = psiFromAreaCircle
            self.psiPrimeFromArea = psiPrimeFromAreaCircle
            self.areaFromPsi = areaFromPsiCircle

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
            if slope < 0.0001:
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
        self.G.es["S_back"] = 0.02 
        self.G.es["Sx"] = 0.02 

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

        # used for manning equation
        self.G.es["beta"] = np.array([np.sqrt(e["slope"]) / e["n"] for e in self.G.es]) 
        self.G.es["qFull"] = np.array([e["beta"] * e["PsiFull"] for e in self.G.es]) 

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
    def update(self, t, dt, coupling):
        """
        Updates the A1,A2,Q1,Q2 of the network.

        Parameters:
        -----------
        t : float
            initial time
        dt : float
            time between initial time and desired end time
        coupling : list(float)
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
        def getIncomingEdges(n,G):
            # total incoming is 0 if nothing is incoming
            if n.indegree() == 0:
                return 0.0
            totalIncoming = 0.0
            for incomingEdge in n.incident(mode="in"):
                totalIncoming += incomingEdge["Q2"]
            pprint(f"totalIncoming for {n.index}: {totalIncoming}")
            return totalIncoming
        def getIncomingCoupled(n,coupling):
            incomingCoupled = 0.0
            #1. add runoff
            incomingCoupled += coupling["subcatchmentRunoff"][n["coupledID"] - 1]
            #2. add drain
            incomingCoupled += coupling["drainCapture"][n["coupledID"] - 1]
            #3. add overflow
            incomingCoupled += coupling["drainOverflow"][n["coupledID"] - 1]
            pprint(f"incoming coupled for {n.index}: {incomingCoupled}")
            return incomingCoupled

        def solveContinuity(dt,eid):
            """
            Solves continuity equation for normalized area at downstream end.
            Uses normalized flow (divided by qFull).
            
            Returns: aNew (normalized area, 0-1)
            """
            # Initial guess from previous timestep (normalize it)
            aNew = eid["A2"] / eid["Afull"]
            
            #1. Compute constants for finite difference equation
            c1 = (eid["length"] * self.THETA) / (dt * self.PHI)
            
            # Normalize all terms
            c2 = (
                c1 * (
                    (1 - self.THETA) * ((eid["A1"] - eid["A1Prev"]) / eid["Afull"])
                    - (self.THETA * eid["A2Prev"] / eid["Afull"])
                )
                + ((1 - self.PHI) / self.PHI) * ((eid["Q2Prev"] - eid["Q1Prev"]) / eid["qFull"])
                - (eid["Q1"] / eid["qFull"])
            )
            
            betaScaled = (np.sqrt(eid["slope"]) / eid["n"]) / eid["qFull"]
            
            #2. Set up bounds for root finding
            aHi = 1.0  # Full flow
            # For full flow: psi/psiFull = 1.0 (approximately)
            fHi = betaScaled * 1.0 + c1 * aHi + c2
            
            # Lower bound: try area that gives maximum section factor (typically ~0.94 for pipes)
            try:
                aMaxPsi = self.areaFromPsi(eid["PsiFull"], eid) / eid["Afull"]
                aLo = min(aMaxPsi, 0.94)
            except:
                aLo = 0.5  # fallback
            
            fLo = betaScaled * (eid["PsiFull"] / eid["PsiFull"]) + c1 * aLo + c2
            
            # Check if bounds bracket the root
            if fLo * fHi > 0.0:
                # Not bracketed - try extending lower bound to zero
                aLo = 0.0
                fLo = c2  # psi(0) = 0
                
                if fLo * fHi > 0.0:
                    # Still not bracketed - make decision based on signs
                    if fLo < 0.0 and fHi < 0.0:
                        # Both negative - equation wants more than pipe can handle
                        return 1.0
                    else:
                        # Both positive/zero - equation wants no flow
                        return 0.0
            
            # Ensure correct ordering: fLo < 0 < fHi (typically)
            if fLo > fHi:
                aLo, aHi = aHi, aLo
                fLo, fHi = fHi, fLo
            
            # Validate initial guess is in range
            if aNew < aLo or aNew > aHi:
                aNew = (aLo + aHi) / 2.0
            
            # Solve using Newton-Bisection
            aNew, n = newtonBisection(
                aLo, aHi, continuity, 
                p={
                    "c1": c1,
                    "c2": c2,
                    "betaScaled": betaScaled,
                    "eid": eid,
                }, 
                xinit=aNew
            )
            
            # Clamp to valid range
            aNew = np.clip(aNew, 0.0, 1.0)
            
            return aNew

        def continuity(x,p):
            """
            Continuity equation for Newton-Bisection solver.
            
            Parameters:
            x : float - normalized area (0-1)
            p : dict - parameters containing c1, c2, betaScaled, eid
            
            Returns:
            f : float - function value
            fp : float - derivative value
            """
            # Convert normalized area to actual area for geometry functions
            A_actual = x * p["eid"]["Afull"]
            
            # Get psi (returns actual value)
            psi_actual = self.psiFromArea(A_actual, p["eid"])
            
            # Normalize psi for equation
            psi_normalized = psi_actual / p["eid"]["PsiFull"]
            
            # Get derivative
            psiPrime_actual = self.psiPrimeFromArea(A_actual, p["eid"])
            
            # Scale derivative: d(psi_norm)/d(a_norm) = d(psi_actual)/d(A_actual) * dA/da
            # where dA/da = Afull
            psiPrime_scaled = psiPrime_actual * p["eid"]["Afull"] / p["eid"]["PsiFull"]
            
            # Equation in normalized form
            f = p["betaScaled"] * psi_normalized + p["c1"] * x + p["c2"]
            fp = p["betaScaled"] * psiPrime_scaled + p["c1"]
            
            return f, fp

                
        def validate_timestep():
            """Check for common numerical issues"""
            # Check for negative values
            Q1_vals = np.array(self.G.es["Q1"])
            Q2_vals = np.array(self.G.es["Q2"])
            A1_vals = np.array(self.G.es["A1"])
            A2_vals = np.array(self.G.es["A2"])
            
            if np.any(Q1_vals < -1e-10):
                pprint(f"WARNING: Negative Q1 detected: min={np.min(Q1_vals):.6e}")
            if np.any(Q2_vals < -1e-10):
                pprint(f"WARNING: Negative Q2 detected: min={np.min(Q2_vals):.6e}")
            if np.any(A1_vals < -1e-10):
                pprint(f"WARNING: Negative A1 detected: min={np.min(A1_vals):.6e}")
            if np.any(A2_vals < -1e-10):
                pprint(f"WARNING: Negative A2 detected: min={np.min(A2_vals):.6e}")
            
            # Check for NaN or Inf
            if np.any(np.isnan(Q1_vals)) or np.any(np.isinf(Q1_vals)):
                pprint("WARNING: NaN or Inf in Q1!")
            if np.any(np.isnan(Q2_vals)) or np.any(np.isinf(Q2_vals)):
                pprint("WARNING: NaN or Inf in Q2!")
                
        # Call the kinematic wave solver
        def kinematic(dt):
            """wrapper for kinematic wave model."""
            # check that network is acyclic
            if not self.G.is_dag():
                raise ValueError("Street Network must be acyclic.")

            # save previous values
            self.G.es["Q1Prev"] = np.array(self.G.es["Q1"])
            self.G.es["Q2Prev"] = np.array(self.G.es["Q2"])
            self.G.es["A1Prev"] = np.array(self.G.es["A1"])
            self.G.es["A2Prev"] = np.array(self.G.es["A2"])

            #1. topo sort
            topoOrder = self.G.topological_sorting()
            for nid in topoOrder:
                n = self.G.vs[nid]
                if n.outdegree() != 0:
                    e = self.G.vs[nid].incident(mode="out")[0]
                else:
                    pprint(f"skipping update for nid: {nid}")
                    continue

                #2. Q1 (get incoming edges and coupling terms)
                incomingEdges = getIncomingEdges(n,self.G)
                incomingCoupled = getIncomingCoupled(n,coupling)
                
                # Handle negative inflow
                if incomingEdges + incomingCoupled < 0:
                    e["Q1"] = 0.0
                    qin = 0.0
                else:
                    e["Q1"] = incomingEdges + incomingCoupled
                    qin = e["Q1"] / e["qFull"]

                #3. A1 from inverse Manning equation: Q = beta * psi(A)
                if qin >= 1.0:
                    # Flow at or exceeds capacity
                    ain = 1.0
                elif qin <= 1e-10:
                    # Essentially no flow
                    ain = 0.0
                else:
                    # Use inverse: Q = beta * psi, so psi = Q/beta, then A = areaFromPsi(psi)
                    beta = np.sqrt(e["slope"]) / e["n"]
                    psi_needed = e["Q1"] / beta
                    
                    # Clamp to valid range
                    psi_needed = min(psi_needed, e["PsiFull"])
                    
                    # Get area from psi (areaFromPsi expects actual psi, returns actual area)
                    A1_actual = self.areaFromPsi(psi_needed, e)
                    ain = A1_actual / e["Afull"]
                    
                    pprint(f"Node {nid}: qin={qin:.6f}, psi_needed={psi_needed:.6f}, ain={ain:.6f}")

                #4. solve continuity for a2
                # Only check inflow for zero condition (allow drainage if previous A2 > 0)
                if qin <= 1e-10:
                    # No inflow - check if there's water to drain
                    if e["A2Prev"] > 1e-10:
                        # Allow drainage
                        aout = solveContinuity(dt, e)
                        # Compute outflow from Manning
                        beta = np.sqrt(e["slope"]) / e["n"]
                        A2_actual = aout * e["Afull"]
                        psi2 = self.psiFromArea(A2_actual, e)
                        qout = (beta * psi2) / e["qFull"]
                    else:
                        # No water to drain
                        aout = 0.0
                        qout = 0.0
                else:
                    # Normal solve with inflow
                    aout = solveContinuity(dt, e)
                    
                    # Compute Q2 from A2 using Manning equation
                    beta = np.sqrt(e["slope"]) / e["n"]
                    A2_actual = aout * e["Afull"]
                    psi2 = self.psiFromArea(A2_actual, e)
                    qout = (beta * psi2) / e["qFull"]
                    
                    # Clamp qin for saving
                    qin = min(qin, 1.0)

                #5. save results (convert back to actual values)
                e["Q1"] = qin * e["qFull"]
                e["A1"] = ain * e["Afull"]
                e["Q2"] = qout * e["qFull"]
                e["A2"] = aout * e["Afull"]
                
                #6. Compute depth at node for visualization
                if e["A1"] > 0:
                    n["depth"] = self.depthFromArea(e["A1"], e)
                else:
                    n["depth"] = 0.0

                
        kinematic(dt)
        
        # Validate results for debugging
        validate_timestep()
        
        depth = self.G.vs["depth"]
        averageArea = np.divide(self.G.es["A1"] + self.G.es["A2"], 2.0)
        peakDischarge = np.max(self.G.es["Q2"])
        return depth, averageArea, coupling, peakDischarge


if __name__ == "__main__":
    print("Dont call this directly :(")
