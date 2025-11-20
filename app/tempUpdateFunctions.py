class tempUpdate:
    def __init__(self, graphType, file):
        pass

    def NEWERupdate(self, t, dt, coupledInputs):
        """
        Updates the attributes of the network.

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
                        0, fullAreaStreet(p), A2NewFunc, p=p, xinit=self.G.es[eid]["A2"]
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

    def OLDUpdate(self, t, dt, runoff, drainOverflows):
        """
        Updates the attributes of the Street network using the kinematic Model.

        Parameters:
        -----------
        t : float
            initial time
        dt : float
            time between initial time and desired end time
        rainfall : float
            average rainfall measured over the time [t,t+dt]

        Returns:
        --------
        depths : list
            Updated depths ordered by igraph id

        """
        # keeps track of inflows for sewer coupling
        peakDischarge = 0.0
        drainInflow = np.zeros(self.G.vcount())

        def kineticFlow(t, dt, runoff, drainOverflows, theta=0.6, phi=0.6):
            """
            TODO: List assumptions. Uses mannings equation and continuity of mass to take the total inflow and write it as
            a discharge considering the pipe shape.

            Parameters:
            -----------
            t : float
                the current time in the ode
            x : list(float)
                list of depths
            theta : float
                one of the two weights for numerical pde method
            phi : float
                one of the two weights for numerical pde method
            """

            # 1. check acyclic
            if not self.G.is_dag():
                raise ValueError("Street Network must be acyclic.")

            # 2. top sort
            order = self.G.topological_sorting()
            # pprint(order)

            for nid in order:
                # 3. Get inflows
                # skips any node without outgoing edges
                if self.G.degree(nid, mode="out") == 0:
                    continue
                edge = self.G.vs[nid].out_edges()[0].index
                Q1 = self.G.es[edge]["Q1"]
                A1 = self.G.es[edge]["A1"]
                Q2 = self.G.es[edge]["Q2"]
                A2 = self.G.es[edge]["A2"]
                Q1New = 0.0
                A1New = 0.0
                Q2New = 0.0
                A2New = 0.0
                slope = self.G.es[edge]["slope"]
                Sx = self.G.es[edge]["Sx"]
                n = self.G.es[edge]["n"]
                drainLength = self.G.vs[nid]["drainLength"]
                drainWidth = self.G.vs[nid]["drainWidth"]
                beta = np.power(slope, 0.5) / self.G.es[edge]["n"]

                # check if node has drain
                if self.G.vs[nid]["drain"] == 0:
                    drainOutflow = 0.0
                # check if drain is overflowing
                elif drainOverflows[nid] > 0.0:
                    drainOutflow = drainOverflows[nid]
                # flow water into drain
                else:
                    drainInflow[nid] = capturedFlow(
                        Q1, A1, slope, Sx, drainLength, drainWidth, n
                    )
                    drainOutflow = -1 * drainInflow[nid]
                # pprint(f"Drain Outflow for {nid}: {drainOutflow}")
                # get Q2 of incoming edges
                incomingQs = 0.0
                for e in self.G.vs[nid].in_edges():
                    incomingQs += e["Q2"]
                # pprint(f"Incoming Qs: {incomingQs}")

                Q1New = drainOutflow + runoff[nid] + incomingQs

                Amax = A_tbl[-1]

                # pprint(f"A_tbl: {A_tbl}")
                # pprint(f"A_tbl[-1]: {A_tbl[-1]}")
                # pprint(f"A_tbl[0]: {A_tbl[0]}")
                # pprint(f"Amax: {Amax}")
                def phiInverse(x, p):
                    f = (
                        psiFromAreaStreet(
                            (p["Q1New"] / p["n"]), p["A_tbl"], p["R_tbl"], p["yFull"]
                        )
                        - x
                    )
                    fp = psiPrimeFromAreaStreet(
                        (p["Q1New"] / p["n"]), p["A_tbl"], p["R_tbl"], p["yFull"]
                    )
                    return f, fp

                p = {
                    "Q1New": Q1New,
                    "n": self.G.es[edge]["n"],
                    "A_tbl": A_tbl,
                    "R_tbl": R_tbl,
                    "yFull": self.yFull,
                }
                A1New, _ = newtonBisection(1e-16, Amax, phiInverse, p=p)
                # pprint(f"A1New From Bisection: {A1New}")

                c1 = (drainLength * theta) / (dt * phi)

                c2 = (
                    c1 * ((1 - theta) * (A1New - A1) - theta * A2)
                    + ((1 - phi) / phi) * (Q2 - Q1)
                    - Q1New
                )

                def A2NewFunction(x, p):
                    f = (
                        beta * psiFromAreaStreet(x, p["A_tbl"], p["R_tbl"], p["yFull"])
                        + c1 * x
                        + c2
                    )
                    fp = (
                        beta
                        * psiPrimeFromAreaStreet(x, p["A_tbl"], p["R_tbl"], p["yFull"])
                        + c1
                    )
                    return f, fp

                p = {
                    "beta": beta,
                    "c1": c1,
                    "c2": c2,
                    "A_tbl": A_tbl,
                    "R_tbl": R_tbl,
                    "yFull": self.yFull,
                }
                A2New, _ = newtonBisection(1e-16, Amax, A2NewFunction, p=p, xinit=A2)
                # A2New, _ = newtonBisection(0, Amax, A2NewFunction, tol=Amax*0.0001, p=p, xinit=A2)

                Q2New = beta * psiFromAreaStreet(A2New, A_tbl, R_tbl, self.yFull)
                # TODO: Do depth after all areas computed
                # d1New = depthFromAreaStreet(A2New)
                # d2New = depthFromAreaStreet(A2New)

                pprint(f" A1: {A1}")
                pprint(f" A2: {A2}")
                pprint(f" Q1: {Q1}")
                pprint(f" Q2: {Q2}")
                pprint(f" A1New: {A1New}")
                pprint(f" A2New: {A2New}")
                pprint(f" Q1New: {Q1New}")
                pprint(f" Q2New: {Q2New}")

                self.G.es[edge]["Q1New"] = Q1New
                self.G.es[edge]["A1New"] = A1New
                self.G.es[edge]["Q2New"] = Q2New
                self.G.es[edge]["A2New"] = A2New

                # pprint(self.G.vs[nid].in_edges())
            # Update A,Q's
            self.G.es["A1"] = np.nan_to_num(self.G.es["A1New"])
            self.G.es["A2"] = np.nan_to_num(self.G.es["A2New"])
            self.G.es["Q1"] = np.nan_to_num(self.G.es["Q1New"])
            self.G.es["Q2"] = np.nan_to_num(self.G.es["Q2New"])
            peakDischarge = np.max(np.abs(self.G.es["Q1"] + self.G.es["Q2"]))
            # compute depth's
            for nid in order:
                maxDepth = 0.0
                for edge in self.G.vs[nid].out_edges():
                    tempDepth = depthFromAreaStreet(edge["A1"], A_tbl, self.yFull)
                    if tempDepth > maxDepth:
                        maxDepth = tempDepth
                for edge in self.G.vs[nid].in_edges():
                    tempDepth = depthFromAreaStreet(edge["A2"], A_tbl, self.yFull)
                    if tempDepth > maxDepth:
                        maxDepth = tempDepth
                if self.G.vs[nid]["depth"] > self.yFull:
                    pprint(
                        f"WARNING: Node {nid} lost {self.G.vs[nid]['depth'] - self.yFull} due to overflow. Forcing depth to yFull."
                    )
                    self.G.vs[nid]["depth"] = self.yFull
                else:
                    self.G.vs[nid]["depth"] = maxDepth

        kineticFlow(t, dt, runoff, drainOverflows)
        # TODO: Add more reporting things here
        averageArea = np.divide(self.G.es["A1"] + self.G.es["A2"], 2.0)

        outfallNode = self.G.vs.select(type_eq=1)[0]
        lastEdge = outfallNode.in_edges()[0]
        pprint(f"Outfall Flow: {self.G.es[lastEdge.index]['Q2New']}")
        peakDischarge = self.G.es[lastEdge.index]["Q2New"]
        # pprint(f"Average Area: {averageArea}")
        # pprint(f"New Depth:{self.G.vs['depth']}")
        return self.G.vs["depth"], averageArea, drainInflow, peakDischarge
