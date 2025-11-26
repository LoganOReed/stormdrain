import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
from pprint import pprint
from .newtonBisection import newtonBisection



class SubcatchmentGraph:
    """Class which implements the Subcatchment Network."""

    def __init__(self, file, oldwaterRatio=0.2):
        super(SubcatchmentGraph, self).__init__()
        self.oldwaterRatio = oldwaterRatio
        # TODO: Make csv also include subcatchment edges
        data = pd.read_csv(f"data/{file}.csv")
        street = data[data["type"].str.contains("STREET")]
        data = data[data["type"].str.contains("SUBCATCHMENT")]
        n = data.shape[0]

        # TODO: Add ability to read in subcatchment edges Needed to create edges
        edges = []

        self.hydraulicCoupling = np.array(data["outgoing"].astype(int))
        
        # Read area and width from CSV if available, otherwise use defaults
        if "area" in data.columns:
            areas = np.array(data["area"].astype(float))
        else:
            # Default area of 10000 m² per subcatchment
            areas = np.full(n, 10000.0)
            
        if "width" in data.columns:
            widths = np.array(data["width"].astype(float))
        else:
            # Default width of 100 m per subcatchment
            widths = np.full(n, 100.0)
            
        # Read Manning's n if available
        if "n_Manning" in data.columns:
            n_manning = np.array(data["n_Manning"].astype(float))
        else:
            n_manning = np.full(n, 0.017)
        
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
                "area": areas,
                "width": widths,
                "slope": np.array(data["slope"].astype(float)),
                "n": n_manning,
                "depth": np.zeros(n),
                "runoff": np.zeros(n),
            },
        )
        self.G.vs["coupledStreet"] = np.array(data["outgoing"])
        # pprint(f"outgoing indexes: {np.array(data["outgoing"])}")

    def update(self, t, dt, rainfall):
        """
        Updates the attributes of the network using the ode defined in "ode".

        Parameters:
        -----------
        t : float
            initial time
        dt : float
            time between initial time and desired end time
        rainfall : list(float)
            average rainfall measured over the time [t,t+dt]

        """
        def ode(t, x):
            """
            Solves d_t = f - alpha * (d-ds)^5/3.

            Parameters:
            -----------
            t: time
            x : variable of ode.
            """
            y = np.zeros(self.G.vcount())
            incomingRunoff = np.zeros(self.G.vcount())
            for i in self.G.topological_sorting():
                # calculate incoming runoff, using top sorting to guarantee the previous runoffs are already computed
                inEdges = self.G.vs[i].in_edges()
                for e in inEdges:
                    incomingRunoff[i] += self.G.vs["depth"][e.source]

                # alpha in manning equation
                a = (self.G.vs["width"][i] * np.power(self.G.vs["slope"][i], 0.5)) / (
                    self.G.vs["area"][i] * self.G.vs["n"][i]
                )
                depth_above_invert = np.maximum(x[i] - self.G.vs["invert"][i], 0.0)
                # outgoingRunoff (rate, in m/s)
                outflow_rate = a * np.power(depth_above_invert, 5 / 3)
                # NOTE: We remove a certain percentage of the rainfall as old water (infiltration + evaporation)
                y[i] = rainfall * (1 - self.oldwaterRatio) + incomingRunoff[i] - outflow_rate
            return y

        # NOTE: RK45 returns an iterator we need to use solve_ivp
        solution = sc.integrate.solve_ivp(
            ode, (t, t + dt), self.G.vs["depth"], method="RK45"
        )
        self.G.vs["depth"] = solution.y[:, -1]
        
        # BUGFIX: Compute outflow from the FINAL depths, not from intermediate ODE evaluations
        # RK45 calls the ODE multiple times at intermediate points, so we must recalculate
        # the outflow using the actual final depths
        final_outflow = np.zeros(self.G.vcount())
        for i in range(self.G.vcount()):
            a = (self.G.vs["width"][i] * np.power(self.G.vs["slope"][i], 0.5)) / (
                self.G.vs["area"][i] * self.G.vs["n"][i]
            )
            depth_above_invert = np.maximum(self.G.vs["depth"][i] - self.G.vs["invert"][i], 0.0)
            final_outflow[i] = a * np.power(depth_above_invert, 5 / 3)
        
        # Convert outflow rate (m/s) to volumetric flow (m³/s)
        self.G.vs["runoff"] = np.array(final_outflow * self.G.vs['area'])

        # pprint(f"outflow: {self.G.vs['runoff']}")
        # pprint(f"type: {type(self.G.vs['runoff'])}")

        return solution.y[:, -1], np.array(self.G.vs["runoff"])

    def visualize(self, times, depths, fileName=None):
        """
        Visualize depth over time for each subcatchment.

        Parameters:
        -----------
        times : 1-d list
            Array of time points
        depths : 2-d list
            List where each element is an array of depths at that time point
            Should have shape (n_timesteps, n_vertices)
        """
        depths_array = np.array(depths)

        plt.figure(figsize=(10, 6))

        for i in range(self.G.vcount()):
            plt.plot(
                times,
                depths_array[:, i],
                label=f"Subcatchment {i}",
                # marker='o',
                linewidth=2,
            )

        plt.xlabel("Time (hours)", fontsize=12)
        plt.ylabel("Depth (m)", fontsize=12)
        plt.title("Subcatchment Depth vs Time", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if fileName == None:
            fileName = "subcatchmentGraph"
        plt.savefig(f"figures/{fileName}.png")
        plt.show()
