import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import random
from .newton_bisection import findroot



# TODO: Create numpy docs for each function

        # self.G = ig.Graph(n=5, 
        #                   edges=[[0,1],[2,3],[3,1],[1,4]],
        #                   edge_attrs={
        #                       'flowrate': [0.0, 0.0, 0.0, 0.0]
        #                       'flowarea': [0.0, 0.0, 0.0, 0.0]
        #                       },
        #                   vertex_attrs={
        #                       'type': ['j', 'j', 'j', 'j', 'o'],
        #                       'depth': [0.0,0.0,0.0,0.0,0.0]
        #                       },
        #                   directed=True)

       

class SubcatchmentGraph:
    """General Graph Structure, Parent of the three subgraphs."""
    def __init__(self, n, vertex_attrs=None):
        super(SubcatchmentGraph, self).__init__()
        if vertex_attrs == None:
            self.G = ig.Graph(n=n,edges=[],directed=True,
                              vertex_attrs={
                                  'area': np.array([10000.0,10000.0,10000.0]),
                                  'width': np.array([100.0,100.0,100.0]),
                                  'slope': np.array([0.005,0.002,0.004]),
                                  'n': np.array([0.017,0.017,0.017]),
                                  'invert': np.array([0.0,0.016,0.035]),
                                  'x': np.array([100,200,200]),
                                  'y': np.array([100,100,0]),
                                  'z': np.array([0,0,0]),
                                  'depth': np.array([0.0,0.0,0.0])
                                  })
        else:
            self.G = ig.Graph(n=n,edges=[(0,1),(1,2)], directed=True,
                              vertex_attrs={
                                  'area': np.array([10000.0,10000.0,10000.0]),
                                  'width': np.array([100.0,100.0,100.0]),
                                  'slope': np.array([0.005,0.002,0.004]),
                                  'n': np.array([0.017,0.017,0.017]),
                                  'invert': np.array([0.0,0.016,0.035]),
                                  'x': np.array([100,200,200]),
                                  'y': np.array([100,100,0]),
                                  'z': np.array([0,0,0]),
                                  'depth': np.array([0.0,0.0,0.0])
                                  })
        # Rainfall (in hours 0-6)

        # print(self.G.summary())
        # print(self.G.topological_sorting())
        # print(self.rainfall)

    def update(self, t, dt, rainfall):
        """
        Updates the attributes of the network using the ode defined in "ode".

        Parameters:
        -----------
        t : float
            initial time
        dt : float
            time between initial time and desired end time
        rainfall : float
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
            outflow = np.zeros(self.G.vcount())
            incomingRunoff = np.zeros(self.G.vcount())
            for i in self.G.topological_sorting():
                # calculate incoming runoff, using top sorting to guarantee the previous runoffs are already computed
                inEdges = self.G.vs[i].in_edges()
                for e in inEdges:
                    incomingRunoff[i] += self.G.vs['depth'][e.source]

                # alpha in manning equation
                a = (self.G.vs['width'][i] * np.power(self.G.vs['slope'][i], 0.5)) / (self.G.vs['area'][i] * self.G.vs['n'][i])
                depth_above_invert = np.maximum(x[i] - self.G.vs['invert'][i], 0.0)
                # outgoingRunoff
                outflow[i] = a * np.power(depth_above_invert, 5/3)
                y[i] = rainfall + incomingRunoff[i] - outflow[i]
            print(f"incomingRunoff: {incomingRunoff}")
            return y
    
        # NOTE: RK45 returns an iterator we need to use solve_ivp
        solution = sc.integrate.solve_ivp(
            ode, 
            (t, t + dt), 
            self.G.vs['depth'], 
            method='RK45'
        )
        self.G.vs['depth'] = solution.y[:, -1]
        return solution.y[:,-1]

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
            plt.plot(times, depths_array[:, i], 
                    label=f'Subcatchment {i}', 
                    # marker='o', 
                    linewidth=2)
        
        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('Depth (m)', fontsize=12)
        plt.title('Subcatchment Depth vs Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if fileName == None:
            fileName = "test"
        plt.savefig(f"figures/{fileName}.png")



class SewerGraph:
    """Graph of Sewer portion of Hydraulic Network."""
    def __init__(self, n=None):
        super(SewerGraph, self).__init__()
        if n == None:
            self.G = ig.Graph(n=5,edges=[(0,1),(2,3),(3,1),(1,4)],directed=True,
                              vertex_attrs={
                                  'invert': np.array([0.0,0.016,0.035]),
                                  'x': np.array([100.0,100.0,200.0,200.0,0.0]),
                                  'y': np.array([100.0,0.0,100.0,0.0,0.0]),
                                  # z choice based on subcatchment slope, except for 4
                                  # 4 is ARBITRARY
                                  'z': np.array([0.5,0.0,0.6,0.4,-0.1]),
                                  'depth': np.array([0.0,0.0,0.0,0.0,0.0]),
                                  # 0 - junction
                                  # 1 - outfall
                                  'type': np.array([0,0,0,0,1])
                                  # 'subcatchmentCoupling': np.array([-1,-1,-1,-1,-1])
                                  })
            # calculate the lengths of each pipe
            for e in self.G.es:
                s = np.array([self.G.vs[e.source]['x'], self.G.vs[e.source]['y'], self.G.vs[e.source]['z']])
                d = np.array([self.G.vs[e.target]['x'], self.G.vs[e.target]['y'], self.G.vs[e.target]['z']])
                self.G.es[e.index]['length'] = np.linalg.norm(s - d)
            # calculate the slope of each pipe
            for e in self.G.es:
                slope = self.G.vs[e.source]['z'] - self.G.vs[e.target]['z']
                if slope < 0.0001:
                    print(f"WARNING: slope for edge {e} is too small.")
                self.G.es[e.index]['slope'] = self.G.vs[e.source]['z'] - self.G.vs[e.target]['z']
            print(self.G.es['slope'])
            # TODO: add offset height calculations
            # Needs to be given a priori
            self.G.es['offsetHeight'] = [0.0 for _ in range(self.G.ecount())]
        else:
            self.G = ig.Graph(n=n,edges=[], directed=True)
        # Rainfall (in hours 0-6)

        # print(self.G.summary())
        # print(self.G.topological_sorting())
        # print(self.rainfall)

    def _steadyFlow(self, t, x):
        """
        TODO: List assumptions. Uses mannings equation to take the total inflow and write it as
        a discharge considering the pipe shape.

        Parameters:
        -----------
        t : float
            the current time in the ode
        x : list(float)
            list of depths
        """
        pass


    def _getDepthCircularPipe(self, q, diam, n, s):
        """
        Compute depth for given flow on Circular Pipe using Manning equation.
        Solves Q = (1/n) * A * R^(2/3) * S^(1/2) for depth.
        Using Newton-Raphson.
        """
        pass


    def update(self, t, dt, rainfall):
        """
        Updates the attributes of the network using the ode defined in "ode".

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
        def ode(t, x):
            """
            Solves d_t = f - alpha * (d-ds)^5/3.

            Parameters:
            -----------
            t: time
            x : variable of ode.
            """
            y = np.zeros(self.G.vcount())
            outflow = np.zeros(self.G.vcount())
            incomingRunoff = np.zeros(self.G.vcount())
            for i in self.G.topological_sorting():
                # calculate incoming runoff, using top sorting to guarantee the previous runoffs are already computed
                inEdges = self.G.vs[i].in_edges()
                for e in inEdges:
                    incomingRunoff[i] += self.G.vs['depth'][e.source]

                # alpha in manning equation
                a = (self.G.vs['width'][i] * np.power(self.G.vs['slope'][i], 0.5)) / (self.G.vs['area'][i] * self.G.vs['n'][i])
                depth_above_invert = np.maximum(x[i] - self.G.vs['invert'][i], 0.0)
                # outgoingRunoff
                outflow[i] = a * np.power(depth_above_invert, 5/3)
                y[i] = rainfall + incomingRunoff[i] - outflow[i]
            print(f"incomingRunoff: {incomingRunoff}")
            return y
    
        # NOTE: RK45 returns an iterator we need to use solve_ivp
        solution = sc.integrate.solve_ivp(
            ode, 
            (t, t + dt), 
            self.G.vs['depth'], 
            method='RK45'
        )
        self.G.vs['depth'] = solution.y[:, -1]
        return solution.y[:,-1]

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
            plt.plot(times, depths_array[:, i], 
                    label=f'Subcatchment {i}', 
                    # marker='o', 
                    linewidth=2)
        
        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('Depth (m)', fontsize=12)
        plt.title('Subcatchment Depth vs Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if fileName == None:
            fileName = "test"
        plt.savefig(f"figures/{fileName}.png")





if __name__ == "__main__":
    print("Dont call this directly :(")
    print("use one of the example files")

