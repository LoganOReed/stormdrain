import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import random



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
                                  'depth': np.array([0.0,0.0,0.0])
                                  })
        else:
            self.G = ig.Graph(n=n,edges=[[0,1],[1,2]], directed=True,
                              vertex_attrs={
                                  'area': np.array([10000.0,10000.0,10000.0]),
                                  'width': np.array([100.0,100.0,100.0]),
                                  'slope': np.array([0.005,0.002,0.004]),
                                  'n': np.array([0.017,0.017,0.017]),
                                  'invert': np.array([0.0,0.016,0.035]),
                                  'x': np.array([100,200,200]),
                                  'y': np.array([100,100,0]),
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
            for i in self.G.topological_sorting():
                # calculate incoming runoff, using top sorting to guarantee the previous runoffs are already computed
                inEdges = self.G.vs[i].in_edges()
                incomingRunoff = 0
                for e in inEdges:
                    incomingRunoff += e.source

                # alpha in manning equation
                a = (self.G.vs['width'][i] * np.power(self.G.vs['slope'][i], 0.5)) / (self.G.vs['area'][i] * self.G.vs['n'][i])
                depth_above_invert = np.maximum(x[i] - self.G.vs['invert'][i], 0.0)
                # outgoingRunoff
                outflow[i] = a * np.power(depth_above_invert, 5/3)
                y[i] = rainfall + incomingRunoff - outflow[i]
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

    def visualize(self, times, depths):
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
                    marker='o', 
                    linewidth=2)
        
        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('Depth (m)', fontsize=12)
        plt.title('Subcatchment Depth vs Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"figures/test.png")



if __name__ == "__main__":
    print("Dont call this directly :(")

