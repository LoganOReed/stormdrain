import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import random


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
    def __init__(self, n):
        super(SubcatchmentGraph, self).__init__()
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
            for i in range(self.G.vcount()):
                a = (self.G.vs['width'][i] * np.power(self.G.vs['slope'][i], 0.5)) / (self.G.vs['area'][i] * self.G.vs['n'][i])
                depth_above_invert = np.maximum(x[i] - self.G.vs['invert'][i], 0.0)
                outflow[i] = a * np.power(depth_above_invert, 5/3)
                y[i] = rainfall - outflow[i]
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
    rainfall = [0.0,0.5,1.0,0.75,0.5,0.25,0.0]
    rainfall = [e * 0.0254 for e in rainfall]
    g = SubcatchmentGraph(3)
    subcatchment = []
    for i in range(len(rainfall)):
        subcatchment.append(g.update(2*i,0.5,rainfall[i]))
        subcatchment.append(g.update(2*i+1,0.5,rainfall[i]))
    print(f"list of depths at each time:{subcatchment}")
    # print(f"After 2 step: {g.G.vs['depth']}")
    ts = []
    for i in range(14):
        ts.append(i*0.5)
    g.visualize(ts, subcatchment)

# TODO: Create numpy docs for each function
