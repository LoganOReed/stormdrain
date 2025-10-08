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

# TODO:
# Functions:
# 1. get_edge_weights - usually involves solving odes on node weights
# 2. 
class Graph:
    """General Graph Structure, Parent of the three subgraphs."""
    def __init__(self, name):
        super(Graph, self).__init__()
        random.seed(0)
        self.G = ig.Graph.Erdos_Renyi(n=15, m=30, directed=False, loops=False)
        self.G.to_directed(mode="acyclic")
        self.G["name"] = name
        print(self.G.summary())
        print(self.G.topological_sorting())



        

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
        self.rainfall = [0.0,0.5,1.0,0.75,0.5,0.25,0.0]
        self.rainfall = [e * 0.0254 for e in self.rainfall]
        print(self.G.summary())
        print(self.G.topological_sorting())
        print(self.rainfall)

    # TODO: Write this
    def update(self, t, dt, rainfall):
        def ode(t, x):
            y = np.zeros(self.G.vcount())
            outflow = np.zeros(self.G.vcount())
            for i in range(self.G.vcount()):
                a = (self.G.vs['width'][i] * np.power(self.G.vs['slope'][i], 0.5)) / (self.G.vs['area'][i] * self.G.vs['n'][i])
                # Calculate depth above invert, ensuring it's non-negative
                depth_above_invert = np.maximum(x[i] - self.G.vs['invert'][i], 0.0)
                # Calculate outflow term
                outflow[i] = a * np.power(depth_above_invert, 5/3)
                y[i] = rainfall - outflow[i]
            print(f"rhs: {y}")
            print(f"outflow: {outflow}")
            return y
    
        # Use solve_ivp instead of RK45 directly
        solution = sc.integrate.solve_ivp(
            ode, 
            (t, t + dt), 
            self.G.vs['depth'], 
            method='RK45'
        )
        
        self.G.vs['depth'] = solution.y[:, -1]





if __name__ == "__main__":
    g = SubcatchmentGraph(3)
    g.update(0,1,0.3)
    print(f"After 1 step: {g.G.vs['depth']}")
    g.update(1,2,0.5)
    print(f"After 2 step: {g.G.vs['depth']}")
    print("Dont call this directly :(")

