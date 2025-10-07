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
                              'area': np.array([10000,10000,10000]),
                              'width': np.array([100,100,100]),
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
    def update(self, t, dt, rainfall ):
        #d_t = f - e - i - \a (d - d_s)^5/3
        # calculating runoff
        def ode(t,x):
            a = (self.G.vs['width']*np.power(self.G.vs['slope'],0.5))/ np.multiply(self.G.vs['area'],self.G.vs['n'])
            print(10000000000*np.subtract(rainfall, np.power(np.clip(np.subtract(x, self.G.vs['invert']),0,None),5/3)))
            return np.subtract(rainfall, np.power(np.clip(np.subtract(x, self.G.vs['invert']),0,None),5/3))
        #print(ode(self.G.vs['depth']))

        self.G.vs['depth'] = sc.integrate.RK45(ode, t, self.G.vs['depth'], t + dt).y






if __name__ == "__main__":
    g = SubcatchmentGraph(3)
    g.update(0,1,0.00000003)
    g.update(1,2,0.000000005)
    print(g.G.vs['depth'])
    print("Dont call this directly :(")

