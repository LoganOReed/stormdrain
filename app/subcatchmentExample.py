import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import random
from .network import SubcatchmentGraph



if __name__ == "__main__":
    rainfall = [0.0,0.5,1.0,0.75,0.5,0.25,0.0]
    rainfall = [e * 0.0254 for e in rainfall]
    g = SubcatchmentGraph(3, 2)
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


