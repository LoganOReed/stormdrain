from pprint import pprint
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import random
from .subcatchmentGraph import SubcatchmentGraph
from .rain import normalizeRainfall


if __name__ == "__main__":
    file = "largerExample"
    rainfall = [0.0, 0.5, 1.0, 0.75, 0.5]
    # rainfall = [0.0,0.5,1.0,0.75,0.5,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    # rainfall = np.array([0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05, 0.0,0.0,0.0])
    rainfallTimes = np.array([i for i in range(len(rainfall))])
    rainfall, rainfallTimes = normalizeRainfall(rainfall, rainfallTimes)

    # Create plot for disconnected subcatchments
    g = SubcatchmentGraph(file)
    subcatchment = []
    runoff = []
    dt = 600
    T = max(rainfallTimes)
    pprint(f"0 to {T}, total steps = {T / dt}")
    N = int(T / dt)
    ts = np.linspace(0, T, N)
    pprint(f"len(ts): {len(ts)}... int(T/dt): {int(T / dt)}")
    rain = np.interp(ts, rainfallTimes, rainfall)
    pprint(f"rain: {rain}")
    for n in range(len(ts)):
        subcatchmentDepth, runoffUnsorted = g.update(ts[n], dt, rain[n])
        pprint(f"it={n},  {ts[n]}, {rain[n]}")
        subcatchment.append(subcatchmentDepth)
        runoff.append(runoffUnsorted)
    # print(f"list of depths at each time:{subcatchment}")
    # print(f"list of runoff at each time:{runoff}")
    # print(f"After 2 step: {g.G.vs['depth']}")
    ts = []
    # Create plot for subcatchments 0->1->2
    # rainfall = [0.0,0.5,1.0,0.75,0.5,0.25,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    # rainfall = [e * 0.0254 for e in rainfall]
    # g = SubcatchmentGraph(3,2) # temp, 2 just indicates not None
    # subcatchment = []
    # for i in range(len(rainfall)):
    #     subcatchment.append(g.update(2*i,0.5,rainfall[i]))
    #     subcatchment.append(g.update(2*i+1,0.5,rainfall[i]))
    # print(f"list of depths at each time:{subcatchment}")
    # # print(f"After 2 step: {g.G.vs['depth']}")
    # ts = []
    # for i in range(2*len(rainfall)):
    #     ts.append(i*0.5)
    # g.visualize(ts, subcatchment, "connected")
    #
