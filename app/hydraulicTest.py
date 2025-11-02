import imageio as iio
import igraph as ig
import pandas as pd
import networkx as nx
import numpy as np
from sys import platform
import matplotlib
if platform == "linux":
    matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import scipy as sc
import random
import csv
from pprint import pprint
from .network import SubcatchmentGraph, SewerGraph, StreetGraph
from .hydraulicGraph import HydraulicGraph
from .newtonBisection import newtonBisection
from .visualize import visualize
from .streetGeometry import depthFromAreaStreet, psiFromAreaStreet, psiPrimeFromAreaStreet

if __name__ == "__main__":
    file = "largerExample"
    data = pd.read_csv(f"data/{file}.csv")
    coupledInputs = {"subcatchments": np.zeros(data.shape[0]), "drainCapture": np.zeros(data.shape[0]), "drainOverflow": np.zeros(data.shape[0])}  
    pprint(f"coupledInputs: {coupledInputs}")



    street = HydraulicGraph("STREET", file)

    street.update(0, 0, coupledInputs)
