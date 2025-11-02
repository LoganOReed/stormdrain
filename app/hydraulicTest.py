import imageio as iio
import igraph as ig
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
from .network import SubcatchmentGraph, SewerGraph, StreetGraph, HydraulicGraph
from .newtonBisection import newtonBisection
from .visualize import visualize
from .streetGeometry import depthFromAreaStreet, psiFromAreaStreet, psiPrimeFromAreaStreet

if __name__ == "__main__":
    g = HydraulicGraph(depthFromAreaStreet, psiFromAreaStreet, psiPrimeFromAreaStreet, 0.345, "largerExample")
    g.update(0, 0, 0, 0)
