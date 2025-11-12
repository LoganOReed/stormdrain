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
from .network import SubcatchmentGraph
from .hydraulicGraph import HydraulicGraph
from .newtonBisection import newtonBisection
from .visualize import visualize
from .rain import normalizeRainfall


class Model:
    """Wraps the coupling and timestepping."""
    def __init__(self, file, oldwaterRatio=0.2):


if __name__ == "__main__":
    pprint(f"Dont run this directly :(")
