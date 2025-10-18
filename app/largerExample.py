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
from .network import SubcatchmentGraph, SewerGraph
from .newton_bisection import findroot




if __name__ == "__main__":
    subcatchment = SubcatchmentGraph("largerExample")
    subcatchment.visualize()
    sewer = SewerGraph("largerExample")

