import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import random
from pprint import pprint
from .newton_bisection import findroot

# TODO: Create enum of desired drains
# TODO: Have v0 depend on enum parameter
def getCapturedFlow(Q, A, Sx, L, W):
    """Computes Flow Captured from a P-50x100 drain on grade where street has slope Sx, A is cross sectional area, Q is flow, L,W are length and width of drain."""
    # NOTE: This is specific to P-50x100, to use other drains change this
    v0 = 0.74 + 2.44 * L - 0.27 * L*L + 0.02 * L*L*L
    v = Q / A
    rs = 1 / (1 + (0.15 * np.power(v,1.8) / (Sx * np.power(L,2.3))))
    rf = 1 - 0.09 * np.maximum(0, v-v0)
    alpha = 2*A*Sx*(1 + (1 / np.power(Sx,2)))
    e1 = alpha - 2*np.power(alpha,0.5)*W + W*W
    e1 = e1 / alpha
    e0 = A - A * e1 * e1
    pprint(e0)
    qc = Q*(rf * e0 + rs * (1 - e0))
    return qc

    

if __name__ == "__main__":
    pprint("Hello")
    for i in range(1,100):
        # pprint(f"Captured Flow: {getCapturedFlow(3*i, i, 0.003, 0.1,0.1 )}")
        # pprint(f"Total Flow: {3*i}")
        pprint(f"Captured Flow Percentage: {100*getCapturedFlow(3*i, i, 0.003, 0.1,0.1 ) / (3*i)}%")
