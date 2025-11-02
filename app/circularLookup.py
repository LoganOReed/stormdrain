import numpy as np
import scipy as sp
from . import circleTable
from pprint import pprint
from sys import platform
import matplotlib
if platform == "linux":
    matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt

from .streetGeometry import depthFromAreaStreet, psiFromAreaStreet, psiPrimeFromAreaStreet

CIRCLE_N = 51

def _angleFromArea(A, Yfull):
    """computes the central angle by cross sectional flow area. A = A_{full} (theta - sin theta) / 2 pi"""
    Afull = 0.7854 * Yfull * Yfull
    a = np.divide(A, Afull)
    # This is an initial guess from the pdf
    theta = 0.031715 - 12.79384 * a + 8.28479 * np.power(a,0.5)
    theta = sp.optimize.newton(lambda x: 2*np.pi*A - (Afull * (x - np.sin(x)) ), theta, rtol=1e-4, maxiter = 100)
    return theta 



def depthFromAreaCircle(A, Yfull):
    """Get depth from cross sectional area."""
    Afull = 0.7854 * Yfull * Yfull
    Aratio = A / Afull
    if A < 0.04 * Afull:
        theta = _angleFromArea(A, Yfull)
        depth = 0.5 * Yfull * (1 - np.cos(0.5 * theta))
    else:
        depth = np.interp(Aratio, circleTable["A"], circleTable["Y"]) * Yfull
    return depth
        


def plotCircleFunctions(diam):
    "Creates plots to show Geometric Functions in terms of cross sectional area."
    Afull = 0.7854 * diam * diam
    res = 1000 # plot resolution
    As = np.linspace(0,1,res)
    Ys = [depthFromAreaCircle(a*Afull, diam) / diam for a in As]

    plt.plot(As,Ys, label="d / d_full", color="blue")
    # plt.plot(As,Ys2, label="Table", color="red")
    plt.legend()
    plt.xlabel("A/Afull")
    plt.ylabel("Y/Yfull")
    plt.title("Circular Geometry")

    plt.show()

    

    


if __name__ == "__main__":
    plotCircleFunctions(5)
