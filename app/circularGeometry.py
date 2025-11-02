import numpy as np
import scipy as sp
from . import circleTable
from pprint import pprint
from sys import platform
import matplotlib
if platform == "linux":
    matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt

from . import A_tbl, R_tbl, STREET_Y_FULL, STREET_LANE_SLOPE

CIRCLE_N = 51

def _angleFromArea(A, Yfull):
    """computes the central angle by cross sectional flow area. A = A_{full} (theta - sin theta) / 2 pi"""
    Afull = 0.7854 * Yfull * Yfull
    Aratio = np.divide(A, Afull)
    # This is an initial guess from the pdf
    theta = 0.031715 - 12.79384 * Aratio + 8.28479 * np.power(Aratio,0.5)
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

def psiFromAreaCircle(A, Yfull):
    """Section Factor (Psi = A * R(A)^2/3) from cross sectional area."""
    Afull = 0.7854 * Yfull * Yfull
    Rfull = 0.25 * Yfull
    PsiFull = Afull * np.power(Rfull,2/3)
    Aratio = A / Afull

    if A < 0.04 * Afull:
        theta = _angleFromArea(A, Yfull)
        psi = (PsiFull * np.power(theta - np.sin(theta), 5/3)) / (2*np.pi*np.power(theta,2/3))
    else:
        psi = np.interp(Aratio, circleTable["A"], circleTable["P"]) * PsiFull
    return psi

def psiPrimeFromAreaCircle(A, Yfull):
    """Section Factor (Psi = A * R(A)^2/3) from cross sectional area."""
    Afull = 0.7854 * Yfull * Yfull
    Rfull = 0.25 * Yfull
    PsiFull = Afull * np.power(Rfull,2/3)
    Aratio = A / Afull

    if A < 0.04 * Afull:
        theta = _angleFromArea(A, Yfull)
        P = 0.5 * theta * Yfull
        PPrime = 4 / (Yfull * (1 - np.cos(theta)))
        R = A / P
        psiPrime = ( (5/3) - (2/3) * PPrime * R) * np.power(R, 2/3)
    else:
        # psi = np.interp(Aratio, circleTable["A"], circleTable["P"]) * PsiFull
        i = int(Aratio * (CIRCLE_N - 1))
        psiPrime = (circleTable["P"][i] - circleTable["P"][i - 1])*(CIRCLE_N - 1) * (PsiFull / Afull)
    return psiPrime
 
        

def hydraulicRadiusFromAreaCircle(A, Yfull):
    """Get HydraulicRadius from area."""
    Afull = 0.7854 * Yfull * Yfull
    Rfull = 0.25 * Yfull
    PsiFull = Afull * np.power(Rfull,2/3)
    Aratio = A / Afull

    if A < 0.04 * Afull:
        theta = _angleFromArea(A, Yfull)
        r = A / (theta * Yfull * 0.5)
    else:
        rTable = np.power(circleTable["P"] / circleTable["A"], 1.5)
        pprint(rTable)
        r = np.interp(Aratio, circleTable["A"], rTable) * Rfull
    return r
    


def plotCircleFunctions(diam):
    "Creates plots to show Geometric Functions in terms of cross sectional area."
    Afull = 0.7854 * diam * diam
    res = 1000 # plot resolution
    As = np.linspace(0,1,res)
    Ys = [depthFromAreaCircle(a*Afull, diam) / diam for a in As]

    # Get PsiFull
    Afull = 0.7854 * diam * diam
    Rfull = 0.25 * diam
    PsiFull = Afull * np.power(Rfull,2/3)
    # TODO: Change or remove this 
    PsiPrimeFull = 1

    Psis = [psiFromAreaCircle(a*Afull, diam) / PsiFull for a in As]
    # Here incase they ask about hydraulicRadius
    Hs = [hydraulicRadiusFromAreaCircle(a*Afull, diam) / Rfull for a in As]
    PsiPrimes = [psiPrimeFromAreaCircle(a*Afull, diam) / PsiPrimeFull for a in As]



    plt.plot(As,Ys, label="d / d_full", color="blue")
    plt.plot(As,Psis, label="Psi / Psi_full", color="red")
    # plt.plot(As,Hs, label="H / H_full", color="purple")
    # plt.plot(As,PsiPrimes, label="Psi' / Psi'_full", color="purple")
    plt.legend()
    plt.grid(True)
    plt.xlabel("A/A_full")
    # plt.ylabel("Y/Yfull")
    plt.title("Circular Geometry from Cross Sectional Area")

    plt.savefig(f"figures/circularGeometry.png")
    plt.show()

    

    


if __name__ == "__main__":
    pprint("Don't call this directly. Or, if you want the geometry plots uncomment the code.")
    plotCircleFunctions(5)

