import numpy as np
import scipy as sp
from sys import platform
import matplotlib

if platform == "linux":
    matplotlib.use("module://matplotlib-backend-kitty")
import matplotlib.pyplot as plt
from pprint import pprint


# open channel, so max psi is full
def maxPsiStreet(ps):
    return psiFromAreaStreet(fullAreaStreet(ps), ps)

def fullAreaStreet(ps):
    belowCrownArea = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
    betweenCrownAndCurbArea = belowCrownArea + (
        ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"])
    )
    return betweenCrownAndCurbArea


def depthFromAreaStreet(A, ps):
    """Gets depth from cross sectional area. ps is street parameters"""
    # Case 1, cross sectional area for depth below crown
    # below crown
    belowCrownArea = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
    # between crown and curb
    betweenCrownAndCurbArea = belowCrownArea + (
        ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"])
    )
    # water is below street crown
    if A <= belowCrownArea:
        d = np.sqrt(2 * ps["Sx"] * A)
    # water is above crown but not curb
    else:
        d = (A - belowCrownArea) / ps["T_crown"]
        d = d + ps["Sx"] * ps["T_crown"]
    # above curb but on sidewalk
    return d

# NOTE: The slope of the curb is 0 to simplify the analytic solution in the last case
def psiFromAreaStreet(A, ps):
    belowCrownArea = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
    betweenCrownAndCurbArea = belowCrownArea + (
        ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"])
    )
    if A <= belowCrownArea:
        c = np.sqrt(2 * ps["Sx"]) * (1 + np.sqrt(1 + np.power(ps["Sx"], -2)))
        psi = np.power(A, 4 / 3) * np.power(c, -2 / 3)
    else:
        k = 0.5 * ps["Sx"] * ps["T_crown"] * ps["T_crown"] + ps["T_crown"] * ps[
            "T_crown"
        ] * np.sqrt(1 + ps["Sx"] * ps["Sx"])
        psi = (
            np.power(ps["T_crown"], 2 / 3)
            * np.power(A, 5 / 3)
            * np.power(A + k, -2 / 3)
        )
    return psi



def psiPrimeFromAreaStreet(A, ps):
    belowCrownArea = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
    betweenCrownAndCurbArea = belowCrownArea + (
        ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"])
    )
    if A <= belowCrownArea:
        c = np.sqrt(2 * ps["Sx"]) * (1 + np.sqrt(1 + np.power(ps["Sx"], -2)))
        psiPrime = (4 / 3) * np.power(A, 1 / 3) * np.power(c, -2 / 3)
    else:
        k = 0.5 * ps["Sx"] * ps["T_crown"] * ps["T_crown"] + ps["T_crown"] * ps[
            "T_crown"
        ] * np.sqrt(1 + ps["Sx"] * ps["Sx"])
        psiPrime = (
            (1 / 3)
            * np.power(ps["T_crown"], 2 / 3)
            * np.power(A, 2 / 3)
            * np.power(A + k, -5 / 3)
            * (3 * A + 5 * k)
        )
    return psiPrime

def areaFromPsiStreet(psi, ps):
    """Gets cross sectional area from psi. ps is street parameters"""
    belowCrownArea = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
    
    # Calculate psi at the boundary between below-crown and above-crown regimes
    c = np.sqrt(2 * ps["Sx"]) * (1 + np.sqrt(1 + np.power(ps["Sx"], -2)))
    psi_boundary = np.power(belowCrownArea, 4 / 3) * np.power(c, -2 / 3)
    
    if psi <= psi_boundary:
        # Below crown - closed form solution
        # psi = A^(4/3) * c^(-2/3)
        # A = (psi * c^(2/3))^(3/4)
        A = np.power(psi * np.power(c, 2/3), 3/4)
    else:
        # Above crown - need numerical solver
        k = 0.5 * ps["Sx"] * ps["T_crown"] * ps["T_crown"] + ps["T_crown"] * ps[
            "T_crown"
        ] * np.sqrt(1 + ps["Sx"] * ps["Sx"])
        
        # Define the equation to solve: psi(A) - psi = 0
        def equation(A):
            return (
                np.power(ps["T_crown"], 2 / 3)
                * np.power(A, 5 / 3)
                * np.power(A + k, -2 / 3)
                - psi
            )
        
        # Use belowCrownArea as initial guess
        A = sp.optimize.fsolve(equation, belowCrownArea)[0]
    
    return A

if __name__ == "__main__":
    ftToM = 0.3048
    ps = {
        "T_curb": 8 * ftToM,
        "T_crown": 15 * ftToM,
        "H_curb": 1 * ftToM,
        "S_back": 0.02 ,
        "Sx": 0.02,
    }
    belowCrownArea = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
    betweenCrownAndCurbArea = belowCrownArea + (
        ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"])
    )
    onSidewalkArea = (
        betweenCrownAndCurbArea
        + 0.5 * ps["T_curb"] * (ps["H_curb"] + ps["S_back"] * ps["T_curb"])
        + ps["T_crown"] * (ps["S_back"] * ps["T_curb"])
    )
    xs = np.linspace(0, betweenCrownAndCurbArea, 500)
    ds = [depthFromAreaStreet(x, ps) for x in xs]
    ys = [psiFromAreaStreet(x, ps) for x in xs]
    yps = [psiPrimeFromAreaStreet(x, ps) for x in xs]
    plt.plot(xs, ds, label="depth")
    plt.plot(xs, ys, label="psi")
    plt.plot(xs, yps, label="psi prime")
    plt.legend()
    plt.savefig(f"figures/CorrectedStreetGeometry.png")
    plt.show()
    c = np.sqrt(2 * ps["Sx"]) * (1 + np.sqrt(1 + np.power(ps["Sx"], -2)))
    belowCrownPsi = np.power(
        0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"], 4 / 3
    ) * np.power(c, -2 / 3)
