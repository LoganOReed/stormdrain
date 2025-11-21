import numpy as np
import scipy as sp
from . import circleTable
from pprint import pprint
from sys import platform
import matplotlib

if platform == "linux":
    matplotlib.use("module://matplotlib-backend-kitty")
import matplotlib.pyplot as plt
import bisect
import math

from . import A_tbl, R_tbl, STREET_Y_FULL, STREET_LANE_SLOPE

CIRCLE_N = 51

def getThetaOfAlpha(alpha):
    """
    Calculate theta angle from normalized area alpha using iterative method
    
    Args:
        alpha: Normalized area (a/aFull)
    
    Returns:
        theta: Central angle in radians
    """
    # Initial approximation based on alpha value
    if alpha > 0.04:
        theta = 1.2 + 5.08 * (alpha - 0.04) / 0.96
    else:
        theta = 0.031715 - 12.79384 * alpha + 8.28479 * math.sqrt(alpha)
    
    theta1 = theta  # Store initial guess
    ap = (2.0 * math.pi) * alpha
    
    # Newton-Raphson iteration (max 40 iterations)
    for k in range(1, 41):
        # Calculate correction term
        d = -(ap - theta + math.sin(theta)) / (1.0 - math.cos(theta))
        
        # Modification to improve convergence for large theta
        if d > 1.0:
            d = math.copysign(1.0, d)  # Apply sign of d to 1.0
        
        theta = theta - d
        
        # Check for convergence
        if abs(d) <= 0.0001:
            return theta
    
    # If didn't converge, return initial guess
    return theta1

def getYcircular(alpha):
    """
    Calculate normalized depth Y from normalized area alpha for circular cross-section
    
    Args:
        alpha: Normalized area (a/aFull)
    
    Returns:
        Normalized depth (y/yFull)
    """
    if alpha >= 1.0:
        return 1.0
    if alpha <= 0.0:
        return 0.0
    
    # For very small alpha, use approximate formula
    if alpha <= 1.0e-5:
        theta = (37.6911 * alpha) ** (1.0/3.0)  # Cube root
        return theta * theta / 16.0
    
    # For larger alpha, use theta-based calculation
    theta = getThetaOfAlpha(alpha)
    return (1.0 - math.cos(theta / 2.0)) / 2.0

def getScircular(alpha):
    """
    Calculate normalized hydraulic parameter S from normalized area alpha for circular cross-section
    
    Args:
        alpha: Normalized area (a/aFull)
    
    Returns:
        Normalized S parameter (s/sFull)
    """
    if alpha >= 1.0:
        return 1.0
    if alpha <= 0.0:
        return 0.0
    
    # For very small alpha, use approximate formula
    if alpha <= 1.0e-5:
        theta = pow(37.6911 * alpha, 1.0/3.0)
        return pow(theta, 13.0/3.0) / 124.4797
    
    # For larger alpha, use theta-based calculation
    theta = getThetaOfAlpha(alpha)
    return pow((theta - math.sin(theta)), 5.0/3.0) / (2.0 * math.pi) / pow(theta, 2.0/3.0)

def getAcircular(psi):
    """
    Calculate normalized area alpha from normalized parameter psi for circular cross-section
    
    Args:
        psi: Normalized hydraulic parameter (s/sFull)
    
    Returns:
        Normalized area (a/aFull)
    """
    if psi >= 1.0:
        return 1.0
    if psi <= 0.0:
        return 0.0
    
    # For very small psi, use approximate formula
    if psi <= 1.0e-6:
        theta = pow(124.4797 * psi, 3.0/13.0)
        return theta * theta * theta / 37.6911
    
    # For larger psi, use theta-based calculation
    theta = getThetaOfPsi(psi)
    return (theta - math.sin(theta)) / (2.0 * math.pi)

def getThetaOfPsi(psi):
    """
    Calculate theta angle from normalized parameter psi using iterative method
    
    Args:
        psi: Normalized hydraulic parameter (s/sFull)
    
    Returns:
        theta: Central angle in radians
    """
    # Initial approximation based on psi value
    if psi > 0.90:
        theta = 4.17 + 1.12 * (psi - 0.90) / 0.176
    elif psi > 0.5:
        theta = 3.14 + 1.03 * (psi - 0.5) / 0.4
    elif psi > 0.015:
        theta = 1.2 + 1.94 * (psi - 0.015) / 0.485
    else:
        theta = 0.12103 - 55.5075 * psi + 15.62254 * math.sqrt(psi)
    
    theta1 = theta  # Store initial guess
    ap = (2.0 * math.pi) * psi
    
    # Newton-Raphson iteration (max 40 iterations)
    for k in range(1, 41):
        theta = abs(theta)
        tt = theta - math.sin(theta)
        tt23 = pow(tt, 2.0/3.0)
        t3 = pow(theta, 1.0/3.0)
        d = ap * theta / t3 - tt * tt23
        d = d / (ap * (2.0/3.0) / t3 - (5.0/3.0) * tt23 * (1.0 - math.cos(theta)))
        theta = theta - d
        
        # Check for convergence
        if abs(d) <= 0.0001:
            return theta
    
    # If didn't converge, return initial guess
    return theta1

def _angleFromArea(A, p):
    """computes the central angle by cross sectional flow area. A = A_{full} (theta - sin theta) / 2 pi"""
    Yfull = p["yFull"]
    Afull = 0.7854 * Yfull * Yfull
    Aratio = np.divide(A, Afull)
    theta = getThetaOfAlpha(A/Afull)

    # # This is an initial guess from the pdf
    # theta = 0.031715 - 12.79384 * Aratio + 8.28479 * np.power(Aratio, 0.5)
    # theta = sp.optimize.newton(
    #     lambda x: 2 * np.pi * A - (Afull * (x - np.sin(x))),
    #     theta,
    #     rtol=1e-6,
    #     maxiter=100,
    # )
    return theta


def depthFromAreaCircle(A, p):
    """Get depth from cross sectional area."""
    Yfull = p["yFull"]
    Afull = 0.7854 * Yfull * Yfull
    Aratio = A / Afull

    depth = getYcircular(Aratio) * Yfull
    # if A < 0.04 * Afull:
    #     theta = _angleFromArea(A, Yfull)
    #     depth = 0.5 * Yfull * (1 - np.cos(0.5 * theta))
    # else:
    #     depth = np.interp(Aratio, circleTable["A"], circleTable["Y"]) * Yfull
    return depth


def psiFromAreaCircle(A, p):
    """Section Factor (Psi = A * R(A)^2/3) from cross sectional area."""
    Yfull = p["yFull"]
    Afull = 0.7854 * Yfull * Yfull
    Rfull = 0.25 * Yfull
    PsiFull = Afull * np.power(Rfull, 2 / 3)
    Aratio = A / Afull

    psi = getScircular(Aratio)*PsiFull

    # if A < 0.04 * Afull:
    #     theta = _angleFromArea(A, Yfull)
    #     psi = (PsiFull * np.power(theta - np.sin(theta), 5 / 3)) / (
    #         2 * np.pi * np.power(theta, 2 / 3)
    #     )
    # else:
    #     psi = np.interp(Aratio, circleTable["A"], circleTable["P"]) * PsiFull
    return psi


def areaFromPsiCircle(Psi, p):
    """Reverse search for area from psi."""
    Yfull = p["yFull"]
    Afull = 0.7854 * Yfull * Yfull
    Rfull = 0.25 * Yfull
    PsiFull = Afull * np.power(Rfull, 2 / 3)
    if PsiFull == 0.0:
        return 0.0
    elif PsiFull >= 1.0:
        return Afull
    A = getAcircular(Psi/PsiFull) * Afull
    # A = getThetaOfPsi
    # A = np.interp(Psi / PsiFull, circleTable["P"], circleTable["A"]) * Afull
    return A


def psiPrimeFromAreaCircle(A, p):
    """Section Factor (Psi = A * R(A)^2/3) from cross sectional area."""
    Yfull = p["yFull"]
    Afull = 0.7854 * Yfull * Yfull
    Rfull = 0.25 * Yfull
    PsiFull = Afull * np.power(Rfull, 2 / 3)
    Aratio = A / Afull

    if Aratio <= 1e-30:
        return 1e-30
    
    theta = getThetaOfAlpha(Aratio)
    p = theta*Yfull / 2.0
    r = A / p
    dPdA = 4.0 / (Yfull * (1.0 - np.cos(theta)))
    psiPrime = ((5.0/3.0) - (2.0/3.0) * dPdA * r) * np.power(r,2.0/3.0)

    # if A < 0.04 * Afull:
    #     theta = _angleFromArea(A, Yfull)
    #     P = 0.5 * theta * Yfull
    #     PPrime = 4 / (Yfull * (1 - np.cos(theta)))
    #     R = A / P
    #     psiPrime = ((5 / 3) - (2 / 3) * PPrime * R) * np.power(R, 2 / 3)
    # else:
    #     # psi = np.interp(Aratio, circleTable["A"], circleTable["P"]) * PsiFull
    #     i = min(int(Aratio * (CIRCLE_N - 1)), len(circleTable["P"]) - 1)
    #     psiPrime = (
    #         (circleTable["P"][i] - circleTable["P"][i - 1])
    #         * (CIRCLE_N - 1)
    #         * (PsiFull / Afull)
    #     )
    return psiPrime


# def hydraulicRadiusFromAreaCircle(A, p):
#     """Get HydraulicRadius from area."""
#     Yfull = p["yFull"]
#     Afull = 0.7854 * Yfull * Yfull
#     Rfull = 0.25 * Yfull
#     PsiFull = Afull * np.power(Rfull, 2 / 3)
#     Aratio = A / Afull
#
#     if A < 0.04 * Afull:
#         theta = _angleFromArea(A, p)
#         r = A / (theta * Yfull * 0.5)
#     else:
#         rTable = np.power(circleTable["P"] / circleTable["A"], 1.5)
#         r = np.interp(Aratio, circleTable["A"], rTable) * Rfull
#     return r


def plotCircleFunctions(p):
    "Creates plots to show Geometric Functions in terms of cross sectional area."
    diam = p["yFull"]
    Afull = 0.7854 * diam * diam
    res = 1000  # plot resolution
    As = np.linspace(0, 1, res)
    Ys = [depthFromAreaCircle(a * Afull, p) / diam for a in As]

    # Get PsiFull
    Afull = 0.7854 * diam * diam
    Rfull = 0.25 * diam
    PsiFull = Afull * np.power(Rfull, 2 / 3)
    # TODO: Change or remove this
    PsiPrimeFull = 1

    Psis = [psiFromAreaCircle(a * Afull, p) / PsiFull for a in As]
    # Here incase they ask about hydraulicRadius
    # Hs = [hydraulicRadiusFromAreaCircle(a * Afull, p) / Rfull for a in As]
    PsiPrimes = [psiPrimeFromAreaCircle(a * Afull, p) / PsiPrimeFull for a in As]

    plt.plot(As, Ys, label="d / d_full", color="blue")
    plt.plot(As, Psis, label="Psi / Psi_full", color="red")
    plt.plot(As, PsiPrimes, label="Psi_prime / Psi_prime_full", color="purple")
    # plt.plot(As,PsiPrimes, label="Psi' / Psi'_full", color="purple")
    plt.legend()
    plt.grid(True)
    plt.xlabel("A/A_full")
    # plt.ylabel("Y/Yfull")
    plt.ylim(-0.5,1.5)
    plt.title("Circular Geometry from Cross Sectional Area")

    plt.savefig(f"figures/circularGeometry.png")
    plt.show()


if __name__ == "__main__":
    pprint(
        "Don't call this directly. Or, if you want the geometry plots uncomment the code."
    )
    plotCircleFunctions({"yFull":0.5})
