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

def depthFromAreaStreet(A, A_tbl, Y_full):
    """
    Calculate depth from area using a lookup table.
    
    Parameters:
    -----------
    A : float
        Area value to look up
    A_tbl : array-like
        Area lookup table (must have length 50 or 51)
    Y_full : float
        Full depth value
        
    Returns:
    --------
    Y : float
        Interpolated depth value
    """
    # Convert to numpy array and flatten to 1D row
    At = np.array(A_tbl).flatten()
    A_full = At[-1]
    N = len(At)
    
    if N != 51:
        raise ValueError(f'A_tbl must have length 50 or 51 (got {N}).')
    
    At_norm = At / A_full
    
    # Edge cases
    a = A / A_full
    if a <= 0:
        return 0.0
    elif a >= 1:
        return Y_full
    
    # Bisection to find i with At[i] <= a <= At[i+1]
    lo = 0  # Python uses 0-based indexing
    hi = N - 1
    
    while (hi - lo) > 1:
        mid = (lo + hi) // 2
        if a >= At_norm[mid]:
            lo = mid
        else:
            hi = mid
    
    i = lo
    denom = At_norm[i+1] - At_norm[i]
    
    if denom <= 0:
        frac = 0.0  # Avoid division by zero if plateau
    else:
        frac = (a - At_norm[i]) / denom
    
    Y = (Y_full / (N - 1)) * (i + frac)
    
    return Y

def R_of_Y(Y, R_tbl, Y_full):
    """
    Calculate R value from Y (depth) using a lookup table.
    
    Parameters:
    -----------
    Y : float
        Depth value
    R_tbl : array-like
        R lookup table
    Y_full : float
        Full depth value
        
    Returns:
    --------
    RY : float
        Interpolated R value
    """
    # Convert to numpy array and flatten to 1D row
    Rt = np.array(R_tbl).flatten()
    R_full = Rt[-1]
    N = len(Rt)
    
    # Clamp Y
    Y = max(0, min(Y, Y_full))
    
    # Edge cases
    if Y <= 0:
        RY = Rt[0]
    elif Y >= Y_full:
        RY = Rt[-1]
    else:
        # Integer portion
        dy = Y_full / (N - 1)
        k = int(np.floor(Y / dy))
        Yk = k * dy
        
        # Linear interpolation
        Rk = Rt[k]
        Rk1 = Rt[k + 1]
        RY = Rk + (Y - Yk) * (Rk1 - Rk) / dy
    
    RY = RY * R_full
    
    return RY

def psiFromAreaStreet(A, A_tbl, R_tbl, Y_full):
    """
    Calculate section factor (Psi) from area using lookup tables.
    
    Parameters:
    -----------
    A : float
        Area value
    A_tbl : array-like
        Area lookup table
    R_tbl : array-like
        Hydraulic radius lookup table
    Y_full : float
        Full depth value
        
    Returns:
    --------
    Psi : float
        Section factor (A * R^(2/3))
    """
    # Convert to numpy arrays and flatten to 1D
    At = np.array(A_tbl).flatten()
    Rt = np.array(R_tbl).flatten()
    
    # Depth from area
    Y = depthFromAreaStreet(A, At, Y_full)
    
    # Hydraulic radius at that depth
    RY = R_of_Y(Y, Rt, Y_full)
    
    # Section factor
    Psi = A * (RY)**(2/3)
    
    return max(Psi,1e-10)


def areaFromPsiStreet(Psi, A_tbl, R_tbl, Y_full):
    """Reverse search for area from psi."""
    def f(x):
        return psiFromAreaStreet(x, A_tbl, R_tbl, Y_full) - Psi

    sol = sp.optimize.root_scalar(f, method="brentq", bracket=(0.0, A_tbl[-1]), rtol=0.0001*A_tbl[-1])
    A = sol.root
    if sol.converged == False:
        raise ValueError(f"ERROR: Brentq Failed to converge for psi {Psi}")
    pprint(f"A = {A} vs Atrue = 0.291169")
    return A


    


def psiPrimeFromAreaStreet(A, A_tbl, R_tbl, Y_full):
    """
    Calculate derivative of section factor (dPsi/dA) with respect to area
    using numerical differentiation.
    
    Parameters:
    -----------
    A : float
        Area value
    A_tbl : array-like
        Area lookup table
    R_tbl : array-like
        Hydraulic radius lookup table
    Y_full : float
        Full depth value
        
    Returns:
    --------
    dPsi : float
        Derivative of section factor with respect to area
    """
    # Convert to numpy arrays and flatten to 1D
    At = np.array(A_tbl).flatten()
    Rt = np.array(R_tbl).flatten()
    
    # Full area and step size ΔA
    A_full = At[-1]
    dA = 0.001 * A_full
    A = max(0, min(A, A_full))
    
    # Handle edges: use one-sided diff if we're too close to 0 or A_full
    if A <= dA:
        Psi_p = psiFromAreaStreet(A + dA, At, Rt, Y_full)
        Psi_0 = psiFromAreaStreet(A, At, Rt, Y_full)
        dPsi = (Psi_p - Psi_0) / dA  # forward diff
        return dPsi
    elif A >= A_full - dA:
        Psi_0 = psiFromAreaStreet(A, At, Rt, Y_full)
        Psi_m = psiFromAreaStreet(A - dA, At, Rt, Y_full)
        dPsi = (Psi_0 - Psi_m) / dA  # backward diff
        return dPsi
    
    # Central difference in the interior
    Psi_p = psiFromAreaStreet(A + dA, At, Rt, Y_full)
    Psi_m = psiFromAreaStreet(A - dA, At, Rt, Y_full)
    dPsi = (Psi_p - Psi_m) / (2 * dA)
    
    return dPsi


def plotStreetFunctions(diam):
    "Creates plots to show Geometric Functions in terms of cross sectional area."
    Afull = A_tbl[-1]
    res = 1000 # plot resolution
    As = np.linspace(0,1,res)
    Ys = [depthFromAreaStreet(a*Afull, A_tbl, diam) / diam for a in As]

    # Get PsiFull
    
    Rfull = 0.25 * diam
    PsiFull = Afull * np.power(Rfull,2/3)
    # TODO: Change or remove this 
    PsiPrimeFull = 1

    Psis = [psiFromAreaStreet(a*Afull, A_tbl, R_tbl, diam) for a in As]
    PsiFull = max(Psis)
    Psis = np.array(Psis) / PsiFull
    # Here incase they ask about hydraulicRadius
    Hs = [R_of_Y(a*Afull, R_tbl, depthFromAreaStreet(a*Afull, A_tbl, diam)) / R_tbl[-1] for a in As]
    PsiPrimes = [psiPrimeFromAreaStreet(a*Afull, A_tbl, R_tbl, diam) / PsiPrimeFull for a in As]



    plt.plot(As,Ys, label="d / d_full", color="blue")
    plt.plot(As,Psis, label="Psi / Psi_full", color="red")
    # plt.plot(As,Hs, label="H / H_full", color="purple")
    # plt.plot(As,PsiPrimes, label="Psi' / Psi'_full", color="purple")
    plt.legend()
    plt.grid(True)
    plt.xlabel("A/A_full")
    # plt.ylabel("Y/Yfull")
    plt.title("Street Geometry from Cross Sectional Area")

    plt.savefig(f"figures/streetGeometry.png")
    plt.show()

    

    


if __name__ == "__main__":
    pprint("Don't call this directly. Or, if you want the geometry plots uncomment the code.")
    plotStreetFunctions(A_tbl[-1])

