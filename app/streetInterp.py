import numpy as np
import scipy as sp
from sys import platform
import matplotlib
if platform == "linux":
    matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
from pprint import pprint

def depthFromAreaStreet(A, ps):
    """Gets depth from cross sectional area. ps is street parameters"""
    # Case 1, cross sectional area for depth below crown
    # below crown
    belowCrownArea = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
    # between crown and curb
    betweenCrownAndCurbArea = belowCrownArea + (ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"]))
    pprint(f"{betweenCrownAndCurbArea}")
    # above curb
    onSidewalkArea = betweenCrownAndCurbArea + 0.5*ps["T_curb"]*(ps["H_curb"] + ps["S_back"]*ps["T_curb"]) + ps["T_crown"]*(ps["S_back"]*ps["T_curb"])

    # water is below street crown
    if A <= belowCrownArea:
        d = np.sqrt(2*ps["Sx"]*A)
    # water is above crown but not curb
    elif A > belowCrownArea and A <= betweenCrownAndCurbArea:
        d = (A - belowCrownArea) / ps["T_crown"]
        d = d + ps["Sx"] * ps["T_crown"]
    # above curb but on sidewalk
    elif A > betweenCrownAndCurbArea and A <= onSidewalkArea:
        d = (A - betweenCrownAndCurbArea) / (ps["T_crown"] + ps["T_curb"])
        d = d + ps["H_curb"]
        # d = ()

    return d

def psiFromAreaStreet(A, ps):
    belowCrownArea = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
    betweenCrownAndCurbArea = belowCrownArea + (ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"]))
    onSidewalkArea = betweenCrownAndCurbArea + 0.5*ps["T_curb"]*(ps["H_curb"] + ps["S_back"]*ps["T_curb"]) + ps["T_crown"]*(ps["S_back"]*ps["T_curb"])

    if A <= belowCrownArea:
        c = np.sqrt(2*ps["Sx"])*(1+np.sqrt(1 + np.power(ps["Sx"],-2)))
        psi = np.power(A,4/3) * np.power(c,-2/3)
        psiPrime = (4/3)*np.power(A,1/3)*np.power(c,-2/3)
    elif A <= betweenCrownAndCurbArea:
        k = 0.5 * ps["Sx"] * ps["T_crown"] * ps["T_crown"] + ps["T_crown"] * ps["T_crown"] * np.sqrt(1 + ps["Sx"]*ps["Sx"])
        psi = np.power(ps["T_crown"],2/3)*np.power(A,5/3) * np.power(A + k,-2/3)
        psiPrime = (1/3) * np.power(ps["T_crown"], 2/3) * np.power(A,2/3) * np.power(A+k,-5/3) * (3*A + 5*k)
    else:
        pStreet = ps["Sx"] * ps["T_crown"] * np.sqrt(np.power(ps["Sx"],-1) + 1) + ps["H_curb"]
        psi = A*A*np.power(pStreet + ps["T_curb"],-1)
    return psi

def psiPrimeFromAreaStreet(A, ps):
    belowCrownArea = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
    betweenCrownAndCurbArea = belowCrownArea + (ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"]))
    onSidewalkArea = betweenCrownAndCurbArea + 0.5*ps["T_curb"]*(ps["H_curb"] + ps["S_back"]*ps["T_curb"]) + ps["T_crown"]*(ps["S_back"]*ps["T_curb"])

    if A <= belowCrownArea:
        c = np.sqrt(2*ps["Sx"])*(1+np.sqrt(1 + np.power(ps["Sx"],-2)))
        psiPrime = (4/3)*np.power(A,1/3)*np.power(c,-2/3)
    elif A <= betweenCrownAndCurbArea:
        k = 0.5 * ps["Sx"] * ps["T_crown"] * ps["T_crown"] + ps["T_crown"] * ps["T_crown"] * np.sqrt(1 + ps["Sx"]*ps["Sx"])
        psiPrime = (1/3) * np.power(ps["T_crown"], 2/3) * np.power(A,2/3) * np.power(A+k,-5/3) * (3*A + 5*k)
    else:
        pStreet = ps["Sx"] * ps["T_crown"]* np.sqrt(np.power(ps["Sx"],-1) + 1) + ps["H_curb"]
        psiPrime = 2*A*np.power(pStreet + ps["T_curb"],-1)
    return psiPrime


if __name__ == "__main__":
    ftToM = 0.3048
    ps = {
        "T_curb": 8*ftToM,
        "T_crown": 15*ftToM,
        "H_curb":  1*ftToM,
        "S_back" : 0.02*ftToM,
        "Sx": 0.02*ftToM
            }
    belowCrownArea = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]
    betweenCrownAndCurbArea = belowCrownArea + (ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"]))
    onSidewalkArea = betweenCrownAndCurbArea + 0.5*ps["T_curb"]*(ps["H_curb"] + ps["S_back"]*ps["T_curb"]) + ps["T_crown"]*(ps["S_back"]*ps["T_curb"])
    xs = np.linspace(0,onSidewalkArea,50)
    ds = [depthFromAreaStreet(x,ps) for x in xs]
    ys = [psiFromAreaStreet(x,ps) for x in xs]
    yps = [psiPrimeFromAreaStreet(x,ps) for x in xs]
    plt.plot(xs,ds,label="depth")
    plt.plot(xs,ys,label="psi")
    plt.plot(xs,yps,label="psi prime")
    plt.legend()
    plt.savefig(f"figures/CorrectedStreetGeometry.png")
    plt.show()
    # pprint(f"max height: {ps["Sx"]*ps["T_crown"]} vs max of function: {depthFromAreaStreet(0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"],ps)}")
    # pprint(f"max height in street: {ps["H_curb"]} vs max of function: {depthFromAreaStreet(maxAreaInStreet,ps)}")

