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
    # elif A > betweenCrownAndCurbArea and A <= onSidewalkArea:
        # d = ()

        

    return d

if __name__ == "__main__":
    ftToM = 0.3048
    ps = {
        "T_curb": 8*ftToM,
        "T_crown": 15*ftToM,
        "H_curb":  1*ftToM,
        "S_back" : 0.02*ftToM,
        "Sx": 0.02*ftToM
            }
    maxAreaInStreet = 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"] + (ps["T_crown"] * (ps["H_curb"] - ps["Sx"] * ps["T_crown"]))
    xs = np.linspace(0,maxAreaInStreet,50)
    ys = [depthFromAreaStreet(x,ps) for x in xs]
    plt.plot(xs,ys,label="depth from area")
    plt.show()
    pprint(f"max height: {ps["Sx"]*ps["T_crown"]} vs max of function: {depthFromAreaStreet(0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"],ps)}")
    pprint(f"max height in street: {ps["H_curb"]} vs max of function: {depthFromAreaStreet(maxAreaInStreet,ps)}")

