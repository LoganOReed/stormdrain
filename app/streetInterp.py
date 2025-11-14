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
    betweenCrownAndCurbArea = belowCrownArea + (ps["T_crown"] * (ps["H_curb"] - ps["H_curb"]))
    # above curb
    # aboveCurbArea = 

    if A <= 0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"]:
        d = np.sqrt(2*ps["Sx"]*A)
    # elif A > 
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
    # theta = np.arcsin(S_x)
    # pprint(S_x)
    # pprint(np.tan(theta))

    xs = np.linspace(0,ps["Sx"]*ps["T_crown"],50)
    ys = [depthFromAreaStreet(x,ps) for x in xs]
    # plt.plot(xs,ys,label="depth from area")
    # plt.show()
    pprint(f"max height: {ps["Sx"]*ps["T_crown"]} vs max of function: {depthFromAreaStreet(0.5 * ps["T_crown"] * ps["Sx"] * ps["T_crown"],ps)}")



    N = 100 
    # get original street function
    
    xs = np.linspace(-ps["T_curb"],ps["T_crown"],N)
    ys = np.zeros(len(xs))
    for i in range(len(xs)):
        if xs[i] < 0:
            ys[i] = - ps["S_back"] * xs[i] + ps["H_curb"]
        else:
            ys[i] = ps["Sx"] * xs[i]
    
    cs = sp.interpolate.CubicSpline(xs,ys)
    
    plt.plot(xs,ys, label="orig")
    plt.plot(xs, cs(xs), label="S")
    plt.plot(xs, cs(xs,1), label="S'")
    plt.saveas(f"figures/streetCubicSpline.png")
    plt.show()
