import numpy as np
import scipy as sp
from oct2py import octave

def octaveTest():
    octave.addpath(octave.genpath("./octave"))
    A_tbl = sp.io.loadmat('./octave/A_tbl51.mat')
    R_tbl = sp.io.loadmat('./octave/R_tbl51.mat')
    A = 2
    Y_full = 10
    psi = octave.psi_from_area(A, A_tbl, R_tbl, Y_full)
    return psi 


if __name__ == "__main__":
    print(octaveTest())

