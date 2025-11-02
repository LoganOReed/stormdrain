# from oct2py import Oct2Py
import scipy as sp
from enum import Enum
# import oct2py

# oc = Oct2Py(logger=None)
# oc.addpath(oc.genpath("./octave"))
# oc.eval("silent_functions = 1;")
# oc.eval("page_screen_output(0);")

# depthFromAreaStreet = oc.depth_Y_from_area
# psiFromAreaStreet = oc.psi_from_area
# psiPrimeFromAreaStreet = oc.psi_prime_from_area
A_tbl = sp.io.loadmat('./octave/A_tbl51.mat')['A_tbl51'][0]
R_tbl = sp.io.loadmat('./octave/R_tbl51.mat')['R_tbl51'][0]
