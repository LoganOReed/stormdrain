from oct2py import Oct2Py
import oct2py

oc = Oct2Py()
oc.clear()
oc.addpath(oc.genpath("./octave"))

depthFromAreaStreet = oc.depth_Y_from_area
psiFromAreaStreet = oc.psi_from_area
psiPrimeFromAreaStreet = oc.psi_prime_from_area
