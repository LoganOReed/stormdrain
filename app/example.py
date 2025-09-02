from .fortran import lapack_sort, two_norm
import numpy as np
from oct2py import octave

def sort():
    """docstring for norm"""
    a = np.array([0,2, 199, 54, 123, 3], dtype=np.float64)
    print(f"Sorting {a}...")
    lapack_sort.lapack_sort(a)
    print(f"\t{a}")
    print("\n")
    
def norm():
    a = np.array([4.0,4.0, 4.0], dtype=np.float64)
    n = two_norm.two_norm(np.array([4.0,4.0, 4.0]))
    print(f"norm of {a}: {n}")
    print(f"norm of {[1.0, 1.0]}: {two_norm.two_norm([1.0, 1.0])}")

def octaveTest():
    octave.addpath(octave.genpath("./octave"))
    out = octave.test()
    return out 



if __name__ == "__main__":
    print(np.__config__.show())
    print("\n\n")
    sort()
    norm()
    print(octaveTest())
