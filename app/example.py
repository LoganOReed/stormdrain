import numpy as np
from oct2py import octave

def octaveTest():
    octave.addpath(octave.genpath("./octave"))
    out = octave.test()
    return out 



if __name__ == "__main__":
    print(np.__config__.show())
    print("\n\n")
    print(octaveTest())
