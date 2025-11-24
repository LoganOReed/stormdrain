import imageio as iio
import igraph as ig
import networkx as nx
import numpy as np
from sys import platform
import matplotlib

if platform == "linux":
    matplotlib.use("module://matplotlib-backend-kitty")
import matplotlib.pyplot as plt
import scipy as sc
import random
import csv
from pprint import pprint
from .newtonBisection import newtonBisection
from .visualize import visualize


def normalizeRainfall(
    rainfall, rainfallTimes, spaceConversion=0.0254, timeConversion=3600
):
    """Converts rainfall array data with sample times to m/s."""
    # to meters
    rainfall = np.array([e * spaceConversion for e in rainfall])
    rainfallTimes = np.array(rainfallTimes) * timeConversion
    rainfall = [e / timeConversion for e in rainfall]
    return rainfall, rainfallTimes


if __name__ == "__main__":
    # rainfall = [0.0,0.5,1.0,0.75,0.5,0.25,0.0]
    # rainfall = [0.01,0.5,1.0,1.1,1.3,1.5,1.8,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,1.6,1.3,1.2,1.1,0.85,0.75,0.5,0.3,0.1,0.1,0.1,0.1,0.0,0.0,0.0]
    rainfall = np.array(
        [
            0.10,
            0.15,
            0.25,
            0.40,
            0.60,
            0.80,
            0.70,
            0.50,
            0.30,
            0.20,
            0.10,
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    rainfallTimes = [2 * i for i in range(len(rainfall))]
    # pprint(f"rainfall: {len(rainfall)}, rainfallTimes: {len(rainfallTimes)}")

    rainfall, rainfallTimes = normalizeRainfall(rainfall, rainfallTimes)
    pprint(f"rainfall: {len(rainfall)}, rainfallTimes: {len(rainfallTimes)}")
    pprint(rainfall)
    pprint(rainfallTimes)

    pprint(max(rainfall))
    sample = np.linspace(0, max(rainfallTimes))
    rain = np.interp(sample, rainfallTimes, rainfall)
    areaSeconds = np.trapezoid(y=rain, x=sample)
    # rainfall = [0.01,0.2,0.3,0.5,0.6,0.8,1.0,1.0,1.0,1.5,1.8,2.0,2.0,2.0,2.0,2.0,2.0]
    # rainfall = rainfall + rainfall[::-1]
    plt.plot(rainfallTimes, rainfall, label="rain", color="blue")
    plt.plot(sample, rain, label="sample", color="red")
    # plt.plot(As,Psis, label="Psi / Psi_full", color="red")
    # plt.plot(As,Hs, label="H / H_full", color="purple")
    # plt.plot(As,PsiPrimes, label="Psi' / Psi'_full", color="purple")
    plt.legend()
    plt.grid(True)
    plt.xlabel("seconds")
    plt.ylabel("meters")
    plt.title("Testing Rain Unit Conversion")

    # plt.savefig(f"figures/circularGeometry.png")
    plt.show()
