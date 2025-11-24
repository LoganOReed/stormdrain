import igraph as ig
import numpy as np
import scipy as sc
import random
from pprint import pprint
from .newtonBisection import newtonBisection
from . import A_tbl, R_tbl, STREET_Y_FULL
from .streetGeometry import psiFromAreaStreet


# TODO: Create enum of desired drains
# TODO: Have v0 depend on enum parameter
def capturedFlow(Q, S0, Sx, L, W, n=0.017):
    """Computes Flow Captured from a P-50x100 drain on grade where street has slope Sx, Q is flow, L,W are length and width of drain."""
    # NOTE: This is specific to P-50x100, to use other drains change this
    if L == 0 or W == 0:
        return 0
    # T = width of water in street
    T = np.power(Q * n / (0.376 * np.power(Sx, 5 / 3) * np.sqrt(S0)), 3 / 8)
    # e0 = ratio of flow within W width from curb over total flow
    # also handle division by 0
    e0 = 1 if T == 0 else 1 - np.power(max(0, (1 - (W / T))), 8 / 3)
    # velocity over the grate
    v = (0.752 / n) * np.sqrt(S0) * np.power(Sx, 2 / 3) * np.power(T, 2 / 3)
    # Velocity where splash-back begins to occur
    v0 = 0.3048 * (
        0.74
        + 2.44 * (3.281 * L)
        - 0.27 * (3.281 * L) * (3.281 * L)
        + 0.02 * (3.281 * L) * (3.281 * L) * (3.281 * L)
    )
    # side/front capture efficiency
    rs = 1 / (1 + (0.0828 * np.power(v, 1.8) / (Sx * np.power(L, 2.3))))
    rf = 1 - 0.295 * np.maximum(0, v - v0)
    # captured flow (m^3/s)
    qc = Q * (rf * e0 + rs * (1 - e0))
    return qc


def plotCapturedFlow(S0, Sx, L, W, n, A_tbl, R_tbl, STREET_Y_FULL, psiFromAreaStreet):
    """
    Creates a smooth plot showing total Q, captured Q, and percentage captured vs A.
    Uses 1000 interpolated A values between min(A_tbl) and max(A_tbl).
    """
    from sys import platform
    import matplotlib

    if platform == "linux":
        matplotlib.use("module://matplotlib-backend-kitty")
    import matplotlib.pyplot as plt

    # Interpolate A values
    A_vals = np.linspace(min(A_tbl), max(A_tbl), 1000)

    # Compute Q via interpolation of psiFromAreaStreet across A_tbl
    psi_vals = [psiFromAreaStreet(A, A_tbl, R_tbl, STREET_Y_FULL) for A in A_tbl]
    Q_interp = np.interp(A_vals, A_tbl, psi_vals)

    Q_list = []
    Qc_list = []
    percent_captured = []

    for A, Q in zip(A_vals, Q_interp):
        Qc = capturedFlow(Q, A, Sx, S0, L, W, n)
        Q_list.append(Q)
        Qc_list.append(Qc)
        percent_captured.append(100 * Qc / Q if Q != 0 else 0)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot Q and Qc on the left y-axis
    ax1.plot(A_vals, Q_list, label="Total Q", color="tab:blue")
    ax1.plot(A_vals, Qc_list, label="Captured Q", color="tab:orange")
    ax1.set_xlabel("Cross-sectional Area (m^2)")
    ax1.set_ylabel("Flow (m^3/s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Secondary axis for % captured
    ax2 = ax1.twinx()
    ax2.plot(
        A_vals, percent_captured, label="% Captured", color="tab:green", linestyle="--"
    )
    ax2.set_ylabel("Captured Flow (%)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.title("Captured Flow vs Area")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Optional: save figure
    fig.savefig("figures/CapturedFlowVsArea.png", dpi=300)


if __name__ == "__main__":
    Sx = 0.04
    S0 = 0.02
    n = 0.013
    beta = np.power(Sx, 0.5) / n
    L = 0.6
    W = 0.6
    plotCapturedFlow(S0, Sx, L, W, n, A_tbl, R_tbl, STREET_Y_FULL, psiFromAreaStreet)
