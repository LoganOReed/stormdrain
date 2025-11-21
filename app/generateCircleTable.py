import numpy as np
from pprint import pprint
from sys import platform
import matplotlib

if platform == "linux":
    matplotlib.use("module://matplotlib-backend-kitty")
import matplotlib.pyplot as plt

def _angleFromArea(area, areaFull):
    """computes the central angle by cross sectional flow area. A = A_{full} (theta - sin theta) / 2 pi"""
    a = np.divide(area, areaFull)
    theta = 0.031715 - 12.79384 * a + 8.28479 * np.power(a, 0.5)
    dtheta = 1e15
    max_iter = 100
    iter_count = 0
    while np.any(np.abs(dtheta) > 0.0001) and iter_count < max_iter:
        denom = 1 - np.cos(theta)
        if np.any(denom == 0):
            raise ValueError("Division by zero: getAngleFromArea theta is 0")
        dtheta = (2 * np.pi * a - (theta - np.sin(theta))) / denom
        theta += dtheta
        iter_count += 1
    return theta


def _areaFromAngle(theta, areaFull):
    return np.multiply(areaFull, (theta - np.sin(theta))) / (2 * np.pi)


def _depth(theta, diam):
    depth = 0.5 * np.multiply(diam, (1 - np.cos(theta / 2)))
    return depth


def _sectionFactor(theta, sectionFactorFull):
    return np.multiply(
        sectionFactorFull, np.power(theta - np.sin(theta), 5 / 3)
    ) / (2 * np.pi * np.power(theta, 2 / 3))


def _wettedPerimeter(theta, diam):
    return 0.5 * np.multiply(theta, diam)


def _wettedPerimeterDerivative(theta, diam):
    return np.divide(4, np.multiply(diam, (1 - np.cos(theta))))


def _hydraulicRadius(theta, areaFull, diam):
    area = _areaFromAngle(theta, areaFull)
    perimeter = _wettedPerimeter(theta, diam)
    return np.divide(area, perimeter)


def _sectionFactorDerivative(theta, areaFull, diam):
    R = _hydraulicRadius(theta, areaFull, diam)
    Pprime = _wettedPerimeterDerivative(theta, diam)
    return ((5 / 3) - (2 / 3) * np.multiply(Pprime, R)) * np.power(R, 2 / 3)


def plot_circle_geometry(diam=1.0, n_points=1000):
    """
    Create plots showing circular pipe geometry relationships.
    
    Parameters
    ----------
    diam : float
        Pipe diameter (default: 1.0)
    n_points : int
        Number of points for plotting (default: 1000)
    """
    # Calculate full pipe properties
    areaFull = np.pi * diam**2 / 4
    Rfull = diam / 4
    sectionFactorFull = areaFull * np.power(Rfull, 2/3)
    
    # Generate theta values from 0 to 2*pi
    theta = np.linspace(0.001, 2*np.pi - 0.001, n_points)
    
    # Calculate geometric properties
    A = _areaFromAngle(theta, areaFull)
    Y = _depth(theta, diam)
    Psi = _sectionFactor(theta, sectionFactorFull)
    R = _hydraulicRadius(theta, areaFull, diam)
    
    # Normalize values
    A_norm = A / areaFull
    Y_norm = Y / diam
    Psi_norm = Psi / sectionFactorFull
    R_norm = R / Rfull
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: All functions in terms of Y (Depth)
    axes[0].plot(Y_norm, A_norm, label='A/A_full', color='blue', linewidth=2)
    axes[0].plot(Y_norm, R_norm, label='R/R_full', color='green', linewidth=2)
    axes[0].plot(Y_norm, Psi_norm, label='Ψ/Ψ_full', color='red', linewidth=2)
    axes[0].set_xlabel('Y/Y_full (Normalized Depth)', fontsize=12)
    axes[0].set_ylabel('Normalized Value', fontsize=12)
    axes[0].set_title('Geometric Properties vs Depth', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.5)
    
    # Plot 2: All functions in terms of A (Area)
    axes[1].plot(A_norm, Y_norm, label='Y/Y_full', color='purple', linewidth=2)
    axes[1].plot(A_norm, R_norm, label='R/R_full', color='green', linewidth=2)
    axes[1].plot(A_norm, Psi_norm, label='Ψ/Ψ_full', color='red', linewidth=2)
    axes[1].set_xlabel('A/A_full (Normalized Area)', fontsize=12)
    axes[1].set_ylabel('Normalized Value', fontsize=12)
    axes[1].set_title('Geometric Properties vs Area', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.5)
    
    plt.tight_layout()
    plt.savefig('circle_geometry_plots.png', dpi=300, bbox_inches='tight')
    print("Plots saved to 'circle_geometry_plots.png'")
    plt.show()
    
    # Print some key values
    print("\n" + "="*60)
    print("Key Values at Different Fill Levels:")
    print("="*60)
    
    fill_levels = [0.25, 0.5, 0.75, 1.0]
    for fill in fill_levels:
        idx = np.argmin(np.abs(A_norm - fill))
        print(f"\nAt A/A_full = {fill:.2f}:")
        print(f"  Y/Y_full = {Y_norm[idx]:.4f}")
        print(f"  R/R_full = {R_norm[idx]:.4f}")
        print(f"  Ψ/Ψ_full = {Psi_norm[idx]:.4f}")
    
    return {
        'theta': theta,
        'A_norm': A_norm,
        'Y_norm': Y_norm,
        'Psi_norm': Psi_norm,
        'R_norm': R_norm
    }



def generate_lookup_table(diam=0.5, n_points=100, output_file='circleTable'):
    """
    Generate a lookup table for circular pipe geometry as functions of normalized area.
    
    Parameters
    ----------
    diam : float
        Pipe diameter (default: 0.5)
    n_points : int
        Number of points in the table (default: 100)
    output_file : str
        Output CSV filename
        
    Returns
    -------
    dict
        Dictionary with keys 'A', 'Y', 'P', 'R' containing the table data
    """
    import pandas as pd
    
    # Calculate full pipe properties
    areaFull = np.pi * diam**2 / 4
    Rfull = diam / 4
    sectionFactorFull = areaFull * np.power(Rfull, 2/3)
    
    # Generate normalized area values from 0 to 1
    A_norm = np.linspace(0, 1, n_points)
    
    # Initialize arrays
    Y_norm = np.zeros(n_points)
    Psi_norm = np.zeros(n_points)
    R_norm = np.zeros(n_points)
    
    # For each area, find the corresponding theta and calculate properties
    for i, a_norm in enumerate(A_norm):

        # Find theta for this area
        area = a_norm * areaFull
        theta = _angleFromArea(area, areaFull)
            
        # Calculate normalized properties
        Y_norm[i] = _depth(theta, diam) / diam
        Psi_norm[i] = _sectionFactor(theta, sectionFactorFull) / sectionFactorFull
        R_norm[i] = _hydraulicRadius(theta, areaFull, diam) / Rfull
    
    # Create DataFrame
    df = pd.DataFrame({
        'A': A_norm,
        'Y': Y_norm,
        'P': Psi_norm,  # Using 'P' for Psi to match your existing code
        'R': R_norm
    })
    
    # Save to CSV
    df.to_csv(f"data/{output_file}.csv", index=False)
    print(f"\nLookup table generated and saved to '{output_file}'")
    print(f"Number of points: {n_points}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nLast few rows:")
    print(df.tail(10))
    
    # Verify key relationships
    print("\n" + "="*60)
    print("Verification of Key Values:")
    print("="*60)
    for a in [0.25, 0.5, 0.75, 1.0]:
        idx = np.argmin(np.abs(A_norm - a))
        print(f"\nAt A/A_full = {a:.2f}:")
        print(f"  Y/Y_full = {Y_norm[idx]:.4f}")
        print(f"  Ψ/Ψ_full = {Psi_norm[idx]:.4f}")
        print(f"  R/R_full = {R_norm[idx]:.4f}")
    
    # Return as dictionary for use in code
    circle_table = {
        'A': A_norm,
        'Y': Y_norm,
        'P': Psi_norm,
        'R': R_norm
    }
    
    return circle_table


if __name__ == "__main__":
    # Generate the plots
    data = plot_circle_geometry(diam=0.5, n_points=1000)
    generate_lookup_table(output_file='circleTable')
