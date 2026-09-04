"""
Spectral Radius Eigenvalue Verification

Author: Rae Chipera
Affiliation: National University, School of Technology and Engineering

Companion code to:
    "Fractal and Chaotic Activation Functions in Echo State Networks: 
    Preprocessing Topology Governs the Echo State Property"
    Chipera, Du, & Tsapara (2025)
    arXiv:2512.14675

Generates Figure 9: confirms that reservoir weight matrices constructed
with target spectral radii of ρ=10 and ρ=100 (N=500) actually achieve
those spectral radii, via direct eigenvalue computation. Produces an
eigenvalue scatter for each target radius plus a magnitude-distribution
histogram comparing the two.
"""

import numpy as np
import matplotlib.pyplot as plt


def create_reservoir_matrix(N, spectral_radius, sparsity=0.1):
    """Create a sparse random reservoir matrix scaled to an exact spectral radius"""
    W = np.random.randn(N, N) * (np.random.rand(N, N) < sparsity)
    eigenvalues = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigenvalues))
    W = W * (spectral_radius / current_radius)
    return W


def verify_spectral_radii():
    """
    Eigenvalue verification for extreme spectral radii.
    Demonstrates that reservoirs truly have the claimed spectral radii.
    """

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    N = 500  # Reservoir size
    fig.suptitle(f'Spectral Radius Eigenvalue Analysis (N={N})',
                 fontsize=16, fontweight='bold')

    np.random.seed(42)  # For reproducibility

    # Create matrices
    W_rho10 = create_reservoir_matrix(N, 10.0)
    W_rho100 = create_reservoir_matrix(N, 100.0)

    # Compute eigenvalues
    eigenvals_10 = np.linalg.eigvals(W_rho10)
    eigenvals_100 = np.linalg.eigvals(W_rho100)

    # Use viridis-like colors for colorblind accessibility
    color_10 = plt.cm.viridis(0.3)   # Greenish
    color_100 = plt.cm.viridis(0.7)  # Purplish

    # Plot 1: Eigenvalue spectrum for ρ=10
    axes[0].scatter(eigenvals_10.real, eigenvals_10.imag,
                   alpha=0.6, s=8, color=color_10)
    circle_10 = plt.Circle((0, 0), 10, fill=False, color='red',
                           linestyle='--', linewidth=2)
    axes[0].add_patch(circle_10)
    axes[0].set_xlim([-12, 12])
    axes[0].set_ylim([-12, 12])
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('Real(λ)', fontsize=12)
    axes[0].set_ylabel('Imag(λ)', fontsize=12)
    axes[0].set_title(f'ρ = 10\n(Verified: max|λ| = {np.max(np.abs(eigenvals_10)):.3f})',
                     fontsize=14)

    # Plot 2: Eigenvalue spectrum for ρ=100
    axes[1].scatter(eigenvals_100.real, eigenvals_100.imag,
                   alpha=0.6, s=8, color=color_100)
    circle_100 = plt.Circle((0, 0), 100, fill=False, color='red',
                            linestyle='--', linewidth=2)
    axes[1].add_patch(circle_100)
    axes[1].set_xlim([-120, 120])
    axes[1].set_ylim([-120, 120])
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Real(λ)', fontsize=12)
    axes[1].set_ylabel('Imag(λ)', fontsize=12)
    axes[1].set_title(f'ρ = 100\n(Verified: max|λ| = {np.max(np.abs(eigenvals_100)):.3f})',
                     fontsize=14)

    # Plot 3: Eigenvalue magnitude distribution, log scale
    axes[2].hist(np.abs(eigenvals_10), bins=30, alpha=0.7,
                color=color_10, label='ρ = 10', density=True)
    axes[2].hist(np.abs(eigenvals_100), bins=30, alpha=0.7,
                color=color_100, label='ρ = 100', density=True)
    axes[2].axvline(x=10, color=color_10, linestyle='--', alpha=0.7, linewidth=2)
    axes[2].axvline(x=100, color=color_100, linestyle='--', alpha=0.7, linewidth=2)
    axes[2].set_xlabel('|λ| (Eigenvalue Magnitude)', fontsize=12)
    axes[2].set_ylabel('Probability Density (log scale)', fontsize=12)
    axes[2].set_yscale('log')
    axes[2].set_title('Eigenvalue Magnitude Distribution', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eigenvalue_verification.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print verification statistics
    print("=" * 60)
    print(f"EIGENVALUE ANALYSIS RESULTS (N={N})")
    print("=" * 60)
    print(f"ρ = 10 reservoir:")
    print(f"  Target spectral radius: 10.000")
    print(f"  Actual spectral radius: {np.max(np.abs(eigenvals_10)):.6f}")
    print(f"  Number of eigenvalues: {len(eigenvals_10)}")
    print(f"  Eigenvalues at boundary: {np.sum(np.abs(np.abs(eigenvals_10) - 10) < 0.01)}")

    print(f"\nρ = 100 reservoir:")
    print(f"  Target spectral radius: 100.000")
    print(f"  Actual spectral radius: {np.max(np.abs(eigenvals_100)):.6f}")
    print(f"  Number of eigenvalues: {len(eigenvals_100)}")
    print(f"  Eigenvalues at boundary: {np.sum(np.abs(np.abs(eigenvals_100) - 100) < 0.1)}")

    return W_rho10, W_rho100


if __name__ == "__main__":
    W_rho10, W_rho100 = verify_spectral_radii()
