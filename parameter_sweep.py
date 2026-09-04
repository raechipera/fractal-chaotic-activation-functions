"""
Parameter Sweep Analysis for Echo State Property Validation

Author: Rae Chipera
Affiliation: National University, School of Technology and Engineering

Companion code to:
    "Fractal and Chaotic Activation Functions in Echo State Networks: 
    Preprocessing Topology Governs the Echo State Property"
    Chipera, Du, & Tsapara (2025)
    arXiv:2512.14675

Generates:
    - Figure 7 (a-f): base parameter sweep heatmaps (ρ in [0.5, 5.0], N=100)
      for Tanh, Cantor Set, Mandelbrot Continuous, Mandelbrot Discrete,
      Logistic Modulo, and Weierstrass.
    - Figure 8: extended parameter sweep at extreme spectral radii
      (ρ up to 100) for Logistic Sigmoid and Cantor Function.

Set EXTENDED_SWEEP = True to also run the Figure 8 extreme sweep.
Set MULTISEED_SWEEP = True to run the full multi-seed robustness
analysis (seed_comparison / robust_boundaries plots) in addition to
Figures 7 and 8.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm

from activations import FractalActivations
from ReservoirComputer import ReservoirComputer

# MUST set font before any plotting
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 14
plt.style.use('seaborn-v0_8-whitegrid')

# Flags controlling which sweeps run when this script is executed directly
EXTENDED_SWEEP = True
MULTISEED_SWEEP = False


class SimplifiedParameterSweep:
    """
    Base parameter sweep producing Figure 7 (a-f): mean convergence rate
    heatmaps across spectral radius and leak rate, for N=100.
    """

    def __init__(self, reservoir_computer_class):
        self.RC = reservoir_computer_class

        self.activations = [
            'tanh', 'mandelbrot_discrete', 'mandelbrot_continuous',
            'logistic_modulo', 'logistic_sigmoid', 'weierstrass',
            'cantor_function', 'cantor_set', 'relu'
        ]

    def quick_esp_test(self, activation, spectral_radius, leak_rate,
                       n_trials=50, n_timesteps=100, n_reservoir=100,
                       random_seed=42):
        """Quick ESP test for a single (spectral_radius, leak_rate) point"""
        np.random.seed(random_seed)

        convergence_count = 0

        for trial in range(n_trials):
            input_dim = 10
            input_sequence = np.random.randn(n_timesteps, input_dim) * 0.5

            rc = self.RC(
                n_reservoir=n_reservoir,
                spectral_radius=spectral_radius,
                activation=activation,
                leak_rate=leak_rate,
                random_state=random_seed + trial
            )

            rc._initialize_weights(input_dim)

            state_1 = np.zeros(n_reservoir)
            state_2 = np.random.randn(n_reservoir) * 2.0

            for t in range(n_timesteps):
                state_1 = rc._update_state(state_1, input_sequence[t])
                state_2 = rc._update_state(state_2, input_sequence[t])

            final_dist = np.linalg.norm(state_1 - state_2)

            if final_dist < 0.1:
                convergence_count += 1

        return convergence_count / n_trials

    def plot_heatmap(self, activation, convergence_grid, spectral_radii, leak_rates, save_path=None):
        """Create a single publication-ready heatmap for one activation"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        sns.heatmap(convergence_grid,
                   xticklabels=[f'{lr:.1f}' for lr in leak_rates],
                   yticklabels=[f'{sr:.1f}' if sr < 1 else f'{sr:.0f}' for sr in spectral_radii],
                   annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1,
                   annot_kws={'fontsize': 14, 'weight': 'bold'},
                   cbar_kws={'label': 'Convergence Rate'},
                   ax=ax, square=False,
                   linewidths=0.5, linecolor='gray')

        ax.set_xlabel('Leak Rate (a)', fontsize=18, weight='bold')
        ax.set_ylabel('Spectral Radius (ρ)', fontsize=18, weight='bold')

        clean_name = activation.replace('_', ' ').title()
        ax.set_title(f'{clean_name} Activation Function', fontsize=20, weight='bold')

        ax.tick_params(axis='both', labelsize=14)

        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.set_ylabel('Convergence Rate', fontsize=16, weight='bold')

        # Mark the classical stability boundary at ρ=1
        if 1.0 in spectral_radii:
            idx = list(spectral_radii).index(1.0)
            ax.axhline(y=idx+0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
            ax.text(-0.5, idx+0.5, 'ρ=1', color='red', fontsize=10,
                   ha='right', va='center', weight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  Saved: {save_path}")

        plt.show()
        return fig

    def run_figure_7(self, spectral_radii=None, leak_rates=None, n_trials=50):
        """
        Run the base sweep for every activation and produce Figure 7 (a-f):
        one heatmap per activation function.
        """
        if spectral_radii is None:
            spectral_radii = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0])
        if leak_rates is None:
            leak_rates = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        print("=" * 60)
        print("BASE PARAMETER SWEEP (FIGURE 7)")
        print("=" * 60)

        for activation in self.activations:
            print(f"\nProcessing {activation}...")

            convergence_grid = np.zeros((len(spectral_radii), len(leak_rates)))

            with tqdm(total=len(spectral_radii)*len(leak_rates),
                     desc=f"{activation}") as pbar:
                for i, sr in enumerate(spectral_radii):
                    for j, lr in enumerate(leak_rates):
                        conv_rate = self.quick_esp_test(
                            activation, sr, lr, n_trials=n_trials
                        )
                        convergence_grid[i, j] = conv_rate
                        pbar.update(1)

            filename = f'param_sweep_{activation.replace("_", "")}.png'
            self.plot_heatmap(activation, convergence_grid,
                             spectral_radii, leak_rates,
                             save_path=filename)

            robust_percentage = (np.sum(convergence_grid > 0.95) / convergence_grid.size) * 100
            max_sr_stable = None
            for i, sr in enumerate(spectral_radii):
                if np.any(convergence_grid[i, :] > 0.95):
                    max_sr_stable = sr

            print(f"  Robust regions (>95%): {robust_percentage:.1f}% of parameter space")
            if max_sr_stable is not None:
                print(f"  Max stable spectral radius: {max_sr_stable:.2f}")
            else:
                print("  No spectral radius in this range achieved >95% convergence")

        print("\n" + "=" * 60)
        print("FIGURE 7 SWEEP COMPLETE")
        print("=" * 60)


class ParameterSweepAnalyzer:
    """
    Extended, multi-seed parameter sweep analysis. Used for Figure 8
    (extreme spectral radii) and, optionally, seed-robustness plots.
    """

    def __init__(self, reservoir_computer_class):
        self.RC = reservoir_computer_class

        self.available_activations = [
            'tanh', 'mandelbrot_discrete', 'mandelbrot_continuous',
            'weierstrass', 'logistic_modulo', 'logistic_sigmoid',
            'cantor_function', 'cantor_set', 'brownian_motion', 'relu'
        ]

    def quick_esp_test(self, activation, spectral_radius, leak_rate,
                       n_trials=50, n_timesteps=100, n_reservoir=50,
                       random_seed=42):
        """Quick ESP test for parameter sweep with a configurable seed"""
        np.random.seed(random_seed)

        convergence_count = 0
        final_distances = []

        for trial in range(n_trials):
            input_dim = 10
            input_sequence = np.random.randn(n_timesteps, input_dim) * 0.5

            rc = self.RC(
                n_reservoir=n_reservoir,
                spectral_radius=spectral_radius,
                activation=activation,
                leak_rate=leak_rate,
                random_state=random_seed + trial
            )

            rc._initialize_weights(input_dim)

            state_1 = np.zeros(n_reservoir)
            state_2 = np.random.randn(n_reservoir) * 2.0

            for t in range(n_timesteps):
                state_1 = rc._update_state(state_1, input_sequence[t])
                state_2 = rc._update_state(state_2, input_sequence[t])

            final_dist = np.linalg.norm(state_1 - state_2)
            final_distances.append(final_dist)

            if final_dist < 0.1:
                convergence_count += 1

        convergence_rate = convergence_count / n_trials
        mean_final_distance = np.mean(final_distances)

        return convergence_rate, mean_final_distance

    def parameter_sweep_with_seeds(self, activations, spectral_radii, leak_rates,
                                   random_seeds, n_trials=50):
        """
        Sweep across parameter space for the given activations, repeating
        across multiple random seeds and averaging. Also returns the raw
        per-seed grids so callers can check robustness across ALL seeds
        rather than just the seed-averaged convergence rate.
        """
        results = {}

        for activation in activations:
            print(f"\nSweeping parameters for {activation}...")

            seed_results = {}

            for seed in random_seeds:
                convergence_grid = np.zeros((len(spectral_radii), len(leak_rates)))
                distance_grid = np.zeros((len(spectral_radii), len(leak_rates)))

                with tqdm(total=len(spectral_radii)*len(leak_rates),
                         desc=f"{activation} (seed {seed})") as pbar:
                    for i, sr in enumerate(spectral_radii):
                        for j, lr in enumerate(leak_rates):
                            conv_rate, mean_dist = self.quick_esp_test(
                                activation, sr, lr,
                                n_trials=n_trials,
                                random_seed=seed
                            )
                            convergence_grid[i, j] = conv_rate
                            distance_grid[i, j] = mean_dist
                            pbar.update(1)

                seed_results[seed] = {
                    'convergence_grid': convergence_grid,
                    'distance_grid': distance_grid
                }

            avg_convergence = np.mean([sr['convergence_grid'] for sr in seed_results.values()], axis=0)
            avg_distance = np.mean([sr['distance_grid'] for sr in seed_results.values()], axis=0)
            std_convergence = np.std([sr['convergence_grid'] for sr in seed_results.values()], axis=0)

            # Robust ESP requires EVERY seed to individually clear the 0.95
            # threshold, not just the seed-averaged grid. This matters:
            # a seed-average above 0.95 does not guarantee every seed
            # individually converged robustly.
            robust_esp = np.ones_like(avg_convergence)
            for seed_data in seed_results.values():
                robust_esp = robust_esp * (seed_data['convergence_grid'] > 0.95)

            results[activation] = {
                'convergence_grid': avg_convergence,
                'distance_grid': avg_distance,
                'convergence_std': std_convergence,
                'robust_esp': robust_esp,
                'seed_results': seed_results
            }

        return results

    def plot_seed_comparison(self, results, activations, spectral_radii, leak_rates, save_path=None):
        """Plot showing mean, variance, and robust-ESP status across seeds for each activation"""
        n_activations = len(activations)
        fig = plt.figure(figsize=(16, 3*n_activations))
        gs = GridSpec(n_activations, 3, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Parameter Sensitivity: Mean and Variance Across Random Seeds\n'
                    'Shows robustness of results to initialization',
                    fontsize=16, fontweight='bold')

        for idx, activation in enumerate(activations):
            data = results[activation]

            ax1 = fig.add_subplot(gs[idx, 0])
            sns.heatmap(data['convergence_grid'],
                       xticklabels=[f'{lr:.1f}' for lr in leak_rates],
                       yticklabels=[f'{sr:.2f}' for sr in spectral_radii],
                       annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                       cbar_kws={'label': 'Mean Convergence Rate'},
                       ax=ax1)
            ax1.set_xlabel('Leak Rate (α)', fontsize=11)
            ax1.set_ylabel('Spectral Radius (ρ)', fontsize=11)
            ax1.set_title(f'{activation}: Mean Convergence', fontsize=12)

            ax2 = fig.add_subplot(gs[idx, 1])
            sns.heatmap(data['convergence_std'],
                       xticklabels=[f'{lr:.1f}' for lr in leak_rates],
                       yticklabels=[f'{sr:.2f}' for sr in spectral_radii],
                       annot=True, fmt='.3f', cmap='plasma', vmin=0,
                       cbar_kws={'label': 'Std Dev Across Seeds'},
                       ax=ax2)
            ax2.set_xlabel('Leak Rate (α)', fontsize=11)
            ax2.set_ylabel('Spectral Radius (ρ)', fontsize=11)
            ax2.set_title(f'{activation}: Seed Variance', fontsize=12)

            ax3 = fig.add_subplot(gs[idx, 2])
            robust_esp = data['robust_esp']

            sns.heatmap(robust_esp.astype(int),
                       xticklabels=[f'{lr:.1f}' for lr in leak_rates],
                       yticklabels=[f'{sr:.2f}' for sr in spectral_radii],
                       annot=True, fmt='d', cmap='RdYlGn', vmin=0, vmax=1,
                       cbar_kws={'label': 'Robust ESP', 'ticks': [0, 1]},
                       ax=ax3)
            ax3.set_xlabel('Leak Rate (α)', fontsize=11)
            ax3.set_ylabel('Spectral Radius (ρ)', fontsize=11)
            ax3.set_title(f'{activation}: Robust ESP (ALL seeds)', fontsize=12)

            robust_percentage = (np.sum(robust_esp) / robust_esp.size) * 100
            ax3.text(1.02, 0.5, f'{robust_percentage:.1f}%\nrobust',
                    transform=ax3.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_comparative_boundaries_all(self, results, spectral_radii, leak_rates, save_path=None):
        """Plot ESP boundaries for all activations, using the robust (all-seeds) metric"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        n_activations = len(results)
        cmap = cm.get_cmap('tab10') if n_activations <= 10 else cm.get_cmap('tab20')
        colors = {act: cmap(i/max(n_activations-1, 1))
                 for i, act in enumerate(results.keys())}

        fig.suptitle('ESP Satisfaction Boundaries: All Activation Functions',
                    fontsize=14, fontweight='bold')

        X, Y = np.meshgrid(leak_rates, spectral_radii)

        for activation, data in results.items():
            if np.max(data['convergence_grid']) < 0.5:
                continue

            CS1 = ax1.contour(X, Y, data['convergence_grid'],
                             levels=[0.5, 0.95],
                             colors=[colors[activation]],
                             alpha=0.7, linewidths=[1, 2])

            robust_esp = data['robust_esp']

            if np.any(robust_esp > 0):
                CS2 = ax2.contour(X, Y, robust_esp,
                                 levels=[0.5], colors=[colors[activation]],
                                 linewidths=2.5, label=activation)

        ax1.set_xlabel('Leak Rate (α)', fontsize=12)
        ax1.set_ylabel('Spectral Radius (ρ)', fontsize=12)
        ax1.set_title('Mean Convergence Boundaries', fontsize=12)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Leak Rate (α)', fontsize=12)
        ax2.set_ylabel('Spectral Radius (ρ)', fontsize=12)
        ax2.set_title('Robust ESP Boundaries (ALL seeds >95%)', fontsize=12)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def run_figure_8(self, n_trials=30):
        """
        Extended parameter sweep at extreme spectral radii (ρ up to 100)
        for the activations that survived the base sweep. Produces
        Figure 8.
        """
        print("=" * 60)
        print("EXTENDED PARAMETER SWEEP (FIGURE 8): TESTING AT EXTREME SPECTRAL RADII")
        print("=" * 60)

        spectral_radii = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0])
        leak_rates = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        random_seeds = [42, 123, 456]

        survivors = [
            'logistic_sigmoid',
            'cantor_function'
        ]

        print(f"\nTesting survivors at extreme parameters:")
        print(f"  Spectral radius: up to {max(spectral_radii)}")
        print(f"  Testing: {survivors}")

        results = self.parameter_sweep_with_seeds(
            survivors,
            spectral_radii,
            leak_rates,
            random_seeds,
            n_trials=n_trials
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))

        fig.suptitle('Extended Parameter Sweep: Testing ESP at Extreme Spectral Radii\n' +
                     'ρ up to 100',
                     fontsize=14, fontweight='bold', y=0.98)

        for idx, activation in enumerate(survivors):
            data = results[activation]

            ax = axes[idx]
            sns.heatmap(data['convergence_grid'],
                       xticklabels=[f'{lr:.1f}' for lr in leak_rates],
                       yticklabels=[f'{sr:.0f}' for sr in spectral_radii],
                       annot=True, fmt='.2f', cmap='viridis', vmin=0, vmax=1,
                       cbar_kws={'label': 'Convergence Rate'},
                       ax=ax)
            ax.set_xlabel('Leak Rate (a)', fontsize=12)
            ax.set_ylabel('Spectral Radius (ρ)', fontsize=12)
            ax.set_title(f'{activation.replace("_", " ").title()}', fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 0.98, 0.96])
        plt.savefig('esp_extended_sweep_extreme.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n" + "=" * 60)
        print("EXTREME PARAMETER RESULTS:")
        print("=" * 60)

        for activation in survivors:
            data = results[activation]

            # Robust ESP requires ALL seeds to individually exceed 0.95,
            # not just the seed-averaged grid (see parameter_sweep_with_seeds).
            robust_mask = data['robust_esp'] > 0
            if np.any(robust_mask):
                max_stable_indices = np.where(robust_mask)
                max_rho_idx = np.max(max_stable_indices[0])
                max_rho = spectral_radii[max_rho_idx]

                if max_rho >= 100:
                    print(f"\n{activation.upper()}: Robustly maintains ESP at ρ=100 (all seeds >95%)")
                elif max_rho >= 20:
                    print(f"\n{activation.upper()}: Robustly stable up to ρ≈{max_rho:.0f}")
                    print(f"  Exceeds classical bounds")
                else:
                    print(f"\n{activation.upper()}: Robustly stable up to ρ≈{max_rho:.1f}")
            else:
                print(f"\n{activation.upper()}: No spectral radius achieved robust ESP (all seeds >95%)")

            # Also report where the seed-averaged convergence rate degrades,
            # since this shows partial/degraded ESP-consistent behavior
            # beyond the robust boundary (e.g. Cantor's gradual decay to
            # ~0.5-0.6 convergence at rho=100, per the manuscript).
            avg_mask = data['convergence_grid'] > 0.5
            if np.any(avg_mask):
                avg_indices = np.where(avg_mask)
                max_avg_rho = spectral_radii[np.max(avg_indices[0])]
                print(f"  Partial convergence (>50% avg) persists up to ρ≈{max_avg_rho:.0f}")

        print("\n" + "=" * 60)
        print("FIGURE 8 SWEEP COMPLETE")
        print("=" * 60)

        return results


def main():
    """Run Figure 7 (always) plus Figure 8 and/or the multi-seed robustness
    plots if the corresponding flags are set."""

    base_analyzer = SimplifiedParameterSweep(ReservoirComputer)
    base_analyzer.run_figure_7()

    if EXTENDED_SWEEP or MULTISEED_SWEEP:
        extended_analyzer = ParameterSweepAnalyzer(ReservoirComputer)

        if EXTENDED_SWEEP:
            extended_analyzer.run_figure_8()

        if MULTISEED_SWEEP:
            spectral_radii = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0, 1.1, 1.25, 1.5])
            leak_rates = np.array([0.3, 0.5, 0.7, 0.9])
            random_seeds = [42, 123, 456, 789, 2024]
            activations = [
                'tanh', 'mandelbrot_discrete', 'mandelbrot_continuous',
                'logistic_modulo', 'logistic_sigmoid', 'weierstrass',
                'cantor_function', 'cantor_set', 'relu', 'brownian_motion'
            ]

            print("\n" + "=" * 60)
            print("MULTI-SEED ROBUSTNESS SWEEP")
            print("=" * 60)

            results = extended_analyzer.parameter_sweep_with_seeds(
                activations, spectral_radii, leak_rates, random_seeds, n_trials=20
            )

            extended_analyzer.plot_seed_comparison(
                results, activations, spectral_radii, leak_rates,
                save_path='esp_seed_comparison.png'
            )
            extended_analyzer.plot_comparative_boundaries_all(
                results, spectral_radii, leak_rates,
                save_path='esp_robust_boundaries.png'
            )

            print("\n" + "=" * 60)
            print("ROBUSTNESS SUMMARY")
            print("=" * 60)

            for activation in activations:
                data = results[activation]
                robust_percentage = (np.sum(data['robust_esp']) / data['robust_esp'].size) * 100
                max_variance_idx = np.unravel_index(
                    np.argmax(data['convergence_std']), data['convergence_std'].shape
                )
                max_var_sr = spectral_radii[max_variance_idx[0]]
                max_var_lr = leak_rates[max_variance_idx[1]]

                print(f"\n{activation.upper()}:")
                print(f"  Robust ESP (ALL seeds): {robust_percentage:.1f}% of parameter space")
                print(f"  Mean convergence: {np.mean(data['convergence_grid']):.3f}")
                print(f"  Max seed variance: {np.max(data['convergence_std']):.3f}")
                print(f"    at ρ={max_var_sr:.2f}, α={max_var_lr:.1f}")

            print("\n" + "=" * 60)
            print("MULTI-SEED SWEEP COMPLETE")
            print("=" * 60)


if __name__ == "__main__":
    main()
