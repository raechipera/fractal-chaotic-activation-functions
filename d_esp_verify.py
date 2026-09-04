"""
Echo State Property Analysis: Statistical Validation

Author: Rae Chipera
Affiliation: National University, School of Technology and Engineering

Companion code to:
    "Fractal and Chaotic Activation Functions in Echo State Networks: 
    Preprocessing Topology Governs the Echo State Property"
    Chipera, Du, & Tsapara (2025)
    arXiv:2512.14675

Tests whether fractal activations satisfy the echo state property (ESP)
with statistical rigor: multiple input distributions, many trials per
configuration, and significance testing on convergence.

Note: This code will take a long time to run, especially with large N.
"""

import os
import pickle

import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from activations import FractalActivations
from ReservoirComputer import ReservoirComputer

# Create results directory
os.makedirs('results_cache', exist_ok=True)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class EchoStateAnalyzer:
    """
    Analyze echo state property for different activation functions
    with statistical rigor
    """

    def __init__(self, reservoir_computer_class):
        """
        Parameters:
        -----------
        reservoir_computer_class : class
            The ReservoirComputer class with fractal activations
        """
        self.RC = reservoir_computer_class

    def analyze_state_convergence_statistical(self,
                                             activation='tanh',
                                             n_reservoir=100,
                                             spectral_radius=0.95,
                                             leak_rate=0.7,
                                             n_timesteps=200,
                                             n_trials=1000,
                                             input_dim=10,
                                             convergence_threshold=0.1,
                                             random_seed=42):
        """
        Statistically rigorous analysis of ESP convergence
        """

        np.random.seed(random_seed)

        print(f"\nAnalyzing {activation} with {n_trials} trials...")

        # Storage for convergence metrics
        convergence_times = []
        final_distances = []
        convergence_rates = []
        failed_trials = 0

        # Test different input distributions
        input_types = ['gaussian', 'uniform', 'sparse']
        results_by_input = {}

        for input_type in input_types:
            print(f"  Testing {input_type} input distribution...")

            trial_distances = []

            for trial in tqdm(range(n_trials // len(input_types)),
                            desc=f"    {input_type}",
                            leave=False):

                # Generate input based on type
                if input_type == 'gaussian':
                    input_sequence = np.random.randn(n_timesteps, input_dim) * 0.5
                elif input_type == 'uniform':
                    input_sequence = np.random.uniform(-1, 1, (n_timesteps, input_dim))
                else:  # sparse
                    input_sequence = np.random.randn(n_timesteps, input_dim) * 0.5
                    mask = np.random.random((n_timesteps, input_dim)) > 0.7
                    input_sequence[mask] = 0

                # Create reservoir with new random initialization
                rc = self.RC(
                    n_reservoir=n_reservoir,
                    spectral_radius=spectral_radius,
                    activation=activation,
                    leak_rate=leak_rate,
                    random_state=random_seed + trial
                )

                # Initialize weights
                rc._initialize_weights(input_dim)

                # Run with two different initial states
                initial_state_1 = np.zeros(n_reservoir)
                initial_state_2 = np.random.randn(n_reservoir) * 2.0

                state_1 = initial_state_1.copy()
                state_2 = initial_state_2.copy()

                distances = np.zeros(n_timesteps)

                for t in range(n_timesteps):
                    state_1 = rc._update_state(state_1, input_sequence[t])
                    state_2 = rc._update_state(state_2, input_sequence[t])
                    distances[t] = np.linalg.norm(state_1 - state_2)

                trial_distances.append(distances)

                # Track convergence metrics
                final_dist = distances[-1]
                final_distances.append(final_dist)

                # Find convergence time (first time below threshold)
                converged_indices = np.where(distances < convergence_threshold)[0]
                if len(converged_indices) > 0:
                    conv_time = converged_indices[0]
                    convergence_times.append(conv_time)

                    # Estimate convergence rate (exponential fit)
                    if conv_time > 10:
                        t_fit = np.arange(min(conv_time, 50))
                        y_fit = distances[:min(conv_time, 50)]
                        if np.all(y_fit > 0):
                            log_y = np.log(y_fit + 1e-10)
                            rate = np.polyfit(t_fit, log_y, 1)[0]
                            convergence_rates.append(rate)
                else:
                    failed_trials += 1

            results_by_input[input_type] = np.array(trial_distances)

        # Calculate statistics
        all_distances = np.vstack([results_by_input[k] for k in input_types])
        mean_distance = np.mean(all_distances, axis=0)
        std_distance = np.std(all_distances, axis=0)
        percentiles = np.percentile(all_distances, [5, 25, 50, 75, 95], axis=0)

        # Statistical tests
        convergence_ratio = 1 - (failed_trials / n_trials)

        # Test if final distances are significantly different from zero
        if len(final_distances) > 0:
            t_stat, p_value = stats.ttest_1samp(final_distances, 0)
            converged = (np.mean(final_distances) < convergence_threshold and
                        convergence_ratio > 0.95)
        else:
            p_value = 1.0
            converged = False

        results = {
            'activation': activation,
            'spectral_radius': spectral_radius,
            'leak_rate': leak_rate,
            'n_trials': n_trials,
            'n_reservoir': n_reservoir,
            'all_distances': all_distances,
            'mean_distance': mean_distance,
            'std_distance': std_distance,
            'percentiles': percentiles,
            'convergence_times': np.array(convergence_times) if convergence_times else np.array([]),
            'convergence_rates': np.array(convergence_rates) if convergence_rates else np.array([]),
            'final_distances': np.array(final_distances),
            'convergence_ratio': convergence_ratio,
            'failed_trials': failed_trials,
            'converged': converged,
            'p_value': p_value,
            'time': np.arange(n_timesteps),
            'results_by_input': results_by_input
        }

        return results

    def test_parameter_sensitivity(self, activation, param_ranges):
        """
        Test ESP across parameter ranges for robustness
        """
        results = []

        for sr in param_ranges['spectral_radius']:
            for lr in param_ranges['leak_rate']:
                print(f"Testing {activation} with ρ={sr}, α={lr}")

                res = self.analyze_state_convergence_statistical(
                    activation=activation,
                    spectral_radius=sr,
                    leak_rate=lr,
                    n_trials=100,  # Fewer trials for parameter sweep
                    n_timesteps=100
                )

                results.append({
                    'spectral_radius': sr,
                    'leak_rate': lr,
                    'converged': res['converged'],
                    'mean_final_distance': np.mean(res['final_distances'])
                })

        return results

    def plot_comparative_analysis(self, all_results, save_path=None):
        """
        Publication-ready comparative plot using Paul Tol's muted palette
        """
        # Set up font properties for better readability
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

        n_neurons = all_results[0].get('n_reservoir', 1)
        n_trials = all_results[0]['n_trials']
        sr = all_results[0]['spectral_radius']
        lr = all_results[0]['leak_rate']

        fig.suptitle(f'Echo State Property Analysis: Comparative Study\n' +
                     f'N={n_trials} trials per activation, ρ={sr}, α={lr}, Neurons={n_neurons}',
                     fontsize=14, fontweight='bold')

        # Paul Tol's muted palette - colorblind safe and publication ready
        paul_tol_muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE',
                          '#882255', '#44AA99', '#999933', '#AA4499', '#DDDDDD']
        colors = paul_tol_muted[:len(all_results)]

        # Top - Mean trajectories (full width)
        ax1 = fig.add_subplot(gs[0, :])
        for i, res in enumerate(all_results):
            clean_name = res['activation'].replace('_', ' ').title()
            ax1.plot(res['time'], res['mean_distance'], color=colors[i],
                   linewidth=2, label=clean_name)

        ax1.axhline(y=0.1, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='Convergence threshold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('State Distance')
        ax1.set_title('Mean State Distance Evolution')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

        # Middle left - Final Distance Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        for i, res in enumerate(all_results):
            if len(res['final_distances']) > 0:
                clean_name = res['activation'].replace('_', ' ').title()
                ax2.hist(res['final_distances'], bins=30, alpha=0.5,
                        label=clean_name, color=colors[i])

        ax2.set_xlabel('Final Distance')
        ax2.set_ylabel('Density')
        ax2.set_title('Final Distance Distribution')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Middle right - Percentile Bands
        ax3 = fig.add_subplot(gs[1, 1])
        for i, res in enumerate(all_results):
            time = res['time']
            p5, p25, p50, p75, p95 = res['percentiles']
            clean_name = res['activation'].replace('_', ' ').title()

            ax3.plot(time, p50, '-', color=colors[i], linewidth=2, label=clean_name)
            ax3.fill_between(time, p5, p95, color=colors[i], alpha=0.1)
            ax3.fill_between(time, p25, p75, color=colors[i], alpha=0.2)

        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Distance (median & quartiles)')
        ax3.set_title('Percentile Bands (5th, 25th, 50th, 75th, 95th)')
        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        # Bottom left - Convergence Speed Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        conv_times_data = []
        labels = []
        for res in all_results:
            if len(res['convergence_times']) > 0:
                conv_times_data.append(res['convergence_times'])
                labels.append(res['activation'].replace('_', ' ').title())

        if conv_times_data:
            bp = ax4.boxplot(conv_times_data, labels=labels, patch_artist=True)
            for patch, i in zip(bp['boxes'], range(len(conv_times_data))):
                patch.set_facecolor(colors[i])
                patch.set_alpha(0.7)

            if len(labels) > 5:
                ax4.set_xticklabels(labels, rotation=45, ha='right')

        ax4.set_ylabel('Convergence Time')
        ax4.set_title('Convergence Speed Distribution')
        ax4.grid(True, alpha=0.3)

        # Bottom right - Statistical Summary Table
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('tight')
        ax5.axis('off')

        summary_data = []
        for res in all_results:
            p_val_str = f"{res['p_value']:.4f}" if not np.isnan(res['p_value']) and res['p_value'] != 1.0 else "—"

            summary_data.append([
                res['activation'].replace('_', ' ').title()[:12],
                f"{res['convergence_ratio']*100:.1f}%",
                f"{np.mean(res['final_distances']):.4f}",
                p_val_str,
                "YES" if res['converged'] else "NO"
            ])

        ax5.set_title('Statistical Summary', fontsize=12, pad=10)

        table = ax5.table(cellText=summary_data,
                         colLabels=['Activation', 'Conv %', 'Final Dist', 'p-value', 'ESP'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 0.95])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Color code ESP column
        for i, res in enumerate(all_results):
            if res['converged']:
                table[(i+1, 4)].set_facecolor('#90EE90')
            else:
                table[(i+1, 4)].set_facecolor('#FFB6C1')

        plt.subplots_adjust(right=0.88)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        return fig


def main_analysis():
    """
    Main analysis function for publication-quality ESP validation.
    """
    print("=" * 60)
    print("ECHO STATE PROPERTY STATISTICAL VALIDATION")
    print("=" * 60)

    analyzer = EchoStateAnalyzer(ReservoirComputer)

    activations = ['tanh', 'mandelbrot_discrete', 'mandelbrot_continuous',
                   'logistic_modulo', 'logistic_sigmoid', 'weierstrass',
                   'cantor_function', 'cantor_set',
                   'brownian_motion', 'relu']

    reservoir_sizes = [1, 10, 50, 100, 500, 1000, 2000]

    # Store all results for the scaling comparison
    all_results_by_size = {}

    for n_neurons in reservoir_sizes:
        print(f"\n{'='*60}")
        print(f"TESTING WITH {n_neurons} NEURONS")
        print(f"{'='*60}")

        all_results = []

        for activation in activations:
            print(f"\nTesting {activation} with {n_neurons} neurons...")

            # Reduce trials for larger networks to keep runtime manageable
            n_trials = 1000 if n_neurons <= 500 else 500 if n_neurons <= 2000 else 200

            results = analyzer.analyze_state_convergence_statistical(
                activation=activation,
                spectral_radius=0.95,
                leak_rate=0.7,
                n_timesteps=200,
                n_trials=n_trials,
                n_reservoir=n_neurons
            )

            all_results.append(results)

            # Print summary
            print(f"  Convergence rate: {results['convergence_ratio']*100:.1f}%")
            print(f"  Mean final distance: {np.mean(results['final_distances']):.6f}")
            print(f"  ESP satisfied: {results['converged']}")

            # Track memory usage for large reservoirs
            if n_neurons >= 1000:
                process = psutil.Process()
                mem_usage = process.memory_info().rss / 1024 / 1024 / 1024  # GB
                print(f"  Memory usage: {mem_usage:.2f} GB")

            # Save individual results as we go
            with open(f'results_cache/{activation}_{n_neurons}.pkl', 'wb') as f:
                pickle.dump(results, f)

            # Also save summary to CSV
            summary_df = pd.DataFrame([{
                'activation': activation,
                'neurons': n_neurons,
                'convergence_ratio': results['convergence_ratio'],
                'mean_final_dist': np.mean(results['final_distances']),
                'mean_conv_time': np.mean(results['convergence_times']) if len(results['convergence_times']) > 0 else np.nan,
                'converged': results['converged']
            }])
            summary_df.to_csv(f'results_cache/summary_{n_neurons}.csv',
                              mode='a', header=not os.path.exists(f'results_cache/summary_{n_neurons}.csv'),
                              index=False)

        all_results_by_size[n_neurons] = all_results

        # Create plots for this reservoir size
        fig = analyzer.plot_comparative_analysis(
            all_results,
            save_path=f'esp_comparison_{n_neurons}_neurons.png'
        )

    # Create a scaling analysis plot
    print("\n" + "="*60)
    print("SCALING ANALYSIS")
    print("="*60)

    create_scaling_plot(all_results_by_size, activations, reservoir_sizes)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"Tested {len(activations)} activations across {len(reservoir_sizes)} network sizes")
    print(f"Total reservoir neurons tested: {sum(reservoir_sizes) * len(activations)}")
    print("=" * 60)


def create_scaling_plot(all_results_by_size, activations, reservoir_sizes):
    """
    Create a plot showing how convergence scales with reservoir size.
    Uses Paul Tol's palette for consistency.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    paul_tol_muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE',
                      '#882255', '#44AA99', '#999933', '#AA4499', '#DDDDDD']

    colors = {act: paul_tol_muted[i % len(paul_tol_muted)]
              for i, act in enumerate(activations)}

    for activation in activations:
        convergence_rates = []
        final_distances = []

        for size in reservoir_sizes:
            try:
                results = next(r for r in all_results_by_size[size]
                              if r['activation'] == activation)
                convergence_rates.append(results['convergence_ratio'])
                final_distances.append(np.mean(results['final_distances']))
            except StopIteration:
                print(f"Warning: No results for {activation} at size {size}")
                convergence_rates.append(np.nan)
                final_distances.append(np.nan)

        clean_name = activation.replace('_', ' ').title()

        ax1.semilogx(reservoir_sizes, convergence_rates,
                     marker='o', label=clean_name, color=colors[activation],
                     linewidth=2, markersize=8)
        ax2.loglog(reservoir_sizes, final_distances,
                   marker='s', label=clean_name, color=colors[activation],
                   linewidth=2, markersize=8)

    ax1.set_xlabel('Reservoir Size (neurons)')
    ax1.set_ylabel('Convergence Rate')
    ax1.set_title('Scaling: Convergence Rate vs Network Size')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    ax2.set_xlabel('Reservoir Size (neurons)')
    ax2.set_ylabel('Final Distance')
    ax2.set_title('Scaling: Final Distance vs Network Size')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('ESP Performance Scaling Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig('esp_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Scaling plot saved with {len(activations)} activation functions")


if __name__ == "__main__":
    main_analysis()
