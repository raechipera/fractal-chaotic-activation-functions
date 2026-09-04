"""
Mandelbrot Attractor Comparison Analysis

Author: Rae Chipera
Affiliation: National University, School of Technology and Engineering

Companion code to:
    "Fractal and Chaotic Activation Functions in Echo State Networks: 
    Preprocessing Topology Governs the Echo State Property"
    Chipera, Du, & Tsapara (2025)
    arXiv:2512.14675

Generates Figure 11: comparative state-distance evolution for Tanh,
Mandelbrot Discrete, and Mandelbrot Continuous at N=500 neurons,
highlighting the discrete variant's convergence to a finite attractor
around ~0.033 rather than true ESP convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from activations import FractalActivations
from ReservoirComputer import ReservoirComputer


class EchoStateAnalyzer:
    """Analyzer for the Tanh vs. Mandelbrot (discrete/continuous) comparison"""

    def __init__(self, reservoir_computer_class):
        self.RC = reservoir_computer_class

    def analyze_state_convergence_statistical(self,
                                             activation='tanh',
                                             n_reservoir=500,
                                             spectral_radius=0.95,
                                             leak_rate=0.7,
                                             n_timesteps=200,
                                             n_trials=1000,
                                             input_dim=10,
                                             convergence_threshold=0.1,
                                             random_seed=42):
        """Run convergence analysis for a single activation"""
        np.random.seed(random_seed)

        all_trial_distances = []

        input_types = ['gaussian', 'uniform', 'sparse']

        for input_type in input_types:
            for trial in tqdm(range(n_trials // len(input_types)),
                            desc=f"    {activation} - {input_type}",
                            leave=False):

                if input_type == 'gaussian':
                    input_sequence = np.random.randn(n_timesteps, input_dim) * 0.5
                elif input_type == 'uniform':
                    input_sequence = np.random.uniform(-1, 1, (n_timesteps, input_dim))
                else:  # sparse
                    input_sequence = np.random.randn(n_timesteps, input_dim) * 0.5
                    mask = np.random.random((n_timesteps, input_dim)) > 0.7
                    input_sequence[mask] = 0

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

                distances = np.zeros(n_timesteps)

                for t in range(n_timesteps):
                    state_1 = rc._update_state(state_1, input_sequence[t])
                    state_2 = rc._update_state(state_2, input_sequence[t])
                    distances[t] = np.linalg.norm(state_1 - state_2)

                all_trial_distances.append(distances)

        all_distances = np.array(all_trial_distances)
        mean_distance = np.mean(all_distances, axis=0)

        return {
            'time': np.arange(n_timesteps),
            'mean_distance': mean_distance,
            'all_distances': all_distances
        }


def plot_mandelbrot_comparison(n_neurons=500, n_trials=1000):
    """
    Create the Tanh vs. Mandelbrot (discrete/continuous) comparison plot
    for Figure 11.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13

    fig, ax = plt.subplots(figsize=(10, 5))

    analyzer = EchoStateAnalyzer(ReservoirComputer)

    activations_to_test = ['tanh', 'mandelbrot_discrete', 'mandelbrot_continuous']

    colors = {
        'tanh': '#332288',                 # Blue
        'mandelbrot_discrete': '#CC6677',  # Red/Pink
        'mandelbrot_continuous': '#117733',  # Green
    }

    print(f"Analyzing at N={n_neurons} neurons...")

    stored_results = {}

    for activation in activations_to_test:
        print(f"  Testing {activation}...")

        results = analyzer.analyze_state_convergence_statistical(
            activation=activation,
            spectral_radius=0.95,
            leak_rate=0.7,
            n_timesteps=200,
            n_trials=n_trials,
            n_reservoir=n_neurons
        )

        stored_results[activation] = results

    # Subtle shading for the plateau region
    ax.axhspan(0.025, 0.04, alpha=0.1, color='#CC6677', zorder=0)

    ax.plot(stored_results['tanh']['time'],
            stored_results['tanh']['mean_distance'],
            color=colors['tanh'], linewidth=2,
            label='Tanh', zorder=3)

    ax.plot(stored_results['mandelbrot_continuous']['time'],
            stored_results['mandelbrot_continuous']['mean_distance'],
            color=colors['mandelbrot_continuous'], linewidth=2,
            label='Mandelbrot Continuous', zorder=4)

    ax.plot(stored_results['mandelbrot_discrete']['time'],
            stored_results['mandelbrot_discrete']['mean_distance'],
            color=colors['mandelbrot_discrete'], linewidth=2.5,
            linestyle='--', label='Mandelbrot Discrete', zorder=5)

    ax.axhline(y=0.1, color='black', linestyle=':', alpha=0.5,
              linewidth=1.5, label='Convergence threshold')

    ax.annotate('Finite attractor\n(~0.033)',
                xy=(100, 0.033), xytext=(50, 0.003),
                arrowprops=dict(arrowstyle='->', color='#CC6677', lw=1.5, alpha=0.7),
                fontsize=12, color='#CC6677')

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('State Distance ||x(t) - x\'(t)||', fontsize=12)
    ax.set_title('Comparative Analysis: Quantization-Induced ESP Failure\n' +
                 f'N={n_neurons} neurons, ρ=0.95, α=0.7, {n_trials} trials per activation',
                 fontsize=13, fontweight='bold')

    ax.set_yscale('log')
    ax.set_ylim([1e-16, 1e0])
    ax.set_xlim([0, 200])
    ax.grid(True, alpha=0.3, which='both')

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=11)

    # Subtle shading for the convergence region
    ax.axhspan(1e-16, 0.1, alpha=0.03, color='green')

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig('mandelbrot_attractor_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


if __name__ == "__main__":
    fig = plot_mandelbrot_comparison()
    print("\nFigure saved as 'mandelbrot_attractor_comparison.png'")
