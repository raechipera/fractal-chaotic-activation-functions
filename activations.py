"""
Activation Functions for Fractal Reservoir Computing

Author: Rae Chipera
Affiliation: National University, School of Technology and Engineering

Companion code to:
    "Fractal and Chaotic Activation Functions in Echo State Networks: 
    Preprocessing Topology Governs the Echo State Property"
    Chipera, Du, & Tsapara (2025)
    arXiv:2512.14675

Activation functions implemented:
    MandelbrotDiscrete    - Quantized escape time (fractal)
    MandelbrotContinuous  - Smooth escape time interpolation (fractal)
    CantorFunction        - Devil's staircase (fractal, monotonic)
    CantorSet             - Cantor set membership indicator (fractal, binary)
    LogisticSigmoid       - Logistic map with sigmoid wrapper (chaotic)
    LogisticModulo        - Logistic map with modulo wrapper (chaotic)
    Weierstrass           - Nowhere-differentiable function (fractal)
    BrownianMotion        - Stochastic activation (negative control)
    ReLU                  - Standard rectified linear unit (baseline)
    Tanh                  - Standard hyperbolic tangent (baseline)
"""

import numpy as np
from scipy.special import expit as sigmoid


class FractalActivations:
    """Activation functions for fractal and chaotic reservoir computing."""

    @staticmethod
    def mandelbrot_discrete(x, max_iter=20, scale=2.0):
        """
        Discrete Mandelbrot activation using integer escape times.

        Maps input to escape time normalized to [0, 1], producing at most
        max_iter+1 distinct output levels (quantized).

        Parameters
        ----------
        x : np.ndarray
            Input array.
        max_iter : int
            Maximum iterations before assuming non-escape.
        scale : float
            Input scaling factor; maps input range to Mandelbrot boundary region.

        Returns
        -------
        np.ndarray
            Escape time normalized to [0, 1].
        """
        c = x / scale + 0j
        z = np.zeros_like(c, dtype=complex)
        escape_time = np.ones_like(x) * max_iter

        for i in range(max_iter):
            mask = np.abs(z) <= 2.0
            z[mask] = z[mask] ** 2 + c[mask]
            escape_time[mask & (np.abs(z) > 2.0)] = i

        return escape_time / max_iter

    @staticmethod
    def mandelbrot_continuous(x, max_iter=20, scale=2.0):
        """
        Continuous Mandelbrot activation using smooth escape time interpolation.

        Uses the formula T(c) = n - log2(log2(|z_n|)) for escaping points,
        providing infinite resolution within [0, 1].

        Parameters
        ----------
        x : np.ndarray
            Input array.
        max_iter : int
            Maximum iterations before assuming non-escape.
        scale : float
            Input scaling factor.

        Returns
        -------
        np.ndarray
            Smoothly interpolated escape time normalized to [0, 1].
        """
        c = x / scale + 0j
        z = np.zeros_like(c, dtype=complex)
        escape_time = np.ones_like(x) * max_iter

        for i in range(max_iter):
            mask = np.abs(z) <= 2.0
            z[mask] = z[mask] ** 2 + c[mask]

            escaping = mask & (np.abs(z) > 2.0)
            if np.any(escaping):
                escape_time[escaping] = i - np.log2(np.log2(np.abs(z[escaping])))

        return escape_time / max_iter

    @staticmethod
    def weierstrass(x, a=0.5, b=3, N=10, scale=1.0):
        """
        Weierstrass function activation.

        Continuous everywhere, differentiable nowhere. Exhibits extreme
        local variation (high Lipschitz constant) that disrupts ESP.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        a : float
            Amplitude decay factor (0 < a < 1).
        b : int
            Frequency scaling factor.
        N : int
            Number of terms in the partial sum.
        scale : float
            Input scaling factor.

        Returns
        -------
        np.ndarray
            Output normalized by (1 - a).
        """
        x_scaled = x * scale
        result = np.zeros_like(x)

        for n in range(N):
            result += a ** n * np.cos(b ** n * np.pi * x_scaled)

        return result / (1 - a)

    @staticmethod
    def logistic_sigmoid(x, r=3.7):
        """
        Logistic map activation with sigmoid preprocessing.

        Applies sigmoid to bound input to (0, 1), then applies the logistic
        map f(y) = r*y*(1-y). The sigmoid wrapper produces a compressive,
        monotone preprocessing topology that maintains ESP.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        r : float
            Logistic map parameter. r=3.7 is in the chaotic regime (r > 3.57).

        Returns
        -------
        np.ndarray
            Output in (0, r/4].
        """
        x_sigmoid = sigmoid(x)
        return r * x_sigmoid * (1 - x_sigmoid)

    @staticmethod
    def logistic_modulo(x, r=3.7):
        """
        Logistic map activation with modulo preprocessing.

        Uses fractional part of |x| to map input to (0, 1), introducing
        countably many discontinuities at integer boundaries. This dispersive
        preprocessing topology breaks ESP at scale.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        r : float
            Logistic map parameter. r=3.7 is in the chaotic regime (r > 3.57).

        Returns
        -------
        np.ndarray
            Output in (0, r/4].
        """
        x_mod = np.abs(x) % 1.0
        x_mod = np.clip(x_mod, 1e-10, 1 - 1e-10)
        return r * x_mod * (1 - x_mod)

    @staticmethod
    def cantor_function(x, depth=10):
        """
        Cantor function (devil's staircase) activation.

        Continuous and monotonically increasing, but has zero derivative
        almost everywhere and is not Lipschitz continuous. Despite these
        irregular properties, maintains ESP at extreme spectral radii.

        Input is mapped to [0, 1] via sigmoid before applying the
        recursive Cantor construction.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        depth : int
            Recursion depth for the Cantor construction.

        Returns
        -------
        np.ndarray
            Output in [0, 1].
        """
        x_norm = sigmoid(x)

        def cantor_recursive(y, d):
            if d == 0:
                return y
            if y <= 1 / 3:
                return 0.5 * cantor_recursive(3 * y, d - 1)
            elif y >= 2 / 3:
                return 0.5 + 0.5 * cantor_recursive(3 * y - 2, d - 1)
            else:
                return 0.5

        result = np.zeros_like(x_norm)
        for i in range(len(x_norm.flat)):
            result.flat[i] = cantor_recursive(x_norm.flat[i], depth)
        return result

    @staticmethod
    def cantor_set(x, depth=10):
        """
        Cantor set membership activation.

        Returns 1 if the sigmoid-mapped input lies in the Cantor set,
        0 otherwise. Binary output (k=2 quantization levels) with
        increasing instability at large reservoir sizes.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        depth : int
            Approximation depth for Cantor set membership test.

        Returns
        -------
        np.ndarray
            Binary output in {0, 1}.
        """
        x_norm = sigmoid(x)
        in_cantor = np.ones_like(x_norm, dtype=bool)

        for d in range(depth):
            scaled = x_norm * (3 ** d)
            middle_third = ((scaled % 3) >= 1) & ((scaled % 3) < 2)
            in_cantor = in_cantor & ~middle_third

        return in_cantor.astype(float)

    @staticmethod
    def brownian_motion(x, dt=0.01, scale=1.0):
        """
        Brownian motion activation (stochastic negative control).

        Adds a Gaussian random increment at each call, making this
        activation non-deterministic. Included to demonstrate that
        internal stochasticity is fundamentally incompatible with ESP.

        Parameters
        ----------
        x : np.ndarray
            Input array.
        dt : float
            Time step controlling noise variance (increment ~ N(0, dt)).
        scale : float
            Output scaling factor.

        Returns
        -------
        np.ndarray
            Stochastic output; not reproducible across calls.
        """
        increment = np.random.normal(0, np.sqrt(dt), size=x.shape)
        return scale * (np.tanh(x) + increment)

    @staticmethod
    def relu(x):
        """
        Rectified linear unit (baseline).

        Parameters
        ----------
        x : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            max(0, x).
        """
        return np.maximum(0, x)

    @staticmethod
    def tanh(x):
        """
        Hyperbolic tangent (baseline).

        Parameters
        ----------
        x : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            tanh(x) in (-1, 1).
        """
        return np.tanh(x)
