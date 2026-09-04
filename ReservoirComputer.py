"""
Echo State Network for Fractal Reservoir Computing

Author: Rae Chipera
Affiliation: National University, School of Technology and Engineering

Companion code to:
    "Fractal and Chaotic Activation Functions in Echo State Networks: 
    Preprocessing Topology Governs the Echo State Property"
    Chipera, Du, & Tsapara (2025)
    arXiv:2512.14675
"""

import numpy as np
from scipy import sparse
from scipy.special import expit as sigmoid
from sklearn.linear_model import Ridge

from activations import FractalActivations


class ReservoirComputer:
    """
    Leaky Echo State Network with configurable activation function.

    Implements the standard leaky ESN update:
        x(t) = (1 - a) * x(t-1) + a * f(W_in * u(t) + W_res * x(t-1))

    where f is any activation function from FractalActivations.
    Output weights are trained via Ridge regression.
    
    Note: ReservoirComputer._initialize_weights() and compute_memory_capacity()
    call np.random.seed() directly, which mutates the *global* NumPy random
    state. If you're generating your own random data elsewhere in a script
    that also uses this class, calls to fit() or compute_memory_capacity()
    will reset that global state as a side effect. Keep this in mind if you
    see unexpected reproducibility (or a lack thereof) in code that mixes
    this class with other random number generation.
    """

    def __init__(self,
                 n_reservoir=500,
                 spectral_radius=0.95,
                 sparsity=0.1,
                 activation='tanh',
                 input_scaling=1.0,
                 leak_rate=0.7,
                 ridge_alpha=1e-6,
                 random_state=42):
        """
        Parameters
        ----------
        n_reservoir : int
            Number of reservoir neurons.
        spectral_radius : float
            Spectral radius of reservoir weight matrix.
        sparsity : float
            Fraction of non-zero reservoir connections.
        activation : str or callable
            Activation function. Valid strings:
            'tanh', 'relu',
            'mandelbrot_discrete', 'mandelbrot_disc',
            'mandelbrot_continuous', 'mandelbrot_cont',
            'logistic_sigmoid', 'logistic',
            'logistic_modulo',
            'cantor_function', 'cantor_set',
            'brownian_motion', 'brownian',
            'weierstrass'.
            Also accepts any callable f(x) -> array.
        input_scaling : float
            Scaling factor for input weights.
        leak_rate : float
            Leak rate a in [0, 1]. a=1.0 gives no leak (standard ESN).
        ridge_alpha : float
            Ridge regression regularization parameter.
        random_state : int
            Random seed for reproducibility.
        """
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state

        self._set_activation(activation)

        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.last_state = None
        self.last_inputs = None
        self.last_outputs = None

    def _set_activation(self, activation):
        """Map activation name to FractalActivations method."""
        activation_map = {
            'tanh':                  FractalActivations.tanh,
            'relu':                  FractalActivations.relu,
            'weierstrass':           FractalActivations.weierstrass,
            'logistic_sigmoid':      FractalActivations.logistic_sigmoid,
            'logistic':              FractalActivations.logistic_sigmoid,
            'logistic_modulo':       FractalActivations.logistic_modulo,
            'cantor_function':       FractalActivations.cantor_function,
            'cantor_set':            FractalActivations.cantor_set,
            'brownian_motion':       FractalActivations.brownian_motion,
            'brownian':              FractalActivations.brownian_motion,
            'mandelbrot_discrete':   FractalActivations.mandelbrot_discrete,
            'mandelbrot_disc':       FractalActivations.mandelbrot_discrete,
            'mandelbrot_continuous': FractalActivations.mandelbrot_continuous,
            'mandelbrot_cont':       FractalActivations.mandelbrot_continuous,
        }

        if isinstance(activation, str):
            if activation not in activation_map:
                raise ValueError(
                    f"Unknown activation: '{activation}'. "
                    f"Valid options: {sorted(activation_map.keys())}"
                )
            self.activation_func = activation_map[activation]
            self.activation_name = activation
        elif callable(activation):
            self.activation_func = activation
            self.activation_name = getattr(activation, '__name__', 'custom')
        else:
            raise ValueError(
                f"activation must be a string or callable, got {type(activation)}"
            )

    def _initialize_weights(self, n_inputs):
        """
        Initialize input and reservoir weight matrices.

        Reservoir matrix is sparse and scaled to the target spectral radius
        using W_scaled = W * (rho_target / rho_current).

        Parameters
        ----------
        n_inputs : int
            Dimensionality of input signal.
        """
        np.random.seed(self.random_state)

        self.W_in = (np.random.uniform(-1, 1, (self.n_reservoir, n_inputs))
                     * self.input_scaling)

        W = sparse.random(self.n_reservoir, self.n_reservoir,
                          density=self.sparsity,
                          random_state=self.random_state).toarray()
        W[W != 0] = np.random.uniform(-1, 1, np.sum(W != 0))

        current_radius = np.max(np.abs(np.linalg.eigvals(W)))
        if current_radius > 0:
            W = W * (self.spectral_radius / current_radius)
        self.W_res = W

    def _update_state(self, state, input_data):
        """
        Apply one leaky ESN update step.

        x(t) = (1 - a) * x(t-1) + a * f(W_in * u(t) + W_res * x(t-1))

        Parameters
        ----------
        state : np.ndarray, shape (n_reservoir,)
            Current reservoir state.
        input_data : np.ndarray, shape (n_inputs,)
            Current input vector.

        Returns
        -------
        np.ndarray, shape (n_reservoir,)
            Updated reservoir state.
        """
        pre_activation = (np.dot(self.W_in, input_data) +
                          np.dot(self.W_res, state))
        new_state = self.activation_func(pre_activation)
        return (1 - self.leak_rate) * state + self.leak_rate * new_state

    def _compute_reservoir_states(self, X, initial_state=None):
        """
        Drive the reservoir with an input sequence and collect states.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_inputs)
            Input sequence.
        initial_state : np.ndarray, shape (n_reservoir,), optional
            Initial reservoir state. Defaults to zeros.

        Returns
        -------
        np.ndarray, shape (n_samples, n_reservoir)
            Reservoir states for each timestep.
        """
        n_samples = X.shape[0]
        state = np.zeros(self.n_reservoir) if initial_state is None else initial_state
        states = np.zeros((n_samples, self.n_reservoir))

        for i in range(n_samples):
            state = self._update_state(state, X[i])
            states[i] = state

        self.last_state = state
        return states

    def fit(self, X, y, validation_split=0.0):
        """
        Train output weights via Ridge regression on reservoir states.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training input data.
        y : array-like, shape (n_samples, n_outputs)
            Target values.
        validation_split : float
            Fraction of data to hold out for validation (0 = no validation).

        Returns
        -------
        self
        """
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)

        if y.shape[0] != X.shape[0]:
            y = y.T

        if self.W_in is None:
            self._initialize_weights(X.shape[1])

        if validation_split > 0:
            split_idx = int(X.shape[0] * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        states = self._compute_reservoir_states(X_train)

        self.ridge_model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        self.ridge_model.fit(states, y_train)
        self.W_out = self.ridge_model.coef_

        self.last_inputs = X_train
        self.last_outputs = states

        if X_val is not None:
            val_states = self._compute_reservoir_states(
                X_val, initial_state=self.last_state)
            val_pred = self.ridge_model.predict(val_states)
            self.validation_score = np.mean((val_pred - y_val) ** 2)

        return self

    def predict(self, X, return_states=False):
        """
        Generate predictions for input sequence.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        return_states : bool
            If True, also return reservoir states.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples, n_outputs)
            Predictions.
        states : np.ndarray, shape (n_samples, n_reservoir)
            Reservoir states (only returned if return_states=True).
        """
        X = np.atleast_2d(X)
        states = self._compute_reservoir_states(X)
        y_pred = self.ridge_model.predict(states)

        if return_states:
            return y_pred, states
        return y_pred

    def compute_memory_capacity(self, sequence_length=100, n_trials=10,
                                max_delay=None):
        """
        Compute linear memory capacity (MC) of the reservoir.

        Follows the standard definition from Jaeger (2001):
            MC = sum_{k=1}^{max_delay} r^2(y_pred_k, u_{t-k})

        A single reservoir is driven by the full input sequence per trial.
        A separate linear readout is trained for each delay k. MC measures
        how well the reservoir can reconstruct past inputs from current state.

        Parameters
        ----------
        sequence_length : int
            Length of random uniform input sequence.
        n_trials : int
            Number of independent trials to average over.
        max_delay : int, optional
            Maximum delay to test. Defaults to min(50, sequence_length // 2).

        Returns
        -------
        float
            Linear memory capacity (sum of squared correlations across delays).

        References
        ----------
        Jaeger, H. (2001). The "echo state" approach to analysing and training
        recurrent neural networks. GMD Report No. 148.
        """
        max_delay = max_delay or min(50, sequence_length // 2)

        if self.W_in is None:
            self._initialize_weights(1)

        capacities = []

        for trial in range(n_trials):
            np.random.seed(self.random_state + trial)
            u = np.random.uniform(-1, 1, sequence_length)
            states = self._compute_reservoir_states(u.reshape(-1, 1))

            washout = max_delay
            states_washed = states[washout:]

            correlations = []
            for delay in range(1, max_delay + 1):
                target = u[washout - delay: sequence_length - delay]

                ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
                ridge.fit(states_washed, target)
                y_pred = ridge.predict(states_washed)

                if np.std(y_pred) > 1e-10 and np.std(target) > 1e-10:
                    corr = np.corrcoef(y_pred, target)[0, 1] ** 2
                else:
                    corr = 0.0
                correlations.append(corr)

            capacities.append(np.sum(correlations))

        return np.mean(capacities)
