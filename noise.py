# noise.py
# =============================================================
# Gaussian noise injection for simulated sensor (accelerometer) signals
# Noise is added to node acceleration arrays to mimic real measurement
# noise from optical (DIC / high-speed camera) or electronic sensors.
# =============================================================

import numpy as np


def add_gaussian_noise(node_accels, std, seed=None):
    """
    Add zero-mean Gaussian noise to a node acceleration array.

    The noisy signal is:
        y_noisy[node, t] = y_clean[node, t] + N(0, std)

    Inputs:
        node_accels : np.ndarray, shape (n_nodes, n_steps)
                      Clean acceleration array from FEA simulation.
        std         : float
                      Standard deviation of the noise in the same units
                      as the acceleration signal (e.g. in/s²).
                      Typical values for good speckle + good camera:
                        0.01  — very low noise
                        0.05  — moderate noise
        seed        : int or None
                      Random seed for reproducibility.
                      Pass sim_id-based seed for deterministic results.
                      None = random (not reproducible across runs).

    Output:
        noisy_accels : np.ndarray, shape (n_nodes, n_steps)
                       Acceleration array with added Gaussian noise.
    """
    if std < 0:
        raise ValueError(f"Noise std must be >= 0, got {std}.")

    if std == 0.0:
        return node_accels.copy()

    rng   = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=std, size=node_accels.shape)

    return node_accels + noise
