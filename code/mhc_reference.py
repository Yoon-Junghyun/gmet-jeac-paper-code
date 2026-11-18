"""Adaptive and Gauss–Hermite quadrature for the oxidation integral (SI S2).

These functions return the logarithm of the integral in the original MHC
convention. Subtract log(2 sqrt(pi lambda_tilde)) to normalize by k_infinity,
as in the paper. Detailed balance gives log k(-eta) = log k(+eta) - eta.
"""

import itertools

import numpy as np
from scipy.integrate import quad
from scipy.special import expit, roots_hermite


def log_rate_quad(lam, eta, direct=False):
    """Evaluate one point. With direct=True, integrate the signed branch."""
    lam, eta = float(lam), float(eta)
    if lam <= 0:
        raise ValueError("lam must be positive")
    potential = eta if direct else abs(eta)
    scale = 2 * np.sqrt(lam)

    def integrand(t):
        return np.exp(-t * t) * expit(potential - lam - scale * t)

    # Split at the Gaussian peak and the midpoint of the Fermi transition.
    midpoint = float(np.clip((potential - lam) / scale, -12, 12))
    bounds = sorted({-12.0, midpoint, 0.0, 12.0})
    integral = sum(
        quad(integrand, a, b, epsabs=1e-30, epsrel=3e-12, limit=150)[0]
        for a, b in itertools.pairwise(bounds)
    )
    return np.log(scale * integral) + (0.0 if direct else min(eta, 0.0))


def log_rate(lam, eta):
    """Adaptive quadrature on broadcast-compatible arrays of lambda and eta."""
    lam, eta = np.broadcast_arrays(lam, eta)
    values = [log_rate_quad(energy, potential) for energy, potential in zip(lam.flat, eta.flat)]
    return np.asarray(values).reshape(lam.shape)


def log_rate_hermite(lam, eta, n=512):
    """An independent Gauss–Hermite evaluation with n nodes."""
    lam, eta = np.broadcast_arrays(lam, eta)
    if np.any(lam <= 0):
        raise ValueError("lam must be positive")
    nodes, weights = roots_hermite(n)
    values = np.empty(lam.size)
    for start in range(0, lam.size, 256):
        section = slice(start, start + 256)
        energy, potential = lam.ravel()[section], eta.ravel()[section]
        scale = 2 * np.sqrt(energy)
        argument = np.abs(potential)[:, None] - energy[:, None] - scale[:, None] * nodes
        values[section] = np.log(scale * (expit(argument) @ weights)) + np.minimum(potential, 0)
    return values.reshape(lam.shape)
