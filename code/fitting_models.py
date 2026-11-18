"""Current models and one-dimensional searches used in Section S5."""

import numpy as np
from mhc_closed_form import log_rate_derivatives, normalized_log_rate
from rate_analysis import zeng_derivatives
from scipy.optimize import minimize_scalar
from scipy.special import expit, roots_hermite

MODELS = {"GM-ET": "mixture", "Zeng": "zeng"}
SEARCH = np.unique(np.r_[np.geomspace(0.01, 100, 321), np.linspace(0.1, 100, 401)])


def log_current(energy, potential, method):
    """Log of the absolute cathodic-positive current at equal concentrations."""
    with np.errstate(divide="ignore"):
        return normalized_log_rate(energy, potential, method=method) + np.log(-np.expm1(-potential))


def dlog_current(energy, potential, method):
    """Derivative of log current magnitude with respect to dimensionless energy."""
    if method == "mixture":
        return log_rate_derivatives(energy, potential)[1][0]
    return zeng_derivatives(energy, potential)[1]


class IntegralCurrent:
    """Direct Gauss–Hermite quadrature of the normalized oxidation rate."""

    def __init__(self, order):
        self.nodes, self.weights = roots_hermite(order)
        self.weights /= np.sqrt(np.pi)

    def __call__(self, energy, potential):
        energy, potential = np.broadcast_arrays(energy, potential)
        rate = np.empty(energy.size)
        for start in range(0, energy.size, 128):
            block = slice(start, start + 128)
            lam = energy.flat[block]
            eta = potential.flat[block]
            argument = eta[:, None] - lam[:, None] - 2 * np.sqrt(lam[:, None]) * self.nodes
            rate[block] = expit(argument) @ self.weights
        return rate.reshape(energy.shape) * np.expm1(-potential)

    def log_curve_and_derivative(self, energy, potential):
        """Log current and its energy derivative for one trial energy."""
        root = np.sqrt(energy)
        argument = potential[:, None] - energy - 2 * root * self.nodes
        occupation = expit(argument)
        rate = occupation @ self.weights
        derivative = (occupation * expit(-argument) * (-1 - self.nodes / root)) @ self.weights
        with np.errstate(divide="ignore"):
            log_current = np.log(rate) + np.log(-np.expm1(-potential))
        return log_current, derivative / rate


def refine_minima(grid_values, objective, tolerance):
    """Refine each sampled minimum and compare its loss with both endpoints."""
    indices = (
        np.flatnonzero(
            (grid_values[1:-1] <= grid_values[:-2]) & (grid_values[1:-1] <= grid_values[2:])
        )
        + 1
    )
    candidates = [(grid_values[i], SEARCH[i]) for i in (0, -1)]
    for index in indices:
        fit = minimize_scalar(
            objective,
            bounds=(SEARCH[index - 1], SEARCH[index + 1]),
            method="bounded",
            options={"xatol": tolerance, "maxiter": 300},
        )
        if not fit.success:
            raise RuntimeError(fit.message)
        candidates.append((fit.fun, fit.x))
    return min(candidates)[1]


def fit_signed(observations, curve, grid, free):
    """Ordinary least squares on signed currents, retaining the equilibrium point."""
    scale = np.sqrt(np.mean(observations**2))
    target = observations / scale
    scaled_grid = grid / scale
    floor = np.finfo(float).tiny
    if free:
        amplitude = np.maximum((scaled_grid @ target) / np.sum(scaled_grid**2, axis=1), floor)
    else:
        amplitude = np.ones(len(SEARCH))
    sampled = np.mean((scaled_grid * amplitude[:, None] - target) ** 2, axis=1)

    def profile(energy):
        shape = curve(energy) / scale
        amplitude = max(np.dot(shape, target) / np.dot(shape, shape), floor) if free else 1.0
        return np.mean((amplitude * shape - target) ** 2), amplitude

    energy = refine_minima(sampled, lambda lam: profile(lam)[0], 1e-11)
    loss, amplitude = profile(energy)
    return energy, amplitude, loss


def fit_log_data(target, curve, grid, free, log_residual):
    """Profile the amplitude for exact log-current magnitudes, following S5.1."""

    def profile(shapes):
        if log_residual:
            loga = (
                np.mean(target - shapes, axis=-1, keepdims=True)
                if free
                else np.zeros((*shapes.shape[:-1], 1))
            )
            residual = shapes + loga - target
        else:
            magnitude = np.exp(shapes)
            data = np.exp(target)
            if free:
                amp = np.sum(magnitude * data, axis=-1, keepdims=True)
                amp /= np.sum(magnitude**2, axis=-1, keepdims=True)
            else:
                amp = np.ones((*shapes.shape[:-1], 1))
            loga = np.log(amp)
            residual = (amp * magnitude - data) / np.sqrt(np.mean(data**2))
        return np.mean(residual**2, axis=-1), np.squeeze(loga, axis=-1)

    sampled, _ = profile(grid)
    energy = refine_minima(sampled, lambda lam: profile(curve(lam))[0], 1e-12)
    loss, loga = profile(curve(energy))
    if log_residual:
        loss /= np.log(10) ** 2
    return energy, np.exp(loga), loss


def angle_degrees(current, derivative):
    """Angle between dI/dlambda and -dI/d(k_infinity), as defined in S5.2."""
    cosine = -np.dot(current, derivative) / (np.linalg.norm(current) * np.linalg.norm(derivative))
    return np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
