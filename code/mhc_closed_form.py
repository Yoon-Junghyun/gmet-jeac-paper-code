"""GM-ET and Zeng et al. (2014) rates for symmetric Marcus kinetics.

Reorganization energy and formal overpotential are expressed in thermal units.
Rates are divided by the limiting rate constant. For functions returning two
rates, the order is reduction, oxidation (Eq. 7 in the paper).
"""

import numpy as np
from scipy.special import expit, log_ndtr, logsumexp

FARADAY = 96485.33212  # C/mol
# Table S3, full-precision weights and variances. The paper uses N = 2.
COEFFICIENTS = {
    1: ([1.0], [3.3143101061879254]),
    2: ([0.7227441690552268, 0.27725583094477313], [2.0093643274018693, 6.640460404596436]),
    3: (
        [0.3880072277562808, 0.5443643623083362, 0.06762840993538288],
        [1.4228142786594116, 3.8160986539110486, 9.764870328538871],
    ),
}


def _coordinates(lambda_tilde, eta):
    lam, eta = np.broadcast_arrays(np.asarray(lambda_tilde, float), np.asarray(eta, float))
    if np.any(lam <= 0):
        raise ValueError("lambda_tilde must be positive")
    return lam, eta


def log_even_factor(lambda_tilde, eta_f_tilde, terms=2, method="mixture"):
    """Logarithm of the sum of the normalized reduction and oxidation rates."""
    lam, eta = _coordinates(lambda_tilde, eta_f_tilde)
    if method == "zeng":
        argument = (np.sqrt(eta**2 + 1 + np.sqrt(lam)) - lam) / np.sqrt(2 * lam)
        return log_ndtr(argument)
    if method != "mixture" or terms not in COEFFICIENTS:
        raise ValueError("use method='zeng' or method='mixture' with 1, 2 or 3 terms")

    weights, variances = COEFFICIENTS[terms]
    components = []
    for weight, variance in zip(weights, variances):
        width = np.sqrt(2 * lam + variance)
        pair = np.logaddexp(log_ndtr((eta - lam) / width), log_ndtr((-eta - lam) / width))
        components.append(np.log(weight) + pair)
    return logsumexp(np.stack(components), axis=0)


def normalized_log_rate(lambda_tilde, eta_f_tilde, terms=2, method="mixture"):
    """Log normalized oxidation rate. Reflection gives the reduction rate."""
    return log_even_factor(lambda_tilde, eta_f_tilde, terms, method) - np.logaddexp(
        0.0, -np.asarray(eta_f_tilde)
    )


def normalized_rate(lambda_tilde, eta_f_tilde, terms=2, method="mixture"):
    return np.exp(normalized_log_rate(lambda_tilde, eta_f_tilde, terms, method))


def log_rates(lambda_tilde, eta_f, terms=2, method="mixture"):
    """Log normalized reduction and oxidation rates evaluated from the same sum."""
    total = log_even_factor(lambda_tilde, eta_f, terms, method)
    eta = np.asarray(eta_f, float)
    return total - np.logaddexp(0.0, eta), total - np.logaddexp(0.0, -eta)


def rates(lambda_tilde, eta_f, terms=2, method="mixture"):
    return tuple(np.exp(log_k) for log_k in log_rates(lambda_tilde, eta_f, terms, method))


def standard_rate(lambda_tilde, terms=2, method="mixture"):
    """Standard rate divided by the limiting rate."""
    return normalized_rate(lambda_tilde, 0.0, terms, method)


def log_rate_derivatives(lambda_tilde, eta_f, terms=2):
    """Return (dlog(k)/dlambda, dlog(k)/deta_f) for reduction, then oxidation."""
    lam, eta = _coordinates(lambda_tilde, eta_f)
    total = log_even_factor(lam, eta, terms)
    weights, variances = COEFFICIENTS[terms]
    shape = (-1,) + (1,) * lam.ndim
    weights = np.asarray(weights).reshape(shape)
    variances = np.asarray(variances).reshape(shape)
    width = np.sqrt(4 * lam + 2 * variances)
    minus, plus = (lam - eta) / width, (lam + eta) / width
    offset = np.log(weights / np.sqrt(np.pi)) - total
    fm, fp = np.exp(offset - minus**2), np.exp(offset - plus**2)
    d_lambda = -np.sum(
        fm * (1 / width - 2 * (lam - eta) / width**3)
        + fp * (1 / width - 2 * (lam + eta) / width**3),
        axis=0,
    )
    d_eta = np.sum((fm - fp) / width, axis=0)
    return (d_lambda, d_eta - expit(eta)), (d_lambda, d_eta + expit(-eta))


def current_density(lambda_tilde, eta, c_O, c_R, k_infinity=1.0, terms=2, method="mixture"):
    """Cathodic-positive current density (A/m²) at Nernst-referenced eta.

    Concentrations are in mol/m³ and k_infinity is in m/s. Computing the
    concentration difference with expm1 retains precision near equilibrium.
    """
    lam, eta, co, cr = np.broadcast_arrays(lambda_tilde, eta, c_O, c_R)
    if np.any(co <= 0) or np.any(cr <= 0):
        raise ValueError("Nernst-referenced overpotential requires positive concentrations")
    formal = eta + np.log(co) - np.log(cr)
    total = np.exp(log_even_factor(lam, formal, terms, method))
    difference = np.where(
        eta >= 0,
        cr * expit(formal) * np.expm1(-np.maximum(eta, 0)),
        -co * expit(-formal) * np.expm1(np.minimum(eta, 0)),
    )
    return FARADAY * k_infinity * total * difference


def exchange_current_density(lambda_tilde, c_O, c_R, k_infinity=1.0, terms=2, method="mixture"):
    """Positive equilibrium exchange-current density in A/m²."""
    lam, co, cr = np.broadcast_arrays(lambda_tilde, c_O, c_R)
    if np.any(co <= 0) or np.any(cr <= 0):
        raise ValueError("exchange current requires positive concentrations")
    formal = np.log(co) - np.log(cr)
    total = np.exp(log_even_factor(lam, formal, terms, method))
    return FARADAY * k_infinity * total * co * expit(-formal)
