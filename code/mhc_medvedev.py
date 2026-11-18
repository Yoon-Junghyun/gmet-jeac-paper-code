"""Medvedev's normalized oxidation rates for nonnegative formal overpotential.

I. G. Medvedev, J. Electroanal. Chem. 761 (2016), 46–54. The reorganization
energy lam is in units of k_B T and mu is the formal overpotential divided by
lam. Each function returns k_ox / k_infinity. Section S3.2 of the Supplement
specifies the pieces used in Fig. S1.
"""

import numpy as np
from scipy.special import erfcx


def k_A(lam, mu):
    """Activation-limited expression, Eq. (21), for |mu| <= 0.9."""
    lam, mu = np.broadcast_arrays(lam, mu)
    root = np.sqrt(lam)

    def psi(u):
        return (
            np.sqrt(np.pi)
            * root
            / 2
            * (erfcx((1 - u) * root / 2) + (3 - u) * lam / 2 * erfcx((3 - u) * root / 2))
        )

    integral = (
        np.pi / (2 * np.cos(np.pi * mu / 2))
        - lam
        - 4 * (mu**4 - 5 * mu**2 + 36) / ((9 - mu**2) ** 2 * (1 - mu**2))
        + psi(mu)
        + psi(-mu)
    )
    marcus = np.sqrt(np.pi) / (2 * root) * np.exp(-lam * (1 - mu) ** 2 / 4)
    return marcus * 2 * integral / np.pi


def k_SA(lam, mu):
    """Saturation-side expression, Eq. (22), for 1.1 <= mu <= 2.9."""
    return 1 - k_A(lam, 2 - np.asarray(mu))


def k_Aint(lam, mu):
    """Intermediate expression, Eq. (B.10)."""
    lam, mu = np.broadcast_arrays(lam, mu)
    root = np.sqrt(lam)

    def scaled_phi(n, u):
        # exp[-lam*(1-mu)^2/4] times phi_n(u). Reflection of erfcx
        # keeps its argument nonnegative even above the Marcus maximum.
        z = (n - u) * root / 2
        tail = np.exp(-lam * (1 - mu) ** 2 / 4) * erfcx(np.abs(z))
        exponent = np.where(z < 0, lam * (n - u - 1 + mu) * (n - u + 1 - mu) / 4, 0.0)
        return np.where(z < 0, 2 * np.exp(exponent) - tail, tail)

    return (2 * scaled_phi(1, mu) + scaled_phi(1, -mu) - scaled_phi(3, mu)) / 4


def piecewise_rate(lam, mu):
    """Use the left piece at 0.9 and 2.9 and the right piece at 1.1."""
    lam, mu = np.broadcast_arrays(np.asarray(lam, float), np.asarray(mu, float))
    if np.any(lam <= 0) or np.any(mu < 0):
        raise ValueError("lam must be positive and mu must be nonnegative")
    activation = mu <= 0.9
    saturation = (1.1 <= mu) & (mu <= 2.9)
    rate = np.empty(lam.shape)
    for mask, formula in (
        (activation, k_A),
        (saturation, k_SA),
        (~(activation | saturation), k_Aint),
    ):
        rate[mask] = formula(lam[mask], mu[mask])
    return rate[()]
