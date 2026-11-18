"""Numerical checks on Eq. 7, Table S4, the large-lambda limit and the current sign."""

import numpy as np
from mhc_closed_form import (
    COEFFICIENTS,
    FARADAY,
    current_density,
    log_rate_derivatives,
    log_rates,
    normalized_rate,
    rates,
)
from mhc_reference import log_rate, log_rate_hermite
from numpy.testing import assert_allclose


def main():
    # Use a different quadrature from the one used to measure the model error.
    lam = np.array([0.1, 1.0, 5.0, 30.0, 50.0, 100.0])
    eta = np.array([-0.2, 0.0, 5.0, -10.0, 50.0, 100.0])
    assert_allclose(log_rate_hermite(lam, eta, n=1024), log_rate(lam, eta), atol=2e-6, rtol=0)

    for method in ("mixture", "zeng"):
        red, ox = log_rates(lam, eta, method=method)
        assert_allclose(ox - red, eta, atol=3e-14, rtol=0)
        assert_allclose(red, log_rates(lam, -eta, method=method)[1])

    for energy in (5.0, 50.0):
        h = np.array([-3.0, 0.0, 3.0])
        step = 1e-5
        for i, (dl, dh) in enumerate(log_rate_derivatives(energy, h)):
            dl_fd = (log_rates(energy + step, h)[i] - log_rates(energy - step, h)[i]) / (2 * step)
            dh_fd = (log_rates(energy, h + step)[i] - log_rates(energy, h - step)[i]) / (2 * step)
            assert_allclose(dl, dl_fd, atol=4e-9)
            assert_allclose(dh, dh_fd, atol=4e-9)

    # At unequal concentrations the zero of current is at the Nernst potential.
    co, cr = 3.0, 1.0
    eta_eq = np.array([-1.0, 0.0, 1.0])
    red, ox = rates(10.0, eta_eq + np.log(co / cr))
    current = current_density(10.0, eta_eq, co, cr) / FARADAY
    assert_allclose(current, co * red - cr * ox, atol=1e-15)
    assert current[0] > 0 and current[1] == 0 and current[2] < 0

    lam = np.array([50.0, 100.0])
    reference = np.exp(log_rate(lam, 0.0)) / (2 * np.sqrt(np.pi * lam))
    error_percent = 100 * (normalized_rate(lam, 0.0) / reference - 1)
    assert_allclose(error_percent, [-0.1293, -0.228], rtol=0, atol=0.0005)
    weights, variances = COEFFICIENTS[2]
    limit = 2 / np.pi * np.dot(weights, np.exp(np.array(variances) / 8))
    assert_allclose(limit, 0.99630, rtol=0, atol=0.000005)
    print(
        f"Formal-potential errors at lambda = 50 and 100: {error_percent[0]:.4f}%, "
        f"{error_percent[1]:.4f}%."
    )
    print(f"Large-lambda rate ratio: {limit:.8f}.")
    print("Quadrature, detailed balance, derivatives and current checks passed.")


if __name__ == "__main__":
    main()
