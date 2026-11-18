"""Independent 55-digit calculation for the endpoint-4 Zeng logarithmic fit."""

import csv

import mpmath as mp
import numpy as np
from mhc_closed_form import normalized_log_rate


def verify(precision=55):
    """Compute the stationary point using direct quadrature and erfc formulas."""
    with mp.workdps(precision):
        h = [mp.mpf(i) / 10 for i in range(1, 41)]

        def exact_rate(eta):
            lam = mp.mpf(5)

            def kernel(t):
                return (
                    mp.exp(-t * t) / (1 + mp.exp(lam - eta + 2 * mp.sqrt(lam) * t)) / mp.sqrt(mp.pi)
                )

            return mp.quad(kernel, [-14, -4, 0, 4, 14])

        reference = [mp.log(exact_rate(eta)) + mp.log(-mp.expm1(-eta)) for eta in h]
        k0_reference = exact_rate(mp.mpf(0))

        def log_current(lam, eta):
            z = (lam - mp.sqrt(eta * eta + 1 + mp.sqrt(lam))) / (2 * mp.sqrt(lam))
            return mp.log(mp.erfc(z) / 2) + mp.log(mp.tanh(eta / 2))

        def loss(lam, details=False):
            curve = [log_current(lam, eta) for eta in h]
            loga = sum(y - f for y, f in zip(reference, curve)) / len(h)
            value = sum(((f + loga - y) / mp.log(10)) ** 2 for y, f in zip(reference, curve)) / len(
                h
            )
            if not details:
                return value
            z0 = (lam - mp.sqrt(1 + mp.sqrt(lam))) / (2 * mp.sqrt(lam))
            k0_ratio = mp.exp(loga) * mp.erfc(z0) / 4 / k0_reference
            nrmse = mp.sqrt(
                sum((mp.exp(f + loga) - mp.exp(y)) ** 2 for y, f in zip(reference, curve))
                / sum(mp.exp(2 * y) for y in reference)
            )
            return value, loga, k0_ratio, nrmse

        stationary = mp.findroot(lambda lam: mp.diff(loss, lam), (mp.mpf(120), mp.mpf(140)))
        value, loga, k0_ratio, nrmse = loss(stationary, details=True)
        slope = mp.diff(loss, stationary)
        curvature = mp.diff(loss, stationary, 2)
        assert 120 < stationary < 140 and curvature > 0
        assert abs(slope) < mp.mpf(10) ** (-precision + 10)
        assert value < loss(mp.mpf(100))

        # Compare the independently evaluated erfc expression to the double-precision kernel.
        eta_double = np.array([float(eta) for eta in h])
        expected = np.array([float(log_current(stationary, eta)) for eta in h])
        implemented = normalized_log_rate(float(stationary), eta_double, method="zeng") + np.log(
            -np.expm1(-eta_double)
        )
        expression_error = float(np.max(abs(implemented - expected)))
        assert expression_error < 1e-12
        return dict(
            lambda_fit=str(stationary),
            limiting_rate_ratio=str(mp.exp(loga)),
            loss_log10=str(value),
            first_derivative=str(slope),
            second_derivative=str(curvature),
            original_cap100_loss_log10=str(loss(mp.mpf(100))),
            k0_relative_error_percent=str(100 * (k0_ratio - 1)),
            lambda_relative_error_percent=str(100 * (stationary / 5 - 1)),
            current_NRMSE_percent=str(100 * nrmse),
            reference_precision_digits=precision,
            reference_integration_limit=14,
            double_precision_expression_max_absolute_log_difference=expression_error,
        )


def precision(output):
    results = verify()
    with (output / "window_precision.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results)
        writer.writeheader()
        writer.writerow(results)
