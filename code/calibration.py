"""Reproduce the coefficient fits and the two tables in Supplement section S2.

Run python code/calibration.py. Both output tables distinguish the new
least-squares fits from the paper's full-precision coefficients. The
last digits of a refit can vary with the numerical libraries and platform.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from mhc_closed_form import COEFFICIENTS
from mhc_reference import log_rate, log_rate_hermite
from scipy.optimize import least_squares
from scipy.special import log_ndtr, logsumexp, softmax
from scipy.stats import qmc


def sample(power, seed, low=0.1, high=50.0, max_eta=20.0):
    """Scrambled Sobol sample with extra resolution near zero and half saturation."""
    u, v, region = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power).T
    energy = np.exp(np.log(low) + u * np.log(high / low))
    eta = max_eta * v
    central = region < 0.25
    eta[central] = 3 * v[central] ** 2
    transition = (region >= 0.25) & (region < 0.5)
    near_half = energy + 3 * (2 * v - 1) * np.sqrt(energy)
    eta[transition] = np.clip(near_half[transition], 0, max_eta)
    return energy, eta


def reference(energy, eta):
    """Logarithm of the normalized oxidation rate from the integral in section S2."""
    return log_rate(energy, eta) - np.log(2 * np.sqrt(np.pi * energy))


def mixture(energy, eta, weights, variances):
    terms = []
    for weight, variance in zip(weights, variances):
        scale = np.sqrt(2 * energy + variance)
        positive = log_ndtr((eta - energy) / scale)
        reflected = log_ndtr((-eta - energy) / scale)
        terms.append(np.log(weight) + np.logaddexp(positive, reflected))
    return logsumexp(terms, axis=0) - np.logaddexp(0, -eta)


def fit_mixture(energy, eta, truth, initial_weights, initial_variances):
    n = len(initial_weights)
    weights = np.asarray(initial_weights)
    start = np.r_[np.log(initial_variances), np.log(weights[:-1] / weights[-1])]

    def coefficients(parameters):
        return softmax(np.r_[parameters[n:], 0.0]), np.exp(parameters[:n])

    def residuals(parameters):
        return mixture(energy, eta, *coefficients(parameters)) - truth

    fits = []
    for seed in range(4):
        initial = start.copy()
        if seed:
            rng = np.random.default_rng(500 + 10 * n + seed)
            initial += rng.normal(0, 0.12, len(start))
        fit = least_squares(residuals, initial, max_nfev=600, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if not fit.success:
            raise RuntimeError(f"{n}-component fit: {fit.message}")
        fits.append(fit)
    weights, variances = coefficients(min(fits, key=lambda fit: fit.cost).x)
    order = np.argsort(variances)
    return weights[order], variances[order]


def grid(max_eta, count):
    energy, eta = np.meshgrid(
        np.geomspace(0.1, 50, 151), np.linspace(0, max_eta, count), indexing="ij"
    )
    return energy.ravel(), eta.ravel()


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    energy, eta = sample(12, 52061001)
    truth = reference(energy, eta)
    # Check the training reference against an independent Gauss-Hermite rule.
    hermite = log_rate_hermite(energy, eta, n=2048) - np.log(2 * np.sqrt(np.pi * energy))
    if np.max(np.abs(np.expm1(hermite - truth))) > 1e-10:
        raise ArithmeticError("Quadrature disagreement on the calibration sample")
    starts = [
        ([1.0], [3.3]),
        ([0.70, 0.30], [1.96, 6.43]),
        ([0.36, 0.56, 0.08], [1.39, 3.67, 9.37]),
    ]
    fitted = {}
    with (output / "calibration_coefficients.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["components", "component", "coefficients", "weight", "variance"])
        for n, initial in enumerate(starts, 1):
            fitted[n] = fit_mixture(energy, eta, truth, *initial)
            for source, (weights, variances) in (("paper", COEFFICIENTS[n]), ("refit", fitted[n])):
                writer.writerows(
                    (n, i, source, w, v) for i, (w, v) in enumerate(zip(weights, variances), 1)
                )

    # Monahan and Stefanski (1989), Table 1: their entries are inverse scales.
    ms_weight = 0.564424843456014
    ms = ([ms_weight, 1 - ms_weight], np.array([0.768621151201617, 0.435255180069097]) ** -2)
    two_component = [
        ("GM-ET", "paper", COEFFICIENTS[2]),
        ("GM-ET", "refit", fitted[2]),
        ("Monahan-Stefanski", "published", ms),
    ]
    calibrated = [
        (f"GM-{n}", source, models[n])
        for n in (1, 2, 3)
        for source, models in (("paper", COEFFICIENTS), ("refit", fitted))
    ]
    test_sets = [
        ("held_out", sample(13, 52061003), two_component),
        ("calibration_grid", grid(20, 201), two_component),
        ("primary_grid", grid(100, 1001), calibrated),
    ]
    with (output / "calibration_errors.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample",
                "method",
                "coefficients",
                "points",
                "rms_percent",
                "max_abs_percent",
                "lambda_tilde_at_max",
                "eta_f_tilde_at_max",
            ]
        )
        for name, (energy, eta), methods in test_sets:
            truth = reference(energy, eta)
            for method, source, parameters in methods:
                error = 100 * np.expm1(mixture(energy, eta, *parameters) - truth)
                index = np.argmax(np.abs(error))
                writer.writerow(
                    [
                        name,
                        method,
                        source,
                        len(error),
                        np.sqrt(np.mean(error**2)),
                        abs(error[index]),
                        energy[index],
                        eta[index],
                    ]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    main(parser.parse_args().output)
