"""Reproduce the current-fitting results in Figures 3–4 and Section S5.

python code/fitting.py reference   Recalculate the integral curves and seeded observations.
python code/fitting.py noiseless   Fit the three potential windows and compute sensitivity angles.
python code/fitting.py windows     Fit endpoints 1–10 at lambda = 5, including range checks.
python code/fitting.py precision   Check the endpoint-4 stationary point at 55 digits.
python code/fitting.py noise       Fit the same signed observations with each rate model.

The fitting commands read retained inputs from data/ by default. Set --inputs
to the output directory of the reference command to use fresh quadrature.
In Table S5, the Zeng median fitted energy is 73.714769 (73.71 to two decimals).
The manuscript prints 73.72.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from fitting_models import (
    MODELS,
    SEARCH,
    IntegralCurrent,
    angle_degrees,
    dlog_current,
    fit_log_data,
    fit_signed,
    log_current,
)
from mhc_closed_form import normalized_log_rate
from mhc_reference import log_rate
from rate_analysis import exact_derivatives

ROOT = Path(__file__).resolve().parents[1]
ENERGIES = np.r_[0.1, 0.2, 0.5, 0.75, np.arange(1.0, 51.0)]
POTENTIAL = np.linspace(0.0, 100.0, 1001)


def make_reference(output):
    rows = [log_rate(lam, POTENTIAL) - np.log(2 * np.sqrt(np.pi * lam)) for lam in ENERGIES]
    log_oxidation = np.asarray(rows)
    np.savez_compressed(
        output / "inference_reference.npz",
        lambda_true=ENERGIES,
        eta_f_tilde=POTENTIAL,
        log_integral_kox=log_oxidation,
    )

    rng = np.random.default_rng(52100701)
    observations = {}
    for limit in (10, 100):
        mask = POTENTIAL <= limit
        eta = POTENTIAL[mask]
        observations[f"h_H{limit}"] = eta
        for energy in (10, 30, 50):
            index = np.flatnonzero(ENERGIES == energy)[0]
            current = np.exp(log_oxidation[index, mask]) * np.expm1(-eta)
            sigma = 0.01 * np.max(np.abs(current))
            # The negative draw preserves the realization in the cathodic-positive convention.
            noise = rng.normal(0, sigma, (100, len(eta)))
            observations[f"exact_L{energy}_H{limit}"] = current
            observations[f"noisy_L{energy}_H{limit}"] = current[None, :] - noise
    np.savez_compressed(output / "noise_observations.npz", **observations)


def noiseless(inputs, output):
    fits, curves, angles = [], [], []
    with np.load(inputs / "inference_reference.npz") as reference:
        energy_values = reference["lambda_true"]
        potential = reference["eta_f_tilde"]
        log_rates = reference["log_integral_kox"]
    log_rates_50 = log_rates[energy_values == 50][0]
    current_50 = np.exp(log_rates_50) * np.expm1(-potential)
    dlog_rate_50 = np.array([exact_derivatives(50, eta)[1] for eta in potential])
    for limit in (10, 20, 100):
        for residual, free in (("linear", True), ("linear", False), ("log", True)):
            mask = potential <= limit
            if residual == "log":
                mask &= potential > 0
            eta = potential[mask]
            with np.errstate(divide="ignore"):
                current_factor = np.log(-np.expm1(-eta))
            for model, method in MODELS.items():

                def model_curve(lam, eta=eta, method=method):
                    return log_current(lam, eta, method)

                grid = model_curve(SEARCH[:, None])
                for true_energy, log_rate_row in zip(energy_values, log_rates):
                    target = log_rate_row[mask] + current_factor
                    energy, amplitude, loss = fit_log_data(
                        target, model_curve, grid, free, residual == "log"
                    )
                    fitted = -amplitude * np.exp(model_curve(energy))
                    reference_current = -np.exp(target)
                    nrmse = np.linalg.norm(fitted - reference_current) / np.linalg.norm(
                        reference_current
                    )
                    k0 = np.exp(normalized_log_rate(energy, 0, method=method) - log_rate_row[0])
                    fits.append(
                        dict(
                            eta_max=limit,
                            lambda_true=true_energy,
                            model=model,
                            weighting=residual,
                            prefactor="fitted" if free else "fixed",
                            lambda_fit=energy,
                            prefactor_ratio=amplitude,
                            loss=loss,
                            lambda_bias_percent=100 * (energy / true_energy - 1),
                            k0_bias_percent=100 * (amplitude * k0 - 1),
                            nrmse=nrmse,
                        )
                    )
                    if true_energy == 50 and free and residual == "linear":
                        angle = angle_degrees(fitted, fitted * dlog_current(energy, eta, method))
                        angles.append((limit, model, energy, angle))
                        if limit == 10:
                            curves.extend(
                                zip(np.repeat(model, len(eta)), eta, reference_current, fitted)
                            )
        mask = potential <= limit
        current = current_50[mask]
        angles.append(
            (limit, "Full integral", 50.0, angle_degrees(current, current * dlog_rate_50[mask]))
        )

    pd.DataFrame(fits).to_csv(output / "noiseless_fits.csv", index=False)
    pd.DataFrame(
        curves, columns=["model", "eta_f_tilde", "integral_current", "fitted_current"]
    ).to_csv(output / "figure3_curves.csv", index=False)
    pd.DataFrame(angles, columns=["eta_max", "model", "lambda_fit", "angle_deg"]).to_csv(
        output / "compensation_angles.csv", index=False
    )


def noise(inputs, output):
    rows = []
    with np.load(inputs / "noise_observations.npz") as data:
        for limit in (10, 100):
            eta = data[f"h_H{limit}"]
            for model, method in MODELS.items():

                def curve(lam, eta=eta, method=method):
                    return -np.exp(log_current(lam, eta, method))

                grid = curve(SEARCH[:, None])
                for energy in (10, 30, 50):
                    noisy = data[f"noisy_L{energy}_H{limit}"]
                    for free in (False, True):
                        for replicate, observed in enumerate(noisy, 1):
                            lam, amp, loss = fit_signed(observed, curve, grid, free)
                            rows.append(
                                (
                                    model,
                                    energy,
                                    limit,
                                    "fitted" if free else "fixed",
                                    replicate,
                                    lam,
                                    amp,
                                    loss,
                                )
                            )
        eta = data["h_H10"]
        observed_curves = data["noisy_L50_H10"]
        integral = IntegralCurrent(4096)

        def integral_curve(lam):
            return integral(lam, eta)

        grid = integral_curve(SEARCH[:, None])
        for replicate, observed in enumerate(observed_curves, 1):
            lam, amp, loss = fit_signed(observed, integral_curve, grid, True)
            rows.append(("Full integral", 50, 10, "fitted", replicate, lam, amp, loss))

    fits = pd.DataFrame(
        rows,
        columns=[
            "model",
            "lambda_true",
            "eta_max",
            "prefactor",
            "replicate",
            "lambda_fit",
            "prefactor_ratio",
            "loss",
        ],
    )
    fits["lambda_bias_percent"] = 100 * (fits.lambda_fit / fits.lambda_true - 1)
    fits.to_csv(output / "noisy_fits.csv", index=False)
    selected = fits[(fits.lambda_true == 50) & (fits.eta_max == 10) & (fits.prefactor == "fitted")]
    columns = ["lambda_fit", "prefactor_ratio"]
    exact = selected[selected.model == "Full integral"].set_index("replicate")[columns]
    summary = []
    for model, group in selected.groupby("model", sort=False):
        values = group.set_index("replicate")[columns].copy()
        if model != "Full integral":
            for column in columns:
                values[column + "_difference"] = values[column] - exact[column]
        for column in values:
            summary.append((model, column, *np.percentile(values[column], [2.5, 50, 97.5])))
    pd.DataFrame(summary, columns=["model", "quantity", "p025", "median", "p975"]).to_csv(
        output / "paired_noise_table.csv", index=False
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "study", choices=["reference", "noiseless", "windows", "precision", "noise"]
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--inputs", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / "data").resolve():
        parser.error("Choose a separate output directory to preserve the archived inputs.")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.study == "reference":
        make_reference(args.output)
    elif args.study == "noiseless":
        noiseless(args.inputs, args.output)
    elif args.study == "noise":
        noise(args.inputs, args.output)
    elif args.study == "windows":
        from window_fits import windows

        windows(args.inputs, args.output)
    else:
        from window_precision import precision

        precision(args.output)
    print(f"{args.study} results written to {args.output}")


if __name__ == "__main__":
    main()
