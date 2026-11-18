"""Endpoint fits for Figures 4 and S4, including the competing Zeng minima."""

import numpy as np
import pandas as pd
from fitting_models import MODELS, SEARCH, IntegralCurrent, dlog_current, log_current
from mhc_closed_form import normalized_log_rate
from scipy.optimize import brentq, minimize_scalar


def search_grid(lower, upper):
    """Dense logarithmic grid plus linear coverage of the competing minima."""
    linear_lower, linear_upper = max(lower, 0.1), min(upper, 150.0)
    linear = (
        np.linspace(linear_lower, linear_upper, 1501)
        if linear_lower <= linear_upper
        else np.empty(0)
    )
    return np.unique(np.r_[np.geomspace(lower, upper, 3001), linear])


def scaled_profile(logf, logy, weighting):
    """Profile the positive amplitude after removing the model log-rate scale.

    Returns the natural-log canonical loss, log amplitude and fitted current.
    The zero-current point is allowed for linear fits and excluded for log fits.
    No raw vanishing model current or enormous amplitude is multiplied together.
    """
    shift = np.max(logf, axis=-1, keepdims=True)
    centered = logf - shift
    if weighting == "log":
        logb = np.mean(logy - centered, axis=-1, keepdims=True)
        residual = centered + logb - logy
        loss = np.mean(residual**2, axis=-1)
        fitted = np.exp(centered + logb)
    else:
        scaled = np.exp(centered)
        target = np.exp(logy)
        b = np.sum(scaled * target, axis=-1, keepdims=True) / np.sum(
            scaled**2, axis=-1, keepdims=True
        )
        logb = np.log(b)
        fitted = scaled * b
        loss = np.sum((fitted - target) ** 2, axis=-1) / np.sum(target**2)
    return loss, np.squeeze(logb - shift, axis=-1), fitted


def fit_window(h, logy, curve_and_derivative, weighting, grid, log_grid):
    losses, _, _ = scaled_profile(log_grid, logy, weighting)
    minima = np.flatnonzero((losses[1:-1] <= losses[:-2]) & (losses[1:-1] <= losses[2:])) + 1

    def detail(lam):
        logf, derivative = curve_and_derivative(lam, h)
        loss, loga, fitted = scaled_profile(logf, logy, weighting)
        if weighting == "log":
            # Center first to preserve precision when the model rate is small.
            centered = logf - np.max(logf)
            residual = centered + np.mean(logy - centered) - logy
            score = np.mean(residual * derivative)
        else:
            target = np.exp(logy)
            score = np.sum((fitted - target) * fitted * derivative) / np.sum(target**2)
        return float(loss), float(loga), fitted, float(score), derivative

    candidates = [
        dict(
            lambda_candidate=float(grid[i]),
            canonical_objective=float(losses[i]),
            refinement="endpoint",
            converged=True,
            bounded_to_score_lambda_change=0.0,
        )
        for i in [0, -1]
    ]
    for index in minima:
        lo, hi = grid[index - 1], grid[index + 1]
        fit = minimize_scalar(
            lambda u: detail(np.exp(u))[0],
            bounds=(np.log(lo), np.log(hi)),
            method="bounded",
            options={"xatol": 1e-13, "maxiter": 300},
        )
        lam = float(np.exp(fit.x))
        change = 0.0
        refinement = "bounded_log_lambda"
        if detail(lo)[3] < 0 < detail(hi)[3]:
            root = brentq(lambda trial: detail(trial)[3], lo, hi, xtol=2e-13, rtol=1e-14)
            change = abs(root - lam)
            lam = float(root)
            refinement = "analytic_score_root"
        candidates.append(
            dict(
                lambda_candidate=lam,
                canonical_objective=detail(lam)[0],
                refinement=refinement,
                converged=bool(fit.success),
                bounded_to_score_lambda_change=change,
            )
        )
    selected = min(candidates, key=lambda c: c["canonical_objective"])
    lam = selected["lambda_candidate"]
    loss, loga, fitted, score, derivative = detail(lam)
    target = np.exp(logy)
    if weighting == "log":
        sensitivity = np.sqrt(np.mean((lam * (derivative - np.mean(derivative))) ** 2))
    else:
        tangent = fitted * lam * derivative
        projected = tangent - fitted * np.dot(fitted, tangent) / np.dot(fitted, fitted)
        sensitivity = np.linalg.norm(projected) / np.linalg.norm(target)
    result = dict(
        lambda_fit=lam,
        log_k_infinity_fit_over_true=loga,
        k_infinity_fit_over_true=float(np.exp(loga)),
        objective_value=loss / (np.log(10.0) ** 2) if weighting == "log" else loss,
        normalized_current_RMSE=float(np.linalg.norm(fitted - target) / np.linalg.norm(target)),
        boundary_status="lower" if lam == grid[0] else "upper" if lam == grid[-1] else "interior",
        local_minima_refined=len(minima),
        convergence_status="converged" if selected["converged"] else "failed",
        selected_refinement=selected["refinement"],
        bounded_to_score_lambda_change=selected["bounded_to_score_lambda_change"],
        projected_shape_sensitivity=float(sensitivity),
        canonical_score=score,
    )
    for candidate in candidates:
        candidate["selected"] = candidate is selected
    return result, candidates


def windows(inputs, output):
    """Fit forty curves on each interval used to check the upper search limit."""
    with np.load(inputs / "inference_reference.npz") as reference:
        eta = reference["eta_f_tilde"][:101]
        reference_rates = reference["log_integral_kox"][reference["lambda_true"] == 5, :101][0]
    with np.errstate(divide="ignore"):
        target = reference_rates + np.log(-np.expm1(-eta))
    rows, candidates = [], []
    for upper in (200, 500, 1000):
        search = search_grid(0.01, upper)
        for model, method in MODELS.items():

            def curve_and_derivative(energy, potential, method=method):
                return log_current(energy, potential, method), dlog_current(
                    energy, potential, method
                )

            grid = log_current(search[:, None], eta, method)
            for endpoint in range(1, 11):
                for weighting in ("linear", "log"):
                    mask = eta <= endpoint
                    if weighting == "log":
                        mask &= eta > 0
                    fit, alternatives = fit_window(
                        eta[mask],
                        target[mask],
                        curve_and_derivative,
                        weighting,
                        search,
                        grid[:, mask],
                    )
                    energy = fit["lambda_fit"]
                    log_amplitude = fit["log_k_infinity_fit_over_true"]
                    log_k0 = (
                        log_amplitude
                        + normalized_log_rate(energy, 0, method=method)
                        - reference_rates[0]
                    )
                    settings = dict(
                        eta_max=endpoint,
                        model=model,
                        weighting=weighting,
                        search_lower=0.01,
                        search_upper=upper,
                    )
                    rows.append(
                        dict(
                            **settings,
                            **fit,
                            lambda_relative_error_percent=100 * (energy / 5 - 1),
                            k_infinity_relative_error_percent=100 * np.expm1(log_amplitude),
                            k0_relative_error_percent=100 * np.expm1(log_k0),
                        )
                    )
                    candidates.extend(dict(**settings, **candidate) for candidate in alternatives)
    fits = pd.DataFrame(rows)
    fits[fits.search_upper == 500].to_csv(output / "window_fits.csv", index=False)
    fits.to_csv(output / "window_sensitivity.csv", index=False)
    pd.DataFrame(candidates).to_csv(output / "window_candidates.csv", index=False)

    # Independent quadrature self-fits use the narrower SI S5.1 search interval.
    integral = IntegralCurrent(4096)
    with np.errstate(divide="ignore"):
        grid = np.log(-integral(SEARCH[:, None], eta))
    controls = []
    for endpoint in range(1, 11):
        for weighting in ("linear", "log"):
            mask = eta <= endpoint
            if weighting == "log":
                mask &= eta > 0
            fit, _ = fit_window(
                eta[mask],
                target[mask],
                integral.log_curve_and_derivative,
                weighting,
                SEARCH,
                grid[:, mask],
            )
            controls.append(
                dict(
                    eta_max=endpoint,
                    weighting=weighting,
                    search_lower=SEARCH[0],
                    search_upper=SEARCH[-1],
                    **fit,
                )
            )
    pd.DataFrame(controls).to_csv(output / "window_integral_controls.csv", index=False)
