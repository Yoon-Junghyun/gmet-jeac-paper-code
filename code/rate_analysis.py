"""Calculate the rates, currents and errors in Figs. 1, 2, S1 and S2.

Run python code/rate_analysis.py --output results. Rates are normalized by
k_infinity, energies by k_B*T and currents in Fig. S2 by A*F*k_infinity*c_T.
Errors in the CSV files are signed percentages unless marked as absolute.

The two supplied NPZ files contain adaptive-quadrature values on the primary
and extended grids. --recompute-grids repeats those integrations from scratch.
All rate slices, currents, derivatives and local searches use fresh quadrature.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from mhc_closed_form import (
    COEFFICIENTS,
    FARADAY,
    current_density,
    log_rate_derivatives,
    normalized_log_rate,
)
from mhc_medvedev import k_A, k_Aint, k_SA, piecewise_rate
from mhc_reference import log_rate
from scipy.integrate import quad
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr

DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = {"GM_ET": "mixture", "Zeng": "zeng"}


def reference_log_rate(lam, eta):
    """Log of the full-integral oxidation rate divided by k_infinity."""
    return log_rate(lam, eta) - np.log(2 * np.sqrt(np.pi * np.asarray(lam)))


def equilibrium_log_rate(lam):
    """Factored integral J(lambda) in Supplement S3.1."""

    def integrand(x):
        return np.exp(-x * x / (4 * lam) - x / 2) * expit(x)

    integral = 2 * quad(integrand, 0, np.inf, epsabs=3e-14, epsrel=3e-13)[0]
    return -lam / 4 + np.log(integral / (2 * np.sqrt(np.pi * lam)))


def exact_derivatives(lam, eta):
    """d ln(k_ox)/d eta and d ln(k_ox)/d lambda at one point (S6.1)."""
    root = np.sqrt(lam)
    transition = np.clip((eta - lam) / (2 * root), -12, 12)
    breaks = np.unique([-12.0, 0.0, transition, 12.0])

    def integral(kernel):
        def integrand(t):
            u = eta - lam - 2 * root * t
            return np.exp(-t * t) * expit(u) * kernel(t, u)

        return sum(
            quad(integrand, a, b, epsabs=1e-32, epsrel=2e-11)[0]
            for a, b in itertools.pairwise(breaks)
        )

    rate = integral(lambda t, u: 1.0)
    d_eta = integral(lambda t, u: expit(-u))
    d_lam = integral(lambda t, u: -expit(-u) * (1 + t / root))
    return d_eta / rate, d_lam / rate


def zeng_derivatives(lam, eta):
    """Log derivatives of Zeng et al. Eq. (17), ordered as in exact_derivatives."""
    b = np.sqrt(eta * eta + 1 + np.sqrt(lam))
    scale = np.sqrt(2 * lam)
    z = (b - lam) / scale
    mills_ratio = np.exp(-z * z / 2 - np.log(2 * np.pi) / 2 - log_ndtr(z))
    d_eta = mills_ratio * eta / (b * scale) + expit(-eta)
    d_lam = mills_ratio * ((1 / (4 * np.sqrt(lam) * b) - 1) / scale - z / (2 * lam))
    return d_eta, d_lam


def rate_frame(lam, eta, reference):
    frame = pd.DataFrame(
        {"lambda_tilde": lam, "eta_f_tilde": eta, "integral_rate": np.exp(reference)}
    )
    for name, method in MODELS.items():
        estimate = normalized_log_rate(lam, eta, method=method)
        frame[name + "_rate"] = np.exp(estimate)
        frame[name + "_error_percent"] = 100 * np.expm1(estimate - reference)
    return frame


def rate_slices(output):
    frames, derivatives = [], []
    for lam in (5.0, 10.0, 50.0):
        # Fig. 1 mirrors the oxidation rate at eta > 0 to show reduction at -eta.
        eta = np.linspace(0, 2 * lam, 1001)
        frames.append(rate_frame(lam, eta, reference_log_rate(lam, eta)))
        for h in np.linspace(0, 10, 101):
            ref_h, ref_l = exact_derivatives(lam, h)
            gm_l, gm_h = log_rate_derivatives(lam, h)[1]
            zeng_h, zeng_l = zeng_derivatives(lam, h)
            derivatives.append(
                (lam, h, ref_h, ref_l, gm_h - ref_h, gm_l - ref_l, zeng_h - ref_h, zeng_l - ref_l)
            )
    pd.concat(frames).to_csv(output / "fig1_rates.csv", index=False)
    pd.DataFrame(
        derivatives,
        columns=[
            "lambda_tilde",
            "eta_f_tilde",
            "integral_dlnk_deta",
            "integral_dlnk_dlambda",
            "GM_ET_deta_error",
            "GM_ET_dlambda_error",
            "Zeng_deta_error",
            "Zeng_dlambda_error",
        ],
    ).to_csv(output / "s6_derivatives.csv", index=False)


def refined_maximum(lams, etas, errors, method):
    """Refine local maxima above one tenth of the largest grid error (S3)."""
    magnitude = np.abs(errors)
    seeds = np.argwhere(
        (magnitude == maximum_filter(magnitude, size=3, mode="nearest"))
        & (magnitude > 0.1 * magnitude.max())
    )
    i, j = np.unravel_index(magnitude.argmax(), magnitude.shape)
    best = (magnitude[i, j], lams[i], etas[j])

    def objective(coords):
        lam, eta = np.exp(coords[0]), coords[1]
        delta = normalized_log_rate(lam, eta, method=method) - reference_log_rate(lam, eta)
        return -abs(np.expm1(delta))

    for i, j in seeds:
        bounds = [
            (np.log(lams[max(i - 1, 0)]), np.log(lams[min(i + 1, len(lams) - 1)])),
            (etas[max(j - 1, 0)], etas[min(j + 1, len(etas) - 1)]),
        ]
        fit = minimize(
            objective,
            [np.log(lams[i]), etas[j]],
            method="Nelder-Mead",
            bounds=bounds,
            options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 400},
        )
        if -fit.fun > best[0]:
            best = (-fit.fun, np.exp(fit.x[0]), fit.x[1])
    return best


def error_grids(output, recompute):
    summaries = []
    grids = [
        (
            "primary",
            np.geomspace(0.1, 50, 151),
            np.linspace(0, 100, 1001),
            "wide_grid.npz",
            "eta_f_abs",
            "log_rate_integral",
        ),
        (
            "extended",
            np.arange(1.0, 101.0),
            np.linspace(0, 100, 401),
            "review_map.npz",
            "eta_f_tilde",
            "log_rate_exact",
        ),
    ]
    for domain, lams, etas, filename, eta_key, log_key in grids:
        lam, eta = (v.ravel() for v in np.meshgrid(lams, etas, indexing="ij"))
        if recompute:
            reference = reference_log_rate(lam, eta)
        else:
            with np.load(DATA / filename) as values:
                np.testing.assert_allclose(values["lambda_tilde"], lam, rtol=5e-15, atol=0)
                np.testing.assert_allclose(values[eta_key], eta, rtol=5e-15, atol=0)
                lam, eta, reference = values["lambda_tilde"], values[eta_key], values[log_key]
                lams, etas = np.unique(lam), np.unique(eta)
        frame = rate_frame(lam, eta, reference)
        if domain == "extended":
            frame.to_csv(output / "fig2_error_map.csv", index=False)
        for name, method in MODELS.items():
            error = (frame[name + "_error_percent"].to_numpy() / 100).reshape(len(lams), -1)
            maximum, energy, potential = refined_maximum(lams, etas, error, method)
            summaries.append(
                (domain, name, 100 * np.max(abs(error)), 100 * maximum, energy, potential)
            )
        if domain == "primary":
            # S4: at fixed k^0 remove each row's formal-potential rate error.
            delta = (normalized_log_rate(lam, eta) - reference).reshape(len(lams), -1)
            fixed_error = 100 * np.expm1(delta - delta[:, :1])
            i, j = np.unravel_index(abs(fixed_error).argmax(), fixed_error.shape)
            pd.DataFrame(
                [
                    {
                        "lambda_tilde": lams[i],
                        "eta_f_tilde": etas[j],
                        "GM_ET_error_percent_fixed_k0": fixed_error[i, j],
                    }
                ]
            ).to_csv(output / "s4_fixed_standard_rate.csv", index=False)
    pd.DataFrame(
        summaries,
        columns=[
            "domain",
            "model",
            "grid_max_abs_error_percent",
            "refined_max_abs_error_percent",
            "lambda_tilde",
            "eta_f_tilde",
        ],
    ).to_csv(output / "s3_peak_errors.csv", index=False)

    # Include exactly the energies sampled for panel 2(a).
    lams = np.unique(np.r_[np.linspace(0.1, 100, 1000), [1, 12, 20, 30, 40, 50, 60, 100]])
    lams = lams[lams >= 1.0]
    ref = np.array([equilibrium_log_rate(lam) for lam in lams])
    rate_frame(lams, np.zeros_like(lams), ref).to_csv(output / "fig2_equilibrium.csv", index=False)
    weights, variances = map(np.asarray, COEFFICIENTS[2])
    ratio = (2 / np.pi) * np.dot(weights, np.exp(variances / 8))
    pd.DataFrame(
        [{"asymptotic_rate_ratio": ratio, "relative_error_percent": 100 * (ratio - 1)}]
    ).to_csv(output / "s3_large_lambda.csv", index=False)


def medvedev_comparison(output):
    switches, slices = [], []
    for lam in (4.0, 10.0, 20.0, 40.0):
        mu = np.linspace(0, 3, 601)
        # Set exact switch coordinates before applying the inclusion convention.
        for boundary, left, right, side in [
            (0.9, k_A, k_Aint, "left"),
            (1.1, k_Aint, k_SA, "right"),
            (2.9, k_SA, k_Aint, "left"),
        ]:
            mu[np.argmin(abs(mu - boundary))] = boundary
            ref = np.exp(reference_log_rate(lam, boundary * lam))
            left_rate, right_rate = left(lam, boundary), right(lam, boundary)
            switches.append(
                (
                    lam,
                    boundary,
                    side,
                    left_rate,
                    right_rate,
                    100 * (left_rate / ref - 1),
                    100 * (right_rate / ref - 1),
                    100 * (right_rate / left_rate - 1),
                )
            )
        eta = lam * mu
        frame = rate_frame(lam, eta, reference_log_rate(lam, eta))
        frame["mu"] = mu
        for name, model in [("Medvedev_piecewise", piecewise_rate), ("Medvedev_B10", k_Aint)]:
            frame[name + "_rate"] = model(lam, mu)
            frame[name + "_error_percent"] = 100 * (frame[name + "_rate"] / frame.integral_rate - 1)
        slices.append(frame)
    pd.concat(slices).to_csv(output / "figS1_rates.csv", index=False)
    pd.DataFrame(
        switches,
        columns=[
            "lambda_tilde",
            "mu",
            "assigned_side",
            "left_rate",
            "right_rate",
            "left_error_percent",
            "right_error_percent",
            "jump_percent",
        ],
    ).to_csv(output / "figS1_switches.csv", index=False)

    lams = np.array([4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0])
    frame = rate_frame(lams, np.zeros_like(lams), reference_log_rate(lams, 0.0))
    for name, model in [("Medvedev_piecewise", piecewise_rate), ("Medvedev_B10", k_Aint)]:
        frame[name + "_error_percent"] = 100 * (model(lams, 0.0) / frame.integral_rate - 1)
    frame.to_csv(output / "tableS4_standard_rates.csv", index=False)


def concentration_currents(output):
    """Fig. S2 uses unit total concentration, with ratios 1 and 100."""
    curves = []
    for lam in (1.0, 50.0):
        eta_max = max(10, 2 * lam)
        h_positive = np.unique(np.r_[np.linspace(0, eta_max, 1001), lam])
        for ratio in (1.0, 100.0):
            c_ox, c_red = ratio / (1 + ratio), 1 / (1 + ratio)
            # Include the Nernst equilibrium itself at each concentration ratio.
            equilibrium = np.log(ratio)
            h = np.unique(np.r_[-h_positive, h_positive, equilibrium])
            eta = h - equilibrium
            favored = np.exp(reference_log_rate(lam, abs(h)))
            k_sum = favored * (1 + np.exp(-abs(h)))
            # expm1 preserves the small difference between the directional currents.
            bracket = np.where(
                eta >= 0,
                c_red * expit(h) * np.expm1(-np.maximum(eta, 0)),
                -c_ox * expit(-h) * np.expm1(np.minimum(eta, 0)),
            )
            curve = pd.DataFrame(
                {
                    "lambda_tilde": lam,
                    "c_O_over_c_R": ratio,
                    "eta_f_tilde": h,
                    "eta_tilde": eta,
                    "integral_I_over_Istar": k_sum * bracket,
                }
            )
            for name, method in MODELS.items():
                scaled_current = current_density(lam, eta, c_ox, c_red, method=method) / FARADAY
                curve[name + "_I_over_Istar"] = scaled_current
                curve[name + "_residual_percent"] = 100 * (scaled_current - k_sum * bracket)
            curves.append(curve)
    pd.concat(curves).to_csv(output / "figS2_currents.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--recompute-grids", action="store_true", help="repeat all 191,251 reference-grid integrals"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rate_slices(args.output)
    error_grids(args.output, args.recompute_grids)
    medvedev_comparison(args.output)
    concentration_currents(args.output)


if __name__ == "__main__":
    main()
