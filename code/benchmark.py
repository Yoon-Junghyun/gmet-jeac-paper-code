"""Reproduce the accuracy and CPU timing comparison in Supplement section S6.

Run python code/benchmark.py with a C11 compiler on PATH (default: gcc).
The accuracy test uses 40,100 grid points and 4,096 held-out conditions.
Times in data/fair_wallclock_summary.csv and data/fair_timing_by_energy.csv
are the paper's measurements on one AMD EPYC 9474F core with GCC 12.2.0.
The 342 batches underlying those tables are in data/benchmark_epyc_batches.csv.
New measurements depend on the processor, compiler and system load.
"""

import argparse
import csv
import ctypes
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from calibration import reference, sample
from mhc_closed_form import normalized_log_rate
from scipy.special import roots_hermite


@dataclass
class Method:
    name: str
    kernel: int
    order: int
    nodes: np.ndarray
    weights: np.ndarray


def compile_kernel(compiler, directory):
    library = Path(directory) / "benchmark.so"
    source = Path(__file__).with_name("review_benchmark.c")
    subprocess.run(
        [
            compiler,
            "-O3",
            "-std=c11",
            "-fPIC",
            "-shared",
            "-fno-math-errno",
            "-ffp-contract=off",
            str(source),
            "-o",
            str(library),
            "-lm",
        ],
        check=True,
    )
    evaluate = ctypes.CDLL(str(library)).evaluate_many
    vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    evaluate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t] + [vector] * 5
    evaluate.restype = None
    return evaluate


def arguments(method, energy, eta, out):
    return (
        method.kernel,
        method.order,
        len(energy),
        energy,
        eta,
        out,
        method.nodes,
        method.weights,
    )


def tune(evaluate, output):
    energy, eta = sample(11, 52100612, low=1, high=100, max_eta=100)
    edges = np.geomspace(1, 100, 101)
    energy = np.r_[energy, np.repeat(edges, 3)]
    eta = np.r_[eta, np.column_stack([np.zeros(101), edges, np.full(101, 100)]).ravel()]
    truth = reference(energy, eta)
    ceiling = np.max(np.abs(np.expm1(normalized_log_rate(energy, eta) - truth)))
    truth = np.exp(truth)

    blank = np.zeros(1)
    selected = {}
    candidates = []
    for order in range(2, 23):
        # Euler acceleration of the Migliore–Nitzan series (Fourmond and Léger, Eq. 13).
        weights = np.array(
            [
                (-1) ** j * sum(math.comb(k, j) / 2 ** (k + 1) for k in range(j, order))
                for j in range(order)
            ]
        )
        candidates.append(Method("Migliore_Nitzan", 2, order, blank, weights))
        candidates.append(
            Method("Migliore_Nitzan_renormalized", 2, order, blank, weights / (1 - 2.0 ** (-order)))
        )
    for order in (16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024):
        nodes, weights = roots_hermite(order)
        candidates.append(Method("Gauss_Hermite", 4, order, nodes, weights / np.sqrt(np.pi)))
    out = np.empty_like(energy)
    with (output / "benchmark_order_selection.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "order", "max_relative_error", "gm_et_error_ceiling"])
        for candidate in candidates:
            evaluate(*arguments(candidate, energy, eta, out))
            error = np.max(np.abs(out / truth - 1))
            writer.writerow([candidate.name, candidate.order, error, ceiling])
            if error <= ceiling:
                selected.setdefault(candidate.name, candidate)
    return [
        Method("Two_term", 0, 0, blank, blank),
        Method("Zeng2014", 1, 0, blank, blank),
        selected["Migliore_Nitzan"],
        selected["Migliore_Nitzan_renormalized"],
        Method("Fourmond_Leger_TS1", 3, 0, blank, blank),
        selected["Gauss_Hermite"],
    ]


def test_set(rng):
    grid_energy, grid_eta = np.meshgrid(
        np.arange(1.0, 101.0), np.linspace(0, 100, 401), indexing="ij"
    )
    test_energy, test_eta = sample(12, 52100613, low=1, high=100, max_eta=100)
    energy = np.r_[grid_energy.ravel(), test_energy]
    eta = np.r_[grid_eta.ravel(), test_eta]
    truth = np.exp(reference(energy, eta))
    order = rng.permutation(len(energy))
    return energy[order], eta[order], truth[order]


def timed_batch(evaluate, args, repeats):
    start = perf_counter()
    for _ in range(repeats):
        evaluate(*args)
    return (perf_counter() - start) / repeats


def benchmark(evaluate, output):
    methods = tune(evaluate, output)
    rng = np.random.default_rng(52100617)
    energy, eta, truth = test_set(rng)
    out = np.empty_like(truth)
    calls = [arguments(method, energy, eta, out) for method in methods]
    repeats, errors = [], []
    for method, args in zip(methods, calls):
        evaluate(*args)
        if not np.all(np.isfinite(out) & (out > 0)):
            raise ArithmeticError(f"Nonpositive or nonfinite rate from {method.name}")
        if method.kernel in (0, 1):
            formula = "mixture" if method.kernel == 0 else "zeng"
            np.testing.assert_allclose(
                out, np.exp(normalized_log_rate(energy, eta, method=formula)), rtol=1e-10, atol=0
            )
        error = out / truth - 1
        errors.append((np.sqrt(np.mean(error**2)), np.max(np.abs(error))))
        evaluate(*args)
        evaluate(*args)
        pilot = timed_batch(evaluate, args, 1)
        repeats.append(max(1, math.ceil(0.1 / pilot)))

    times = np.empty((len(methods), 15))
    with (output / "benchmark_batches.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "method", "repeats", "ns_per_rate"])
        for batch in range(times.shape[1]):
            for i in rng.permutation(len(methods)):
                evaluate(*calls[i])
                times[i, batch] = 1e9 * timed_batch(evaluate, calls[i], repeats[i]) / len(energy)
                writer.writerow([batch + 1, methods[i].name, repeats[i], times[i, batch]])

    with (output / "benchmark_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "order",
                "points",
                "RMS_relative",
                "maximum_relative",
                "median_ns_per_rate",
                "Q1_ns",
                "Q3_ns",
            ]
        )
        for method, error, timing in zip(methods, errors, times):
            writer.writerow(
                [
                    method.name,
                    method.order,
                    len(energy),
                    *error,
                    *np.quantile(timing, [0.5, 0.25, 0.75]),
                ]
            )

    with (output / "benchmark_by_energy.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lambda_tilde", "method", "median_ns_per_rate"])
        for value in (1.0, 10.0, 20.0, 30.0, 50.0, 100.0):
            energy, eta = np.full(401, value), np.linspace(0, 100, 401)
            out = np.empty_like(energy)
            for method in methods:
                args = arguments(method, energy, eta, out)
                batches = [timed_batch(evaluate, args, 60) for _ in range(7)]
                writer.writerow([value, method.name, 1e9 * np.median(batches) / len(energy)])


def main(output, compiler):
    output.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    with TemporaryDirectory() as directory:
        benchmark(compile_kernel(compiler, directory), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="gcc", help="C11 compiler executable")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    main(args.output, args.compiler)
