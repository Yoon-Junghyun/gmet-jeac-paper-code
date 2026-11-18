"""Numerical results for "A Gaussian-mixture approximation for Marcus kinetics
of electron transfer at metal electrodes" by Junghyun Yoon and Martin Z. Bazant.

From the repository directory, use Python 3.13 and a C11 compiler:

    python -m pip install -r requirements.txt
    python reproduce.py check
    python reproduce.py all --output results

The 'all' command writes numerical tables for Figures 1–4, Figures S1–S5
and the supplementary tables. Add --fresh to regenerate the quadrature grids
and the shared noisy observations before fitting. In data/, benchmark_epyc_batches.csv
contains the paper's single-core AMD EPYC measurements. fair_wallclock_summary.csv
and fair_timing_by_energy.csv summarize these measurements. The benchmark writes
new timings for your machine. The ts1 comparator adapts QSoas. Its source
notice and license are in code/review_benchmark.c and code/COPYING.
The calculation commands below accept --help.

    code/calibration.py        SI S2, mixture calibration and held-out errors
    code/rate_analysis.py      Figs. 1, 2, S1, S2; SI S3, S4 and S6 derivatives
    code/fitting.py noiseless  Figs. 3 and S3; SI S5 sensitivity angles
    code/fitting.py windows    Figs. 4 and S4; competing minima and range checks
    code/fitting.py precision  SI S5.4, 55-digit check of the endpoint-4 fit
    code/fitting.py noise      Fig. S5 and Table S5, paired noisy fits
    code/benchmark.py          Table S6, single-core cost and accuracy
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script, *options):
    print("Running", script, *map(str, options), flush=True)
    subprocess.run([sys.executable, ROOT / "code" / script, *options], check=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=("check", "all"))
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="recompute the archived numerical inputs before fitting",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results",
        help="directory for recalculated CSV and NPZ files",
    )
    parser.add_argument("--compiler", default="gcc", help="C11 compiler for the benchmark")
    args = parser.parse_args()
    if args.stage == "check" and args.fresh:
        parser.error("--fresh applies to the full calculation")
    output = args.output.resolve()
    if (
        output == ROOT
        or output.is_relative_to(ROOT / "data")
        or output.is_relative_to(ROOT / "code")
    ):
        parser.error("choose a results directory separate from the release sources and data")

    run("check_rates.py")
    if args.stage == "check":
        return
    run("calibration.py", "--output", output)
    grid_options = ["--recompute-grids"] if args.fresh else []
    run("rate_analysis.py", "--output", output, *grid_options)
    inputs = ROOT / "data"
    if args.fresh:
        inputs = output / "reference"
        run("fitting.py", "reference", "--output", inputs)
    for stage in ("noiseless", "windows", "precision", "noise"):
        run("fitting.py", stage, "--inputs", inputs, "--output", output)
    run("benchmark.py", "--compiler", args.compiler, "--output", output)
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
