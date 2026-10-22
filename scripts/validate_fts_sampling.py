#!/usr/bin/env python3
"""One sampling-validation configuration per invocation (submit sweeps as separate jobs)."""
import argparse
import contextlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.io import loadmat
from polymerfts import LFTS, CLFTS


def mean_error(values):
    """Mean, conservative initial-positive-pair ESS and Monte Carlo standard error."""
    x = np.asarray(values, dtype=float)
    if x.size < 4 or not np.all(np.isfinite(x)):
        raise ValueError("At least four finite post-equilibration samples are required")
    centered = x - np.mean(x)
    variance = np.var(x, ddof=1)
    if variance == 0:
        return {"mean": float(np.mean(x)), "mcse": 0.0, "ess": float(x.size), "samples": int(x.size)}
    fft = np.fft.rfft(centered, n=2 * x.size)
    autocorrelation = np.fft.irfft(fft * np.conj(fft))[:x.size]
    autocorrelation /= autocorrelation[0]
    positive_sum = 0.0
    for lag in range(1, x.size - 1, 2):
        pair = autocorrelation[lag] + autocorrelation[lag + 1]
        if pair <= 0:
            break
        positive_sum += pair
    ess = x.size / max(1.0, 1.0 + 2.0 * positive_sum)
    return {"mean": float(np.mean(x)), "mcse": float(np.sqrt(variance / ess)),
            "ess": float(ess), "samples": int(x.size)}


def rpa_aa(nx, length, zeta_n, modes=4):
    """Continuum-chain RPA reference, chi N=6, symmetric AB, b=1."""
    values = []
    for n in range(1, min(modes + 1, nx // 2)):
        x = (2 * np.pi * n / length)**2 / 6
        g_a = 2 * (0.5 * x + np.expm1(-0.5 * x)) / x**2
        g_total = 2 * (x + np.expm1(-x)) / x**2
        g_ab = 0.5 * (g_total - 2 * g_a)
        if zeta_n is None:
            value = 1 / (g_total / (g_a**2 - g_ab**2) - 12.0)
        else:
            g = np.array([[g_a, g_ab], [g_ab, g_a]])
            interaction = np.array([[zeta_n, zeta_n + 6], [zeta_n + 6, zeta_n]])
            value = np.linalg.inv(np.linalg.inv(g) + interaction)[0, 0]
        values.append(value)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("lfts", "clfts"), required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ds", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--burn-in", type=int, default=4000)
    parser.add_argument("--block", type=int, default=1000)
    parser.add_argument("--saddle-tolerance", type=float, default=1e-6)
    parser.add_argument("--failure-policy", choices=("retry", "abort", "resample"), default="retry")
    parser.add_argument("--platform", default="cpu-fftw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.nx < 8 or args.block < 1 or args.burn_in < 0 or args.steps - args.burn_in < 4 * args.block:
        parser.error("Use nx>=8, block>=1, burn-in>=0 and at least four blocks after burn-in")
    first_full_block = (args.burn_in + args.block - 1) // args.block
    if args.steps // args.block - first_full_block < 4:
        parser.error("At least four complete recorded blocks must lie entirely after burn-in")
    if args.model == "lfts" and args.alpha != 0:
        parser.error("--alpha applies to CL-FTS only")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / (output.stem + "_data")
    work.mkdir(exist_ok=True)
    if any(work.iterdir()):
        parser.error(f"Refusing to mix a new run with existing files in {work}")
    params = {
        "nx": [args.nx], "lx": [4.0], "chain_model": "continuous", "ds": args.ds,
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 6.0},
        "distinct_polymers": [{"volume_fraction": 1.0, "blocks": [
            {"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
        "platform": args.platform,
        "langevin": {"nbar": 1e6, "dt": args.dt, "max_step": args.steps},
        "recording": {"dir": str(work), "recording_period": args.steps,
                      "sf_computing_period": 1, "sf_recording_period": args.block},
        "verbose_level": 0,
    }
    if args.model == "lfts":
        params.update(saddle={"max_iter": 200, "tolerance": args.saddle_tolerance,
                              "failure_policy": args.failure_policy},
                      compressor={"name": "lram", "max_hist": 20, "start_error": 0.5,
                                  "mix_min": 0.01, "mix_init": 0.01})
    else:
        params.update(zeta_n=100.0, dynamic_stabilization=args.alpha)
    previous = Path.cwd()
    try:
        os.chdir(work)
        with (work / "run.log").open("w") as log, contextlib.redirect_stdout(log):
            sim = (LFTS if args.model == "lfts" else CLFTS)(params, random_seed=args.seed)
            status = sim.run({"A": np.zeros(args.nx), "B": np.zeros(args.nx)})
    finally:
        os.chdir(previous)
    h_real, h_imag, drift, sf_real, sf_imag = [], [], [], [], []
    for end in range(args.block, args.steps + 1, args.block):
        data = loadmat(work / f"dH_{end:06d}.mat", squeeze_me=True)
        start = end - args.block
        keep = max(0, args.burn_in - start)
        if args.model == "lfts":
            h_real.extend(np.atleast_1d(data["H_history"])[keep:])
        else:
            h_real.extend(np.atleast_1d(data["H_history_real"])[keep:])
            h_imag.extend(np.atleast_1d(data["H_history_imag"])[keep:])
        drift.extend(np.atleast_1d(data["drift_max_history"])[keep:])
        if start >= args.burn_in:
            sf = loadmat(work / f"structure_function_{end:06d}.mat", squeeze_me=True)
            suffix = "" if args.model == "lfts" else "_real"
            sf_real.append(np.ravel(sf["structure_function_A_A" + suffix])[1:5])
            if args.model == "clfts":
                sf_imag.append(np.ravel(sf["structure_function_A_A_imag"])[1:5])
    reference = rpa_aa(args.nx, 4.0, None if args.model == "lfts" else 100.0)
    summary = {"configuration": vars(args) | {"output": str(output)}, "run_status": status,
               "metadata": sim.metadata, "hamiltonian_real": mean_error(h_real),
               "sampled_langevin_time": len(h_real) * args.dt,
               "structure_factor_A_A": [mean_error(np.asarray(sf_real)[:, i]) for i in range(len(reference))],
               "rpa_A_A": reference,
               "drift_max_quantiles": dict(zip(("q50", "q90", "q99", "q999", "max"),
                                               np.quantile(drift, [0.5, 0.9, 0.99, 0.999, 1]).tolist()))}
    if h_imag:
        summary["hamiltonian_imag"] = mean_error(h_imag)
        summary["structure_factor_A_A_imag"] = [mean_error(np.asarray(sf_imag)[:, i]) for i in range(len(reference))]
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
