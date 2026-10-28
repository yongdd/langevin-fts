#!/usr/bin/env python3
r"""Distribution of the largest CL-FTS drift, for the criterion of correct convergence.

    python scripts/analysis/drift_tail.py data_simulation
    python scripts/analysis/drift_tail.py data_simulation --skip 20000 --plot drift_tail.png

Reads `drift_max_step_history` (every CL-FTS run; one column per auxiliary field) and
`psi_drift_max_step_history` (with electrostatics) of the `dH_*.mat` files in a recording
directory. They hold the largest drift of every Langevin step over the grid. For each field
the script prints the complementary distribution S(u) = P(drift > u) and its log-log slope per
interval of u. With canonical single-site species (discrete chains of one segment, point ions)
it also summarizes their partition ratios |Q_s|/<|e^(-u_s)|> (`partition_ratio_step_history`):
the smallest value and the fractions of steps below 0.1 and 0.01.

Nagata, Nishimura and Shimasaki, Phys. Rev. D 94, 114515 (2016), require the distribution of
the drift magnitude to fall off exponentially or faster. An exponential tail steepens without
bound on a log-log plot; a slope that stays constant over the largest decades is a power-law
tail, and the averages cannot be trusted. Near a simple zero of a partition function the drift
grows as the inverse distance to it. If the samples reach the zero with a finite density, S(u)
falls off as u^-2 far out; a density that vanishes at the zero gives a steeper power law. In
toy models with such a zero, run in their wrong regime with 10^7 steps or more, the slopes
between 3 and 100 times the median were about -2 to -3 (the one-pole models of Aarts, Seiler,
Sexty and Stamatescu, JHEP 05 (2017) 044, and of Nagata et al.; tests/scripts/TestDriftTail.py),
approaching -2 only in the last decade. Runs close to the boundary of the correct regime left
about 10^-4 of the steps beyond 10 times the median and too few beyond that for a slope: such
borderline cases cannot be classified with 10^7 steps. The weight of a power-law tail sets the
size of the bias, so compare with an independent reference where one exists. See
docs/theory/ComplexLangevinSimulation.md (Section 3).
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat

EDGES = [1, 2, 3.16, 10, 31.6, 100, 316, 1000, 3162, 10000]


def read(directory, skip, ratios=None):
    """Step values of each field, ordered by step; steps <= skip are left out. A dictionary given as
    ratios receives the partition ratios of each single-site species in the same way."""
    files = sorted(Path(directory).glob("dH_*.mat"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if not files:
        raise SystemExit(f"No dH_*.mat files in {directory}")
    values = {}
    for path in files:
        data = loadmat(path)   # without squeeze_me: (steps, fields) and (1, steps)
        last = int(re.findall(r"\d+", path.stem)[-1])
        columns, ratio_columns = {}, {}
        if "drift_max_step_history" in data:
            record = np.asarray(data["drift_max_step_history"], dtype=float)
            columns.update({f"auxiliary field {i}": record[:, i] for i in range(record.shape[1])})
        if "psi_drift_max_step_history" in data:
            columns["psi"] = np.asarray(data["psi_drift_max_step_history"], dtype=float).ravel()
        if "partition_ratio_step_history" in data:
            record = np.asarray(data["partition_ratio_step_history"], dtype=float)
            species = json.loads(str(data["partition_ratio_species_json"][0]))
            ratio_columns.update({name: record[:, i] for i, name in enumerate(species)})
        for target, found in ((values, columns), (ratios, ratio_columns)):
            if target is None:
                continue
            for name, record in found.items():
                steps = np.arange(last - record.size + 1, last + 1)
                target.setdefault(name, []).append(record[steps > skip])
    if ratios is not None:
        ratios.update({name: np.concatenate(parts) for name, parts in ratios.items()})
    return {name: np.concatenate(parts) for name, parts in values.items()}


def tail(values):
    """Median, S(u) at multiples of the median and log-log slopes of S per half decade of u/median."""
    u = np.sort(values)
    n, median = u.size, float(np.median(u))

    def survival(x):
        return (n - np.searchsorted(u, x, side="right")) / n

    slopes = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        sa, sb = survival(a * median), survival(b * median)
        slopes.append(np.log(sb / sa) / np.log(b / a) if sb * n >= 20 else None)
    return n, median, survival, slopes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", help="recording.dir of a CL-FTS run")
    parser.add_argument("--skip", type=int, default=0, help="leave out the steps up to this one (equilibration)")
    parser.add_argument("--plot", help="also plot S(u) into this image file")
    args = parser.parse_args()
    ratios = {}
    records = read(args.directory, args.skip, ratios)
    results = {}
    for name, values in records.items():
        if values.size == 0:
            continue
        if not np.any(values > 0):
            print(f"{name}: no drift (zero mobility)")
            continue
        n, median, survival, slopes = tail(values)
        results[name] = (values, median, survival)
        print(f"{name}: {n} steps, median {median:.4g}, maximum {values.max():.4g}")
        print("  S(u) at u = 10, 100, 1000 x median: " +
              ", ".join(f"{survival(k * median):.3g}" for k in (10, 100, 1000)))
        print("  log-log slope of S(u) per interval of u/median:")
        for a, b, slope in zip(EDGES[:-1], EDGES[1:], slopes):
            print(f"    {a:>6g} - {b:<6g} " + ("(fewer than 20 steps beyond)" if slope is None else f"{slope:7.2f}"))
    # Partition ratios |Q_s|/<|e^(-u_s)|>; NaN marks steps of a checkpoint without the record
    for name, values in ratios.items():
        values = values[~np.isnan(values)]
        if values.size:
            print(f"partition ratio of {name}: {values.size} steps, minimum {values.min():.3g}, "
                  f"steps below 0.1: {np.mean(values < 0.1):.3g}, below 0.01: {np.mean(values < 0.01):.3g}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for name, (values, median, survival) in results.items():
            u = np.logspace(np.log10(max(values.min(), 1e-300)), np.log10(values.max()), 200)
            s = survival(u)
            ax.loglog(u[s > 0] / median, s[s > 0], label=name)
        ax.set(xlabel="drift / median", ylabel="P(drift > u)", title="largest drift of every step")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)


if __name__ == "__main__":
    main()
