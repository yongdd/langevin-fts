#!/usr/bin/env python3
r"""Distribution of the largest CL-FTS drift, for the criterion of correct convergence.

    python scripts/analysis/drift_tail.py data_simulation
    python scripts/analysis/drift_tail.py data_simulation --skip 20000 --plot drift_tail.png

Reads `drift_max_step_history` (every CL-FTS run; one column per auxiliary field) and
`psi_drift_max_step_history` (with electrostatics) of the `dH_*.mat` files in a recording
directory. They hold the largest drift of every Langevin step over the grid. For each field
the script prints the complementary distribution S(u) = P(drift > u) and its log-log slope per
interval of u.

Nagata, Nishimura and Shimasaki, Phys. Rev. D 94, 114515 (2016), require the distribution of
the drift magnitude to fall off exponentially or faster. An exponential tail steepens without
bound on a log-log plot. A slope that stays constant over the largest decades is a power-law
tail and the averages cannot be trusted: near a simple zero of a partition function the drift
grows as the inverse distance and S(u) falls off as u^-2. The weight of such a tail sets the
size of the bias, so compare with an independent reference where one exists. See
docs/theory/ComplexLangevinSimulation.md (Section 3).
"""

import argparse
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat

EDGES = [1, 2, 3.16, 10, 31.6, 100, 316, 1000, 3162, 10000]


def read(directory, skip):
    """Step values of each field, ordered by step; steps <= skip are left out."""
    files = sorted(Path(directory).glob("dH_*.mat"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if not files:
        raise SystemExit(f"No dH_*.mat files in {directory}")
    values = {}
    for path in files:
        data = loadmat(path)   # without squeeze_me: (steps, fields) and (1, steps)
        last = int(re.findall(r"\d+", path.stem)[-1])
        columns = {}
        if "drift_max_step_history" in data:
            record = np.asarray(data["drift_max_step_history"], dtype=float)
            columns.update({f"auxiliary field {i}": record[:, i] for i in range(record.shape[1])})
        if "psi_drift_max_step_history" in data:
            columns["psi"] = np.asarray(data["psi_drift_max_step_history"], dtype=float).ravel()
        for name, record in columns.items():
            steps = np.arange(last - record.size + 1, last + 1)
            values.setdefault(name, []).append(record[steps > skip])
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
    records = read(args.directory, args.skip)
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
