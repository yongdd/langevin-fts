#!/usr/bin/env python3
"""
Benchmark CrysFFT speedup for propagator computation (triplet-heavy path).

This mimics the 2020 Qiang & Li sample by timing only propagator computation
for a single AB diblock, using a BCC initial field when available.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from polymerfts import PropagatorSolver, SpaceGroup


def load_bcc_fields(path: Path, nx):
    data = np.loadtxt(path, dtype=np.float64)
    size = int(np.prod(nx))
    if data.shape[0] != size or data.shape[1] < 4:
        raise ValueError(f"Unexpected field file shape: {data.shape}")
    w_a = data[:, 2].copy()
    w_b = data[:, 3].copy()
    return w_a, w_b


def sphere_init(nx, lx):
    w_a = np.zeros(int(np.prod(nx)), dtype=np.float64)
    w_b = np.zeros_like(w_a)
    positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    for x, y, z in positions:
        mx, my, mz = np.round((np.array([x, y, z]) * nx)).astype(np.int32)
        mx = mx % nx[0]
        my = my % nx[1]
        mz = mz % nx[2]
        idx = (mx * nx[1] + my) * nx[2] + mz
        w_a[idx] = -1.0 / (np.prod(lx) / np.prod(nx))
    return w_a, w_b


def run_case(use_space_group, nx, lx, ds, f_a, iters, warmup, fields_path, sg, w_full_sym, w_red):

    solver = PropagatorSolver(
        nx=nx,
        lx=lx,
        ds=ds,
        bond_lengths={"A": 1.0, "B": 1.0},
        bc=["periodic"] * 6,
        chain_model="continuous",
        numerical_method="rqm4",
        platform="cpu-fftw",
        space_group=sg if use_space_group else None,
    )
    solver.add_polymer(1.0, [["A", f_a, 0, 1], ["B", 1.0 - f_a, 1, 2]])

    if use_space_group:
        w_a = w_red["A"]
        w_b = w_red["B"]
    else:
        w_a = w_full_sym["A"]
        w_b = w_full_sym["B"]

    w_fields = {"A": w_a, "B": w_b}

    for _ in range(warmup):
        solver.compute_propagators(w_fields)

    t0 = time.perf_counter()
    for _ in range(iters):
        solver.compute_propagators(w_fields)
    elapsed = time.perf_counter() - t0

    q = solver.get_partition_function(0)
    return {
        "space_group": use_space_group,
        "max_iter": iters,
        "elapsed_sec": elapsed,
        "iter_sec": elapsed / max(iters, 1),
        "Q": float(q),
        "n_grid": solver.n_grid,
        "total_grid": solver.total_grid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--lx", type=float, nargs=3, default=[1.9, 1.9, 1.9])
    parser.add_argument("--ds", type=float, default=1.0 / 90.0)
    parser.add_argument("--fa", type=float, default=24.0 / 90.0)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--fields", type=str, default=None)
    parser.add_argument("--space-group-basis", type=str, default="irreducible",
                        choices=["irreducible", "pmmm-physical"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    fields_path = Path(args.fields) if args.fields else Path(
        "/home/yongdd/polymer/langevin-fts/references/crysfft/BCC_fields_test.txt"
    )

    sg = SpaceGroup(args.nx, "Im-3m", 529)
    if args.space_group_basis == "pmmm-physical":
        sg.enable_pmmm_physical_basis()
    if fields_path and fields_path.exists():
        w_a_full, w_b_full = load_bcc_fields(fields_path, args.nx)
    else:
        w_a_full, w_b_full = sphere_init(args.nx, args.lx)

    # Symmetrize full-grid fields to match reduced-basis representation.
    w_a_red = sg.to_reduced_basis(w_a_full.reshape(1, -1))[0]
    w_b_red = sg.to_reduced_basis(w_b_full.reshape(1, -1))[0]
    w_a_sym = sg.from_reduced_basis(w_a_red.reshape(1, -1))[0]
    w_b_sym = sg.from_reduced_basis(w_b_red.reshape(1, -1))[0]

    w_full_sym = {"A": w_a_sym, "B": w_b_sym}
    w_red = {"A": w_a_red, "B": w_b_red}

    on = run_case(True, args.nx, args.lx, args.ds, args.fa, args.iters, args.warmup, fields_path, sg, w_full_sym, w_red)
    off = run_case(False, args.nx, args.lx, args.ds, args.fa, args.iters, args.warmup, fields_path, sg, w_full_sym, w_red)

    speedup = off["iter_sec"] / on["iter_sec"] if on["iter_sec"] > 0 else float("inf")
    result = {"sg_on": on, "sg_off": off, "speedup": speedup, "percent": (speedup - 1.0) * 100.0}

    print(json.dumps(result, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
