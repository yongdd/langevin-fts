#!/usr/bin/env python3
"""
Benchmark SCFT runtime for space-group on/off and CPU/CUDA platforms.

Phase parameters are based on examples/scft/phases/*.py.
This script runs a fixed number of SCFT iterations and reports timing.
"""

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import scipy.io

# Ensure consistent CPU threading
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")

from polymerfts import scft


def _set_common_params(params: Dict, platform: str, max_iter: int) -> Dict:
    params = dict(params)
    params["platform"] = platform
    params["max_iter"] = max_iter
    # Force fixed iteration count (avoid early convergence)
    params["tolerance"] = 0.0
    params["verbose_level"] = 0
    return params


def _sphere_init(nx, lx, positions):
    w_A = np.zeros(list(nx), dtype=np.float64)
    w_B = np.zeros(list(nx), dtype=np.float64)
    for x, y, z in positions:
        mx, my, mz = np.round((np.array([x, y, z]) * nx)).astype(np.int32)
        mx = mx % nx[0]
        my = my % nx[1]
        mz = mz % nx[2]
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx) / 15, mode="wrap")
    return {"A": w_A, "B": w_B}


def phase_bcc():
    f = 24 / 90
    params = {
        "nx": [32, 32, 32],
        "lx": [1.9, 1.9, 1.9],
        "reduce_memory": False,
        "box_is_altering": True,
        "stress_interval": 1,
        "chain_model": "continuous",
        "ds": 1 / 90,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 18.1},
        "distinct_polymers": [
            {
                "volume_fraction": 1.0,
                "blocks": [
                    {"type": "A", "length": f},
                    {"type": "B", "length": 1 - f},
                ],
            }
        ],
        "space_group": {"symbol": "Im-3m", "number": 529},
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
    }
    positions = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ]
    w_init = _sphere_init(params["nx"], params["lx"], positions)
    return params, w_init


def phase_dg():
    f = 0.4
    input_data = scipy.io.loadmat(
        os.path.join(os.path.dirname(__file__), "..", "examples", "scft", "phases", "DG.mat"),
        squeeze_me=True,
    )
    params = {
        "nx": [32, 32, 32],
        "lx": input_data["lx"].tolist(),
        "reduce_memory": False,
        "box_is_altering": True,
        "stress_interval": 1,
        "chain_model": "continuous",
        "ds": 1 / 100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15.0},
        "distinct_polymers": [
            {
                "volume_fraction": 1.0,
                "blocks": [
                    {"type": "A", "length": f},
                    {"type": "B", "length": 1 - f},
                ],
            }
        ],
        "space_group": {"symbol": "Ia-3d", "number": 530},
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
    }

    w_A = input_data["w_A"]
    w_B = input_data["w_B"]
    w_A = zoom(
        np.reshape(w_A, input_data["nx"]),
        np.array(params["nx"]) / np.array(input_data["nx"]),
    )
    w_B = zoom(
        np.reshape(w_B, input_data["nx"]),
        np.array(params["nx"]) / np.array(input_data["nx"]),
    )
    w_init = {"A": w_A, "B": w_B}
    return params, w_init


def phase_sigma():
    f = 0.25
    eps = 2.0
    params = {
        "nx": [64, 64, 32],
        "lx": [7.0, 7.0, 4.0],
        "reduce_memory": False,
        "box_is_altering": True,
        "stress_interval": 1,
        "chain_model": "continuous",
        "ds": 1 / 100,
        "segment_lengths": {
            "A": np.sqrt(eps * eps / (eps * eps * f + (1 - f))),
            "B": np.sqrt(1.0 / (eps * eps * f + (1 - f))),
        },
        "chi_n": {"A,B": 25},
        "distinct_polymers": [
            {
                "volume_fraction": 1.0,
                "blocks": [
                    {"type": "B", "length": 1 - f},
                    {"type": "A", "length": f},
                ],
            }
        ],
        "space_group": {"symbol": "P4_2/mnm", "number": 419},
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
    }

    positions = [
        [0.00, 0.00, 0.00], [0.50, 0.50, 0.50],
        [0.40, 0.40, 0.00], [0.60, 0.60, 0.00], [0.90, 0.10, 0.50], [0.10, 0.90, 0.50],
        [0.46, 0.13, 0.00], [0.13, 0.46, 0.00], [0.87, 0.54, 0.00], [0.54, 0.87, 0.00],
        [0.63, 0.04, 0.50], [0.04, 0.63, 0.50], [0.96, 0.37, 0.50], [0.37, 0.96, 0.50],
        [0.74, 0.07, 0.00], [0.07, 0.74, 0.00], [0.93, 0.26, 0.00], [0.26, 0.93, 0.00],
        [0.43, 0.24, 0.50], [0.24, 0.43, 0.50], [0.76, 0.57, 0.50], [0.56, 0.77, 0.50],
        [0.18, 0.18, 0.25], [0.82, 0.82, 0.25], [0.68, 0.32, 0.25], [0.32, 0.68, 0.25],
        [0.18, 0.18, 0.75], [0.82, 0.82, 0.75], [0.68, 0.32, 0.75], [0.32, 0.68, 0.75],
    ]
    w_init = _sphere_init(params["nx"], params["lx"], positions)
    return params, w_init


def phase_hcp_hexagonal():
    f = 0.25
    params = {
        "nx": [48, 48, 48],
        "lx": [1.72, 1.72, 2.8],
        "angles": [90.0, 90.0, 120.0],
        "crystal_system": "Hexagonal",
        "reduce_memory": False,
        "box_is_altering": True,
        "stress_interval": 1,
        "chain_model": "continuous",
        "ds": 1 / 100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 20},
        "distinct_polymers": [
            {
                "volume_fraction": 1.0,
                "blocks": [
                    {"type": "A", "length": f},
                    {"type": "B", "length": 1 - f},
                ],
            }
        ],
        "space_group": {"symbol": "P6_3/mmc", "number": 488},
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
    }
    positions = [
        [1 / 3, 2 / 3, 1 / 4],
        [2 / 3, 1 / 3, 3 / 4],
    ]
    w_init = _sphere_init(params["nx"], params["lx"], positions)
    return params, w_init


PHASES = {
    "BCC": phase_bcc,
    "DG": phase_dg,
    "Sigma": phase_sigma,
    "HCP": phase_hcp_hexagonal,
}


def run_benchmark(phase: str, platform: str, use_space_group: bool, max_iter: int, warmup: int) -> Dict:
    params, w_init = PHASES[phase]()

    if not use_space_group:
        params.pop("space_group", None)

    params = _set_common_params(params, platform, max_iter)

    if warmup > 0:
        warm_params = dict(params)
        warm_params["max_iter"] = warmup
        warm_params["tolerance"] = 0.0
        warm_calc = scft.SCFT(params=warm_params)
        warm_calc.run(initial_fields=w_init)

    calc = scft.SCFT(params=params)
    start = time.perf_counter()
    calc.run(initial_fields=w_init)
    elapsed = time.perf_counter() - start

    q = calc.prop_solver.get_partition_function(0)

    return {
        "phase": phase,
        "platform": platform,
        "space_group": use_space_group,
        "max_iter": max_iter,
        "elapsed_sec": elapsed,
        "iter_sec": elapsed / max_iter,
        "Q": float(q),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=PHASES.keys())
    parser.add_argument("--platform", required=True, choices=["cpu-fftw", "cuda"])
    parser.add_argument("--space-group", required=True, choices=["on", "off"])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = run_benchmark(
        phase=args.phase,
        platform=args.platform,
        use_space_group=args.space_group == "on",
        max_iter=args.iters,
        warmup=args.warmup,
    )

    print(json.dumps(result, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
