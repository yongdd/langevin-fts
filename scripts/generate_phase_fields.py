#!/usr/bin/env python3
"""
Generate converged .mat files for all space group phases.

This script runs SCFT to convergence for phases that don't have .mat files:
- BCC, FCC, SC, A15, Sigma, HCP, PL

Run this script once to generate the .mat files, then use them for fast testing.

Usage:
    python generate_phase_fields.py [phase_name]  # Generate specific phase
    python generate_phase_fields.py               # Generate all phases
"""

import os
import sys
import numpy as np
from scipy.io import savemat
from scipy.ndimage import gaussian_filter

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

from polymerfts import scft

# Output directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def run_and_save(name, params, w_init):
    """Run SCFT to convergence and save results."""
    print(f"\n{'='*60}")
    print(f"Generating: {name}")
    print(f"{'='*60}")

    calc = scft.SCFT(params=params)
    calc.run(initial_fields=w_init)

    output_path = os.path.join(DATA_DIR, f"{name}.mat")
    calc.save_results(output_path)
    print(f"Saved: {output_path}")
    print(f"Free energy: {calc.free_energy:.7f}")
    return calc.free_energy


def generate_bcc():
    """Generate BCC phase (Im-3m)."""
    f = 24/90
    nx = [32, 32, 32]
    lx = [1.9, 1.9, 1.9]

    params = {
        "nx": nx, "lx": lx,
        "box_is_altering": True,
        "chain_model": "continuous",
        "ds": 1/90,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 18.1},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Im-3m", "number": 529},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 2000, "tolerance": 1e-8,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    sphere_positions = [[0, 0, 0], [0.5, 0.5, 0.5]]
    for x, y, z in sphere_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    return run_and_save("BCC", params, {"A": w_A.flatten(), "B": w_B.flatten()})


def generate_fcc():
    """Generate FCC phase (Fm-3m)."""
    f = 24/90
    nx = [32, 32, 32]
    lx = [1.91, 1.91, 1.91]

    params = {
        "nx": nx, "lx": lx,
        "box_is_altering": True,
        "chain_model": "continuous",
        "ds": 1/90,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 18.1},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Fm-3m", "number": 523},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 2000, "tolerance": 1e-8,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    fcc_positions = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    for x, y, z in fcc_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    return run_and_save("FCC", params, {"A": w_A.flatten(), "B": w_B.flatten()})


def generate_sc():
    """Generate SC phase (Pm-3m)."""
    f = 0.2
    nx = [32, 32, 32]
    lx = [1.5, 1.5, 1.5]

    params = {
        "nx": nx, "lx": lx,
        "box_is_altering": True,
        "stress_interval": 1,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 25},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Pm-3m", "number": 517},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 2000, "tolerance": 1e-8,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    sphere_positions = [[0, 0, 0]]
    for x, y, z in sphere_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    return run_and_save("SC", params, {"A": w_A.flatten(), "B": w_B.flatten()})


def generate_a15():
    """Generate A15 phase (Pm-3n)."""
    f = 0.3
    eps = 2.0
    nx = [64, 64, 64]
    lx = [4.0, 4.0, 4.0]

    params = {
        "nx": nx, "lx": lx,
        "box_is_altering": True,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {
            "A": np.sqrt(eps*eps/(eps*eps*f + (1-f))),
            "B": np.sqrt(1.0/(eps*eps*f + (1-f))),
        },
        "chi_n": {"A,B": 25},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "Pm-3n", "number": 520},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 2000, "tolerance": 1e-8,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    sphere_positions = [[0,0,0],[1/2,1/2,1/2],
        [1/4,1/2,0],[3/4,1/2,0],[1/2,0,1/4],[1/2,0,3/4],[0,1/4,1/2],[0,3/4,1/2]]
    for x, y, z in sphere_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    return run_and_save("A15", params, {"A": w_A.flatten(), "B": w_B.flatten()})


def generate_sigma():
    """Generate Sigma phase (P4_2/mnm)."""
    f = 0.25
    eps = 2.0
    nx = [64, 64, 32]
    lx = [7.0, 7.0, 4.0]

    params = {
        "nx": nx, "lx": lx,
        "box_is_altering": True,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {
            "A": np.sqrt(eps*eps/(eps*eps*f + (1-f))),
            "B": np.sqrt(1.0/(eps*eps*f + (1-f))),
        },
        "chi_n": {"A,B": 25},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "B", "length": 1-f},
                {"type": "A", "length": f},
            ],
        }],
        "space_group": {"symbol": "P4_2/mnm", "number": 419},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 2000, "tolerance": 1e-8,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    sphere_positions = [
        [0.00, 0.00, 0.00], [0.50, 0.50, 0.50],
        [0.40, 0.40, 0.00], [0.60, 0.60, 0.00], [0.90, 0.10, 0.50], [0.10, 0.90, 0.50],
        [0.46, 0.13, 0.00], [0.13, 0.46, 0.00], [0.87, 0.54, 0.00], [0.54, 0.87, 0.00],
        [0.63, 0.04, 0.50], [0.04, 0.63, 0.50], [0.96, 0.37, 0.50], [0.37, 0.96, 0.50],
        [0.74, 0.07, 0.00], [0.07, 0.74, 0.00], [0.93, 0.26, 0.00], [0.26, 0.93, 0.00],
        [0.43, 0.24, 0.50], [0.24, 0.43, 0.50], [0.76, 0.57, 0.50], [0.56, 0.77, 0.50],
        [0.18, 0.18, 0.25], [0.82, 0.82, 0.25], [0.68, 0.32, 0.25], [0.32, 0.68, 0.25],
        [0.18, 0.18, 0.75], [0.82, 0.82, 0.75], [0.68, 0.32, 0.75], [0.32, 0.68, 0.75]
    ]
    for x, y, z in sphere_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    return run_and_save("Sigma", params, {"A": w_A.flatten(), "B": w_B.flatten()})


def generate_hcp():
    """Generate HCP phase (P6_3/mmc)."""
    f = 0.25
    nx = [24, 24, 24]
    lx = [1.7186, 1.7186, 2.7982]

    params = {
        "nx": nx, "lx": lx,
        "angles": [90.0, 90.0, 120.0],
        "box_is_altering": True,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 20},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "P6_3/mmc", "number": 488},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 2000, "tolerance": 1e-8,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    sphere_positions = [[1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]
    for x, y, z in sphere_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    return run_and_save("HCP", params, {"A": w_A.flatten(), "B": w_B.flatten()})


def generate_pl():
    """Generate PL phase (P6/mmm)."""
    f = 0.4
    nx = [24, 24, 36]
    lx = [1.958, 1.958, 2.981]

    params = {
        "nx": nx, "lx": lx,
        "angles": [90.0, 90.0, 120.0],
        "box_is_altering": True,
        "chain_model": "continuous",
        "ds": 1/100,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "space_group": {"symbol": "P6/mmm", "number": 485},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 2000, "tolerance": 1e-8,
    }

    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)
    sphere_positions = [
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.5],
        [1/3, 2/3, 0.0], [2/3, 1/3, 0.0],
        [1/3, 2/3, 0.5], [2/3, 1/3, 0.5],
    ]
    for x, y, z in sphere_positions:
        mx, my, mz = np.round(np.array([x, y, z]) * nx).astype(np.int32) % nx
        w_A[mx, my, mz] = -1 / (np.prod(lx) / np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')

    return run_and_save("PL", params, {"A": w_A.flatten(), "B": w_B.flatten()})


GENERATORS = {
    "BCC": generate_bcc,
    "FCC": generate_fcc,
    "SC": generate_sc,
    "A15": generate_a15,
    "Sigma": generate_sigma,
    "HCP": generate_hcp,
    "PL": generate_pl,
}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Generate specific phase
        phase = sys.argv[1]
        if phase in GENERATORS:
            GENERATORS[phase]()
        else:
            print(f"Unknown phase: {phase}")
            print(f"Available: {list(GENERATORS.keys())}")
            sys.exit(1)
    else:
        # Generate all phases
        print("Generating converged fields for all phases")
        print("=" * 60)
        for name, generator in GENERATORS.items():
            generator()
        print("\n" + "=" * 60)
        print("All phases generated!")
        print(f"Output directory: {DATA_DIR}")
