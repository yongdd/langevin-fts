"""
Free energy convergence benchmark for SDC method - 3D Gyroid phase.

Tests SDC-4 method and compares with RQM4, ETDRK4, CN-ADI2, CN-ADI4.
Usage: python benchmark_sdc_gyroid_3d.py --method <method> --ds <ds>
"""

import os
import sys
import argparse
import json
import time
import numpy as np

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

from polymerfts.scft import SCFT

def run_scft_gyroid(method, ds, chi_n=18.0, f=0.375, nx=32, lx=3.65, platform="cuda"):
    """Run SCFT for AB diblock in gyroid phase (3D)."""

    params = {
        "nx": [nx, nx, nx],
        "lx": [lx, lx, lx],
        "chain_model": "continuous",
        "ds": ds,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": chi_n},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f, "v": 0, "u": 1},
                {"type": "B", "length": 1.0-f, "v": 1, "u": 2},
            ]
        }],
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
        "max_iter": 3000,
        "tolerance": 1e-9,
        "platform": platform,
        "numerical_method": method,
        "verbose": False,
        "box_is_altering": False,
    }

    # Initialize SCFT
    scft = SCFT(params)

    # Set initial fields (gyroid approximation using level-set function)
    x = np.linspace(0, lx, nx, endpoint=False)
    y = np.linspace(0, lx, nx, endpoint=False)
    z = np.linspace(0, lx, nx, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Gyroid level-set function
    k = 2 * np.pi / lx
    gyroid = np.sin(k*X) * np.cos(k*Y) + np.sin(k*Y) * np.cos(k*Z) + np.sin(k*Z) * np.cos(k*X)

    # Initial fields
    w_A = chi_n * 0.3 * gyroid
    w_B = -w_A

    # Run SCFT with timing
    start_time = time.perf_counter()
    scft.run(initial_fields={"A": w_A, "B": w_B})
    elapsed_time = time.perf_counter() - start_time

    # Get results from attributes
    result = {
        "method": method,
        "ds": ds,
        "N": int(round(1/ds)),
        "chi_n": chi_n,
        "f": f,
        "nx": nx,
        "lx": lx,
        "dim": 3,
        "phase": "gyroid",
        "platform": platform,
        "free_energy": float(scft.free_energy),
        "error_level": float(scft.error_level),
        "iterations": int(scft.iter),
        "converged": bool(scft.error_level < 1e-9),
        "elapsed_time": elapsed_time,
    }

    return result

def main():
    parser = argparse.ArgumentParser(description="SDC free energy convergence benchmark (3D Gyroid)")
    parser.add_argument("--method", type=str, required=True,
                        help="Numerical method (rqm4, etdrk4, cn-adi2, cn-adi4, sdc-4, etc.)")
    parser.add_argument("--ds", type=float, required=True,
                        help="Contour step size")
    parser.add_argument("--platform", type=str, default="cuda",
                        help="Platform (cuda or cpu-fftw)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    args = parser.parse_args()

    print(f"Running SCFT (3D Gyroid): method={args.method}, ds={args.ds}, platform={args.platform}")

    result = run_scft_gyroid(args.method, args.ds, platform=args.platform)

    print(f"  Free energy: {result['free_energy']:.12f}")
    print(f"  Error level: {result['error_level']:.2e}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Converged: {result['converged']}")
    print(f"  Time: {result['elapsed_time']:.2f} s")

    # Save result
    if args.output:
        output_file = args.output
    else:
        output_file = f"result_{args.method}_N{result['N']}.json"

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved to: {output_file}")

if __name__ == "__main__":
    main()
