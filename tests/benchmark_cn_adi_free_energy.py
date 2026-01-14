"""
Free energy convergence benchmark for CN-ADI methods.

Compares cn-adi2, cn-adi4-lr, and cn-adi4-gr methods.
Usage: python benchmark_cn_adi_free_energy.py --method <method> --ds <ds>
"""

import os
import sys
import argparse
import json
import numpy as np

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"

from polymerfts.scft import SCFT

def run_scft(method, ds, chi_n=20.0, f=0.5, nx=32, lx=3.2):
    """Run SCFT for AB diblock in lamellar phase."""

    params = {
        "nx": [nx],
        "lx": [lx],
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
        "max_iter": 2000,
        "tolerance": 1e-9,
        "platform": "cpu-mkl",
        "numerical_method": method,
        "verbose": False,
        "box_is_altering": False,
    }

    # Initialize SCFT
    scft = SCFT(params)

    # Set initial fields (lamellar)
    x = np.linspace(0, lx, nx, endpoint=False)
    w_A = chi_n * 0.5 * np.cos(2*np.pi*x/lx)
    w_B = -w_A

    # Run SCFT
    scft.run(initial_fields={"A": w_A, "B": w_B})

    # Get results from attributes
    result = {
        "method": method,
        "ds": ds,
        "N": int(1/ds),
        "chi_n": chi_n,
        "f": f,
        "nx": nx,
        "lx": lx,
        "free_energy": float(scft.free_energy),
        "error_level": float(scft.error_level),
        "iterations": int(scft.iter),
        "converged": bool(scft.error_level < 1e-9),
    }

    return result

def main():
    parser = argparse.ArgumentParser(description="CN-ADI free energy convergence benchmark")
    parser.add_argument("--method", type=str, required=True,
                        choices=["cn-adi2", "cn-adi4-lr", "cn-adi4-gr"],
                        help="Numerical method")
    parser.add_argument("--ds", type=float, required=True,
                        help="Contour step size")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    args = parser.parse_args()

    print(f"Running SCFT: method={args.method}, ds={args.ds}")

    result = run_scft(args.method, args.ds)

    print(f"  Free energy: {result['free_energy']:.12f}")
    print(f"  Error level: {result['error_level']:.2e}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Converged: {result['converged']}")

    # Save result
    if args.output:
        output_file = args.output
    else:
        output_file = f"result_{args.method}_ds{args.ds:.6f}.json"

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved to: {output_file}")

if __name__ == "__main__":
    main()
