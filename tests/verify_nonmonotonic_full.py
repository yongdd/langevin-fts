#!/usr/bin/env python3
"""
Full verification of non-monotonic convergence fix.

Runs SCFT for Ns = 10 to 40 with f = 0.375 and χN = 18.
Verifies that same-segment pairs produce identical free energies.
"""

import os
import sys
import math
import numpy as np

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

from polymerfts import scft

def run_scft_for_ns(Ns, f=0.375, chi_n=18.0):
    """Run SCFT for a given Ns value and return the free energy."""
    ds = 1.0 / Ns

    params = {
        "nx": [32, 32, 32],
        "lx": [4.0, 4.0, 4.0],
        "box_is_altering": False,
        "chain_model": "continuous",
        "ds": ds,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": chi_n},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1.0 - f},
            ],
        }],
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
        "max_iter": 3000,
        "tolerance": 1e-8,
        "verbose_level": 0,
    }

    calculation = scft.SCFT(params=params)

    # Initialize fields with lamellar guess
    w_A = np.zeros(list(params["nx"]), dtype=np.float64)
    w_B = np.zeros(list(params["nx"]), dtype=np.float64)
    for i in range(params["nx"][2]):
        w_A[:,:,i] =  5.0 * np.cos(2*np.pi*i/params["nx"][2])
        w_B[:,:,i] = -5.0 * np.cos(2*np.pi*i/params["nx"][2])

    calculation.run(initial_fields={"A": w_A, "B": w_B})

    return calculation.free_energy

def round_half_away_from_zero(x):
    """Round using round-half-away-from-zero (like C++ lround)."""
    if x >= 0:
        return int(math.floor(x + 0.5))
    else:
        return int(math.ceil(x - 0.5))

def compute_segment_counts(Ns, f=0.375):
    """Compute segment counts for each block using C++-style rounding."""
    n_A = round_half_away_from_zero(f * Ns)
    n_B = round_half_away_from_zero((1 - f) * Ns)
    return n_A, n_B, n_A + n_B

def main():
    f = 0.375
    chi_n = 18.0

    print("=" * 80)
    print("Full Verification of Non-Monotonic Convergence Fix")
    print("=" * 80)
    print(f"Parameters: f = {f}, χN = {chi_n}")
    print()

    # Run SCFT for Ns = 10 to 40
    results = {}

    print("| Ns | f×Ns | n_A | n_B | Total | Free Energy F | Note |")
    print("|---:|-----:|----:|----:|------:|--------------:|------|")

    for Ns in range(10, 41):
        n_A, n_B, total = compute_segment_counts(Ns, f)
        f_times_Ns = f * Ns

        # Check if half-integer
        is_half_integer = abs(f_times_Ns - round(f_times_Ns)) > 0.4
        note = "half-integer" if is_half_integer else ""

        # Run SCFT
        F = run_scft_for_ns(Ns, f, chi_n)
        results[Ns] = {"F": F, "n_A": n_A, "n_B": n_B, "total": total}

        print(f"| {Ns} | {f_times_Ns:.2f} | {n_A} | {n_B} | {total} | {F:.8f} | {note} |")
        sys.stdout.flush()

    print()
    print("=" * 80)
    print("Same-Segment Pair Verification")
    print("=" * 80)
    print()
    print("| Ns pair | Total segments | |F(Ns) - F(Ns+1)| |")
    print("|---------|----------------|-------------------|")

    # Check same-segment pairs
    pairs_to_check = [(12, 13), (20, 21), (28, 29), (36, 37)]
    all_passed = True

    for Ns1, Ns2 in pairs_to_check:
        total1 = results[Ns1]["total"]
        total2 = results[Ns2]["total"]
        F1 = results[Ns1]["F"]
        F2 = results[Ns2]["F"]
        diff = abs(F1 - F2)

        if total1 != total2:
            print(f"| {Ns1}, {Ns2} | {total1} vs {total2} | DIFFERENT SEGMENTS |")
            all_passed = False
        else:
            print(f"| {Ns1}, {Ns2}  | {total1} | {diff:.2e} |")
            if diff > 1e-7:
                print(f"  WARNING: Difference {diff:.2e} > 1e-7")
                all_passed = False

    print()
    if all_passed:
        print("**Result: All 4 same-segment pairs have |ΔF| < 1e-7 (PASSED)**")
    else:
        print("**Result: FAILED - Some pairs have different free energies**")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
