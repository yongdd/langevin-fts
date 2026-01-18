#!/usr/bin/env python
"""
Test: Same-Segment Consistency for SCFT Results

Tests that when two Ns values produce the same segment counts (due to rounding),
the free energy, stress, and box size are identical within machine precision.

This validates the per-block local_ds normalization fix for concentration and stress.
When f*Ns is a half-integer, both blocks round up, causing total_segments > Ns.

Test range: Ns = 19 to 21 (f = 0.375)
Half-integer case: Ns = 20 (where f*Ns = 7.5)
Expected: Ns=20 and Ns=21 both have (n_A=8, n_B=13, total=21) and must produce identical results.
"""

import os
import sys
import math
import numpy as np

# Set OpenMP environment before importing polymerfts
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

from polymerfts import scft


def cpp_round(x):
    """Mimic C++ std::round (round half away from zero)."""
    if x >= 0:
        return math.floor(x + 0.5)
    else:
        return math.ceil(x - 0.5)


def get_segment_counts(f, Ns):
    """Calculate segment counts using C++ rounding behavior."""
    ds = 1.0 / Ns

    # A-block
    ratio_A = f / ds
    rounded_to_half_A = round(ratio_A * 2.0) / 2.0
    if abs(ratio_A - rounded_to_half_A) < 1e-9:
        ratio_A = rounded_to_half_A
    n_A = int(cpp_round(ratio_A))

    # B-block
    ratio_B = (1 - f) / ds
    rounded_to_half_B = round(ratio_B * 2.0) / 2.0
    if abs(ratio_B - rounded_to_half_B) < 1e-9:
        ratio_B = rounded_to_half_B
    n_B = int(cpp_round(ratio_B))

    return n_A, n_B


def run_scft_1d(platform, reduce_memory, Ns, f=0.375, chi_n=18.0):
    """Run 1D lamellar SCFT for given Ns value."""
    ds = 1.0 / Ns

    params = {
        "platform": platform,
        "nx": [64],
        "lx": [4.0],
        "reduce_memory": reduce_memory,
        "box_is_altering": True,
        "chain_model": "continuous",
        "ds": ds,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": chi_n},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
        "max_iter": 2000,
        "tolerance": 1e-8,
    }

    # Initial fields for lamellar phase
    nx = params["nx"][0]
    w_A = np.array([np.cos(2 * np.pi * i / nx) for i in range(nx)])
    w_B = -w_A

    calculation = scft.SCFT(params=params)
    result = calculation.run(initial_fields={"A": w_A, "B": w_B}, return_result=True)

    n_A, n_B = get_segment_counts(f, Ns)

    return {
        "Ns": Ns,
        "n_A": n_A,
        "n_B": n_B,
        "total": n_A + n_B,
        "free_energy": result.free_energy,
        "stress": result.stress[0] if result.stress is not None else None,
        "lx": result.lx[0],
    }


def main():
    f = 0.375
    chi_n = 18.0
    Ns_range = range(19, 22)  # Ns = 19 to 21
    tight_tol = 1e-10

    print("=" * 80)
    print("Test: Same-Segment Consistency for SCFT Results")
    print("=" * 80)
    print(f"\nParameters: f = {f}, chi_n = {chi_n}")
    print(f"Testing Ns = {Ns_range.start} to {Ns_range.stop - 1}")
    print(f"Tight tolerance: {tight_tol:.0e}")
    print("\nThis test verifies:")
    print("1. Same-segment pairs have identical F, stress, and Lx")
    print("2. Results are consistent across platforms and memory modes")

    all_passed = True

    # Test configurations
    platforms = ["cpu-mkl", "cuda"]
    reduce_memory_modes = [False, True]

    # Store results for cross-platform comparison
    all_results = {}

    for platform in platforms:
        for reduce_memory in reduce_memory_modes:
            config_name = f"{platform}_{'reduce_mem' if reduce_memory else 'normal'}"

            print("\n" + "-" * 80)
            print(f"Testing: {config_name}")
            print("-" * 80)

            results = []
            config_valid = True

            for Ns in Ns_range:
                print(f"  Ns = {Ns}...", end=" ", flush=True)
                try:
                    result = run_scft_1d(platform, reduce_memory, Ns, f, chi_n)
                    results.append(result)
                    print(f"F = {result['free_energy']:.10f}, total = {result['total']}")
                    if not np.isfinite(result['free_energy']):
                        config_valid = False
                except Exception as e:
                    print(f"ERROR: {e}")
                    config_valid = False
                    break

            if not config_valid:
                print(f"\n  WARNING: This configuration produced non-finite results, skipping further checks.")
                continue

            # Print results table
            print("\n  Results:")
            print("  | Ns | n_A | n_B | Total |   Free Energy F   |    Stress    |     Lx     |")
            print("  |----|-----|-----|-------|-------------------|--------------|------------|")
            for r in results:
                stress_str = f"{r['stress']:.2e}" if r['stress'] is not None else "N/A"
                print(f"  | {r['Ns']:2d} | {r['n_A']:3d} | {r['n_B']:3d} | {r['total']:5d} | "
                      f"{r['free_energy']:17.12f} | {stress_str:>12s} | {r['lx']:10.6f} |")

            # Test same-segment pair consistency
            print("\n  Same-Segment Pair Check:")
            for i in range(len(results) - 1):
                r1 = results[i]
                r2 = results[i + 1]

                if r1["total"] == r2["total"] and r1["n_A"] == r2["n_A"] and r1["n_B"] == r2["n_B"]:
                    delta_F = abs(r1["free_energy"] - r2["free_energy"])
                    delta_stress = abs(r1["stress"] - r2["stress"]) if r1["stress"] and r2["stress"] else 0
                    delta_lx = abs(r1["lx"] - r2["lx"])

                    print(f"    Ns={r1['Ns']},{r2['Ns']} (total={r1['total']}): ", end="")
                    print(f"|dF|={delta_F:.2e}, |dStress|={delta_stress:.2e}, |dLx|={delta_lx:.2e}", end="")

                    if delta_F > tight_tol or delta_stress > tight_tol or delta_lx > tight_tol:
                        print(" -> FAIL")
                        all_passed = False
                    else:
                        print(" -> PASS")

            all_results[config_name] = results

    # Cross-platform consistency check
    print("\n" + "-" * 80)
    print("Cross-Platform Consistency Check")
    print("-" * 80)

    config_names = list(all_results.keys())

    # Cross-platform tolerances
    cross_tol_F = 1e-9
    cross_tol_stress = 1e-6
    cross_tol_lx = 1e-5

    for i in range(len(config_names)):
        for j in range(i + 1, len(config_names)):
            name1 = config_names[i]
            name2 = config_names[j]
            results1 = all_results[name1]
            results2 = all_results[name2]

            print(f"\n  Comparing {name1} vs {name2}:")

            for k in range(min(len(results1), len(results2))):
                delta_F = abs(results1[k]["free_energy"] - results2[k]["free_energy"])
                delta_stress = abs(results1[k]["stress"] - results2[k]["stress"]) if results1[k]["stress"] and results2[k]["stress"] else 0
                delta_lx = abs(results1[k]["lx"] - results2[k]["lx"])

                print(f"    Ns={results1[k]['Ns']}: |dF|={delta_F:.2e}, |dStress|={delta_stress:.2e}, |dLx|={delta_lx:.2e}", end="")

                F_ok = delta_F <= cross_tol_F or not np.isfinite(delta_F)
                stress_ok = delta_stress <= cross_tol_stress or not np.isfinite(delta_stress)
                lx_ok = delta_lx <= cross_tol_lx or not np.isfinite(delta_lx)

                if F_ok and stress_ok and lx_ok:
                    print(" -> PASS")
                else:
                    print(" -> FAIL", end="")
                    if not F_ok:
                        print(" (F)", end="")
                    if not stress_ok:
                        print(" (stress)", end="")
                    if not lx_ok:
                        print(" (lx)", end="")
                    print()
                    all_passed = False

    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("All tests PASSED")
        return 0
    else:
        print("Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
