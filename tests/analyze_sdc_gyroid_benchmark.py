"""
Analyze SDC-4 3D Gyroid convergence benchmark results.

Reads JSON result files and outputs convergence tables.
"""

import os
import json
import glob
import numpy as np

def load_results(output_dir="sdc_gyroid_benchmark"):
    """Load all result JSON files."""
    results = {}

    for filepath in glob.glob(os.path.join(output_dir, "result_*.json")):
        with open(filepath, "r") as f:
            result = json.load(f)

        method = result["method"]
        N = result["N"]

        if method not in results:
            results[method] = {}
        results[method][N] = result

    return results

def compute_convergence_order(F1, F2, F_ref, N1, N2):
    """Compute convergence order from two data points."""
    err1 = abs(F1 - F_ref)
    err2 = abs(F2 - F_ref)

    if err1 <= 0 or err2 <= 0:
        return float('nan')

    return np.log(err1 / err2) / np.log(N2 / N1)

def main():
    print("=" * 80)
    print("SDC-4 3D Gyroid Free Energy Convergence Analysis")
    print("=" * 80)
    print()

    results = load_results()

    if not results:
        print("No results found. Run submit_sdc_gyroid_benchmark.sh first.")
        return

    # Get all methods and N values
    methods = sorted(results.keys())
    all_N = set()
    for method_results in results.values():
        all_N.update(method_results.keys())
    N_values = sorted(all_N)

    print(f"Methods found: {methods}")
    print(f"N values: {N_values}")
    print()

    # Reference free energies (use highest N for each method category)
    # Pseudo-spectral methods converge to one value, real-space to another
    pseudo_spectral = ["rqm4", "etdrk4"]
    real_space = ["cn-adi2", "cn-adi4", "sdc-4", "sdc-2", "sdc-3", "sdc-5", "sdc-6", "sdc-8", "sdc-10"]

    # Get reference values from highest N
    F_ref_pseudo = None
    F_ref_real = None

    for method in methods:
        if max(results[method].keys()) in results[method]:
            max_N = max(results[method].keys())
            F = results[method][max_N]["free_energy"]
            if method in pseudo_spectral:
                F_ref_pseudo = F
            elif any(method.startswith(r.split("-")[0]) for r in real_space):
                F_ref_real = F

    print("-" * 80)
    print("Free Energy vs Contour Steps (Ns)")
    print("-" * 80)
    print()

    # Header
    header = "| Method |"
    for N in N_values:
        header += f" Ns={N} |"
    print(header)
    print("|" + "-"*8 + "|" + "|".join(["-"*12 for _ in N_values]) + "|")

    # Data rows
    for method in methods:
        row = f"| **{method}** |"
        for N in N_values:
            if N in results[method]:
                F = results[method][N]["free_energy"]
                row += f" {F:.8f} |"
            else:
                row += "      -      |"
        print(row)

    print()
    print("-" * 80)
    print("Execution Time vs Contour Steps (seconds)")
    print("-" * 80)
    print()

    # Header
    header = "| Method |"
    for N in N_values:
        header += f" Ns={N} |"
    print(header)
    print("|" + "-"*8 + "|" + "|".join(["-"*10 for _ in N_values]) + "|")

    # Data rows
    for method in methods:
        row = f"| **{method}** |"
        for N in N_values:
            if N in results[method]:
                t = results[method][N]["elapsed_time"]
                row += f" {t:8.1f} |"
            else:
                row += "     -    |"
        print(row)

    print()

    # Convergence order analysis
    if F_ref_pseudo or F_ref_real:
        print("-" * 80)
        print("Convergence Order Analysis")
        print("-" * 80)
        print()

        for method in methods:
            if method in pseudo_spectral:
                F_ref = F_ref_pseudo
            else:
                F_ref = F_ref_real

            if F_ref is None:
                continue

            print(f"Method: {method}")
            N_list = sorted(results[method].keys())

            for i in range(len(N_list) - 1):
                N1, N2 = N_list[i], N_list[i+1]
                if N1 in results[method] and N2 in results[method]:
                    F1 = results[method][N1]["free_energy"]
                    F2 = results[method][N2]["free_energy"]
                    order = compute_convergence_order(F1, F2, F_ref, N1, N2)
                    err1 = abs(F1 - F_ref)
                    print(f"  Ns={N1:4d} -> {N2:4d}: order={order:.2f}, |F-F_ref|={err1:.2e}")
            print()

    # Summary table for documentation
    print("=" * 80)
    print("SUMMARY FOR DOCUMENTATION")
    print("=" * 80)
    print()

    print("### Free Energy vs Contour Steps (Ns)")
    print()
    print("| Method |", end="")
    for N in N_values:
        print(f" Ns={N} |", end="")
    print()
    print("|--------|", end="")
    for _ in N_values:
        print("--------|", end="")
    print()

    for method in methods:
        print(f"| **{method.upper()}** |", end="")
        for N in N_values:
            if N in results[method]:
                F = results[method][N]["free_energy"]
                print(f" {F:.8f} |", end="")
            else:
                print(" - |", end="")
        print()

    print()
    print("### Execution Time (seconds)")
    print()
    print("| Method |", end="")
    for N in N_values:
        print(f" Ns={N} |", end="")
    print()
    print("|--------|", end="")
    for _ in N_values:
        print("--------|", end="")
    print()

    for method in methods:
        print(f"| **{method.upper()}** |", end="")
        for N in N_values:
            if N in results[method]:
                t = results[method][N]["elapsed_time"]
                print(f" {t:.1f} s |", end="")
            else:
                print(" - |", end="")
        print()

if __name__ == "__main__":
    main()
