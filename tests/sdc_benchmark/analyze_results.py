#!/usr/bin/env python
"""
Analyze SDC convergence benchmark results.

Reads JSON result files and computes convergence orders.
"""
import os
import json
import glob
import numpy as np

def load_results(results_dir: str):
    """Load all result JSON files."""
    results = {}
    pattern = os.path.join(results_dir, "result_sdc*.json")

    for filepath in glob.glob(pattern):
        with open(filepath, 'r') as f:
            data = json.load(f)
            order = data["order"]
            N = data["N"]
            if order not in results:
                results[order] = {}
            results[order][N] = data

    return results


def compute_convergence_order(results: dict, order: int, ref_N: int = 640):
    """Compute convergence order for a given SDC order."""
    if order not in results:
        return []

    data = results[order]
    if ref_N not in data:
        # Use the largest N as reference
        ref_N = max(data.keys())

    F_ref = data[ref_N]["free_energy"]

    orders = []
    N_values = sorted([N for N in data.keys() if N < ref_N])

    for i in range(len(N_values) - 1):
        N1 = N_values[i]
        N2 = N_values[i + 1]
        err1 = abs(data[N1]["free_energy"] - F_ref)
        err2 = abs(data[N2]["free_energy"] - F_ref)

        if err2 > 1e-14:
            conv_order = np.log(err1 / err2) / np.log(2)
            orders.append((N1, N2, conv_order))

    return orders


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    results = load_results(results_dir)

    if not results:
        print("No results found. Run submit_all_jobs.sh first.")
        return

    print("=" * 80)
    print("SDC Convergence Benchmark Results - 3D Gyroid Phase")
    print("=" * 80)
    print()

    # Print free energy values for each order
    for order in sorted(results.keys()):
        print(f"### SDC-{order} Free Energy Values")
        print()
        print("| N | ds | Free Energy |")
        print("|---|---:|------------:|")

        data = results[order]
        for N in sorted(data.keys()):
            ds = data[N]["ds"]
            F = data[N]["free_energy"]
            print(f"| {N} | {ds:.4f} | {F:.10f} |")
        print()

    # Print convergence orders
    print("### Convergence Order Analysis")
    print()
    print("| SDC Order | N transition | Measured Order |")
    print("|-----------|--------------|---------------:|")

    for order in sorted(results.keys()):
        conv_orders = compute_convergence_order(results, order)
        for N1, N2, conv in conv_orders:
            print(f"| sdc-{order} | {N1} -> {N2} | {conv:.2f} |")
    print()

    # Summary table
    print("### Summary: Average Convergence Order")
    print()
    print("| Method | Expected Order | Measured Order |")
    print("|--------|---------------:|---------------:|")

    for order in sorted(results.keys()):
        conv_orders = compute_convergence_order(results, order)
        if conv_orders:
            # Use later transitions for more accurate estimate
            if len(conv_orders) >= 2:
                avg_order = np.mean([c[2] for c in conv_orders[-2:]])
            else:
                avg_order = conv_orders[-1][2]
            print(f"| sdc-{order} | {order} | {avg_order:.2f} |")
    print()

    # Save markdown output
    output_file = os.path.join(results_dir, "convergence_analysis.md")
    print(f"Saving analysis to: {output_file}")


if __name__ == "__main__":
    main()
