"""
Analyze CN-ADI free energy convergence results.

Reads JSON result files and computes convergence order.
Usage: python analyze_cn_adi_convergence.py [--output-dir <dir>]
"""

import os
import sys
import argparse
import json
import glob
import numpy as np

def load_results(output_dir):
    """Load all result JSON files."""
    results = {}

    for filepath in glob.glob(os.path.join(output_dir, "result_*.json")):
        with open(filepath, "r") as f:
            data = json.load(f)
            method = data["method"]
            if method not in results:
                results[method] = []
            results[method].append(data)

    # Sort by ds (descending, so N is ascending)
    for method in results:
        results[method].sort(key=lambda x: -x["ds"])

    return results

def compute_convergence_order(ds_values, errors):
    """Compute convergence order from log-log slope."""
    if len(ds_values) < 2:
        return None

    # Filter out zero or negative errors
    valid = [(ds, err) for ds, err in zip(ds_values, errors) if err > 0]
    if len(valid) < 2:
        return None

    ds_valid = [v[0] for v in valid]
    err_valid = [v[1] for v in valid]

    log_ds = np.log(ds_valid)
    log_err = np.log(err_valid)

    # Linear regression
    slope, _ = np.polyfit(log_ds, log_err, 1)
    return slope

def analyze_results(results):
    """Analyze convergence for each method."""

    # Find reference (finest grid) free energy for each method
    # Use the finest ds result as reference
    analysis = {}

    for method, data_list in results.items():
        if not data_list:
            continue

        # Sort by N (ascending)
        data_list.sort(key=lambda x: x["N"])

        # Reference: finest grid (largest N)
        ref = data_list[-1]
        F_ref = ref["free_energy"]

        ds_values = []
        N_values = []
        F_values = []
        errors = []

        for data in data_list:
            ds_values.append(data["ds"])
            N_values.append(data["N"])
            F_values.append(data["free_energy"])
            errors.append(abs(data["free_energy"] - F_ref))

        # Compute convergence order (exclude finest grid from fit)
        if len(ds_values) > 2:
            order = compute_convergence_order(ds_values[:-1], errors[:-1])
        else:
            order = None

        analysis[method] = {
            "ds": ds_values,
            "N": N_values,
            "F": F_values,
            "errors": errors,
            "F_ref": F_ref,
            "order": order,
        }

    return analysis

def print_results(analysis):
    """Print formatted results."""

    print("=" * 80)
    print("CN-ADI Free Energy Convergence Analysis")
    print("=" * 80)

    methods = sorted(analysis.keys())

    # Print free energy table
    print("\n--- Free Energy Values ---")
    print(f"{'N':>6} {'ds':>10}", end="")
    for method in methods:
        print(f" {method:>16}", end="")
    print()
    print("-" * (18 + 17 * len(methods)))

    # Get all N values
    all_N = set()
    for method in methods:
        all_N.update(analysis[method]["N"])
    all_N = sorted(all_N)

    for N in all_N:
        ds = 1.0 / N
        print(f"{N:>6} {ds:>10.6f}", end="")
        for method in methods:
            idx = None
            for i, n in enumerate(analysis[method]["N"]):
                if n == N:
                    idx = i
                    break
            if idx is not None:
                F = analysis[method]["F"][idx]
                print(f" {F:>16.10f}", end="")
            else:
                print(f" {'N/A':>16}", end="")
        print()

    # Print error table
    print("\n--- Free Energy Error (relative to finest grid) ---")
    print(f"{'N':>6} {'ds':>10}", end="")
    for method in methods:
        print(f" {method:>16}", end="")
    print()
    print("-" * (18 + 17 * len(methods)))

    for N in all_N:
        ds = 1.0 / N
        print(f"{N:>6} {ds:>10.6f}", end="")
        for method in methods:
            idx = None
            for i, n in enumerate(analysis[method]["N"]):
                if n == N:
                    idx = i
                    break
            if idx is not None:
                err = analysis[method]["errors"][idx]
                if err > 0:
                    print(f" {err:>16.2e}", end="")
                else:
                    print(f" {'(ref)':>16}", end="")
            else:
                print(f" {'N/A':>16}", end="")
        print()

    # Print convergence order
    print("\n--- Convergence Order ---")
    for method in methods:
        order = analysis[method]["order"]
        if order is not None:
            print(f"  {method}: {order:.2f}")
        else:
            print(f"  {method}: N/A")

    print("\n" + "=" * 80)

    return analysis

def save_analysis(analysis, output_file):
    """Save analysis to JSON file."""
    # Convert numpy types to native Python
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return obj

    analysis_json = {}
    for method, data in analysis.items():
        analysis_json[method] = {k: convert(v) for k, v in data.items()}

    with open(output_file, "w") as f:
        json.dump(analysis_json, f, indent=2)

    print(f"Analysis saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze CN-ADI convergence results")
    parser.add_argument("--output-dir", type=str, default="cn_adi_benchmark_results",
                        help="Directory containing result JSON files")
    parser.add_argument("--save", type=str, default=None,
                        help="Save analysis to JSON file")
    args = parser.parse_args()

    # Load results
    results = load_results(args.output_dir)

    if not results:
        print(f"No results found in {args.output_dir}/")
        print("Run submit_cn_adi_benchmark.sh first.")
        sys.exit(1)

    # Analyze
    analysis = analyze_results(results)

    # Print
    print_results(analysis)

    # Save if requested
    if args.save:
        save_analysis(analysis, args.save)

if __name__ == "__main__":
    main()
