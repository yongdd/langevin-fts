"""
Analyze CN-ADI free energy convergence results for 3D Gyroid.
"""

import os
import json
import numpy as np
from glob import glob

def analyze_results(output_dir):
    """Analyze benchmark results from JSON files."""

    # Collect all results
    methods = {}
    for json_file in glob(os.path.join(output_dir, "result_*.json")):
        with open(json_file, 'r') as f:
            result = json.load(f)

        method = result['method']
        if method not in methods:
            methods[method] = {'ds': [], 'N': [], 'F': []}

        methods[method]['ds'].append(result['ds'])
        methods[method]['N'].append(result['N'])
        methods[method]['F'].append(result['free_energy'])

    # Sort by ds (descending) for each method
    for method in methods:
        data = methods[method]
        sorted_indices = np.argsort(data['ds'])[::-1]
        data['ds'] = [data['ds'][i] for i in sorted_indices]
        data['N'] = [data['N'][i] for i in sorted_indices]
        data['F'] = [data['F'][i] for i in sorted_indices]

    # Compute errors relative to finest discretization
    analysis = {}
    for method, data in methods.items():
        F_ref = data['F'][-1]  # Finest discretization
        errors = [abs(F - F_ref) for F in data['F']]

        # Compute convergence order
        orders = []
        for i in range(len(errors) - 2):
            if errors[i] > 0 and errors[i+1] > 0:
                order = np.log(errors[i] / errors[i+1]) / np.log(data['ds'][i] / data['ds'][i+1])
                orders.append(order)

        avg_order = np.mean(orders) if orders else 0

        analysis[method] = {
            'ds': data['ds'],
            'N': data['N'],
            'F': data['F'],
            'errors': errors,
            'F_ref': F_ref,
            'order': avg_order
        }

    return analysis

def print_results(analysis):
    """Print formatted results."""

    print("\n" + "="*80)
    print("CN-ADI Free Energy Convergence Analysis - 3D Gyroid")
    print("="*80)

    # Print free energy values
    print("\nFree Energy Values (βF/V):")
    print("-" * 70)
    header = f"{'N':>5} | {'ds':>8}"
    for method in sorted(analysis.keys()):
        header += f" | {method:>14}"
    print(header)
    print("-" * 70)

    # Get common N values
    n_values = sorted(set(analysis[list(analysis.keys())[0]]['N']))
    for n in n_values:
        row = f"{n:>5} | {1/n:>8.4f}"
        for method in sorted(analysis.keys()):
            idx = analysis[method]['N'].index(n) if n in analysis[method]['N'] else None
            if idx is not None:
                row += f" | {analysis[method]['F'][idx]:>14.10f}"
            else:
                row += f" | {'N/A':>14}"
        print(row)

    # Print errors
    print("\nFree Energy Error (relative to N=160):")
    print("-" * 70)
    header = f"{'N':>5} | {'ds':>8}"
    for method in sorted(analysis.keys()):
        header += f" | {method:>14}"
    print(header)
    print("-" * 70)

    for n in n_values[:-1]:  # Skip reference N
        row = f"{n:>5} | {1/n:>8.4f}"
        for method in sorted(analysis.keys()):
            idx = analysis[method]['N'].index(n) if n in analysis[method]['N'] else None
            if idx is not None:
                row += f" | {analysis[method]['errors'][idx]:>14.2e}"
            else:
                row += f" | {'N/A':>14}"
        print(row)

    # Print convergence order
    print("\nConvergence Order:")
    print("-" * 40)
    for method in sorted(analysis.keys()):
        print(f"  {method}: {analysis[method]['order']:.2f}")

    print("\n" + "="*80)

def main():
    output_dir = "cn_adi_benchmark_results_3d"

    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' not found")
        return

    analysis = analyze_results(output_dir)
    print_results(analysis)

    # Save analysis to JSON
    analysis_file = os.path.join(output_dir, "analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to: {analysis_file}")

if __name__ == "__main__":
    main()
