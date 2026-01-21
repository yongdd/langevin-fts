"""
Plot Real-Space method free energy convergence analysis for 1D Lamellar and 3D Gyroid.
Includes CN-ADI methods and SDC.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def load_results(output_dir):
    """Load benchmark results from JSON files."""
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
        data['ds'] = np.array([data['ds'][i] for i in sorted_indices])
        data['N'] = np.array([data['N'][i] for i in sorted_indices])
        data['F'] = np.array([data['F'][i] for i in sorted_indices])

    return methods

def compute_errors(methods):
    """Compute errors relative to finest discretization."""
    analysis = {}
    for method, data in methods.items():
        F_ref = data['F'][-1]  # Finest discretization
        errors = np.abs(data['F'] - F_ref)

        # Compute convergence order using linear regression (excluding reference)
        valid = errors[:-1] > 0
        if np.sum(valid) >= 2:
            log_ds = np.log(data['ds'][:-1][valid])
            log_err = np.log(errors[:-1][valid])
            # Linear fit
            coeffs = np.polyfit(log_ds, log_err, 1)
            order = coeffs[0]
        else:
            order = 0

        analysis[method] = {
            'ds': data['ds'],
            'N': data['N'],
            'F': data['F'],
            'errors': errors,
            'F_ref': F_ref,
            'order': order
        }

    return analysis

def plot_convergence(analysis_1d, analysis_3d, sdc_analysis_1d, sdc_analysis_3d,
                     output_file='RealSpaceConvergencePlot.png'):
    """Create convergence plots for 1D and 3D including SDC."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color and marker settings for CN-ADI
    colors = {'cn-adi2': 'blue', 'cn-adi4-lr': 'green', 'cn-adi4-gr': 'red', 'sdc': 'purple'}
    markers = {'cn-adi2': 'o', 'cn-adi4-lr': 's', 'cn-adi4-gr': '^', 'sdc': 'D'}
    labels = {'cn-adi2': 'CN-ADI2 (2nd order)',
              'cn-adi4-lr': 'CN-ADI4-LR (Local Richardson)',
              'cn-adi4-gr': 'CN-ADI4-GR (Global Richardson)',
              'sdc': 'SDC (M=3, K=2)'}

    # Plot 1D results
    ax1 = axes[0]

    # Plot CN-ADI methods
    for method in sorted(analysis_1d.keys()):
        data = analysis_1d[method]
        mask = data['errors'] > 0
        if np.sum(mask) > 0:
            ax1.loglog(data['ds'][mask], data['errors'][mask],
                      marker=markers[method], color=colors[method],
                      label=f"{labels[method]} (order={data['order']:.2f})",
                      linewidth=2, markersize=8)

    # Plot SDC
    if sdc_analysis_1d and 'sdc' in sdc_analysis_1d:
        data = sdc_analysis_1d['sdc']
        mask = data['errors'] > 0
        if np.sum(mask) > 0:
            ax1.loglog(data['ds'][mask], data['errors'][mask],
                      marker=markers['sdc'], color=colors['sdc'],
                      label=f"{labels['sdc']} (order={data['order']:.2f})",
                      linewidth=2, markersize=8)

    # Add reference lines for orders
    ds_ref = np.array([0.1, 0.001])
    ax1.loglog(ds_ref, 1e-2 * (ds_ref/0.1)**2, 'k--', alpha=0.3, label='O(ds²)')
    ax1.loglog(ds_ref, 1e-4 * (ds_ref/0.1)**4, 'k:', alpha=0.3, label='O(ds⁴)')

    ax1.set_xlabel('ds (contour step size)', fontsize=12)
    ax1.set_ylabel('|F - F_ref|', fontsize=12)
    ax1.set_title('1D Lamellar Phase (χN=20, f=0.5)', fontsize=14)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([5e-4, 0.2])
    ax1.set_ylim([1e-12, 1e-0])

    # Plot 3D results
    ax2 = axes[1]

    # Plot CN-ADI methods
    for method in sorted(analysis_3d.keys()):
        data = analysis_3d[method]
        mask = data['errors'] > 0
        if np.sum(mask) > 0:
            ax2.loglog(data['ds'][mask], data['errors'][mask],
                      marker=markers[method], color=colors[method],
                      label=f"{labels[method]} (order={data['order']:.2f})",
                      linewidth=2, markersize=8)

    # Plot SDC
    if sdc_analysis_3d and 'sdc' in sdc_analysis_3d:
        data = sdc_analysis_3d['sdc']
        mask = data['errors'] > 0
        if np.sum(mask) > 0:
            ax2.loglog(data['ds'][mask], data['errors'][mask],
                      marker=markers['sdc'], color=colors['sdc'],
                      label=f"{labels['sdc']} (order={data['order']:.2f})",
                      linewidth=2, markersize=8)

    # Add reference lines for orders
    ax2.loglog(ds_ref, 1e-2 * (ds_ref/0.1)**2, 'k--', alpha=0.3, label='O(ds²)')
    ax2.loglog(ds_ref, 1e-4 * (ds_ref/0.1)**4, 'k:', alpha=0.3, label='O(ds⁴)')

    ax2.set_xlabel('ds (contour step size)', fontsize=12)
    ax2.set_ylabel('|F - F_ref|', fontsize=12)
    ax2.set_title('3D Gyroid Phase (χN=20, f=0.45)', fontsize=14)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([5e-4, 0.2])
    ax2.set_ylim([1e-9, 1e-0])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

def print_table(analysis, title, include_sdc=None):
    """Print formatted results table."""
    print(f"\n{'='*100}")
    print(title)
    print('='*100)

    all_analysis = dict(analysis)
    if include_sdc:
        all_analysis.update(include_sdc)

    # Print free energy values
    print("\nFree Energy Values (βF/V):")
    print("-" * 110)
    header = f"{'N':>6} | {'ds':>10}"
    for method in sorted(all_analysis.keys()):
        header += f" | {method:>16}"
    print(header)
    print("-" * 110)

    # Get all N values
    all_N = set()
    for method in all_analysis:
        all_N.update(all_analysis[method]['N'])
    all_N = sorted(all_N)

    for n in all_N:
        row = f"{n:>6} | {1/n:>10.6f}"
        for method in sorted(all_analysis.keys()):
            idx = np.where(all_analysis[method]['N'] == n)[0]
            if len(idx) > 0:
                row += f" | {all_analysis[method]['F'][idx[0]]:>16.12f}"
            else:
                row += f" | {'N/A':>16}"
        print(row)

    # Print errors
    print(f"\nFree Energy Error (relative to finest):")
    print("-" * 110)
    header = f"{'N':>6} | {'ds':>10}"
    for method in sorted(all_analysis.keys()):
        header += f" | {method:>16}"
    print(header)
    print("-" * 110)

    for n in all_N[:-1]:  # Skip reference
        row = f"{n:>6} | {1/n:>10.6f}"
        for method in sorted(all_analysis.keys()):
            idx = np.where(all_analysis[method]['N'] == n)[0]
            if len(idx) > 0:
                row += f" | {all_analysis[method]['errors'][idx[0]]:>16.2e}"
            else:
                row += f" | {'N/A':>16}"
        print(row)

    # Print convergence order
    print("\nConvergence Order:")
    print("-" * 50)
    for method in sorted(all_analysis.keys()):
        print(f"  {method}: {all_analysis[method]['order']:.2f}")

def main():
    # Change to tests directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load CN-ADI 1D results
    methods_1d = load_results("cn_adi_benchmark_results")
    analysis_1d = compute_errors(methods_1d)

    # Load CN-ADI 3D results
    methods_3d = load_results("cn_adi_benchmark_results_3d")
    analysis_3d = compute_errors(methods_3d)

    # Load SDC 1D results
    sdc_methods_1d = load_results("sdc_benchmark_results")
    sdc_analysis_1d = compute_errors(sdc_methods_1d) if sdc_methods_1d else {}

    # Load SDC 3D results
    sdc_methods_3d = load_results("sdc_benchmark_results_3d")
    sdc_analysis_3d = compute_errors(sdc_methods_3d) if sdc_methods_3d else {}

    # Print tables
    print_table(analysis_1d, "CN-ADI Free Energy Convergence - 1D Lamellar (χN=20, f=0.5)",
                include_sdc=sdc_analysis_1d)
    print_table(analysis_3d, "CN-ADI Free Energy Convergence - 3D Gyroid (χN=20, f=0.45)",
                include_sdc=sdc_analysis_3d)

    # Create plots
    plot_convergence(analysis_1d, analysis_3d, sdc_analysis_1d, sdc_analysis_3d,
                     '../docs/RealSpaceConvergencePlot.png')

    # Save analysis to JSON
    with open('sdc_benchmark_results/analysis.json', 'w') as f:
        sdc_1d_serializable = {k: {kk: v.tolist() if isinstance(v, np.ndarray) else v
                                    for kk, v in vv.items()}
                                for k, vv in sdc_analysis_1d.items()}
        json.dump(sdc_1d_serializable, f, indent=2)

    with open('sdc_benchmark_results_3d/analysis.json', 'w') as f:
        sdc_3d_serializable = {k: {kk: v.tolist() if isinstance(v, np.ndarray) else v
                                    for kk, v in vv.items()}
                                for k, vv in sdc_analysis_3d.items()}
        json.dump(sdc_3d_serializable, f, indent=2)

    print("\nAnalysis saved to:")
    print("  - sdc_benchmark_results/analysis.json")
    print("  - sdc_benchmark_results_3d/analysis.json")

if __name__ == "__main__":
    main()
