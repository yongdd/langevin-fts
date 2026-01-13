#!/usr/bin/env python3
"""
Generate Fig. 1 convergence plot for NumericalMethodsPerformance.md

Creates two subplots:
1. Free energy error vs Ns (log-log scale)
2. Execution time vs Ns (log-log scale)
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_results():
    """Load all benchmark results."""
    results = defaultdict(dict)

    for filepath in glob.glob("benchmark_fig1_*.json"):
        with open(filepath, 'r') as f:
            data = json.load(f)

        method = data['metadata']['method']
        for r in data['results']:
            if r.get('success', False):
                Ns = r['Ns']
                # Skip Ns=100 which has non-integer block issues
                if Ns == 100:
                    continue
                results[method][Ns] = {
                    'free_energy': r['free_energy'],
                    'run_time_s': r['run_time_s'],
                    'total_time_s': r['total_time_s'],
                }

    return results


def main():
    results = load_results()

    # Reference free energies
    F_ref_pseudo = -0.47697411  # Pseudo-spectral reference (from RQM4 at Ns=4000)
    F_ref_cnadi = -0.47935136   # CN-ADI reference (from CN-ADI4 at Ns=4000)

    # Method styling
    styles = {
        'rqm4': {'color': '#1f77b4', 'marker': 'o', 'label': 'RQM4', 'linestyle': '-'},
        'etdrk4': {'color': '#ff7f0e', 'marker': 's', 'label': 'ETDRK4', 'linestyle': '-'},
        'cn-adi2': {'color': '#2ca02c', 'marker': '^', 'label': 'CN-ADI2', 'linestyle': '-'},
        'cn-adi4': {'color': '#d62728', 'marker': 'v', 'label': 'CN-ADI4', 'linestyle': '-'},
    }

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Plot 1: Free energy error vs Ns (pseudo-spectral only for convergence)
    # Only show Ns <= 400 for cleaner convergence visualization
    ax1.set_xlabel('Ns (contour steps)', fontsize=12)
    ax1.set_ylabel('|F - F$_{ref}$|', fontsize=12)
    ax1.set_title('(a) Free Energy Convergence', fontsize=12)

    for method in ['rqm4', 'etdrk4']:
        if method in results:
            ns_vals = sorted(results[method].keys())
            # Filter to Ns <= 400
            ns_vals = [ns for ns in ns_vals if ns <= 400]
            errors = [abs(results[method][ns]['free_energy'] - F_ref_pseudo) for ns in ns_vals]

            # Filter out zero errors for log plot
            valid = [(ns, err) for ns, err in zip(ns_vals, errors) if err > 1e-14]
            if valid:
                ns_plot, err_plot = zip(*valid)
                ax1.loglog(ns_plot, err_plot,
                          marker=styles[method]['marker'],
                          color=styles[method]['color'],
                          linestyle=styles[method]['linestyle'],
                          label=styles[method]['label'],
                          markersize=8, linewidth=2)

    # Add reference line for 4th order
    ns_ref = np.array([40, 400])
    err_ref = 1e-4 * (ns_ref / 40) ** (-4)
    ax1.loglog(ns_ref, err_ref, 'k--', alpha=0.5, label='4th order slope')

    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(30, 500)
    ax1.set_ylim(1e-9, 1e-3)

    # Plot 2: CN-ADI convergence
    # Only show Ns <= 400 for cleaner convergence visualization
    ax2.set_xlabel('Ns (contour steps)', fontsize=12)
    ax2.set_ylabel('|F - F$_{ref}$|', fontsize=12)
    ax2.set_title('(b) CN-ADI Convergence', fontsize=12)

    for method in ['cn-adi2', 'cn-adi4']:
        if method in results:
            ns_vals = sorted(results[method].keys())
            # Filter to Ns <= 400
            ns_vals = [ns for ns in ns_vals if ns <= 400]
            errors = [abs(results[method][ns]['free_energy'] - F_ref_cnadi) for ns in ns_vals]

            # Filter out zero errors for log plot
            valid = [(ns, err) for ns, err in zip(ns_vals, errors) if err > 1e-14]
            if valid:
                ns_plot, err_plot = zip(*valid)
                ax2.loglog(ns_plot, err_plot,
                          marker=styles[method]['marker'],
                          color=styles[method]['color'],
                          linestyle=styles[method]['linestyle'],
                          label=styles[method]['label'],
                          markersize=8, linewidth=2)

    # Add reference lines for 2nd and 4th order
    ns_ref = np.array([40, 400])
    err_ref_2 = 2e-3 * (ns_ref / 40) ** (-2)
    err_ref_4 = 1.5e-5 * (ns_ref / 40) ** (-4)
    ax2.loglog(ns_ref, err_ref_2, 'k--', alpha=0.5, label='2nd order slope')
    ax2.loglog(ns_ref, err_ref_4, 'k:', alpha=0.5, label='4th order slope')

    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(30, 500)
    ax2.set_ylim(1e-9, 1e-2)

    # Plot 3: Execution time vs Ns
    ax3.set_xlabel('Ns (contour steps)', fontsize=12)
    ax3.set_ylabel('Time (s)', fontsize=12)
    ax3.set_title('(c) Execution Time', fontsize=12)

    for method in ['rqm4', 'etdrk4', 'cn-adi2', 'cn-adi4']:
        if method in results:
            ns_vals = sorted(results[method].keys())
            times = [results[method][ns]['run_time_s'] for ns in ns_vals]

            ax3.loglog(ns_vals, times,
                      marker=styles[method]['marker'],
                      color=styles[method]['color'],
                      linestyle=styles[method]['linestyle'],
                      label=styles[method]['label'],
                      markersize=8, linewidth=2)

    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(30, 5000)

    plt.tight_layout()

    # Save figure
    output_path = '../docs/figures/figure1_song2018.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

    # Also save PDF version
    pdf_path = '../docs/figures/figure1_song2018.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to {pdf_path}")

    plt.close()


if __name__ == "__main__":
    main()
