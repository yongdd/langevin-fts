#!/usr/bin/env python
"""
Generate Figure 1 for NumericalMethodsPerformance.md

Plots convergence analysis for numerical methods:
- (a) Pseudo-spectral methods (RQM4, ETDRK4)
- (b) Real-space methods (CN-ADI2, CN-ADI4-LR)
- (c) Execution time comparison

Data from Gyroid SCFT benchmark: f=0.375, chi_n=18, M=32^3, L=3.65
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Benchmark results directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'sdc_gyroid_benchmark')

# Methods to plot (only supported methods)
PSEUDO_4TH_METHODS = ['rqm4', 'etdrk4']
PSEUDO_2ND_METHODS = ['rk2']
PSEUDO_METHODS = PSEUDO_4TH_METHODS + PSEUDO_2ND_METHODS
REALSPACE_METHODS = ['cn-adi2', 'cn-adi4-lr']

# Reference free energies (converged values from Ns=1000)
F_REF_PSEUDO = -0.476974113087715  # ETDRK4 at Ns=1000 (used for all pseudo-spectral methods)
F_REF_REALSPACE = -0.479351354862727  # CN-ADI4-LR at Ns=1000

# Plot styling
COLORS = {
    'rqm4': 'C1', 'rk2': 'C4', 'etdrk4': 'C0',
    'cn-adi2': 'C2', 'cn-adi4-lr': 'C3'
}
MARKERS = {
    'rqm4': 's', 'rk2': 'D', 'etdrk4': 'o',
    'cn-adi2': '^', 'cn-adi4-lr': 'v'
}
LABELS = {
    'rqm4': 'RQM4', 'rk2': 'RK2', 'etdrk4': 'ETDRK4',
    'cn-adi2': 'CN-ADI2', 'cn-adi4-lr': 'CN-ADI4-LR'
}


def load_results(method):
    """Load all benchmark results for a method."""
    results = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.startswith(f'result_{method}_N') and filename.endswith('.json'):
            filepath = os.path.join(RESULTS_DIR, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(data)
    return sorted(results, key=lambda x: x['N'])


def plot_figure():
    """Generate the three-panel convergence figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Collect data for all methods
    all_data = {}
    for method in PSEUDO_METHODS + REALSPACE_METHODS:
        results = load_results(method)
        if results:
            all_data[method] = results

    # (a) Pseudo-spectral error convergence
    ax = axes[0]
    # All pseudo-spectral methods use same reference F_REF_PSEUDO
    for method in PSEUDO_METHODS:
        if method not in all_data:
            continue
        results = all_data[method]
        Ns = [r['N'] for r in results]
        errors = [abs(r['free_energy'] - F_REF_PSEUDO) for r in results]
        ax.loglog(Ns, errors, marker=MARKERS[method], color=COLORS[method],
                  label=LABELS[method], linewidth=2, markersize=8)

    # Add reference slopes
    Ns_ref = np.array([40, 1000])
    ax.loglog(Ns_ref, 5e-5 * (Ns_ref/40)**(-4), 'k--', alpha=0.5, label='slope -4')
    ax.loglog(Ns_ref, 2e-3 * (Ns_ref/40)**(-2), 'k:', alpha=0.5, label='slope -2')

    ax.set_xlabel(r'$N_s$ (contour steps)', fontsize=12)
    ax.set_ylabel(r'$|F - F_{ref}|$', fontsize=12)
    ax.set_title('(a) Pseudo-spectral methods', fontsize=12)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(30, 1500)
    ax.set_ylim(1e-10, 1e-2)

    # (b) Real-space error convergence
    ax = axes[1]
    for method in REALSPACE_METHODS:
        if method not in all_data:
            continue
        results = all_data[method]
        # Exclude the reference point (last point used as F_ref)
        Ns = [r['N'] for r in results[:-1]]
        errors = [abs(r['free_energy'] - F_REF_REALSPACE) for r in results[:-1]]
        ax.loglog(Ns, errors, marker=MARKERS[method], color=COLORS[method],
                  label=LABELS[method], linewidth=2, markersize=8)

    # Add reference slopes
    Ns_ref = np.array([40, 1000])
    ax.loglog(Ns_ref, 2e-3 * (Ns_ref/40)**(-2), 'k:', alpha=0.5, label='slope -2')
    ax.loglog(Ns_ref, 2e-5 * (Ns_ref/40)**(-4), 'k--', alpha=0.5, label='slope -4')

    ax.set_xlabel(r'$N_s$ (contour steps)', fontsize=12)
    ax.set_ylabel(r'$|F - F_{ref}|$', fontsize=12)
    ax.set_title('(b) Real-space methods', fontsize=12)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(30, 1500)
    ax.set_ylim(1e-10, 1e-2)

    # (c) Execution time comparison
    ax = axes[2]
    for method in PSEUDO_METHODS + REALSPACE_METHODS:
        if method not in all_data:
            continue
        results = all_data[method]
        Ns = [r['N'] for r in results]
        times = [r['elapsed_time'] for r in results]
        ax.loglog(Ns, times, marker=MARKERS[method], color=COLORS[method],
                  label=LABELS[method], linewidth=2, markersize=8)

    ax.set_xlabel(r'$N_s$ (contour steps)', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('(c) Execution time', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(30, 1500)

    plt.tight_layout()

    # Save to docs/figures
    docs_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures')
    os.makedirs(docs_dir, exist_ok=True)

    plt.savefig(os.path.join(docs_dir, 'figure1_song2018.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(docs_dir, 'figure1_song2018.pdf'), bbox_inches='tight')
    print(f"Saved figures to {docs_dir}/figure1_song2018.png and .pdf")

    plt.close()


if __name__ == "__main__":
    plot_figure()
