#!/usr/bin/env python
"""
Plot Figure 1 and Figure 2 from Song et al. 2018 benchmark results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_results(filepath):
    """Load JSON result file."""
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    return data['results']

def compute_relative_errors(results, key='Ns'):
    """Compute relative errors ΔF = |F(N) - F(N')| for adjacent N values."""
    # Sort by key
    sorted_results = sorted(results, key=lambda x: x[key])

    errors = []
    for i in range(len(sorted_results) - 1):
        r1 = sorted_results[i]
        r2 = sorted_results[i + 1]
        delta_F = abs(r1['free_energy'] - r2['free_energy'])
        errors.append({
            key: r2[key],
            'delta_F': delta_F,
            'time': r2['run_time_s'],
            'method': r2['method'],
        })

    return errors

def plot_figure1(results_dir='.'):
    """Plot Figure 1: Convergence in Ns."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = ['rqm4', 'etdrk4', 'cn-adi4-lr', 'cn-adi2']
    colors = {'etdrk4': 'C0', 'rqm4': 'C1', 'cn-adi2': 'C2', 'cn-adi4-lr': 'C3'}
    markers = {'etdrk4': 'o', 'rqm4': 's', 'cn-adi2': '^', 'cn-adi4-lr': 'v'}
    labels = {'etdrk4': 'ETDRK4', 'rqm4': 'RQM4', 'cn-adi2': 'CN-ADI2', 'cn-adi4-lr': 'CN-ADI4'}

    for method in methods:
        filepath = os.path.join(results_dir, f'benchmark_fig1_{method}_results.json')
        if not os.path.exists(filepath):
            print(f"No file found for {method} ({filepath})")
            continue

        all_results = load_results(filepath)

        if not all_results:
            continue

        # Filter by method and valid Ns (f*Ns must be integer; for f=3/8, Ns must be divisible by 8)
        method_results = [r for r in all_results if r['method'] == method and r['Ns'] % 8 == 0]
        if not method_results:
            continue

        # Compute relative errors
        errors = compute_relative_errors(method_results, 'Ns')

        if not errors:
            continue

        Ns_vals = [e['Ns'] for e in errors]
        delta_F_vals = [e['delta_F'] for e in errors]
        time_vals = [e['time'] for e in errors]

        # Plot (a): ΔF vs Ns
        axes[0].loglog(Ns_vals, delta_F_vals, marker=markers[method],
                       color=colors[method], label=labels[method], linewidth=2, markersize=8)

        # Plot (b): Time vs ΔF
        axes[1].loglog(delta_F_vals, time_vals, marker=markers[method],
                       color=colors[method], label=labels[method], linewidth=2, markersize=8)

    # Add reference slopes to (a)
    Ns_ref = np.array([50, 3000])
    # Order 2 slope
    axes[0].loglog(Ns_ref, 0.1 * (Ns_ref/50)**(-2), 'k--', alpha=0.5, label='slope -2')
    # Order 4 slope
    axes[0].loglog(Ns_ref, 1e-4 * (Ns_ref/50)**(-4), 'k:', alpha=0.5, label='slope -4')

    axes[0].set_xlabel(r'$N_s$', fontsize=14)
    axes[0].set_ylabel(r'$\Delta F(N_s)$', fontsize=14)
    axes[0].legend(loc='lower left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('(a) Relative error vs contour steps')

    axes[1].set_xlabel(r'$\Delta F(N_s)$', fontsize=14)
    axes[1].set_ylabel('Computation time (s)', fontsize=14)
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('(b) Time vs accuracy')
    axes[1].invert_xaxis()

    plt.tight_layout()
    plt.savefig('figure1_song2018.png', dpi=150)
    plt.savefig('figure1_song2018.pdf')
    print("Saved figure1_song2018.png/pdf")
    plt.close()


def plot_figure2(results_dir='.'):
    """Plot Figure 2: Convergence in Nx."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = ['rqm4', 'etdrk4', 'cn-adi4-lr', 'cn-adi2']
    colors = {'etdrk4': 'C0', 'rqm4': 'C1', 'cn-adi2': 'C2', 'cn-adi4-lr': 'C3'}
    markers = {'etdrk4': 'o', 'rqm4': 's', 'cn-adi2': '^', 'cn-adi4-lr': 'v'}
    labels = {'etdrk4': 'ETDRK4', 'rqm4': 'RQM4', 'cn-adi2': 'CN-ADI2', 'cn-adi4-lr': 'CN-ADI4'}

    chi_n_values = [40.0, 80.0]

    for ax_idx, chi_n in enumerate(chi_n_values):
        for method in methods:
            filepath = os.path.join(results_dir, f'benchmark_fig2_{method}_results.json')
            if not os.path.exists(filepath):
                print(f"No file found for {method}")
                continue

            all_results = load_results(filepath)

            if not all_results:
                continue

            # Filter by method and chi_n
            method_results = [r for r in all_results
                            if r['method'] == method and abs(r['chi_n'] - chi_n) < 0.1]
            if not method_results:
                continue

            # Compute relative errors
            errors = compute_relative_errors(method_results, 'Nx')

            if not errors:
                continue

            Nx_vals = [e['Nx'] for e in errors]
            delta_F_vals = [e['delta_F'] for e in errors]

            axes[ax_idx].semilogy(Nx_vals, delta_F_vals, marker=markers[method],
                                  color=colors[method], label=labels[method],
                                  linewidth=2, markersize=8)

        axes[ax_idx].set_xlabel(r'$N_x$', fontsize=14)
        axes[ax_idx].set_ylabel(r'$\Delta F(N_x)$', fontsize=14)
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_title(f'$\\chi N = {int(chi_n)}$')

    plt.tight_layout()
    plt.savefig('figure2_song2018.png', dpi=150)
    plt.savefig('figure2_song2018.pdf')
    print("Saved figure2_song2018.png/pdf")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='.',
                        help='Directory containing benchmark results')
    args = parser.parse_args()

    print("Plotting Figure 1...")
    plot_figure1(args.dir)

    print("Plotting Figure 2...")
    plot_figure2(args.dir)


if __name__ == "__main__":
    main()
