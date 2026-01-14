#!/usr/bin/env python3
"""
Generate Fig. 1 convergence plot for NumericalMethodsPerformance.md
Using hard-coded benchmark data from the documentation.
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    # Data from NumericalMethodsPerformance.md tables
    # Free Energy vs Contour Steps (Ns)
    Ns = np.array([40, 80, 160, 320, 640, 1000])

    free_energy = {
        'RQM4': np.array([-0.47701093, -0.47697737, -0.47697436, -0.47697413, -0.47697411, -0.47697411]),
        'ETDRK4': np.array([-0.47693550, -0.47697152, -0.47697394, -0.47697410, -0.47697411, -0.47697411]),
        'CN-ADI2': np.array([-0.47773363, -0.47895081, -0.47925127, -0.47932631, -0.47934509, -0.47934879]),
        'CN-ADI4-LR': np.array([-0.47936255, -0.47935202, -0.47935137, -0.47935135, -0.47935135, -0.47935135]),
        'CN-ADI4-GR': np.array([-0.47934960, -0.47935095, -0.47935129, -0.47935135, -0.47935135, -0.47935135]),
    }

    # Execution Time vs Contour Steps (seconds)
    # CN-ADI4-GR uses same Ns array as free energy data
    exec_time = {
        'RQM4': np.array([4.3, 8.2, 15.7, 30.7, 60.7, 94.3]),
        'ETDRK4': np.array([8.7, 16.7, 32.6, 64.7, 128.4, 199.8]),
        'CN-ADI2': np.array([11.8, 22.9, 45.5, 90.3, 180.1, 282.0]),
        'CN-ADI4-LR': np.array([35.4, 68.7, 136.5, 270.9, 540.3, 846.0]),
        'CN-ADI4-GR': np.array([34.6, 68.3, 135.8, 271.2, 541.2, 844.9]),
    }

    # Reference free energies
    F_ref_pseudo = -0.47697411  # Pseudo-spectral reference
    F_ref_cnadi = -0.47935135   # CN-ADI reference

    # Method styling
    styles = {
        'RQM4': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
        'ETDRK4': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-'},
        'CN-ADI2': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-'},
        'CN-ADI4-LR': {'color': '#d62728', 'marker': 'v', 'linestyle': '-'},
        'CN-ADI4-GR': {'color': '#9467bd', 'marker': 'D', 'linestyle': '--'},
    }

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Plot 1: Free energy error vs Ns (pseudo-spectral)
    ax1.set_xlabel('Ns (contour steps)', fontsize=12)
    ax1.set_ylabel('|F - F$_{ref}$|', fontsize=12)
    ax1.set_title('(a) Pseudo-Spectral Convergence', fontsize=12)

    for method in ['RQM4', 'ETDRK4']:
        errors = np.abs(free_energy[method] - F_ref_pseudo)
        # Filter out zero/tiny errors for log plot
        valid = errors > 1e-12
        if np.sum(valid) > 0:
            ax1.loglog(Ns[valid], errors[valid],
                      marker=styles[method]['marker'],
                      color=styles[method]['color'],
                      linestyle=styles[method]['linestyle'],
                      label=method,
                      markersize=8, linewidth=2)

    # Add reference line for 4th order
    ns_ref = np.array([40, 1000])
    err_ref = 4e-5 * (ns_ref / 40) ** (-4)
    ax1.loglog(ns_ref, err_ref, 'k--', alpha=0.5, label='4th order slope')

    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(30, 1200)
    ax1.set_ylim(1e-10, 1e-3)

    # Plot 2: CN-ADI convergence (including CN-ADI4-GR)
    ax2.set_xlabel('Ns (contour steps)', fontsize=12)
    ax2.set_ylabel('|F - F$_{ref}$|', fontsize=12)
    ax2.set_title('(b) Real-Space Convergence', fontsize=12)

    for method in ['CN-ADI2', 'CN-ADI4-LR', 'CN-ADI4-GR']:
        errors = np.abs(free_energy[method] - F_ref_cnadi)
        # Filter out zero/tiny errors for log plot
        valid = errors > 1e-12
        if np.sum(valid) > 0:
            ax2.loglog(Ns[valid], errors[valid],
                      marker=styles[method]['marker'],
                      color=styles[method]['color'],
                      linestyle=styles[method]['linestyle'],
                      label=method,
                      markersize=8, linewidth=2)

    # Add reference lines for 2nd and 4th order
    ns_ref = np.array([40, 1000])
    err_ref_2 = 2e-3 * (ns_ref / 40) ** (-2)
    err_ref_4 = 1.5e-5 * (ns_ref / 40) ** (-4)
    ax2.loglog(ns_ref, err_ref_2, 'k--', alpha=0.5, label='2nd order slope')
    ax2.loglog(ns_ref, err_ref_4, 'k:', alpha=0.5, label='4th order slope')

    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(30, 1200)
    ax2.set_ylim(1e-8, 1e-1)

    # Plot 3: Execution time vs Ns (including CN-ADI4-GR)
    ax3.set_xlabel('Ns (contour steps)', fontsize=12)
    ax3.set_ylabel('Time (s)', fontsize=12)
    ax3.set_title('(c) Execution Time', fontsize=12)

    for method in ['RQM4', 'ETDRK4', 'CN-ADI2', 'CN-ADI4-LR', 'CN-ADI4-GR']:
        ax3.loglog(Ns, exec_time[method],
                  marker=styles[method]['marker'],
                  color=styles[method]['color'],
                  linestyle=styles[method]['linestyle'],
                  label=method,
                  markersize=8, linewidth=2)

    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(30, 1200)

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
