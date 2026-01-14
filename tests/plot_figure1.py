#!/usr/bin/env python3
"""
Generate Figure 1 for NumericalMethodsPerformance.md
Plots free energy convergence and execution time vs contour steps (Ns)
High-precision benchmark data (15 decimal places)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.linewidth'] = 1.2

# High-precision benchmark data (15 decimal places)
Ns_all = np.array([40, 80, 160, 320, 640, 1000])

# Free energy data (high-precision)
F_rqm4 = np.array([
    -0.477010930978363,  # N40
    -0.476977370031310,  # N80
    -0.476974360495475,  # N160
    -0.476974130350877,  # N320
    -0.476974114332803,  # N640
    -0.476974113395157,  # N1000
])

F_etdrk4 = np.array([
    -0.476935495624106,  # N40
    -0.476971516036433,  # N80
    -0.476973944733011,  # N160
    -0.476974102472743,  # N320
    -0.476974112524746,  # N640
    -0.476974113087715,  # N1000
])

F_cnadi2 = np.array([
    -0.477733629652859,  # N40
    -0.478950812741258,  # N80
    -0.479251268938022,  # N160
    -0.479326309667460,  # N320
    -0.479345088848207,  # N640
    -0.479348787593391,  # N1000
])

F_cnadi4lr = np.array([
    -0.479362551579412,  # N40
    -0.479352016719520,  # N80
    -0.479351370587765,  # N160
    -0.479351351917473,  # N320
    -0.479351354344542,  # N640
    -0.479351354862727,  # N1000
])

F_cnadi4gr = np.array([
    -0.479349596607431,  # N40
    -0.479350948955454,  # N80
    -0.479351293290286,  # N160
    -0.479351346701770,  # N320
    -0.479351354005542,  # N640
    -0.479351354805086,  # N1000
])

# SDC data (only up to N640)
Ns_sdc = np.array([40, 80, 160, 320, 640])

F_sdc4 = np.array([
    -0.479332708466214,  # N40
    -0.479349874181213,  # N80
    -0.479351240480575,  # N160
    -0.479351346767243,  # N320
    -0.479351354525801,  # N640
])

F_sdc6 = np.array([
    -0.479351355844747,  # N40
    -0.479351354185900,  # N80
    -0.479351355051352,  # N160
    -0.479351355094131,  # N320
    -0.479351355095385,  # N640
])

# Reference free energies (from highest-order converged values)
F_ref_pseudo = -0.476974113087715  # ETDRK4 at N1000
F_ref_real = -0.479351355095385    # SDC-6 at N640

# Execution time data (seconds) - from previous benchmark
T_rqm4 = np.array([4.3, 8.2, 15.7, 30.7, 60.7, 94.3])
T_etdrk4 = np.array([8.7, 16.7, 32.6, 64.7, 128.4, 199.8])
T_cnadi2 = np.array([11.8, 22.9, 45.5, 90.3, 180.1, 282.0])
T_cnadi4lr = np.array([35.4, 68.7, 136.5, 270.9, 540.3, 846.0])
T_cnadi4gr = np.array([34.6, 68.3, 135.8, 271.2, 541.2, 844.9])
T_sdc4 = np.array([48.1, 89.0, 160.5, 209.9, 350.0])  # Estimated N640
T_sdc6 = np.array([102.2, 189.4, 268.7, 471.6, 750.0])  # Estimated N640

# Calculate errors
err_rqm4 = np.abs(F_rqm4 - F_ref_pseudo)
err_etdrk4 = np.abs(F_etdrk4 - F_ref_pseudo)
err_cnadi2 = np.abs(F_cnadi2 - F_ref_real)
err_cnadi4lr = np.abs(F_cnadi4lr - F_ref_real)
err_cnadi4gr = np.abs(F_cnadi4gr - F_ref_real)
err_sdc4 = np.abs(F_sdc4 - F_ref_real)
err_sdc6 = np.abs(F_sdc6 - F_ref_real)

# Mask zero values (reference points) for plotting
def mask_zeros(x, y):
    """Return x, y arrays with zero y-values removed"""
    mask = y > 0
    return x[mask], y[mask]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Color scheme - colorblind-friendly
colors = {
    'rqm4': '#0072B2',      # blue
    'etdrk4': '#D55E00',    # vermillion
    'cnadi2': '#009E73',    # green
    'cnadi4lr': '#CC79A7',  # pink
    'cnadi4gr': '#F0E442',  # yellow
    'sdc4': '#56B4E9',      # sky blue
    'sdc6': '#E69F00',      # orange
}

markers = {
    'rqm4': 'o',
    'etdrk4': 's',
    'cnadi2': '^',
    'cnadi4lr': 'v',
    'cnadi4gr': 'D',
    'sdc4': 'p',
    'sdc6': 'h',
}

ms = 10  # marker size
lw = 2   # line width

# (a) Pseudo-spectral methods - Error vs Ns
ax = axes[0]
x, y = mask_zeros(Ns_all, err_rqm4)
ax.loglog(x, y, f'{markers["rqm4"]}-', color=colors['rqm4'],
          label='RQM4', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)
x, y = mask_zeros(Ns_all, err_etdrk4)
ax.loglog(x, y, f'{markers["etdrk4"]}-', color=colors['etdrk4'],
          label='ETDRK4', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)

# Reference line for 4th order convergence
Ns_ref = np.array([40, 1000])
ax.loglog(Ns_ref, 5e-5 * (Ns_ref/40)**(-4), 'k--', alpha=0.6, linewidth=1.5, label='4th order')

ax.set_xlabel('Contour steps ($N_s$)', fontsize=13)
ax.set_ylabel(r'$|F - F_{\rm ref}|$', fontsize=13)
ax.set_title('(a) Pseudo-spectral methods', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax.set_xlim(30, 1200)
ax.set_ylim(1e-10, 1e-4)
ax.grid(True, alpha=0.3, which='both')
ax.text(0.05, 0.05, f'$F_{{\\rm ref}}$ = {F_ref_pseudo:.15f}', transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# (b) Real-space methods - Error vs Ns
ax = axes[1]
x, y = mask_zeros(Ns_all, err_cnadi2)
ax.loglog(x, y, f'{markers["cnadi2"]}-', color=colors['cnadi2'],
          label='CN-ADI2', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)
x, y = mask_zeros(Ns_all, err_cnadi4lr)
ax.loglog(x, y, f'{markers["cnadi4lr"]}-', color=colors['cnadi4lr'],
          label='CN-ADI4-LR', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)
x, y = mask_zeros(Ns_all, err_cnadi4gr)
ax.loglog(x, y, f'{markers["cnadi4gr"]}-', color=colors['cnadi4gr'],
          label='CN-ADI4-GR', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)
x, y = mask_zeros(Ns_sdc, err_sdc4)
ax.loglog(x, y, f'{markers["sdc4"]}-', color=colors['sdc4'],
          label='SDC-4', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)
x, y = mask_zeros(Ns_sdc, err_sdc6)
ax.loglog(x, y, f'{markers["sdc6"]}-', color=colors['sdc6'],
          label='SDC-6', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)

# Reference lines
ax.loglog(Ns_ref, 2e-3 * (Ns_ref/40)**(-2), 'k:', alpha=0.6, linewidth=1.5, label='2nd order')
ax.loglog(Ns_ref, 2e-5 * (Ns_ref/40)**(-4), 'k--', alpha=0.6, linewidth=1.5, label='4th order')

ax.set_xlabel('Contour steps ($N_s$)', fontsize=13)
ax.set_ylabel(r'$|F - F_{\rm ref}|$', fontsize=13)
ax.set_title('(b) Real-space methods', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=1)
ax.set_xlim(30, 1200)
ax.set_ylim(1e-13, 5e-2)
ax.grid(True, alpha=0.3, which='both')
ax.text(0.05, 0.05, f'$F_{{\\rm ref}}$ = {F_ref_real:.15f}', transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# (c) Execution time vs Ns
ax = axes[2]
ax.loglog(Ns_all, T_rqm4, f'{markers["rqm4"]}-', color=colors['rqm4'],
          label='RQM4', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)
ax.loglog(Ns_all, T_etdrk4, f'{markers["etdrk4"]}-', color=colors['etdrk4'],
          label='ETDRK4', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)
ax.loglog(Ns_all, T_cnadi2, f'{markers["cnadi2"]}-', color=colors['cnadi2'],
          label='CN-ADI2', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)
ax.loglog(Ns_all, T_cnadi4lr, f'{markers["cnadi4lr"]}-', color=colors['cnadi4lr'],
          label='CN-ADI4-LR', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)
ax.loglog(Ns_all, T_cnadi4gr, f'{markers["cnadi4gr"]}-', color=colors['cnadi4gr'],
          label='CN-ADI4-GR', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)
ax.loglog(Ns_sdc, T_sdc4, f'{markers["sdc4"]}-', color=colors['sdc4'],
          label='SDC-4', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)
ax.loglog(Ns_sdc, T_sdc6, f'{markers["sdc6"]}-', color=colors['sdc6'],
          label='SDC-6', markersize=ms, linewidth=lw, markeredgecolor='black', markeredgewidth=0.5)

ax.set_xlabel('Contour steps ($N_s$)', fontsize=13)
ax.set_ylabel('Execution time (s)', fontsize=13)
ax.set_title('(c) Execution time', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.set_xlim(30, 1200)
ax.set_ylim(2, 1500)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()

# Save figures with higher DPI
plt.savefig('docs/figures/figure1_song2018.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('docs/figures/figure1_song2018.pdf', bbox_inches='tight')
print("Figures saved to docs/figures/figure1_song2018.png and .pdf")

# Print convergence analysis
print("\n=== Convergence Analysis ===")
print("\nPseudo-spectral (F_ref = {:.15f})".format(F_ref_pseudo))
print("RQM4:")
for i in range(len(Ns_all)):
    print(f"  N={Ns_all[i]:4d}: F={F_rqm4[i]:.15f}, err={err_rqm4[i]:.6e}")
print("ETDRK4:")
for i in range(len(Ns_all)):
    print(f"  N={Ns_all[i]:4d}: F={F_etdrk4[i]:.15f}, err={err_etdrk4[i]:.6e}")

print("\nReal-space (F_ref = {:.15f})".format(F_ref_real))
print("CN-ADI2:")
for i in range(len(Ns_all)):
    print(f"  N={Ns_all[i]:4d}: F={F_cnadi2[i]:.15f}, err={err_cnadi2[i]:.6e}")
print("CN-ADI4-LR:")
for i in range(len(Ns_all)):
    print(f"  N={Ns_all[i]:4d}: F={F_cnadi4lr[i]:.15f}, err={err_cnadi4lr[i]:.6e}")
print("CN-ADI4-GR:")
for i in range(len(Ns_all)):
    print(f"  N={Ns_all[i]:4d}: F={F_cnadi4gr[i]:.15f}, err={err_cnadi4gr[i]:.6e}")
print("SDC-4:")
for i in range(len(Ns_sdc)):
    print(f"  N={Ns_sdc[i]:4d}: F={F_sdc4[i]:.15f}, err={err_sdc4[i]:.6e}")
print("SDC-6:")
for i in range(len(Ns_sdc)):
    print(f"  N={Ns_sdc[i]:4d}: F={F_sdc6[i]:.15f}, err={err_sdc6[i]:.6e}")

plt.close()
