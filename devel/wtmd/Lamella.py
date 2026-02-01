"""Example: Well-Tempered Metadynamics for AB diblock copolymer.

This example demonstrates how to use WTMD with L-FTS to enhance sampling
and compute free energy as a function of order parameter.

Parameters match deep-langevin-fts/examples/WTMD/Lamella.py for comparison.

Reference:
T. M. Beardsley and M. W. Matsen, J. Chem. Phys. 157, 114902 (2022).
"""

import os
import numpy as np
from scipy.io import loadmat, savemat

# Set OpenMP environment before importing polymerfts
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

from polymerfts import LFTS

# Simulation parameters (same as deep-langevin-fts)
f = 4.0/9.0     # A-fraction
eps = 1.0       # Conformational asymmetry a_A/a_B

params = {
    "nx": [40, 40, 40],
    "lx": [4.38, 4.38, 4.38],

    "chain_model": "discrete",
    "ds": 1/90,

    "segment_lengths": {
        "A": np.sqrt(eps*eps/(eps*eps*f + (1-f))),
        "B": np.sqrt(1.0/(eps*eps*f + (1-f))),
    },
    "chi_n": {"A,B": 17.148912},

    "distinct_polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": f},
            {"type": "B", "length": 1-f},
        ],
    }],

    "langevin": {
        "max_step": 5000000,    # 5 million steps for WTMD
        "dt": 8.0,
        "nbar": 10000,
    },

    "recording": {
        "dir": "data_wtmd",
        "recording_period": 10000,
        "sf_computing_period": 10,
        "sf_recording_period": 100000,
    },

    "saddle": {
        "max_iter": 100,
        "tolerance": 1e-4,
    },

    "compressor": {
        "name": "lram",
        "max_hist": 20,
        "start_error": 5e-1,
        "mix_min": 0.01,
        "mix_init": 0.01,
    },

    # WTMD parameters (same as deep-langevin-fts)
    "wtmd": {
        "ell": 4,              # ℓ-norm
        "kc": 6.02,            # Screening out frequency
        "delta_t": 5.0,        # ΔT/T
        "sigma_psi": 0.16,     # σ_Ψ
        "psi_min": 0.0,        # Ψ_min
        "psi_max": 10.0,       # Ψ_max
        "n_bins": 10000,       # Number of bins (dpsi = 1e-3)
        "update_freq": 1000,   # Update frequency
    },

    "verbose_level": 1,
}

def run_wtmd_simulation():
    """Run L-FTS with WTMD enhanced sampling."""

    # Set random seed for reproducibility
    random_seed = 12345

    # Initialize LFTS with WTMD
    simulation = LFTS(params=params, random_seed=random_seed)

    # Load initial fields (or generate random)
    try:
        input_data = loadmat("LamellaInput.mat", squeeze_me=True)
        w_A = input_data["w_A"]
        w_B = input_data["w_B"]
        print("Loaded initial fields from file.")
    except FileNotFoundError:
        print("Generating random initial fields...")
        n_grid = np.prod(params["nx"])
        w_A = np.random.randn(n_grid) * 0.1
        w_B = np.random.randn(n_grid) * 0.1

    print("Starting WTMD-enhanced L-FTS simulation...")
    print("-" * 60)

    # Run simulation
    simulation.run(initial_fields={"A": w_A, "B": w_B})

    # After simulation, get free energy from WTMD
    if simulation.wtmd is not None:
        psi_bins, free_energy = simulation.wtmd.get_free_energy()

        # Save free energy data
        np.savez("wtmd_free_energy.npz",
                 psi=psi_bins,
                 free_energy=free_energy,
                 u=simulation.wtmd.u,
                 up=simulation.wtmd.up)
        print("Free energy saved to wtmd_free_energy.npz")

if __name__ == "__main__":
    run_wtmd_simulation()
