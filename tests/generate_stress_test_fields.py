#!/usr/bin/env python3
"""
Generate converged field data for stress tests.
Run this once to create input files that speed up stress tests.
"""

import os
import sys
import numpy as np

# OpenMP settings
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

from polymerfts import scft

def generate_2d_fields():
    """Generate converged fields for 2D stress test."""
    print("=" * 60)
    print("Generating 2D stress test fields")
    print("=" * 60)

    # Parameters matching TestStressLinear2D.cpp
    f = 0.25
    chi_n = 20.0
    nx = [33, 29]
    lx = [3.2, 4.1]
    gamma = 115.0
    ds = 1.0/100

    M = nx[0] * nx[1]

    for chain_model in ["discrete", "continuous"]:
        print(f"\nChain model: {chain_model}")

        # Initialize with 2D pattern
        w_A = np.zeros(nx, dtype=np.float64)
        w_B = np.zeros(nx, dtype=np.float64)
        for i in range(nx[0]):
            xx = (i + 0.5) * 2 * np.pi / nx[0]
            for j in range(nx[1]):
                yy = (j + 0.5) * 2 * np.pi / nx[1]
                phi_a_init = f + 0.2 * np.cos(xx) + 0.2 * np.cos(yy)
                phi_b_init = 1.0 - phi_a_init
                w_A[i, j] = chi_n * phi_b_init
                w_B[i, j] = chi_n * phi_a_init

        # Zero mean
        w_A -= np.mean(w_A)
        w_B -= np.mean(w_B)

        # SCFT parameters
        params = {
            "nx": nx,
            "lx": lx,
            "angles": [gamma],
            "box_is_altering": False,
            "chain_model": chain_model,
            "ds": ds,
            "segment_lengths": {"A": 1.0, "B": 1.0},
            "chi_n": {"A,B": chi_n},
            "distinct_polymers": [{
                "volume_fraction": 1.0,
                "blocks": [
                    {"type": "A", "length": f},
                    {"type": "B", "length": 1.0-f},
                ],
            }],
            "optimizer": {
                "name": "am",
                "max_hist": 20,
                "start_error": 1e-2,
                "mix_min": 0.02,
                "mix_init": 0.02,
            },
            "max_iter": 2000,
            "tolerance": 1e-9,
        }

        # Run SCFT
        solver = scft.SCFT(params=params)
        solver.run(initial_fields={"A": w_A, "B": w_B})

        # Get converged fields (solver.w is indexed by monomer type order)
        # monomer_types are in order: ["A", "B"]
        w_A_conv = solver.w[0].flatten()
        w_B_conv = solver.w[1].flatten()

        # Save to file (in build directory where tests run)
        if chain_model == "continuous":
            filename = "Stress2D_ContinuousInput.txt"
        else:
            filename = "Stress2D_DiscreteInput.txt"

        # Save to tests directory (where stress tests run from)
        tests_dir = os.path.dirname(__file__)
        output_path = os.path.join(tests_dir, filename)
        with open(output_path, 'w') as f_out:
            for val in w_A_conv:
                f_out.write(f"{val:.10e}\n")
            for val in w_B_conv:
                f_out.write(f"{val:.10e}\n")
        print(f"Saved: {output_path}")


def generate_3d_fields():
    """Generate converged fields for 3D stress test."""
    print("=" * 60)
    print("Generating 3D stress test fields")
    print("=" * 60)

    # Parameters matching TestStressLinear3D.cpp
    f = 0.36
    chi_n = 20.0
    nx = [23, 27, 25]
    lx = [3.2, 3.3, 3.4]
    angles = [83.0, 95.0, 98.0]  # alpha, beta, gamma
    ds = 1.0/100

    M = nx[0] * nx[1] * nx[2]

    for chain_model in ["discrete", "continuous"]:
        print(f"\nChain model: {chain_model}")

        # Initialize with gyroid-like pattern
        w_A = np.zeros(nx, dtype=np.float64)
        w_B = np.zeros(nx, dtype=np.float64)
        for i in range(nx[0]):
            xx = (i + 1) * 2 * np.pi / nx[0]
            for j in range(nx[1]):
                yy = (j + 1) * 2 * np.pi / nx[1]
                for k in range(nx[2]):
                    zz = (k + 1) * 2 * np.pi / nx[2]
                    c1 = np.sqrt(8.0/3.0) * (np.cos(xx)*np.sin(yy)*np.sin(2*zz) +
                                              np.cos(yy)*np.sin(zz)*np.sin(2*xx) +
                                              np.cos(zz)*np.sin(xx)*np.sin(2*yy))
                    c2 = np.sqrt(4.0/3.0) * (np.cos(2*xx)*np.cos(2*yy) +
                                              np.cos(2*yy)*np.cos(2*zz) +
                                              np.cos(2*zz)*np.cos(2*xx))
                    w_A[i, j, k] = -0.3164 * c1 + 0.1074 * c2
                    w_B[i, j, k] = 0.3164 * c1 - 0.1074 * c2

        # Zero mean
        w_A -= np.mean(w_A)
        w_B -= np.mean(w_B)

        # SCFT parameters
        params = {
            "nx": nx,
            "lx": lx,
            "angles": angles,
            "box_is_altering": False,
            "chain_model": chain_model,
            "ds": ds,
            "segment_lengths": {"A": 1.0, "B": 1.0},
            "chi_n": {"A,B": chi_n},
            "distinct_polymers": [{
                "volume_fraction": 1.0,
                "blocks": [
                    {"type": "A", "length": f},
                    {"type": "B", "length": 1.0-f},
                ],
            }],
            "optimizer": {
                "name": "am",
                "max_hist": 20,
                "start_error": 1e-1,
                "mix_min": 0.1,
                "mix_init": 0.1,
            },
            "max_iter": 500,
            "tolerance": 1e-9,
        }

        # Run SCFT
        solver = scft.SCFT(params=params)
        solver.run(initial_fields={"A": w_A, "B": w_B})

        # Get converged fields (solver.w is indexed by monomer type order)
        # monomer_types are in order: ["A", "B"]
        w_A_conv = solver.w[0].flatten()
        w_B_conv = solver.w[1].flatten()

        # Save to file (in build directory where tests run)
        if chain_model == "continuous":
            filename = "Stress3D_ContinuousInput.txt"
        else:
            filename = "Stress3D_DiscreteInput.txt"

        # Save to tests directory (where stress tests run from)
        tests_dir = os.path.dirname(__file__)
        output_path = os.path.join(tests_dir, filename)
        with open(output_path, 'w') as f_out:
            for val in w_A_conv:
                f_out.write(f"{val:.10e}\n")
            for val in w_B_conv:
                f_out.write(f"{val:.10e}\n")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    generate_2d_fields()
    generate_3d_fields()
    print("\nDone! Field files generated in tests/ directory.")
