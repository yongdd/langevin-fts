#!/usr/bin/env python
"""Quick test for SDC with direct sparse solver - 2D case for faster results."""
import os
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
from polymerfts.scft import SCFT

def run_sdc_test(N_steps, chi_n=20.0, f=0.5, nx=32, lx=3.2):
    """Run SDC SCFT and return free energy."""
    ds = 1.0 / N_steps
    params = {
        "nx": [nx, nx],
        "lx": [lx, lx],
        "chain_model": "continuous",
        "ds": ds,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": chi_n},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f, "v": 0, "u": 1},
                {"type": "B", "length": 1.0-f, "v": 1, "u": 2},
            ]
        }],
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
        "max_iter": 500,
        "tolerance": 1e-7,
        "platform": "cpu-mkl",
        "numerical_method": "sdc",
        "verbose": False,
        "box_is_altering": False,
    }
    scft = SCFT(params)

    # Initialize with lamellar-like structure
    np.random.seed(42)
    x = np.linspace(0, lx, nx, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')
    # Lamellar approximation
    w_A = chi_n * 0.3 * np.cos(2*np.pi*X/lx).flatten()
    w_B = -w_A

    scft.run(initial_fields={"A": w_A, "B": w_B})
    return scft.free_energy

# Test with different N values
print("Testing SDC convergence on CPU-MKL (2D, nx=32, direct sparse solver)...")
print("=" * 60)

N_values = [20, 40, 80]
results = []

for N in N_values:
    print(f"Running N={N}...", end=" ", flush=True)
    H = run_sdc_test(N)
    results.append((N, 1.0/N, H))
    print(f"done. H={H:.10f}")

print("=" * 60)

# Calculate convergence order
print("\nConvergence analysis:")
for i in range(1, len(results)):
    N1, ds1, H1 = results[i-1]
    N2, ds2, H2 = results[i]
    if abs(H2 - H1) > 1e-14:
        if i < len(results) - 1:
            N3, ds3, H3 = results[i+1]
            err1 = H2 - H1
            err2 = H3 - H2
            if abs(err2) > 1e-14:
                order = np.log(abs(err1/err2)) / np.log(2)
                print(f"N={N1:4d} to {N2:4d}: error ratio gives order ~ {order:.2f}")

# Final comparison with reference
if len(results) >= 2:
    H_ref = results[-1][2]
    print(f"\nFree energy convergence (relative to N={N_values[-1]}):")
    for N, ds, H in results[:-1]:
        err = abs(H - H_ref)
        print(f"  N={N:4d}: error = {err:.2e}")
