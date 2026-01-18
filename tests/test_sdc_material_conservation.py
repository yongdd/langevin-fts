#!/usr/bin/env python
"""
Test material conservation for SDC solver.

Material conservation requires that the forward and backward partition functions
are equal: Q_forward = Q_backward. This test checks this property for 1D, 2D, and 3D
cases using various boundary conditions.

The fully implicit SDC scheme should provide better material conservation than
the original IMEX scheme, especially in 1D where the tridiagonal solve is exact.
"""

import os
import sys
import numpy as np

# Set OpenMP settings before importing the module
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

from polymerfts import PropagatorSolver

def test_material_conservation(platform, nx, lx, bc, ds, method="sdc-3", field_std=5.0):
    """
    Test material conservation for a given configuration.

    Returns True if the check_total_partition() passes (forward == backward within tolerance).
    """
    dim = len(nx)
    n_grid = int(np.prod(nx))

    # Generate random fields with specified standard deviation
    np.random.seed(42)
    w_A = np.random.normal(0.0, field_std, n_grid)
    w_B = -w_A  # Incompressibility

    # Create solver
    solver = PropagatorSolver(
        nx=nx, lx=lx,
        ds=ds,
        bond_lengths={"A": 1.0, "B": 1.0},
        bc=bc,
        chain_model="continuous",
        numerical_method=method,
        platform=platform,
        use_checkpointing=False
    )

    # Add AB diblock polymer
    solver.add_polymer(
        volume_fraction=1.0,
        blocks=[["A", 0.5, 0, 1], ["B", 0.5, 1, 2]]
    )

    # Compute propagators
    solver.compute_propagators({"A": w_A, "B": w_B})

    # Access internal solver to check partition function
    # This returns True if all forward and backward partition functions match
    partition_match = solver._propagator_computation.check_total_partition()

    # Get partition function
    Q_total = solver.get_partition_function(0)

    return {
        'partition_match': partition_match,
        'Q_total': Q_total
    }


def main():
    print("=" * 70)
    print("Testing Material Conservation for SDC Solver")
    print("=" * 70)
    print()

    # Test configurations
    tests = [
        # 1D tests
        {"name": "1D Periodic", "nx": [64], "lx": [4.0],
         "bc": ["periodic", "periodic"], "ds": 1/64},
        {"name": "1D Reflecting", "nx": [64], "lx": [4.0],
         "bc": ["reflecting", "reflecting"], "ds": 1/64},
        {"name": "1D Absorbing", "nx": [64], "lx": [4.0],
         "bc": ["absorbing", "absorbing"], "ds": 1/64},
        # 2D tests
        {"name": "2D Periodic", "nx": [32, 32], "lx": [4.0, 4.0],
         "bc": ["periodic", "periodic", "periodic", "periodic"], "ds": 1/32},
        {"name": "2D Mixed", "nx": [32, 32], "lx": [4.0, 4.0],
         "bc": ["reflecting", "reflecting", "absorbing", "absorbing"], "ds": 1/32},
        # 3D test
        {"name": "3D Periodic", "nx": [16, 16, 16], "lx": [4.0, 4.0, 4.0],
         "bc": ["periodic", "periodic", "periodic", "periodic", "periodic", "periodic"], "ds": 1/16},
    ]

    # Methods to test
    methods = ["sdc-3", "sdc-5", "cn-adi2"]

    # Platforms to test
    platforms = ["cpu-mkl"]
    try:
        # Check if CUDA is available
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        if result.returncode == 0:
            platforms.append("cuda")
    except:
        pass

    print("Testing methods:", methods)
    print("Testing platforms:", platforms)
    print()

    all_passed = True

    for test in tests:
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print(f"Grid: {test['nx']}, Box: {test['lx']}")
        print(f"BC: {test['bc']}, ds: {test['ds']}")
        print(f"{'='*60}")

        for method in methods:
            for platform in platforms:
                try:
                    result = test_material_conservation(
                        platform=platform,
                        nx=test['nx'],
                        lx=test['lx'],
                        bc=test['bc'],
                        ds=test['ds'],
                        method=method,
                        field_std=5.0
                    )

                    status = "PASS" if result['partition_match'] else "FAIL"
                    if not result['partition_match']:
                        all_passed = False

                    print(f"  {method:8s} {platform:8s}: Q={result['Q_total']:.6f} [{status}]")

                except Exception as e:
                    print(f"  {method:8s} {platform:8s}: ERROR - {e}")
                    all_passed = False

    print()
    print("=" * 70)
    if all_passed:
        print("All material conservation tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
