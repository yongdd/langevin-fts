#!/usr/bin/env python3
"""
Minimal reduced-basis safety test for CPU reduce-memory SCFT.

This test is intentionally small and fast. It exercises the reduced-basis
and reduce-memory paths together under a space group, and checks:
1. Partition function consistency.
2. Material conservation (mean total concentration ~ 1).
"""

import os
import numpy as np

# Keep CPU threading predictable in CI
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from polymerfts import scft
from polymerfts import _core


def get_platform():
    """Get available platform (cuda > cpu-mkl > cpu-fftw)."""
    available = _core.PlatformSelector.avail_platforms()
    if "cuda" in available:
        return "cuda"
    if "cpu-mkl" in available:
        return "cpu-mkl"
    return "cpu-fftw"


def main() -> None:
    np.random.seed(1234)

    params = {
        "platform": get_platform(),
        "nx": [16, 16, 16],
        "lx": [2.0, 2.0, 2.0],
        "box_is_altering": False,
        "chain_model": "continuous",
        "ds": 1 / 50,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15.0},
        "distinct_polymers": [
            {
                "volume_fraction": 1.0,
                "blocks": [
                    {"type": "A", "length": 0.3},
                    {"type": "B", "length": 0.7},
                ],
            }
        ],
        "space_group": {"symbol": "Im-3m", "number": 529},
        "reduce_memory": True,
        "optimizer": {
            "name": "am",
            "max_hist": 10,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
        "max_iter": 5,
        "tolerance": 1e-8,
        "verbose": 0,
    }

    calc = scft.SCFT(params=params)

    total_grid = calc.prop_solver.total_grid
    w_init = {
        "A": np.random.normal(0.0, 5.0, size=total_grid),
        "B": np.random.normal(0.0, 5.0, size=total_grid),
    }

    calc.run(initial_fields=w_init)

    # 1) Partition function consistency
    calc.prop_solver._propagator_computation.check_total_partition()

    # 2) Material conservation
    phi_a = calc.prop_solver.get_concentration("A")
    phi_b = calc.prop_solver.get_concentration("B")
    total_phi_mean = float(calc.prop_solver.mean(phi_a + phi_b))

    if not np.isfinite(total_phi_mean):
        raise AssertionError(f"Non-finite total concentration mean: {total_phi_mean}")
    if abs(total_phi_mean - 1.0) > 1e-6:
        raise AssertionError(
            f"Material conservation failed: mean(phi_A + phi_B)={total_phi_mean:.6e}"
        )

    print(
        "Reduced-memory space-group safety check passed: "
        f"mean(phi_A + phi_B)={total_phi_mean:.6e}"
    )


if __name__ == "__main__":
    main()
