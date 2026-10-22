"""Constant potential with a nonzero Fourier mode: reaction AND diffusion."""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from polymerfts import PropagatorSolver, PlatformSelector


def main():
    nx, length, ds, potential = 64, 4.0, 0.05, 3.0
    x = (np.arange(nx) + 0.5) * length / nx
    mode = np.cos(2 * np.pi * x / length)
    initial = 1.0 + 0.2 * mode
    expected = np.exp(-potential) * (1.0 + 0.2 * np.exp(-(2 * np.pi / length)**2 / 6) * mode)
    for platform in PlatformSelector.avail_platforms():
        for method in ("rqm4", "rk2"):
            solver = PropagatorSolver(nx=[nx], lx=[length], ds=ds, bond_lengths={"A": 1.0},
                                      bc=["periodic"] * 2, chain_model="continuous",
                                      numerical_method=method, platform=platform)
            solver.add_polymer(1.0, [["A", 1.0, 0, 1]], {0: "mode"})
            solver.compute_propagators({"A": np.full(nx, potential)}, q_init={"mode": initial})
            q = solver.get_propagator(0, 0, 1, 20)
            assert np.max(np.abs(q - expected)) < 1e-12, (platform, method)
            assert abs(solver.get_partition_function(0) - np.exp(-potential)) < 1e-12
            assert solver.check_total_partition()
            solver.compute_concentrations()
            assert abs(np.mean(solver.get_concentration("A")) - 1.0) < 1e-12
    print("Test passed, analytic propagator including a nonzero Fourier mode")


if __name__ == "__main__":
    main()
