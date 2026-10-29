#!/usr/bin/env python3
"""Contour mappings preserve physical lengths across rounding and aggregation."""

import contextlib
import io
import sys

import numpy as np

from polymerfts import PlatformSelector, PropagatorSolver
from cuda_part import cuda_part_missing


def solve(platform, reduced, aggregate, ds, blocks):
    with contextlib.redirect_stdout(io.StringIO()):
        solver = PropagatorSolver(
            nx=[4], lx=[2.0], bc=["periodic"] * 2, ds=ds,
            bond_lengths={"A": 1.0, "B": 1.0}, chain_model="continuous",
            platform=platform, reduce_memory=reduced,
            aggregate_propagator_computation=aggregate,
        )
        solver.add_polymer(1.0, blocks)
        solver.compute_propagators({"A": np.zeros(4), "B": np.zeros(4)})
        solver.compute_concentrations()
        phi = np.array([solver.get_concentration(m) for m in "AB"])
        solver.compute_stress()
        stress = np.asarray(solver.get_stress())
    expected = np.array([sum(b[1] for b in blocks if b[0] == m) for m in "AB"])
    expected /= expected.sum()
    np.testing.assert_allclose(phi, np.broadcast_to(expected[:, None], phi.shape), rtol=0.0, atol=2e-13)
    assert abs(solver.mean(phi.sum(axis=0)) - 1.0) < 2e-13
    assert solver.check_total_partition()
    return np.asarray(solver.get_partition_function(0)), phi, stress


def main():
    brush = [["A", 0.4, 0, 1], ["A", 0.4, 1, 2]]
    for node in range(3):
        brush.extend([["B", 0.50000004, node, 10 + 2 * node],
                      ["A", 0.1, 10 + 2 * node, 11 + 2 * node]])
    cases = [
        (0.01, [["A", 0.495 - 2e-11, 0, 1], ["B", 0.495 + 2e-11, 1, 2]]),
        (0.001, [["A", 0.5, 0, 1], ["B", 0.50000004, 1, 2]]),
        (0.001, brush),
    ]
    for platform in PlatformSelector.avail_platforms():
        for ds, blocks in cases:
            for aggregate in (False, True):
                full = solve(platform, False, aggregate, ds, blocks)
                reduced = solve(platform, True, aggregate, ds, blocks)
                for a, b in zip(full, reduced):
                    assert a.tobytes() == b.tobytes(), (platform, aggregate, "checkpoint mismatch")
        print(f"Contour precision: {platform} passed")
    return int(cuda_part_missing())


if __name__ == "__main__":
    sys.exit(main())
