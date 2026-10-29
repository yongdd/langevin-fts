#!/usr/bin/env python3
"""A zero grand-canonical partition is regular; canonical zero Q remains invalid."""

import contextlib
import io
import sys

import numpy as np

from polymerfts import PlatformSelector, PropagatorSolver
from cuda_part import cuda_part_missing


def solve(platform, reduced, model, canonical_weight=0.0):
    w = np.array([0.0, 0.0, 1j * np.pi, -1j * np.pi])
    if model == "continuous":
        # A zero graft has Q = 0 at every contour position without cancellation.
        w = np.zeros(4, dtype=complex)
    with contextlib.redirect_stdout(io.StringIO()):
        solver = PropagatorSolver(
            nx=[4], lx=[2.0], bc=["periodic"] * 2,
            ds=1.0 if model == "discrete" else 0.1,
            bond_lengths={"A": 1.0, "B": 1.0}, chain_model=model,
            platform=platform, reduce_memory=reduced, field_type="complex",
            aggregate_propagator_computation=False,
        )
        solver.add_polymer(canonical_weight, [["A", 1.0, 0, 1]],
                           {0: "G"} if model == "continuous" else None)
        solver.add_polymer(1.0 - canonical_weight, [["B", 1.0, 0, 1]])
        solver.compute_propagators(
            {"A": w, "B": np.zeros(4, dtype=complex)},
            {"G": np.zeros(4, dtype=complex)} if model == "continuous" else None,
        )
        assert solver.get_partition_function(0) == 0.0
        if canonical_weight:
            assert not solver.check_total_partition()
            return None
        solver.compute_concentrations()
        canonical = solver.get_concentration("A")
        np.testing.assert_array_equal(canonical, np.zeros(4))
        np.testing.assert_array_equal(solver.propagator_computation.get_total_concentration(0, "A"), np.zeros(4))
        np.testing.assert_array_equal(solver.propagator_computation.get_block_concentration(0), np.zeros((1, 4)))
        np.testing.assert_allclose(solver.get_concentration("B"), 1.0, rtol=0.0, atol=2e-14)
        gce = solver.get_concentration_gce(0.4, 0, "A")
        expected = 0.4 * np.exp(-w) if model == "discrete" else np.zeros(4)
        np.testing.assert_allclose(gce, expected, rtol=2e-14, atol=2e-14)
        assert solver.check_total_partition()
        np.testing.assert_array_equal(solver.partition_errors(), np.zeros(2))
        solver.compute_stress()
        stress = np.asarray(solver.get_stress())
        stress_gce = np.asarray(solver.propagator_computation.get_stress_gce([0.4, 0.0]))
        np.testing.assert_allclose(stress, 0.0, rtol=0.0, atol=2e-14)
        np.testing.assert_allclose(stress_gce, 0.0, rtol=0.0, atol=2e-14)
    return canonical, gce, stress, stress_gce


def solve_cancellation(platform, reduced, real_offset, model):
    """Nonzero spatial integrands cancel, leaving finite contour roundoff."""
    w = real_offset + 1j * (np.array([0.0, 0.0, np.pi, np.pi]) + 0.7)
    graft = None
    if model == "continuous":
        w = np.full(4, real_offset, dtype=complex)
        graft = {"G": np.exp(0.7j) * np.array([1.0, 1.0, -1.0, -1.0])}
    with contextlib.redirect_stdout(io.StringIO()):
        solver = PropagatorSolver(
            nx=[4], lx=[2.0], bc=["periodic"] * 2,
            ds=1.0 if model == "discrete" else 0.1,
            bond_lengths={"A": 1.0, "B": 1.0}, chain_model=model,
            platform=platform, reduce_memory=reduced, field_type="complex",
        )
        solver.add_polymer(1.0, [["A", 1.0, 0, 1]])
        solver.add_polymer(0.0, [["B", 3.0, 0, 1]], {0: "G"} if graft else None)
        solver.compute_propagators({"A": np.zeros(4, dtype=complex), "B": w}, graft)
        solver.compute_concentrations()
        phi = solver.get_concentration_gce(1.0, 1, "B")
        assert np.isfinite(phi).all()
        assert solver.check_total_partition()
        errors = np.asarray(solver.partition_errors())
        assert np.isfinite(errors).all() and np.max(errors) < 1e-13
        solver.compute_stress()
        stress = np.asarray(solver.propagator_computation.get_stress_gce([0.0, 1.0]))
        assert np.isfinite(stress).all()
        partition = np.asarray(solver.get_partition_function(1))
        # The length-three chain multiplies by exp(-3*offset) under a uniform shift.
        # Scaling back also checks that the diagnostic does not impose an absolute
        # numerical floor on a small or large partition.
        scale = np.exp(3.0 * real_offset)
        assert abs(partition * scale) < 1e-14
    return partition, phi, stress, errors


def main():
    for platform in PlatformSelector.avail_platforms():
        for model in ("continuous", "discrete"):
            full = solve(platform, False, model)
            reduced = solve(platform, True, model)
            for a, b in zip(full, reduced):
                assert a.tobytes() == b.tobytes(), (platform, model, "checkpoint mismatch")
            for reduced in (False, True):
                solve(platform, reduced, model, canonical_weight=1.0)
        for model in ("continuous", "discrete"):
            reference = None
            for offset in (0.0, -2.0, 2.0):
                full = solve_cancellation(platform, False, offset, model)
                reduced = solve_cancellation(platform, True, offset, model)
                for a, b in zip(full, reduced):
                    assert a.tobytes() == b.tobytes(), (platform, model, offset, "cancellation checkpoint mismatch")
                scaled = full[1] * np.exp(3.0 * offset)
                if reference is None:
                    reference = scaled
                else:
                    np.testing.assert_allclose(scaled, reference, rtol=2e-13, atol=2e-14)
        print(f"Zero-partition GCE: {platform} passed")
    return int(cuda_part_missing())


if __name__ == "__main__":
    sys.exit(main())
