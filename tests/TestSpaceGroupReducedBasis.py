#!/usr/bin/env python3
"""
Test space group reduced basis functionality.

Tests:
1. PropagatorComputation with space group - n_grid, compute_propagators, get_total_concentration
2. ComputationBox with space group - integral, inner_product with reduced basis
3. SCFT with space group - full workflow with reduced basis
4. save_results with space group - verify full grid output
"""

import sys
import numpy as np
import scipy.io
import os
import tempfile

from polymerfts import SCFT
from polymerfts import _core


def get_platform():
    """Get available platform (cuda if available, otherwise cpu-fftw)."""
    available = _core.PlatformSelector.avail_platforms()
    if "cuda" in available:
        return "cuda"
    return "cpu-fftw"


def test_propagator_computation_reduced_basis():
    """Test PropagatorComputation with space group reduced basis."""
    print("Test 1: PropagatorComputation with space group")
    print("-" * 50)

    # Parameters
    nx = [32, 32, 32]
    lx = [4.0, 4.0, 4.0]
    bc = ["periodic"] * 6
    ds = 0.01
    bond_lengths = {"A": 1.0, "B": 1.0}

    total_grid = int(np.prod(nx))

    # Create platform factory
    platform = get_platform()
    factory = _core.PlatformSelector.create_factory(platform, False)

    # Create molecules
    molecules = factory.create_molecules_information("continuous", ds, bond_lengths)
    molecules.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])

    # Create computation box
    cb = factory.create_computation_box(nx, lx, None, bc)

    # Create space group (Im-3m)
    sg = _core.SpaceGroup(nx, "Im-3m", 529)
    n_irreducible = sg.get_n_irreducible()

    # Create optimizer and propagator computation WITH space group (passed to constructor)
    optimizer = factory.create_propagator_computation_optimizer(molecules, False)
    propagator = factory.create_propagator_computation(cb, molecules, optimizer, "rqm4", sg)

    # Test with space group
    assert propagator.get_cb().get_n_basis() == n_irreducible, f"n_grid should equal {n_irreducible} with space group"
    print(f"  With space group: n_grid = {propagator.get_cb().get_n_basis()} (n_irreducible = {n_irreducible})")

    # Test compute_propagators with reduced basis
    np.random.seed(42)
    w_A = np.random.randn(n_irreducible) * 0.1
    w_B = np.random.randn(n_irreducible) * 0.1

    propagator.compute_propagators({"A": w_A, "B": w_B})
    propagator.compute_concentrations()

    # Test get_total_concentration returns reduced basis
    phi_A = propagator.get_total_concentration("A")
    phi_B = propagator.get_total_concentration("B")

    assert len(phi_A) == n_irreducible, f"phi_A should have size {n_irreducible}"
    assert len(phi_B) == n_irreducible, f"phi_B should have size {n_irreducible}"
    print(f"  phi_A size: {len(phi_A)}, phi_B size: {len(phi_B)}")

    # Test material conservation with weighted mean
    orbit_counts = np.array(sg.get_orbit_counts())
    weighted_mean = np.sum((phi_A + phi_B) * orbit_counts) / np.sum(orbit_counts)
    assert abs(weighted_mean - 1.0) < 0.01, f"Material not conserved: {weighted_mean}"
    print(f"  Material conservation: weighted_mean(phi) = {weighted_mean:.6f}")

    # Test without space group (create separate computation box and propagator)
    cb_no_sg = factory.create_computation_box(nx, lx, None, bc)
    propagator_no_sg = factory.create_propagator_computation(cb_no_sg, molecules, optimizer, "rqm4", None)
    assert propagator_no_sg.get_cb().get_n_basis() == total_grid, "n_grid should equal total_grid without space group"
    print(f"  Without space group: n_grid = {propagator_no_sg.get_cb().get_n_basis()} (total_grid = {total_grid})")

    print("  PASSED\n")


def test_computation_box_reduced_basis():
    """Test ComputationBox integral and inner_product with reduced basis."""
    print("Test 2: ComputationBox with space group")
    print("-" * 50)

    nx = [32, 32, 32]
    lx = [4.0, 4.0, 4.0]
    bc = ["periodic"] * 6
    total_grid = int(np.prod(nx))
    volume = float(np.prod(lx))

    platform = get_platform()
    factory = _core.PlatformSelector.create_factory(platform, False)
    cb = factory.create_computation_box(nx, lx, None, bc)

    # Create space group
    sg = _core.SpaceGroup(nx, "Im-3m", 529)
    n_irreducible = sg.get_n_irreducible()

    # Set space group on computation box
    cb.set_space_group(sg)

    assert cb.get_n_basis() == n_irreducible, f"cb.get_n_basis() should be {n_irreducible}"
    print(f"  n_grid = {cb.get_n_basis()} (n_irreducible = {n_irreducible})")

    # Test integral with reduced basis (constant field = 1.0)
    field_ones = np.ones(n_irreducible)
    integral_result = cb.integral(field_ones)
    expected_integral = volume  # integral of 1 over volume = volume
    assert abs(integral_result - expected_integral) < 1e-10, f"Integral failed: {integral_result} != {expected_integral}"
    print(f"  integral(1) = {integral_result:.6f} (expected: {expected_integral})")

    # Test inner_product with reduced basis
    field_A = np.random.randn(n_irreducible)
    field_B = np.random.randn(n_irreducible)
    inner_prod = cb.inner_product(field_A, field_B)
    print(f"  inner_product computed successfully: {inner_prod:.6f}")

    # Clear space group
    cb.set_space_group(None)
    assert cb.get_n_basis() == total_grid, "n_grid should return to total_grid"
    print(f"  After clearing: n_grid = {cb.get_n_basis()}")

    print("  PASSED\n")


def test_scft_with_space_group():
    """Test SCFT full workflow with space group."""
    print("Test 3: SCFT with space group")
    print("-" * 50)

    params = {
        "platform": get_platform(),
        "nx": [32, 32, 32],
        "lx": [4.68, 4.68, 4.68],
        "chain_model": "continuous",
        "ds": 0.01,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 20.0},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": 0.2},
                {"type": "B", "length": 0.8}
            ]
        }],
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-1,
            "mix_min": 0.1,
            "mix_init": 0.1
        },
        "max_iter": 20,
        "tolerance": 1e-8,
        "box_is_altering": False,
        "space_group": {
            "symbol": "Im-3m",
            "number": 529
        }
    }

    total_grid = int(np.prod(params["nx"]))

    calc = SCFT(params)

    n_grid = calc.prop_solver.n_grid
    assert n_grid < total_grid, "n_grid should be reduced with space group"
    print(f"  n_grid = {n_grid} (reduced from {total_grid})")

    # Run SCFT
    np.random.seed(42)
    w_init = {
        "A": np.random.randn(total_grid) * 0.1,
        "B": np.random.randn(total_grid) * 0.1
    }
    calc.run(initial_fields=w_init)

    # Verify internal storage is reduced basis
    assert calc.w.shape == (2, n_grid), f"w should be (2, {n_grid})"
    assert len(calc.phi["A"]) == n_grid, f"phi['A'] should have size {n_grid}"
    print(f"  Internal w.shape = {calc.w.shape}")
    print(f"  Internal phi['A'] size = {len(calc.phi['A'])}")

    # Verify material conservation
    orbit_counts = np.array(calc.sg.get_orbit_counts())
    weighted_mean = np.sum((calc.phi["A"] + calc.phi["B"]) * orbit_counts) / np.sum(orbit_counts)
    assert abs(weighted_mean - 1.0) < 0.01, f"Material not conserved: {weighted_mean}"
    print(f"  Material conservation: {weighted_mean:.6f}")

    print("  PASSED\n")


def test_save_results_full_grid():
    """Test save_results outputs full grid even with space group."""
    print("Test 4: save_results with space group")
    print("-" * 50)

    params = {
        "platform": get_platform(),
        "nx": [32, 32, 32],
        "lx": [4.68, 4.68, 4.68],
        "chain_model": "continuous",
        "ds": 0.01,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 20.0},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": 0.2},
                {"type": "B", "length": 0.8}
            ]
        }],
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-1,
            "mix_min": 0.1,
            "mix_init": 0.1
        },
        "max_iter": 5,
        "tolerance": 1e-8,
        "box_is_altering": False,
        "space_group": {
            "symbol": "Im-3m",
            "number": 529
        }
    }

    total_grid = int(np.prod(params["nx"]))

    calc = SCFT(params)

    np.random.seed(42)
    w_init = {
        "A": np.random.randn(total_grid) * 0.1,
        "B": np.random.randn(total_grid) * 0.1
    }
    calc.run(initial_fields=w_init)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
        save_path = f.name

    try:
        calc.save_results(save_path)
        data = scipy.io.loadmat(save_path)

        # Verify saved fields are full grid
        w_A = data["w_A"].flatten()
        w_B = data["w_B"].flatten()
        phi_A = data["phi_A"].flatten()
        phi_B = data["phi_B"].flatten()

        assert len(w_A) == total_grid, f"Saved w_A should be full grid"
        assert len(w_B) == total_grid, f"Saved w_B should be full grid"
        assert len(phi_A) == total_grid, f"Saved phi_A should be full grid"
        assert len(phi_B) == total_grid, f"Saved phi_B should be full grid"
        print(f"  Saved field sizes: w_A={len(w_A)}, phi_A={len(phi_A)} (full grid={total_grid})")

        # Verify material conservation in saved data
        mean_phi = np.mean(phi_A + phi_B)
        assert abs(mean_phi - 1.0) < 0.01, f"Material not conserved in saved file: {mean_phi}"
        print(f"  Material conservation in saved file: {mean_phi:.6f}")

    finally:
        os.unlink(save_path)

    print("  PASSED\n")


def main():
    print("=" * 60)
    print("Space Group Reduced Basis Tests")
    print("=" * 60 + "\n")

    try:
        test_propagator_computation_reduced_basis()
        test_computation_box_reduced_basis()
        test_scft_with_space_group()
        test_save_results_full_grid()

        print("=" * 60)
        print("All tests PASSED")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nTest ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
