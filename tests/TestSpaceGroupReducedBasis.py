#!/usr/bin/env python3
"""
Test space group reduced basis functionality.

Tests:
1. PropagatorComputation with space group - n_grid, compute_propagators, get_total_concentration
2. ComputationBox with space group - integral, inner_product with reduced basis
3. SCFT with space group - full workflow with reduced basis
4. save_results with space group - verify full grid output
5. A masked ComputationBox given one space group, another and none
"""

import sys
import numpy as np
import scipy.io
import os
import tempfile

from polymerfts import SCFT
from polymerfts import _core


def get_platform():
    """Get available platform (cuda > cpu-mkl > cpu-fftw)."""
    available = _core.PlatformSelector.avail_platforms()
    if "cuda" in available:
        return "cuda"
    if "cpu-mkl" in available:
        return "cpu-mkl"
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
    n_reduced = sg.get_n_reduced_basis()

    # Create optimizer and propagator computation WITH space group (passed to constructor)
    optimizer = factory.create_propagator_computation_optimizer(molecules, False)
    propagator = factory.create_propagator_computation(cb, molecules, optimizer, "rqm4", sg)

    # Test with space group
    assert propagator.get_cb().get_n_basis() == n_reduced, f"n_grid should equal {n_reduced} with space group"
    print(f"  With space group: n_grid = {propagator.get_cb().get_n_basis()} (n_reduced = {n_reduced})")

    # Test compute_propagators with reduced basis
    np.random.seed(42)
    w_A = np.random.randn(n_reduced) * 0.1
    w_B = np.random.randn(n_reduced) * 0.1

    propagator.compute_propagators({"A": w_A, "B": w_B})
    propagator.compute_concentrations()

    # Test get_total_concentration returns reduced basis
    phi_A = propagator.get_total_concentration("A")
    phi_B = propagator.get_total_concentration("B")

    assert len(phi_A) == n_reduced, f"phi_A should have size {n_reduced}"
    assert len(phi_B) == n_reduced, f"phi_B should have size {n_reduced}"
    print(f"  phi_A size: {len(phi_A)}, phi_B size: {len(phi_B)}")

    # Test material conservation with weighted mean
    orbit_counts = np.array(sg.get_orbit_counts())
    weighted_mean = np.sum((phi_A + phi_B) * orbit_counts) / np.sum(orbit_counts)
    assert abs(weighted_mean - 1.0) < 1e-10, f"Material not conserved: {weighted_mean}"
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
    n_reduced = sg.get_n_reduced_basis()

    # Set space group on computation box
    cb.set_space_group(sg)

    assert cb.get_n_basis() == n_reduced, f"cb.get_n_basis() should be {n_reduced}"
    print(f"  n_grid = {cb.get_n_basis()} (n_reduced = {n_reduced})")

    # Test integral with reduced basis (constant field = 1.0)
    field_ones = np.ones(n_reduced)
    integral_result = cb.integral(field_ones)
    expected_integral = volume  # integral of 1 over volume = volume
    assert abs(integral_result - expected_integral) < 1e-10, f"Integral failed: {integral_result} != {expected_integral}"
    print(f"  integral(1) = {integral_result:.6f} (expected: {expected_integral})")

    # Test inner_product with reduced basis
    field_A = np.random.randn(n_reduced)
    field_B = np.random.randn(n_reduced)
    inner_prod = cb.inner_product(field_A, field_B)
    # Reference: the same product on the full grid the reduced fields expand to
    expected_inner_prod = np.sum(sg.from_reduced_basis(field_A) * sg.from_reduced_basis(field_B)) * volume / total_grid
    assert abs(inner_prod - expected_inner_prod) < 1e-10, f"Inner product failed: {inner_prod} != {expected_inner_prod}"
    print(f"  inner_product = {inner_prod:.6f} (expected: {expected_inner_prod:.6f})")

    # Clear space group
    cb.set_space_group(None)
    assert cb.get_n_basis() == total_grid, "n_grid should return to total_grid"
    print(f"  After clearing: n_grid = {cb.get_n_basis()}")

    print("  PASSED\n")


def test_masked_box_changes_space_group():
    """A masked box given a space group, then another, then none, computes the
    partition function of a box built fresh for each case: the mask is reduced
    from the full grid every time (it used to stay in the first group's basis,
    so the later computations read past it). Also: a computation whose box got
    another group refuses to run, a box-size change with a group set keeps the
    full-grid volume elements, and a group built for another grid is rejected."""
    print("Test 5: masked ComputationBox, space group changed twice and cleared")
    print("-" * 50)

    nx = [16, 16, 16]
    lx = [4.0, 4.0, 4.0]
    bc = ["periodic"] * 6
    platform = get_platform()
    factory = _core.PlatformSelector.create_factory(platform, False)

    # Impenetrable spheres at the corner and the body center (BCC: symmetric
    # under Im-3m and its subgroup Pm-3m), and a field with the same symmetry
    x = (np.arange(nx[0]) + 0.5) / nx[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    def dist2(center):
        d = np.abs(np.stack([X, Y, Z]) - center)
        d = np.minimum(d, 1.0 - d)
        return np.sum(d * d, axis=0)
    mask = np.ones(nx)
    mask[dist2(0.0) < 0.2 ** 2] = 0.0
    mask[dist2(0.5) < 0.2 ** 2] = 0.0
    mask = mask.ravel()
    w_full = (0.5 * (np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y) + np.cos(4 * np.pi * Z))).ravel()

    molecules = factory.create_molecules_information("continuous", 0.05, {"A": 1.0})
    molecules.add_polymer(1.0, [["A", 1.0, 0, 1]])
    optimizer = factory.create_propagator_computation_optimizer(molecules, False)

    def partition(cb, sg):
        computation = factory.create_propagator_computation(cb, molecules, optimizer, "rqm4", sg)
        w = sg.to_reduced_basis(w_full) if sg is not None else w_full
        computation.compute_propagators({"A": w})
        return computation.get_total_partition(0)

    sg_bcc = _core.SpaceGroup(nx, "Im-3m", 529)
    sg_sc = _core.SpaceGroup(nx, "Pm-3m", 517)
    assert sg_bcc.get_n_reduced_basis() != sg_sc.get_n_reduced_basis()

    # Reference: a fresh box for every case. The mask and the field have both
    # symmetries, so all three cases describe the same system.
    q_bcc = partition(factory.create_computation_box(nx, lx, None, bc, mask), sg_bcc)
    q_sc = partition(factory.create_computation_box(nx, lx, None, bc, mask), sg_sc)
    q_none = partition(factory.create_computation_box(nx, lx, None, bc, mask), None)
    print(f"  fresh boxes: Q(Im-3m) = {q_bcc:.12e}, Q(Pm-3m) = {q_sc:.12e}, Q(none) = {q_none:.12e}")
    for name, q in (("Im-3m", q_bcc), ("Pm-3m", q_sc)):
        assert abs(q - q_none) <= 1e-12 * abs(q_none), f"Q({name}) {q} != Q without a group {q_none}"

    # One box, given Im-3m, then Pm-3m, then no group by a computation created
    # without one (the computation sets the box's group, also to none)
    cb = factory.create_computation_box(nx, lx, None, bc, mask)
    q1 = partition(cb, sg_bcc)
    q2 = partition(cb, sg_sc)
    q3 = partition(cb, None)
    print(f"  one box:     Q(Im-3m) = {q1:.12e}, Q(Pm-3m) = {q2:.12e}, Q(none) = {q3:.12e}")
    for name, q, ref in (("Im-3m", q1, q_bcc), ("Pm-3m", q2, q_sc), ("none", q3, q_none)):
        assert abs(q - ref) <= 1e-10 * abs(ref), f"Q({name}) of the reused box {q} != fresh box {ref}"
    assert cb.get_n_basis() == int(np.prod(nx))

    # A computation whose box got another group later must refuse to run
    computation_bcc = factory.create_propagator_computation(cb, molecules, optimizer, "rqm4", sg_bcc)
    factory.create_propagator_computation(cb, molecules, optimizer, "rqm4", sg_sc)
    try:
        computation_bcc.compute_propagators({"A": sg_bcc.to_reduced_basis(w_full)})
        raise AssertionError("a computation ran on a box whose space group had been changed")
    except AssertionError:
        raise
    except Exception as e:
        print(f"  a computation on a regrouped box refuses to run: {str(e).splitlines()[0][:90]}")

    # A box-size change while a group is set: the volume elements come from the
    # full-grid mask, so the box then matches a fresh box of the new size
    lx_new = [4.4, 4.4, 4.4]
    cb.set_space_group(sg_bcc)
    cb.set_lx(lx_new)
    q_resized = partition(cb, sg_bcc)
    q_resized_none = partition(cb, None)
    integral_resized = cb.integral(np.ones(int(np.prod(nx))))
    fresh = factory.create_computation_box(nx, lx_new, None, bc, mask)
    q_fresh = partition(fresh, None)
    integral_fresh = fresh.integral(np.ones(int(np.prod(nx))))
    print(f"  resized box: Q(Im-3m) = {q_resized:.12e}, Q(none) = {q_resized_none:.12e}, fresh {q_fresh:.12e}")
    assert abs(integral_resized - integral_fresh) <= 1e-12 * abs(integral_fresh), \
        f"integral of 1 after set_lx {integral_resized} != fresh box {integral_fresh}"
    for name, q in (("Im-3m", q_resized), ("none", q_resized_none)):
        assert abs(q - q_fresh) <= 1e-10 * abs(q_fresh), f"Q({name}) after set_lx {q} != fresh box {q_fresh}"

    # A group built for another grid is rejected
    try:
        cb.set_space_group(_core.SpaceGroup([8, 8, 8], "Pm-3m", 517))
        raise AssertionError("a space group for an 8x8x8 grid was accepted on a 16x16x16 box")
    except AssertionError:
        raise
    except Exception as e:
        print(f"  a space group for another grid is rejected: {str(e).splitlines()[0][:90]}")

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

    # Verify internal storage is irreducible basis when physical basis is active
    if calc.sg is not None:
        n_irreducible = calc.sg.get_n_reduced_basis_full()
    else:
        n_irreducible = n_grid

    assert calc.w.shape == (2, n_irreducible), f"w should be (2, {n_irreducible})"
    assert len(calc.phi["A"]) == n_irreducible, f"phi['A'] should have size {n_irreducible}"
    print(f"  Internal w.shape = {calc.w.shape}")
    print(f"  Internal phi['A'] size = {len(calc.phi['A'])}")

    # Verify material conservation (convert to reduced basis if needed)
    orbit_counts = np.array(calc.sg.get_orbit_counts())
    if calc.sg.get_n_reduced_basis_full() != calc.sg.get_n_reduced_basis():
        phi_A = calc.sg.irreducible_to_reduced(calc.phi["A"])
        phi_B = calc.sg.irreducible_to_reduced(calc.phi["B"])
    else:
        phi_A = calc.phi["A"]
        phi_B = calc.phi["B"]

    weighted_mean = np.sum((phi_A + phi_B) * orbit_counts) / np.sum(orbit_counts)
    assert abs(weighted_mean - 1.0) < 1e-10, f"Material not conserved: {weighted_mean}"
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
        assert abs(mean_phi - 1.0) < 1e-10, f"Material not conserved in saved file: {mean_phi}"
        print(f"  Material conservation in saved file: {mean_phi:.6f}")

    finally:
        os.unlink(save_path)

    print("  PASSED\n")


def test_save_results_m3_same_size():
    """save_results with an mmm group whose m3 basis has the same size as the
    irreducible basis (Pmmm 16^3: both N^3/8) but a different ordering. The
    irreducible -> reduced conversion must not be skipped on a size comparison,
    otherwise the saved full-grid fields are scrambled."""
    print("Test 5: save_results with the m3 basis of the same size as the irreducible basis")
    print("-" * 50)

    params = {
        "platform": get_platform(),
        "nx": [16, 16, 16],
        "lx": [3.2, 2.8, 2.4],
        "chain_model": "continuous",
        "ds": 0.01,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 20.0},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": 0.4},
                {"type": "B", "length": 0.6}
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
        "space_group": {"symbol": "Pmmm"}
    }
    total_grid = int(np.prod(params["nx"]))

    calc = SCFT(params)
    # The case this test is about: a physical basis of the same size as the irreducible one
    assert calc.sg.physical_basis_name() == "m3", f"expected the m3 basis, got {calc.sg.physical_basis_name()}"
    assert calc.sg.get_n_reduced_basis() == calc.sg.get_n_reduced_basis_full(), \
        "this test needs equal basis sizes (Pmmm 16^3: both N^3/8)"

    np.random.seed(7)
    w_init = {
        "A": np.random.randn(total_grid) * 0.1,
        "B": np.random.randn(total_grid) * 0.1
    }
    calc.run(initial_fields=w_init)

    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
        save_path = f.name
    try:
        calc.save_results(save_path)
        data = scipy.io.loadmat(save_path)
        # Reference: the full-group (irreducible-basis) instance expands calc.w / calc.phi,
        # which scft.py keeps in the irreducible basis, independently of the m3 ordering.
        # (A scrambled m3 expansion is still mirror-symmetric, so a symmetry check alone
        # cannot catch the bug; the values differ from this reference by O(1).)
        sg_full = _core.SpaceGroup(params["nx"], "Pmmm")
        irreducible = {"w_A": calc.w[0], "w_B": calc.w[1], "phi_A": calc.phi["A"], "phi_B": calc.phi["B"]}
        for name in ["w_A", "w_B", "phi_A", "phi_B"]:
            field = data[name].flatten()
            assert len(field) == total_grid, f"Saved {name} should be full grid"
            ref = np.asarray(sg_full.from_reduced_basis(np.asarray(irreducible[name]).flatten())).flatten()
            err = np.max(np.abs(field - ref))
            assert err < 1e-10, f"Saved {name} differs from the full-group expansion by {err:.3e}"
            print(f"  {name}: full grid, max |saved - full-group expansion| = {err:.2e}")
        mean_phi = np.mean(data["phi_A"].flatten() + data["phi_B"].flatten())
        assert abs(mean_phi - 1.0) < 1e-10, f"Material not conserved in saved file: {mean_phi}"
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
        test_save_results_m3_same_size()
        test_masked_box_changes_space_group()

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
