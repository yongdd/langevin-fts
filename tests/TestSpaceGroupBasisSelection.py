#!/usr/bin/env python3
"""Basic regression test for SpaceGroup physical basis selection."""

import numpy as np

from polymerfts import _core


def expect_error(fn, contains):
    try:
        fn()
    except Exception as exc:
        msg = str(exc)
        assert contains in msg, f"Expected '{contains}' in error, got: {msg}"
        return
    raise AssertionError("Expected error, but function succeeded")


def main():
    nx = [8, 8, 8]

    # Pmmm: should allow Pmmm basis + z-mirror; M3 should fail
    hall_nums = _core.SpaceGroup.hall_numbers_from_symbol("Pmmm")
    assert len(hall_nums) > 0, "Pmmm hall numbers not found"
    sg_pmmm = _core.SpaceGroup(nx, hall_nums[0])

    sg_pmmm.enable_pmmm_physical_basis()
    assert sg_pmmm.using_pmmm_physical_basis(), "Pmmm basis not enabled"
    assert sg_pmmm.get_n_reduced_basis() == (nx[0] // 2) * (nx[1] // 2) * (nx[2] // 2)

    sg_pmmm.enable_z_mirror_physical_basis()
    assert sg_pmmm.using_z_mirror_physical_basis(), "Z-mirror basis not enabled"
    assert sg_pmmm.get_n_reduced_basis() == nx[0] * nx[1] * (nx[2] // 2)

    # Im-3m: should allow M3 basis
    sg_m3 = _core.SpaceGroup(nx, "Im-3m", 529)
    sg_m3.enable_m3_physical_basis()
    assert sg_m3.using_m3_physical_basis(), "M3 basis not enabled"
    assert sg_m3.get_n_reduced_basis() == (nx[0] // 2) * (nx[1] // 2) * (nx[2] // 2)

    # I4_132: no 3m translations
    sg_no_m3 = _core.SpaceGroup(nx, "I4_132", 510)
    expect_error(lambda: sg_no_m3.enable_m3_physical_basis(), "3m")

    # Fddd 44^3: the early checks pass but the point mapping fails (the 1/4
    # translations shift odd grid indices by 11, onto odd indices). The failed
    # enable must leave the object untouched; it used to keep the m3 size with
    # a half-filled map and the next symmetrize() corrupted the heap.
    nx44 = [44, 44, 44]
    sg_fddd = _core.SpaceGroup(nx44, "Fddd", 336)
    assert not sg_fddd.m3_crysfft_feasible(), "scft.py would not enable m3 here"
    n_before = sg_fddd.get_n_reduced_basis()
    counts_before = np.asarray(sg_fddd.get_orbit_counts()).copy()
    expect_error(lambda: sg_fddd.enable_m3_physical_basis(), "Failed to map")
    assert sg_fddd.physical_basis_name() == "irreducible", sg_fddd.physical_basis_name()
    assert sg_fddd.get_n_reduced_basis() == n_before, "basis size changed by a failed enable"
    assert np.array_equal(np.asarray(sg_fddd.get_orbit_counts()), counts_before), "orbit counts changed"
    f = np.random.default_rng(0).normal(size=(1, int(np.prod(nx44))))
    f_sym = np.asarray(sg_fddd.symmetrize(f))
    reduced = np.asarray(sg_fddd.to_reduced_basis(f_sym))
    back = np.asarray(sg_fddd.from_reduced_basis(reduced.reshape(1, -1)))
    assert np.max(np.abs(back.reshape(-1) - f_sym.reshape(-1))) < 1e-12, "round trip after a failed enable"

    # The physical basis is frozen once a solver holds the group (the computation
    # box and the solvers snapshot the reduced-basis sizes and maps)
    from polymerfts.propagator_solver import PropagatorSolver
    cpu = [p for p in _core.PlatformSelector.avail_platforms() if p.startswith("cpu")][0]
    sg_used = _core.SpaceGroup(nx, hall_nums[0])
    solver = PropagatorSolver(nx=nx, lx=[2.0, 2.0, 2.0], ds=0.1, bond_lengths={"A": 1.0},
                              bc=["periodic"] * 6, chain_model="continuous", platform=cpu,
                              space_group=sg_used)
    solver.add_polymer(1.0, [["A", 1.0, 0, 1]])
    solver.compute_propagators({"A": np.zeros(sg_used.get_n_reduced_basis())})
    assert sg_used.in_use()
    expect_error(lambda: sg_used.enable_pmmm_physical_basis(), "cannot be changed")
    del solver

    # A physical basis enabled through the C++ API must match the cell: the Pmmm
    # mirrors are no symmetry of a hexagonal cell (scft.py checks the cell first)
    def run_pmmm_basis_on_hexagonal_cell():
        sg_hex = _core.SpaceGroup(nx, hall_nums[0])
        sg_hex.enable_pmmm_physical_basis()
        s = PropagatorSolver(nx=nx, lx=[2.0, 2.0, 2.0], ds=0.1, bond_lengths={"A": 1.0},
                             bc=["periodic"] * 6, chain_model="continuous", platform=cpu,
                             angles=[90.0, 90.0, 120.0], space_group=sg_hex)
        s.add_polymer(1.0, [["A", 1.0, 0, 1]])
        s.compute_propagators({"A": np.zeros(sg_hex.get_n_reduced_basis())})
    expect_error(run_pmmm_basis_on_hexagonal_cell, "orthogonal cell")

    # Grid compatibility is checked per axis: the 6_3 screw of P6_3/mmc (1/2 along z)
    # constrains only nz, so odd in-plane sizes are accepted and an odd nz is not
    sg_odd = _core.SpaceGroup([9, 9, 8], "P6_3/mmc", 488)
    assert int(np.sum(sg_odd.get_orbit_counts())) == 9 * 9 * 8
    expect_error(lambda: _core.SpaceGroup([9, 9, 9], "P6_3/mmc", 488), "z=9 (a multiple of 2)")

    # The field conversions keep the layout of their input (1-D -> 1-D, 2-D -> 2-D)
    sg_shape = _core.SpaceGroup(nx, "Im-3m", 529)
    one = np.zeros(sg_shape.get_total_grid()); two = np.zeros((2, sg_shape.get_total_grid()))
    assert sg_shape.to_reduced_basis(one).shape == (sg_shape.get_n_reduced_basis(),)
    assert sg_shape.to_reduced_basis(two).shape == (2, sg_shape.get_n_reduced_basis())
    assert sg_shape.from_reduced_basis(sg_shape.to_reduced_basis(one)).shape == (sg_shape.get_total_grid(),)
    assert sg_shape.from_reduced_basis(sg_shape.to_reduced_basis(two)).shape == (2, sg_shape.get_total_grid())
    assert sg_shape.reduced_to_irreducible(sg_shape.to_reduced_basis(one)).shape == (sg_shape.get_n_reduced_basis_full(),)
    assert sg_shape.irreducible_to_reduced(np.zeros((2, sg_shape.get_n_reduced_basis_full()))).shape == (2, sg_shape.get_n_reduced_basis())
    assert sg_shape.symmetrize(two).shape == two.shape

    print("TestSpaceGroupBasisSelection: PASSED")


if __name__ == "__main__":
    main()
