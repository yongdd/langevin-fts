#!/usr/bin/env python3
"""Basis lifetime, degenerate grids, ensemble normalization and computation-state regressions."""

import numpy as np
from polymerfts import _core, PropagatorSolver


def rejects(call, message=None):
    try:
        call()
    except (RuntimeError, ValueError) as exc:
        if message is not None:
            assert message in str(exc), (message, str(exc))
        return
    raise AssertionError("Invalid operation was accepted")


def molecules(factory, model="continuous", fraction=1.0):
    mol = factory.create_molecules_information(model, 0.05, {"A": 1.0, "B": 1.0})
    mol.add_polymer(fraction, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
    return mol


def test_basis_lifetime(platform, reduced):
    fac = _core.PlatformSelector.create_factory(platform, reduced, "real")
    nx, n = [8]*3, 512
    mol = molecules(fac)
    opt = fac.create_propagator_computation_optimizer(mol, False)
    cb = fac.create_computation_box(nx, [3.0]*3)
    first = fac.create_propagator_computation(cb, mol, opt, "rqm4")
    rng = np.random.default_rng(32)
    first.compute_propagators({m: rng.normal(0, 5, n) for m in "AB"})
    first.compute_concentrations()
    assert first.check_total_partition()
    assert abs(np.mean(sum(first.get_total_concentration(m) for m in "AB")) - 1) < 1e-10
    sg = _core.SpaceGroup(nx, "Im-3m", 529)
    cb.set_space_group(sg)
    for call in (lambda: first.get_total_concentration("A"),
                 lambda: first.get_total_concentration(0, "A"),
                 lambda: first.get_total_concentration_gce(1.0, 0, "A"),
                 lambda: first.get_block_concentration(0),
                 lambda: first.get_chain_propagator(0, 0, 1, 2),
                 first.compute_concentrations, first.compute_stress,
                 first.check_total_partition, first.update_laplacian_operator):
        rejects(call)

    second = fac.create_propagator_computation(cb, mol, opt, "rqm4", sg)
    third = fac.create_propagator_computation(cb, mol, opt, "rqm4")
    rejects(lambda: second.get_total_concentration("A"))
    # Restoring the original box basis makes its already computed buffers usable.
    assert np.all(np.isfinite(first.get_total_concentration("A")))
    assert third.get_cb().get_n_basis() == n


def test_failed_construction(platform):
    fac = _core.PlatformSelector.create_factory(platform, False, "real")
    mol = molecules(fac)
    opt = fac.create_propagator_computation_optimizer(mol, False)
    cb = fac.create_computation_box([8]*3, [3.0]*3, [90.0, 90.0, 80.0])
    previous = _core.SpaceGroup([8]*3, "P1", 1)
    replacement = _core.SpaceGroup([8]*3, "P1", 1)
    for old in (None, previous):
        cb.set_space_group(old)
        # The factory rejects cn-adi2 on an oblique cell before it attaches the group;
        # the rollback after an attached group fails is checked with nx = 1 below.
        rejects(lambda: fac.create_propagator_computation(cb, mol, opt, "cn-adi2", replacement),
                "cn-adi2 requires an orthogonal computation box")
        assert cb.get_space_group() is old
        assert not replacement.in_use()
        assert cb.get_n_basis() == 512
        assert abs(cb.integral(np.ones(512)) - cb.get_volume()) < 1e-10
    complex_fac = _core.PlatformSelector.create_factory(platform, False, "complex")
    complex_cb = complex_fac.create_computation_box([8]*3, [3.0]*3)
    rejects(lambda: complex_fac.create_propagator_computation(complex_cb, mol, opt, "rqm4", replacement))
    assert complex_cb.get_space_group() is None and not replacement.in_use()

    # Rollback must preserve device integration weights as well as the host
    # maps: these groups have different basis sizes, and CN-ADI rejects nx=1.
    cb = fac.create_computation_box([1, 8, 8], [3.0]*3)
    previous = _core.SpaceGroup([1, 8, 8], "P1", 1)
    replacement = _core.SpaceGroup([1, 8, 8], "Pmmm", 227)
    pc = fac.create_propagator_computation(cb, mol, opt, "rqm4", previous)
    w = {m: np.random.default_rng(36).normal(0, 5, 64) for m in "AB"}
    pc.compute_propagators(w)
    reference = pc.get_total_partition(0)
    rejects(lambda: fac.create_propagator_computation(cb, mol, opt, "cn-adi2", replacement))
    assert cb.get_space_group() is previous and not replacement.in_use()
    pc.compute_propagators(w)
    np.testing.assert_allclose(pc.get_total_partition(0), reference, rtol=1e-12)


def test_method_checks(platform, reduced):
    # The factory checks the numerical method once for every platform; each case
    # names its reason (a backend left without a solver would crash instead)
    mol = molecules(_core.PlatformSelector.create_factory(platform, reduced, "real"))
    for field_type, angles, method, message in (
            ("complex", [90.0]*3, "cn-adi2", "cn-adi2 does not support complex fields"),
            ("real", [90.0]*3, "rk4", "Unknown numerical method 'rk4'"),
            ("real", [90.0, 90.0, 80.0], "cn-adi2", "cn-adi2 requires an orthogonal computation box")):
        fac = _core.PlatformSelector.create_factory(platform, reduced, field_type)
        opt = fac.create_propagator_computation_optimizer(mol, False)
        cb = fac.create_computation_box([8]*3, [3.0]*3, angles)
        rejects(lambda: fac.create_propagator_computation(cb, mol, opt, method), message)


def test_gce(platform, reduced):
    fac = _core.PlatformSelector.create_factory(platform, reduced, "real")
    rng = np.random.default_rng(33)
    w = {m: rng.normal(0, 5, 16) for m in "AB"}
    for model in ("continuous", "discrete"):
        results = []
        for fraction in (0.0, 1.0):
            mol = molecules(fac, model, fraction)
            opt = fac.create_propagator_computation_optimizer(mol, True)
            cb = fac.create_computation_box([16], [4.0])
            pc = fac.create_propagator_computation(cb, mol, opt, "rqm4")
            pc.compute_propagators(w)
            pc.compute_concentrations()
            assert pc.check_total_partition()
            canonical = sum(pc.get_total_concentration(m) for m in "AB")
            assert abs(np.mean(canonical) - fraction) < 1e-10
            total = sum(pc.get_total_concentration_gce(2.0, 0, m) for m in "AB")
            assert np.all(np.isfinite(total))
            assert abs(np.mean(total) - 2.0*pc.get_total_partition(0)) < 1e-9
            results.append(total)
        np.testing.assert_allclose(*results, rtol=0, atol=1e-10)


def test_input_lists(platform, reduced):
    fac = _core.PlatformSelector.create_factory(platform, reduced, "real")
    # Volume fractions are finite and in [0, 1]
    for fraction in (np.nan, np.inf, -0.5, 1.5):
        mol = fac.create_molecules_information("continuous", 0.05, {"A": 1.0, "S": 1.0})
        rejects(lambda: mol.add_polymer(fraction, [["A", 1.0, 0, 1]]), "volume_fraction")
        rejects(lambda: mol.add_solvent(fraction, "S"), "volume_fraction")

    # One fugacity per polymer type: a short list must not read past its end
    mol = fac.create_molecules_information("continuous", 0.05, {"A": 1.0, "B": 1.0})
    mol.add_polymer(0.6, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
    mol.add_polymer(0.4, [["A", 1.0, 0, 1]])
    opt = fac.create_propagator_computation_optimizer(mol, False)
    cb = fac.create_computation_box([16], [4.0])
    pc = fac.create_propagator_computation(cb, mol, opt, "rqm4")
    rng = np.random.default_rng(41)
    pc.compute_propagators({m: rng.normal(0, 5, 16) for m in "AB"})
    pc.compute_stress()
    for fugacities in ([], [1.0], [1.0, np.nan], [1.0, -1.0]):
        rejects(lambda: pc.get_stress_gce(fugacities), "fugacit")
    assert np.all(np.isfinite(pc.get_stress_gce([1.0, 2.0])))

    # A solvent field left out of a later call is not taken from the earlier one
    mol = fac.create_molecules_information("continuous", 0.05, {"A": 1.0, "S": 1.0})
    mol.add_polymer(0.7, [["A", 1.0, 0, 1]])
    mol.add_solvent(0.3, "S")
    opt = fac.create_propagator_computation_optimizer(mol, False)
    cb = fac.create_computation_box([16], [4.0])
    pc = fac.create_propagator_computation(cb, mol, opt, "rqm4")
    w = {m: rng.normal(0, 5, 16) for m in "AS"}
    pc.compute_propagators(w)
    pc.compute_concentrations()
    assert abs(np.mean(pc.get_total_concentration("A") + pc.get_total_concentration("S")) - 1.0) < 1e-10
    pc.compute_propagators({"A": w["A"]})
    rejects(pc.compute_concentrations, 'monomer_type "S" of solvent 0 is not in w_input')


def test_angles_and_grafting(platform):
    fac = _core.PlatformSelector.create_factory(platform, False, "real")
    for nx, angles in (([16], [90, 90, 60]), ([8, 8], [120, 90, 90])):
        rejects(lambda: fac.create_computation_box(nx, [3.0]*len(nx), angles))
        cb = fac.create_computation_box(nx, [3.0]*len(nx))
        rejects(lambda: cb.set_lattice_parameters([3.0]*len(nx), angles))
        assert cb.is_orthogonal()
    for graft in ({7: "G"}, {1: "G"}):
        mol = fac.create_molecules_information("continuous", 0.05, {"A": 1.0})
        rejects(lambda: mol.add_polymer(1.0, [["A", 0.5, 0, 1], ["A", 0.5, 1, 2]], graft))


def test_grafting_conversions(platform, reduced):
    # The binding must retain dtype/contiguity conversion buffers until both
    # full-storage and checkpoint computations finish reading the graft input.
    rng = np.random.default_rng(37)
    n = 128
    for field_type, small, wide in (("real", np.float32, np.float64),
                                   ("complex", np.complex64, np.complex128)):
        graft = rng.uniform(0.5, 1.5, n).astype(small)
        if field_type == "complex":
            graft += 0.1j*rng.normal(size=n).astype(small)
        reference = graft.astype(wide)
        strided = np.column_stack((reference, reference))[:, 0]
        assert not strided.flags.c_contiguous
        w = rng.normal(0, 5, n).astype(wide)
        for model in ("continuous", "discrete"):
            outputs = []
            for value in (reference, graft, strided):
                s = PropagatorSolver([n], [4.0], 0.05, {"A": 1.0}, ["periodic"]*2,
                                     model, platform=platform, reduce_memory=reduced,
                                     field_type=field_type)
                s.add_polymer(1.0, [["A", 1.0, 0, 1]], {0: "G"})
                s.compute_propagators({"A": w}, q_init={"G": value})
                s.compute_concentrations()
                assert s.check_total_partition()
                phi = s.get_concentration("A")
                assert abs(s.mean(phi) - 1) < 1e-10
                outputs.append((s.get_partition_function(0), phi))
            for partition, phi in outputs[1:]:
                np.testing.assert_allclose(partition, outputs[0][0], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(phi, outputs[0][1], rtol=1e-12, atol=1e-12)


def test_cnadi_lattice_update(platform, reduced):
    s = PropagatorSolver([8, 8], [4.0, 4.0], 0.05, {"A": 1.0}, ["periodic"]*4,
                         "continuous", numerical_method="cn-adi2", platform=platform,
                         reduce_memory=reduced)
    s.add_polymer(1.0, [["A", 1.0, 0, 1]])
    w = {"A": np.random.default_rng(38).normal(0, 5, 64)}
    s.compute_propagators(w)
    partition = s.get_partition_function(0)
    # Reject before changing either the box or the solver's cached lengths.
    rejects(lambda: s.set_lattice_parameters([5.0, 6.0], [90, 90, 60]))
    np.testing.assert_allclose(s.get_lx(), [4.0, 4.0], rtol=0, atol=0)
    np.testing.assert_allclose(s.lx, [4.0, 4.0], rtol=0, atol=0)
    np.testing.assert_allclose(s.get_angles_degrees(), [90.0]*3, rtol=0, atol=1e-12)
    s.compute_propagators(w)
    np.testing.assert_allclose(s.get_partition_function(0), partition, rtol=1e-12)
    s.set_lattice_parameters([5.0, 6.0], [90, 90, 90])
    s.compute_propagators(w)
    s.compute_concentrations()
    assert s.check_total_partition()
    assert abs(s.mean(s.get_concentration("A")) - 1) < 1e-10


def test_pmmm_singleton(platform):
    fac = _core.PlatformSelector.create_factory(platform, False, "real")
    nx = [2, 8, 8]
    sg = _core.SpaceGroup(nx, "Pmmm", 227)
    full = sg.symmetrize(np.random.default_rng(20).normal(0, 5, 128))
    sg.enable_pmmm_physical_basis()
    values = []
    for group in (None, sg):
        cb = fac.create_computation_box(nx, [2.0]*3)
        mol = molecules(fac)
        opt = fac.create_propagator_computation_optimizer(mol, False)
        pc = fac.create_propagator_computation(cb, mol, opt, "rqm4", group)
        w = full if group is None else sg.to_reduced_basis(full)
        pc.compute_propagators({"A": w, "B": -w})
        pc.compute_concentrations()
        assert pc.check_total_partition()
        assert abs(cb.mean(sum(pc.get_total_concentration(m) for m in "AB")) - 1) < 1e-10
        values.append(pc.get_total_partition(0))
    np.testing.assert_allclose(*values, rtol=1e-11, atol=1e-11)


def test_box_update_and_cnadi(platform):
    def solver(length, method="rqm4", ds=0.05):
        s = PropagatorSolver([16], [length], ds, {"A": 1.0}, ["periodic"]*2,
                             "continuous", numerical_method=method, platform=platform)
        s.add_polymer(1.0, [["A", 1.0, 0, 1]])
        return s
    w = {"A": np.random.default_rng(34).normal(0, 5, 16)}
    old, fresh = solver(4.0), solver(8.0)
    old.compute_propagators(w)
    old.set_lx([8.0])
    old.compute_propagators(w)
    fresh.compute_propagators(w)
    np.testing.assert_allclose(old.get_partition_function(0), fresh.get_partition_function(0), rtol=1e-12)
    s = solver(1600.0, "cn-adi2", 0.001)
    s.compute_propagators(w)
    s.compute_concentrations()
    assert s.check_total_partition()
    assert abs(s.mean(s.get_concentration("A")) - 1) < 1e-10


def test_native_guards(platform, reduced):
    fac = _core.PlatformSelector.create_factory(platform, reduced, "real")
    mol = molecules(fac)
    opt = fac.create_propagator_computation_optimizer(mol, False)
    cb = fac.create_computation_box([8]*3, [3.0]*3)
    pc = fac.create_propagator_computation(cb, mol, opt, "rqm4")
    # The computation sized its arrays for these molecules
    rejects(lambda: mol.add_solvent(0.1, "A"), "Cannot add a solvent")
    rejects(lambda: mol.add_polymer(0.1, [["A", 1.0, 0, 1]]), "Cannot add a polymer")
    assert mol.get_n_solvent_types() == 0 and mol.get_n_polymer_types() == 1

    # Results exist only for the fields and the box they were computed for
    not_computed = "Propagators not computed"
    for call in (lambda: pc.get_total_partition(0), lambda: pc.get_chain_propagator(0, 0, 1, 2),
                 pc.compute_concentrations, pc.compute_stress, pc.compute_partition_errors):
        rejects(call, not_computed)
    rng = np.random.default_rng(39)
    w = {m: rng.normal(0, 5, 512) for m in "AB"}
    pc.compute_propagators(w)
    rejects(lambda: pc.get_total_concentration("A"), "Concentrations not computed")
    rejects(pc.get_stress, "Stress not computed")
    pc.compute_concentrations()
    pc.compute_stress()
    assert np.all(np.isfinite(pc.get_total_concentration("A"))) and np.all(np.isfinite(pc.get_stress()))
    cb.set_lx([3.5]*3)
    pc.update_laplacian_operator()
    for call in (lambda: pc.get_total_partition(0), lambda: pc.get_total_concentration("A"), pc.get_stress):
        rejects(call, not_computed)
    pc.compute_propagators(w)
    assert pc.check_total_partition()
    rejects(lambda: pc.compute_propagators({"A": w["A"]}), 'monomer_type "B"')
    rejects(lambda: pc.get_total_partition(0), not_computed)

    # Pseudo-spectral methods need every direction periodic or every direction non-periodic
    mixed = fac.create_computation_box(nx=[8]*3, lx=[3.0]*3, bc=["periodic"]*2 + ["reflecting"]*4)
    rejects(lambda: fac.create_propagator_computation(mixed, mol, opt, "rqm4"), "cannot mix periodic")
    fac.create_propagator_computation(mixed, mol, opt, "cn-adi2")

    # A cn-adi2 computation keeps its box orthogonal while it exists
    box = fac.create_computation_box(nx=[8]*3, lx=[3.0]*3, bc=["periodic"]*6)
    cn = fac.create_propagator_computation(box, mol, opt, "cn-adi2")
    rejects(lambda: box.set_lattice_parameters([3.0]*3, [90.0, 90.0, 80.0]), "requires an orthogonal cell")
    np.testing.assert_allclose(box.get_angles_degrees(), [90.0]*3, rtol=0, atol=1e-12)
    del cn
    box.set_lattice_parameters([3.0]*3, [90.0, 90.0, 80.0])


def test_grid_geometry(platform):
    # Coordinates and the cell volume follow the lattice vectors of the box
    s = PropagatorSolver([8, 8], [4.0, 3.0], 0.05, {"A": 1.0}, ["periodic"]*4, "continuous",
                         platform=platform, angles=[90.0, 90.0, 60.0])
    x, y = s.get_grid_points()
    assert x.shape == (8, 8) and y.shape == (8, 8)
    np.testing.assert_allclose(s.get_dv(), 4.0*3.0*np.sin(np.pi/3)/64, rtol=1e-14)
    np.testing.assert_allclose([x[1, 0] - x[0, 0], y[1, 0] - y[0, 0]], [0.5, 0.0], rtol=0, atol=1e-14)
    np.testing.assert_allclose([x[0, 1] - x[0, 0], y[0, 1] - y[0, 0]],
                               [3/8*np.cos(np.pi/3), 3/8*np.sin(np.pi/3)], rtol=0, atol=1e-14)
    np.testing.assert_allclose([x[0, 0], y[0, 0]], [0.25 + 3/16*np.cos(np.pi/3), 3/16*np.sin(np.pi/3)],
                               rtol=0, atol=1e-14)
    # The default Gaussian sits at the center of the cell, half the sum of the lattice vectors
    center = [0.5*(4.0 + 3.0*np.cos(np.pi/3)), 0.5*3.0*np.sin(np.pi/3)]
    np.testing.assert_array_equal(s.gaussian_initial_condition(), s.gaussian_initial_condition(center=center))


def test_n_grid(platform):
    # Fields have the reduced-basis size already before the first computation
    sg = _core.SpaceGroup([8]*3, "Im-3m", 529)
    s = PropagatorSolver([8]*3, [3.0]*3, 0.05, {"A": 1.0, "B": 1.0}, ["periodic"]*6, "continuous",
                         platform=platform, space_group=sg)
    s.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
    n = s.n_grid
    assert n == sg.get_n_reduced_basis() and n < s.total_grid, (n, s.total_grid)
    w = s.to_reduced_basis(sg.symmetrize(np.random.default_rng(40).normal(0, 5, 512)))
    assert w.size == n
    s.compute_propagators({"A": w, "B": -w})
    assert s.n_grid == n and s.check_total_partition()


def main():
    for nx in ([0, 8, 8], [-8, -8, 8], [2**31-1]*3):
        rejects(lambda: _core.SpaceGroup(nx, "P1", 1))
    field = np.random.default_rng(35).normal(0, 5, 512)
    for symbol, enable in (("Im-3m", "enable_m3_physical_basis"),
                           ("Im-3m", "enable_pmmm_physical_basis"),
                           ("P6/mmm", "enable_z_mirror_physical_basis")):
        sg = _core.SpaceGroup([8]*3, symbol)
        projected = sg.symmetrize(field)
        getattr(sg, enable)()
        np.testing.assert_allclose(sg.symmetrize(field), projected, rtol=0, atol=1e-12)
        reduced = sg.to_reduced_basis(np.stack([field, 2*field]))
        assert reduced.shape == (2, sg.get_n_reduced_basis())
        np.testing.assert_allclose(reduced[1], 2*reduced[0], rtol=0, atol=1e-12)
    for value in (np.zeros((512, 2)), np.zeros((8, 8, 8))):
        rejects(lambda: sg.to_reduced_basis(value))
    for platform in _core.PlatformSelector.avail_platforms():
        test_angles_and_grafting(platform)
        test_failed_construction(platform)
        test_pmmm_singleton(platform)
        test_box_update_and_cnadi(platform)
        test_grid_geometry(platform)
        test_n_grid(platform)
        for reduced in (False, True):
            test_native_guards(platform, reduced)
            test_basis_lifetime(platform, reduced)
            test_method_checks(platform, reduced)
            test_gce(platform, reduced)
            test_input_lists(platform, reduced)
            test_grafting_conversions(platform, reduced)
            test_cnadi_lattice_update(platform, reduced)
        print("Computation safety:", platform, "OK")


if __name__ == "__main__":
    main()
