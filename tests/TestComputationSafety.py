#!/usr/bin/env python3
"""Basis lifetime, degenerate grids, and ensemble normalization regressions."""

import numpy as np
from polymerfts import _core, PropagatorSolver


def rejects(call):
    try:
        call()
    except (RuntimeError, ValueError):
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
        # The real-space backend rejects an oblique cell after group attachment.
        rejects(lambda: fac.create_propagator_computation(cb, mol, opt, "cn-adi2", replacement))
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
        for reduced in (False, True):
            test_basis_lifetime(platform, reduced)
            test_gce(platform, reduced)
            test_grafting_conversions(platform, reduced)
            test_cnadi_lattice_update(platform, reduced)
        print("Computation safety:", platform, "OK")


if __name__ == "__main__":
    main()
