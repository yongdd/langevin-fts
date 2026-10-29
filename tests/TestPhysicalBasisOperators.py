"""Public box operations distinguish active physical and irreducible fields."""
import contextlib
import io
import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from polymerfts import PlatformSelector, PropagatorSolver, SpaceGroup
from cuda_part import cuda_part_missing


def check(platform, symbol, hall):
    nx = [16, 16, 16]
    group = SpaceGroup(nx, symbol, hall)
    group.enable_m3_physical_basis()
    if symbol == "Pmmm":
        assert group.get_n_reduced_basis() == group.get_n_reduced_basis_full()
    else:
        assert group.get_n_reduced_basis() != group.get_n_reduced_basis_full()
    rng = np.random.default_rng(917)
    mask = (group.symmetrize(rng.normal(size=np.prod(nx))) > 0).astype(float)
    full = [group.symmetrize(rng.normal(size=np.prod(nx))) for _ in range(2)]
    full = [5.0 * field / field.std() for field in full]
    solver = PropagatorSolver(nx=nx, lx=[2.0] * 3, ds=0.1, bond_lengths={"A": 1.0},
                              bc=["periodic"] * 6, chain_model="continuous",
                              platform=platform, space_group=group, mask=mask)
    solver.add_polymer(1.0, [["A", 1.0, 0, 1]])
    active = [solver.to_reduced_basis(field) for field in full]
    irreducible = [group.reduced_to_irreducible(field) for field in active]
    volume = solver.get_volume()
    for fields, basis in ((active, "active"), (irreducible, "irreducible")):
        for field, reference in zip(fields, full):
            np.testing.assert_allclose(solver.mean(field, basis=basis), np.mean(mask * reference),
                                       rtol=0, atol=2e-13)
            np.testing.assert_allclose(solver.integral(field, basis=basis), volume * np.mean(mask * reference),
                                       rtol=0, atol=2e-12)
            shifted = field.copy()
            solver.zero_mean(shifted, basis=basis)
            expanded = solver.from_reduced_basis(
                group.irreducible_to_reduced(shifted) if basis == "irreducible" else shifted)
            expected = reference - np.sum(mask * reference) / np.sum(mask)
            np.testing.assert_allclose(expanded, expected, rtol=0, atol=2e-13)
            np.testing.assert_allclose(np.mean(mask * expanded), 0.0, rtol=0, atol=2e-13)
        expected_product = volume * np.mean(mask * full[0] * full[1])
        np.testing.assert_allclose(solver.inner_product(*fields, basis=basis), expected_product,
                                   rtol=0, atol=2e-12)
        pairs = np.stack(fields).ravel()
        reverse_pairs = np.stack(fields[::-1]).ravel()
        np.testing.assert_allclose(solver.multi_inner_product(2, pairs, reverse_pairs, basis=basis),
                                   2 * expected_product, rtol=0, atol=4e-12)

    # Fields returned by the public solver use the default active basis.
    np.testing.assert_allclose(solver.mean(active[0]), np.mean(mask * full[0]), rtol=0, atol=2e-13)
    solver.compute_propagators({"A": active[0]})
    assert solver.check_total_partition()
    solver.compute_concentrations()
    phi = solver.get_concentration("A")
    expanded = solver.from_reduced_basis(phi)
    np.testing.assert_allclose(solver.mean(phi), np.mean(mask * expanded), rtol=0, atol=2e-13)
    np.testing.assert_allclose(solver.mean(phi), 1.0, rtol=0, atol=2e-12)
    try:
        solver.mean(active[0], basis="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown field representation was accepted")


def main():
    for platform in PlatformSelector.avail_platforms():
        for symbol, hall in (("Pmmm", 227), ("Im-3m", 529)):
            with contextlib.redirect_stdout(io.StringIO()):
                check(platform, symbol, hall)
            print(f"[PASS] {platform}: {symbol} active/irreducible masked field operations")
    return 1 if cuda_part_missing() else 0


if __name__ == "__main__":
    raise SystemExit(main())
