#!/usr/bin/env python3
"""SCFT with monoclinic and triclinic space groups.

1. Free angles: box relaxation varies the angle between the two axes perpendicular to the
   unique axis of a monoclinic setting (beta for P2/m with unique axis b, gamma for unique
   axis c) and all three angles of a triclinic group (P-1); a cell whose unique axis is not
   perpendicular to the others is rejected.
2. Stress: with identical symmetrized fields of standard deviation 5, all six components of
   SCFT.compute_stress with the group (reduced basis) equal those of the full grid, for
   P2/m (unique axes b and c), P2_1/c, C2/c and P-1 in oblique cells, continuous and discrete
   chains with unequal segment lengths, on every available platform.
3. Relaxation: AB diblock cylinders (f = 1/3, chi N = 18) along the unique axis, started in a
   cell with an angle of 115 degrees, relax with P2/m (unique axes b and c) and P-1 to the
   cell and free energy of the same run without a group: the hexagonal packing a = 1.7019,
   angle 120 degrees.
"""

import contextlib
import io
import os

os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
from scipy.ndimage import gaussian_filter
from polymerfts import SCFT, PlatformSelector, SpaceGroup, ValidationError
from cuda_part import cuda_part_missing


def cpu_platform():
    return next(p for p in PlatformSelector.avail_platforms() if p.startswith("cpu"))


def quiet(make):
    with contextlib.redirect_stdout(io.StringIO()):
        return make()


def parameters(nx, lx, angles, platform, f=0.4, chi_n=15.0, chain_model="continuous", relax_cell=False, **extra):
    params = {"nx": nx, "lx": lx, "angles": angles, "chain_model": chain_model,
              "ds": 1 / 20 if chain_model == "discrete" else 1 / 40,
              "segment_lengths": {"A": 1.0, "B": 1.2}, "chi_n": {"A,B": chi_n},
              "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": f}, {"type": "B", "length": 1 - f}]}],
              "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.05, "mix_init": 0.05},
              "platform": platform, "relax_cell": relax_cell, "max_iter": 2000, "tolerance": 1e-8}
    params.update(extra)
    return params


def group(symbol, hall):
    return {"symbol": symbol, "hall_number": hall}


def free_angles():
    cases = (("P2/m", 57, [90.0, 100.0, 90.0], [1]), ("P2/m", 58, [90.0, 90.0, 100.0], [2]),
             ("P2/m", 57, [90.0, 90.0, 90.0], [1]), ("P-1", 2, [80.0, 100.0, 110.0], [0, 1, 2]))
    for symbol, hall, angles, expected in cases:
        sim = quiet(lambda: SCFT(parameters([12, 10, 8], [3.0, 3.2, 2.6], angles, cpu_platform(), relax_cell=True,
                                            space_group=group(symbol, hall))))
        assert sim.crystal_system in ("Monoclinic", "Triclinic"), sim.crystal_system
        assert sim.angles_reduced_indices == expected, f"{symbol} ({hall}) at {angles}: free angles {sim.angles_reduced_indices}"
    try:
        quiet(lambda: SCFT(parameters([12, 10, 8], [3.0, 3.2, 2.6], [90.0, 90.0, 100.0], cpu_platform(),
                                      space_group=group("P2/m", 57))))
    except ValidationError as error:
        assert "unique axis" in str(error), str(error)
    else:
        raise AssertionError("P2/m with unique axis b accepted a cell with gamma = 100 degrees")


def stress():
    cases = (("P2/m", 57, [12, 10, 8], [90.0, 110.0, 90.0]), ("P2/m", 58, [12, 10, 8], [90.0, 90.0, 110.0]),
             ("P2_1/c", 81, [12, 10, 8], [90.0, 110.0, 90.0]), ("C2/c", 90, [12, 12, 8], [90.0, 110.0, 90.0]),
             ("P-1", 2, [12, 10, 8], [80.0, 100.0, 110.0]))
    rng = np.random.default_rng(7)
    for platform in PlatformSelector.avail_platforms():
        for symbol, hall, nx, angles in cases:
            sg = quiet(lambda: SpaceGroup(nx, symbol, hall))
            w_full = sg.symmetrize(rng.normal(0.0, 5.0, (2, int(np.prod(nx)))))
            w_basis = sg.reduced_to_irreducible(sg.to_reduced_basis(w_full))
            for chain_model in ("continuous", "discrete"):
                values = []
                for space_group, w in ((None, w_full), (group(symbol, hall), w_basis)):
                    extra = {"crystal_system": "Triclinic"} if space_group is None else {"space_group": space_group}
                    sim = quiet(lambda: SCFT(parameters(nx, [3.0, 3.2, 2.6], angles, platform, chain_model=chain_model,
                                                        **extra)))
                    phi, _ = quiet(lambda: sim.compute_concentrations(w))
                    values.append(np.array(sim.compute_stress(w, phi)))
                full, reduced = values
                scale = np.max(np.abs(full))
                assert np.max(np.abs(reduced - full)) < 1e-9 * scale, \
                    f"{platform} {symbol} ({hall}) {chain_model}: stress {reduced} against {full}"
                assert np.max(np.abs(full[3:])) > 1e-3 * scale, f"{symbol} ({hall}): no angle stress to compare"
        print(f"  {platform}: six stress components with and without the group agree")


def cylinders(axis, angle, nx_cylinder=24, nz=8):
    """Cylinders along `axis` (1: b, 2: c) through the origin, in a cell with `angle` between the other two axes."""
    f, chi_n, length = 1 / 3, 18.0, 1.8
    plane = [i for i in range(3) if i != axis]
    nx, lx, angles = [nx_cylinder] * 3, [length] * 3, [90.0] * 3
    nx[axis], lx[axis] = nz, length / 2
    angles[3 - sum(plane)] = angle      # the angle between the two axes of the plane
    u = np.arange(nx_cylinder)[:, None] / nx_cylinder
    v = np.arange(nx_cylinder)[None, :] / nx_cylinder
    u, v = u - np.round(u), v - np.round(v)
    r = np.sqrt(np.maximum(u ** 2 + v ** 2 + 2 * np.cos(np.radians(angle)) * u * v, 0.0))
    w = -gaussian_filter(0.5 * (1.0 - np.tanh((r - 0.2) / 0.05)), sigma=1.0, mode="wrap")
    w = np.expand_dims(0.5 * chi_n / np.abs(w).max() * w, axis).repeat(nz, axis=axis)
    params = parameters(nx, lx, angles, cpu_platform(), f=f, chi_n=chi_n, relax_cell=True, stress_scale=0.2)
    params["segment_lengths"] = {"A": 1.0, "B": 1.0}
    params["ds"] = 1 / 90
    return params, {"A": w, "B": -w}


def relaxation():
    for axis, space_groups in ((1, [group("P2/m", 57)]), (2, [group("P2/m", 58), group("P-1", 2)])):
        params, fields = cylinders(axis, 115.0)
        plane = [i for i in range(3) if i != axis]
        free = 3 - sum(plane)
        reference = quiet(lambda: SCFT(dict(params, crystal_system="Triclinic")).run(initial_fields=fields,
                                                                                    return_result=True))
        assert reference.converged, "relaxation without a group"
        assert abs(reference.lx[plane[0]] - 1.7019) < 1e-4 and abs(reference.angles[free] - 120.0) < 1e-2, \
            f"cell without a group: {reference.lx}, {reference.angles}"
        for space_group in space_groups:
            result = quiet(lambda: SCFT(dict(params, space_group=space_group)).run(initial_fields=fields,
                                                                                 return_result=True))
            assert result.converged, f"relaxation with {space_group}"
            assert abs(result.free_energy - reference.free_energy) < 1e-8, \
                f"{space_group}: F {result.free_energy:.12f} against {reference.free_energy:.12f}"
            assert np.allclose(result.lx, reference.lx, rtol=0, atol=1e-5) and \
                np.allclose(result.angles, reference.angles, rtol=0, atol=1e-3), \
                f"{space_group}: cell {result.lx}, {result.angles} against {reference.lx}, {reference.angles}"
            print(f"  {space_group['symbol']} ({space_group['hall_number']}): a = {result.lx[plane[0]]:.5f}, "
                  f"angle {result.angles[free]:.3f}, F = {result.free_energy:.10f}")


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    for name, check in (("free angles", free_angles), ("stress", stress), ("relaxation", relaxation)):
        check()
        print(f"[PASS] {name}")
    return 1 if cuda_part_missing() else 0


if __name__ == "__main__":
    raise SystemExit(main())
