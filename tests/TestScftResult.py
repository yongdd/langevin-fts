#!/usr/bin/env python3
"""SCFTResult returned by SCFT.run(return_result=True).

1D lamella on the CPU: the result must match the attributes of the run, its
fields must be on the full grid, and its w fields must restart the run at the
converged solution. 3D BCC with the Im-3m space group: one iteration must
give the same full-grid fields as the same iteration without the space group.
"""

import contextlib
import io
import sys
import json
import tempfile
from pathlib import Path
import yaml

import numpy as np

from polymerfts import SCFT, SCFTResult, SpaceGroup, PlatformSelector


def cpu_platform():
    avail = PlatformSelector.avail_platforms()
    cpu = [p for p in avail if p.startswith("cpu")]
    return "cpu-mkl" if "cpu-mkl" in avail else (cpu[0] if cpu else None)


def params(nx, lx, **extra):
    p = {
        "nx": nx, "lx": lx,
        "chain_model": "continuous", "ds": 1/50,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 20.0},
        "polymers": [{"volume_fraction": 1.0,
                               "blocks": [{"type": "A", "length": 0.3}, {"type": "B", "length": 0.7}]}],
        "platform": cpu_platform(),
        "relax_cell": False,
        "max_iter": 300, "tolerance": 1e-8,
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
    }
    p.update(extra)
    return p


def main():
    if cpu_platform() is None:
        print("SKIP: no CPU platform in this build")
        sys.exit(77)  # ctest SKIP_RETURN_CODE
    failed = []

    # A max_iter exit must return fields, densities, box, Q and stress from the
    # same iteration, including when the final step is between stress updates.
    x_short = (np.arange(32) + 0.5)/32
    for altering in (False, True):
        p = params([np.int64(32)], [1.7], max_iter=5, tolerance=1e-11,
                   relax_cell=altering, stress_interval=2)
        with contextlib.redirect_stdout(io.StringIO()):
            short = SCFT(p)
            r = short.run({"A": np.sqrt(2)*5*np.cos(2*np.pi*x_short),
                           "B": -np.sqrt(2)*5*np.cos(2*np.pi*x_short)}, return_result=True)
            phi, _ = short.compute_concentrations(short.w)
        for m in ("A", "B"):
            np.testing.assert_allclose(r.phi[m], phi[m], rtol=0, atol=1e-11)
        np.testing.assert_allclose(r.partition_functions,
                                   [short.prop_solver.get_partition_function(0)], rtol=1e-12)
        if altering:
            short.prop_solver.compute_stress()
            np.testing.assert_allclose(r.stress, short.prop_solver.get_stress(), rtol=0, atol=1e-11)
        with tempfile.TemporaryDirectory() as tmp:
            for extension, load in (("json", json.load), ("yaml", yaml.safe_load)):
                path = Path(tmp)/("fields." + extension)
                short.save_results(str(path))
                with path.open() as stream:
                    saved = load(stream)
                assert saved["initial_params"]["nx"] == [32]
                np.testing.assert_allclose(saved["phi_A"], phi["A"], rtol=0, atol=1e-11)
            assert isinstance(short.params["nx"][0], np.int64)

    # Callback stops between stress updates must also return the final stress,
    # including its smearing contribution, and report the corresponding error.
    for smearing in (None, {"type": "gaussian", "a_int": 0.1}):
        p = params([32], [1.7], max_iter=10, tolerance=1e-11,
                   relax_cell=True, stress_interval=4, smearing=smearing)
        with contextlib.redirect_stdout(io.StringIO()):
            short = SCFT(p)
            r = short.run({"A": np.sqrt(2)*5*np.cos(2*np.pi*x_short),
                           "B": -np.sqrt(2)*5*np.cos(2*np.pi*x_short)}, return_result=True,
                          on_iteration=lambda info: False if info.iteration == 3 else None)
            phi, _ = short.compute_concentrations(short.w)
        assert r.iterations == 3
        short.prop_solver.compute_stress()
        stress = np.array(short.prop_solver.get_stress())
        if short.smearing.enabled:
            correction = short.smearing.stress(short.w, [phi[m] for m in short.monomer_types])
            correction[1:] = 0.0
            stress += correction[:len(stress)]
        np.testing.assert_allclose(r.stress, stress, rtol=0, atol=1e-11)
        for m in short.monomer_types:
            np.testing.assert_allclose(r.phi[m], phi[m], rtol=0, atol=1e-11)
        phi_array = np.array([short.smearing.apply_to_dict(phi)[m] for m in short.monomer_types])
        diff = short.matrix_q_perp @ (short.matrix_chi @ phi_array - short.w) \
            + short.pressure_kappa*(sum(phi_array) - 1.0)
        for row in diff:
            row -= short.prop_solver.mean(row)
        field_error = np.sqrt(sum(short.prop_solver.inner_product(row, row) for row in diff) /
                              (1 + sum(short.prop_solver.inner_product(row, row) for row in short.w)))
        np.testing.assert_allclose(r.error_level, field_error + abs(stress[0]), rtol=0, atol=1e-11)

    # ---- 1D lamella: result = attributes, full-grid fields, restart from result.w
    x = (np.arange(64) + 0.5)/64
    with contextlib.redirect_stdout(io.StringIO()):
        calc = SCFT(params([64], [4.0]))
        result = calc.run(initial_fields={"A": np.cos(2*np.pi*x), "B": -np.cos(2*np.pi*x)}, return_result=True)
    if not isinstance(result, SCFTResult) or not result.converged:
        failed.append("1D run did not return a converged SCFTResult")
    if result.free_energy != calc.free_energy or result.iterations != calc.iter:
        failed.append("result differs from the run attributes")
    if sorted(result.phi) != ["A", "B"] or sorted(result.w) != ["A", "B"] \
            or any(np.size(v) != 64 for v in list(result.phi.values()) + list(result.w.values())):
        failed.append("1D fields are not {A, B} on the full grid")
    if abs(np.mean(result.phi["A"] + result.phi["B"]) - 1.0) > 1e-10:
        failed.append("1D concentrations do not sum to 1")

    with contextlib.redirect_stdout(io.StringIO()):
        restart = calc.run(initial_fields=result.w, return_result=True)
    if restart.iterations > 2 or abs(restart.free_energy - result.free_energy) > 1e-8:
        failed.append(f"restart from result.w took {restart.iterations} iterations, "
                      f"free energy {restart.free_energy} vs {result.free_energy}")

    # ---- 3D BCC: the space-group result equals the same iteration without the space group
    n = 16
    g = (np.arange(n) + 0.5)/n
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    w_a = sum(np.cos(2*np.pi*(s*u + t*v)) for u, v in ((X, Y), (Y, Z), (Z, X)) for s, t in ((1, 1), (1, -1)))
    w_a = SpaceGroup([n]*3, "Im-3m", 529).symmetrize(np.reshape(w_a, (1, -1)))[0]
    results = {}
    for name, sg in (("on", {"space_group": {"symbol": "Im-3m", "hall_number": 529}}), ("off", {})):
        with contextlib.redirect_stdout(io.StringIO()):
            calc = SCFT(params([n]*3, [1.9]*3, max_iter=1, **sg))
            results[name] = calc.run(initial_fields={"A": w_a, "B": -w_a}, return_result=True)
    for kind in ("phi", "w"):
        for m in ("A", "B"):
            on, off = getattr(results["on"], kind)[m], getattr(results["off"], kind)[m]
            if np.size(on) != n**3:
                failed.append(f"BCC {kind}[{m}] is not on the full {n}^3 grid")
            elif not np.allclose(on, off, rtol=0, atol=1e-10):
                failed.append(f"BCC {kind}[{m}] differs from the run without space group by {np.max(np.abs(on - off)):.2e}")

    for f in failed:
        print("FAILED:", f)
    if failed:
        sys.exit(1)
    print("SCFTResult: OK")


if __name__ == "__main__":
    main()
