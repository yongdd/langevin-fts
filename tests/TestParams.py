#!/usr/bin/env python3
"""Parameter handling of SCFT, LFTS and CLFTS.

Builds the three simulation classes on the CPU with tiny grids and checks
that the parameter dictionary is read as documented: defaults, chi N
completion, random copolymers, no modification of the user's dictionary,
and ValidationError for invalid input.
"""

import sys
import copy
import io
import contextlib
import numpy as np

from polymerfts import SCFT, LFTS, CLFTS, ValidationError, PlatformSelector


def cpu_platform():
    avail = PlatformSelector.avail_platforms()
    cpu = [p for p in avail if p.startswith("cpu")]
    return "cpu-mkl" if "cpu-mkl" in avail else (cpu[0] if cpu else None)


def base_params():
    return {
        "nx": [8, 8, 8],
        "lx": [4.0, 4.0, 4.0],
        "chain_model": "continuous",
        "ds": 1/10,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15.0},
        "polymers": [{
            "volume_fraction": 1.0,
            "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}],
        }],
        "platform": cpu_platform(),
        "relax_cell": False,
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
    }


def lfts_params():
    params = base_params()
    params.update({
        "langevin": {"nbar": 10000, "dt": 8.0, "max_step": 10},
        "compressor": {"name": "am", "max_hist": 20, "start_error": 1e-1, "mix_min": 0.1, "mix_init": 0.1},
        "saddle": {"max_iter": 100, "tolerance": 1e-4},
        "recording": {"dir": "data", "checkpoint_interval": 10, "sample_interval": 1, "statistics_save_interval": 10},
        "verbose_level": 1,
    })
    return params


def build(cls, params):
    """Construct quietly (the constructors print the simulation summary)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return cls(params)


def expect_error(cls, params, what):
    try:
        build(cls, params)
    except ValidationError:
        return True
    print(f"FAILED: {what} did not raise ValidationError")
    return False


def main():
    if cpu_platform() is None:
        print("SKIP: no CPU platform in this build")
        return 77  # ctest SKIP_RETURN_CODE
    ok = True

    from polymerfts.validation import parse_chi_n
    for bad in ("A,B,C", "A", "A,,B"):
        try:
            parse_chi_n({bad: 10.0}, {"A": 1.0, "B": 1.0, "C": 1.0})
            raise AssertionError(f"Malformed pair {bad} was accepted")
        except ValidationError:
            pass
    names = {"A_a": 1.0, "B_b": 1.0}
    for pair in ("A_a,B_b", "B_b/A_a", "A_a_B_b"):
        assert parse_chi_n({pair: 10.0}, names) == {"A_a,B_b": 10.0}
    try:
        parse_chi_n({"A_B_C": 10.0}, dict.fromkeys(("A", "A_B", "B_C", "C"), 1.0))
        raise AssertionError("Ambiguous underscore pair was accepted")
    except ValidationError:
        pass
    for nx, angles in (([16], [90, 90, 60]), ([8, 8], [120, 90, 90])):
        p = base_params(); p.update(nx=nx, lx=[4.0]*len(nx), angles=angles)
        ok &= expect_error(SCFT, p, "angles involving absent dimensions")
    for angles in (None, []):
        p = base_params(); p.update(nx=[8, 8], lx=[4.0, 4.0], angles=angles)
        assert build(SCFT, p).crystal_system == "Orthorhombic"

    # NaN mass errors must raise, rather than pass the comparison to tolerance.
    for cls in (LFTS, CLFTS):
        sim = build(cls, lfts_params())
        sim.prop_solver.mean = lambda value: np.nan
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sim.compute_concentrations(np.zeros((2, sim.prop_solver.n_grid)))
            raise AssertionError(f"{cls.__name__} accepted NaN mass error")
        except ValidationError:
            pass

    p = lfts_params(); p["verbose_level"] = 0; p["langevin"]["max_step"] = 1
    sim = build(LFTS, p)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        sim.run(initial_fields={m: np.zeros(sim.prop_solver.n_grid) for m in ("A", "B")})
    assert not any(line.startswith("Langevin step:") for line in output.getvalue().splitlines())

    p = base_params(); p["chain_model"] = "discrete"
    assert build(SCFT, p).prop_solver.numerical_method == "discrete"

    # A fully absent composition mode must have a finite zero step size and
    # remain fixed through CL-FTS, including when initial fields contain that mode.
    p = lfts_params(); p["chi_n"] = {"A,B": 0.0}; p["verbose_level"] = 0
    p["langevin"].update(max_step=2, dt=0.01)
    lfts = build(LFTS, p)
    assert np.all(np.isfinite(lfts.dt_scaling))
    assert np.all(lfts.dt_scaling[lfts.mpt.aux_fields_dropped_idx] == 0)
    sim = build(CLFTS, p)
    assert sim.mpt.aux_fields_dropped_idx and np.all(np.isfinite(sim.dt_scaling))
    assert np.all(sim.dt_scaling[sim.mpt.aux_fields_dropped_idx] == 0)
    compute = sim.compute_concentrations
    snapshots = []
    def capture(w_aux):
        snapshots.append(w_aux.copy())
        return compute(w_aux)
    sim.compute_concentrations = capture
    rng = np.random.default_rng(51)
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run(initial_fields={m: rng.normal(0, 5, sim.prop_solver.n_grid) for m in ("A", "B")})
    assert len(snapshots) == 3
    for w_aux in snapshots:
        for i in sim.mpt.aux_fields_dropped_idx:
            np.testing.assert_allclose(w_aux[i], sim.mpt._vector_large_s[i]/2, rtol=0, atol=1e-12)

    # ---- SCFT defaults
    calc = build(SCFT, base_params())
    checks = {
        "max_iter": calc.max_iter == 2000,
        "tolerance": calc.tolerance == 1e-8,
        "stress_interval": calc.stress_interval == 1,
        "stress_scale": calc.stress_scale == 1,
        "zeta_n": calc.zeta_n is None,
        "space_group": calc.sg is None,
        "monomer_types": calc.monomer_types == ["A", "B"],
        "chi_n": calc.chi_n == {"A,B": 15.0},
        "blocks_input": calc.polymers[0]["blocks_input"] == [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]],
        "platform": calc.prop_solver.platform == cpu_platform(),
        "numerical_method": calc.prop_solver.numerical_method == "rqm4",
        "crystal_system": calc.crystal_system == "Orthorhombic" and calc.lx_reduced_indices == [0, 1, 2],
    }
    for name, passed in checks.items():
        if not passed:
            print(f"FAILED: SCFT default {name}")
            ok = False

    # ---- explicit options reach the solver
    params = base_params()
    params.update({"max_iter": 10, "tolerance": 1e-5, "stress_interval": "adaptive", "stress_scale": 2.0,
                   "zeta_n": 100.0, "numerical_method": "rk2",
                   "crystal_system": "Cubic", "relax_cell": True})
    calc = build(SCFT, params)
    if not (calc.max_iter == 10 and calc.tolerance == 1e-5 and calc.stress_interval == "adaptive"
            and calc.stress_scale == 2.0 and calc.zeta_n == 100.0
            and calc.prop_solver.numerical_method == "rk2"
            and calc.lx_reduced_indices == [0] and calc.lx_full_indices == [0, 0, 0]):
        print("FAILED: SCFT explicit options")
        ok = False

    # ---- chi N completion and sorted monomer types (through LFTS: a missing pair makes
    #      the 3x3 chi matrix singular, which incompressible SCFT rejects)
    params = lfts_params()
    params["segment_lengths"] = {"C": 1.0, "A": 1.0, "B": 1.0}
    params["chi_n"] = {"B,A": 15.0, "A,C": 5.0}
    sim = build(LFTS, params)
    if not (sim.monomer_types == ["A", "B", "C"] and sim.chi_n == {"A,B": 15.0, "A,C": 5.0, "B,C": 0.0}):
        print("FAILED: chi_n completion", sim.chi_n)
        ok = False

    # ---- random copolymer: effective segment length and fractions
    params = base_params()
    params["segment_lengths"] = {"A": 1.0, "B": 2.0}
    params["polymers"] = [{
        "volume_fraction": 1.0,
        "blocks": [{"type": "R", "length": 1.0, "monomer_fractions": {"A": 0.5, "B": 0.5}}],
    }]
    calc = build(SCFT, params)
    if not (np.isclose(calc.segment_lengths["R"], np.sqrt(0.5 * 1.0 + 0.5 * 4.0))
            and calc.random_fraction == {"R": {"A": 0.5, "B": 0.5}}
            and calc.monomer_types == ["A", "B"]):
        print("FAILED: random copolymer", calc.segment_lengths, calc.random_fraction)
        ok = False

    # ---- the user's dictionary is left unchanged
    for cls, params in ((SCFT, base_params()), (LFTS, lfts_params()), (CLFTS, lfts_params())):
        snapshot = copy.deepcopy(params)
        build(cls, params)
        if params != snapshot:
            print(f"FAILED: {cls.__name__} modified its input dictionary")
            ok = False

    # ---- LFTS and CLFTS read their own options
    sim = build(LFTS, lfts_params())
    if not (sim.langevin["nbar"] == 10000 and "sigma" in sim.langevin and sim.saddle["max_iter"] == 100
            and sim.recording["dir"] == "data" and sim.verbose_level == 1 and sim.wtmd is None
            and sim.zeta_n is None):
        print("FAILED: LFTS options")
        ok = False
    params = lfts_params()
    params.update({"dynamic_stabilization": 0.05})
    sim = build(CLFTS, params)
    if not (sim.alpha_ds == 0.05 and sim.zeta_n == 100.0 and sim.prop_solver.field_type == "complex"):
        print("FAILED: CLFTS options", sim.alpha_ds, sim.zeta_n)
        ok = False
    params = lfts_params(); params.update({"alpha_ds": 0.02})
    if build(CLFTS, params).alpha_ds != 0.02:
        print("FAILED: CLFTS alpha_ds alias")
        ok = False

    # ---- invalid input raises ValidationError
    params = base_params(); params["polymers"][0]["volume_fraction"] = 0.5
    ok &= expect_error(SCFT, params, "volume fractions not summing to 1")
    params = base_params(); params["chi_n"] = {"A,Z": 1.0}
    ok &= expect_error(SCFT, params, "chi_n with unknown monomer")
    params = base_params(); del params["chain_model"]
    ok &= expect_error(SCFT, params, "missing chain_model")
    params = base_params(); params["lx"] = [4.0, 4.0]
    ok &= expect_error(SCFT, params, "nx/lx length mismatch")
    params = base_params(); params["ds"] = -0.1
    ok &= expect_error(SCFT, params, "negative ds")
    params = lfts_params(); del params["langevin"]
    ok &= expect_error(LFTS, params, "LFTS without langevin")
    params = lfts_params(); del params["saddle"]
    ok &= expect_error(LFTS, params, "LFTS without saddle")
    params = base_params(); del params["optimizer"]
    ok &= expect_error(SCFT, params, "SCFT without optimizer")

    # ---- solver options: defaults and explicit values reach the propagator solver
    calc = build(SCFT, base_params())
    if not (calc.prop_solver.bc == ["periodic"] * 6 and calc.prop_solver.angles is None
            and calc.prop_solver.reduce_memory is False
            and calc.prop_solver.aggregate_propagator_computation is True
            and calc.smearing.enabled is False):
        print("FAILED: SCFT solver defaults")
        ok = False
    params = base_params()
    params.update({"bc": ["reflecting"] * 6, "reduce_memory": True, "aggregate_propagator_computation": False})
    calc = build(SCFT, params)
    if not (calc.prop_solver.bc == ["reflecting"] * 6 and calc.prop_solver.reduce_memory is True
            and calc.prop_solver.aggregate_propagator_computation is False):
        print("FAILED: SCFT solver options (bc, reduce_memory, aggregation)")
        ok = False
    params = base_params(); params["angles"] = [90.0, 90.0, 90.0]
    if build(SCFT, params).prop_solver.angles != [90.0, 90.0, 90.0]:
        print("FAILED: SCFT angles")
        ok = False
    params = base_params(); params["space_group"] = {"symbol": "Im-3m"}
    calc = build(SCFT, params)
    if not (calc.sg is not None and calc.crystal_system == "Cubic" and calc.lx_reduced_indices == [0]):
        print("FAILED: SCFT space group")
        ok = False

    # ---- the cell must match the crystal system of the space group
    params = base_params(); params["space_group"] = {"symbol": "Im-3m"}; params["lx"] = [4.0, 4.0, 3.0]
    ok &= expect_error(SCFT, params, "cubic space group with unequal box lengths")
    params = base_params(); params["space_group"] = {"symbol": "Im-3m"}; params["angles"] = [90.0, 90.0, 120.0]
    ok &= expect_error(SCFT, params, "cubic space group with a 120-degree angle")
    params = base_params(); params["space_group"] = {"symbol": "P6_3/mmc", "hall_number": 488}
    ok &= expect_error(SCFT, params, "hexagonal space group without a 120-degree angle")
    params = base_params(); params["space_group"] = {"symbol": "P6_3/mmc", "hall_number": 488}
    params["angles"] = [90.0, 90.0, 120.0]
    if build(SCFT, params).crystal_system != "Hexagonal":
        print("FAILED: SCFT hexagonal space group with a hexagonal cell")
        ok = False
    params = base_params(); params["space_group"] = {"symbol": "P6_3/mmc", "hall_number": 488}; params["angles"] = [120.0]
    ok &= expect_error(SCFT, params, "3D box with a single angle")
    # trigonal groups: the axes setting of the group decides which cell is valid
    params = base_params(); params["space_group"] = {"symbol": "P-3m1"}
    ok &= expect_error(SCFT, params, "hexagonal-axes trigonal group on a cube")
    params = base_params(); params["space_group"] = {"symbol": "P-3m1"}; params["angles"] = [90.0, 90.0, 120.0]
    if build(SCFT, params).crystal_system != "Trigonal":
        print("FAILED: SCFT hexagonal-axes trigonal group with a hexagonal cell")
        ok = False
    params = base_params(); params["space_group"] = {"symbol": "R-3m", "hall_number": 459}; params["angles"] = [90.0, 90.0, 120.0]
    ok &= expect_error(SCFT, params, "rhombohedral-axes group on a hexagonal cell")
    params = base_params(); params["space_group"] = {"symbol": "R-3m", "hall_number": 459}; params["angles"] = [80.0, 80.0, 80.0]
    if build(SCFT, params).crystal_system != "Trigonal":
        print("FAILED: SCFT rhombohedral-axes group on a rhombohedral cell")
        ok = False
    params = base_params(); params["space_group"] = {"symbol": "R-3m", "hall_number": 459}; params["angles"] = [80.0, 80.0, 80.0]
    params["relax_cell"] = True
    ok &= expect_error(SCFT, params, "box relaxation with a rhombohedral-axes setting")

    # ---- changing the user's dictionary after construction does not affect the object
    params = base_params()
    calc = build(SCFT, params)
    params["max_iter"] = 1
    params["chi_n"]["A,B"] = 0.0
    params["polymers"][0]["volume_fraction"] = 0.2
    if not (calc.max_iter == 2000 and calc.chi_n == {"A,B": 15.0}
            and calc.polymers[0]["volume_fraction"] == 1.0):
        print("FAILED: SCFT shares the user's dictionary")
        ok = False

    # ---- platform "auto" is accepted
    params = base_params(); params["platform"] = "auto"
    if build(SCFT, params).prop_solver.platform not in PlatformSelector.avail_platforms():
        print("FAILED: platform auto")
        ok = False

    # ---- SCFT accepts a singular chi N matrix (a missing pair) and rejects a single
    #      monomer type in the incompressible model
    params = base_params()
    params["segment_lengths"] = {"A": 1.0, "B": 1.0, "C": 1.0}
    params["chi_n"] = {"A,B": 15.0, "A,C": 5.0}
    try:
        build(SCFT, params)
    except ValidationError as e:
        print(f"FAILED: SCFT with a singular chi matrix (missing B,C pair) raised: {e}")
        ok = False
    params = base_params()
    params["segment_lengths"] = {"A": 1.0}
    params["chi_n"] = {}
    params["polymers"] = [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 1.0}]}]
    ok &= expect_error(SCFT, params, "SCFT with a single monomer type")

    # ---- mean concentrations from the chain compositions (random copolymer, alpha != 1)
    params = base_params()
    params["polymers"] = [
        {"volume_fraction": 0.3, "blocks": [{"type": "A", "length": 0.7}]},
        {"volume_fraction": 0.7, "blocks": [{"type": "R", "length": 1.0, "monomer_fractions": {"A": 0.5, "B": 0.5}}]},
    ]
    calc = build(SCFT, params)
    if not np.allclose(calc.mean_concentrations(), [0.3 + 0.35, 0.35]):
        print("FAILED: mean concentrations", calc.mean_concentrations())
        ok = False

    # ---- crystal systems are checked against the dimension
    params = base_params(); params["nx"] = [8, 8]; params["lx"] = [4.0, 4.0]; params["crystal_system"] = "Cubic"
    try:
        build(SCFT, params)
        print("FAILED: Cubic accepted for a 2D box")
        ok = False
    except ValueError:
        pass
    params = base_params(); params["nx"] = [8, 8]; params["lx"] = [4.0, 4.0]; params["angles"] = [90.0, 90.0, 120.0]
    params["relax_cell"] = True
    calc = build(SCFT, params)
    if not (calc.crystal_system == "Hexagonal2D" and calc.lx_reduced_indices == [0]):
        print("FAILED: Hexagonal2D classification", calc.crystal_system, calc.lx_reduced_indices)
        ok = False

    # ---- LFTS / CLFTS options
    for key in ("compressor", "recording", "verbose_level"):
        params = lfts_params(); del params[key]
        ok &= expect_error(LFTS, params, f"LFTS without {key}")
    params = base_params(); del params["relax_cell"]
    ok &= expect_error(SCFT, params, "SCFT without relax_cell")
    params = lfts_params(); del params["langevin"]
    ok &= expect_error(CLFTS, params, "CLFTS without langevin")
    if build(CLFTS, lfts_params()).alpha_ds != 0.0:
        print("FAILED: CLFTS alpha_ds default")
        ok = False
    for level in (0, 1, 2):
        params = lfts_params(); params["verbose_level"] = level
        if not (build(LFTS, params).verbose_level == level and build(CLFTS, params).verbose_level == level):
            print(f"FAILED: verbose_level {level}")
            ok = False
    params = lfts_params(); params["verbose_level"] = 3
    ok &= expect_error(LFTS, params, "LFTS verbose_level 3")
    ok &= expect_error(CLFTS, params, "CLFTS verbose_level 3")
    params = lfts_params(); params["compressor"]["name"] = "foo"
    ok &= expect_error(LFTS, params, "LFTS unknown compressor")
    params = lfts_params(); params["wtmd"] = {"ell": 4}; params["nx"] = [8, 8]; params["lx"] = [4.0, 4.0]
    ok &= expect_error(LFTS, params, "WTMD on a 2D grid")
    params = lfts_params(); params["wtmd"] = {"ell": 4}
    ok &= expect_error(CLFTS, params, "CL-FTS with a wtmd block")
    # A bin width that does not divide the range is refused when a new run starts; the
    # constructor accepts it, so that a run started with such a width still continues
    params = lfts_params(); params["wtmd"] = {"psi_min": 0.0, "psi_max": 1.0, "dpsi": 0.3}
    simulation = build(LFTS, params)
    try:
        simulation.run(initial_fields={})
        print("FAILED: WTMD bin width that does not divide the range did not raise ValidationError")
        ok = False
    except ValidationError:
        pass
    for chi_n in (0.0, -5.0):   # no real auxiliary (exchange) field for the order parameter
        params = lfts_params(); params["wtmd"] = {"ell": 4}; params["chi_n"] = {"A,B": chi_n}
        ok &= expect_error(LFTS, params, f"WTMD at chi_n {chi_n}")
    params = lfts_params(); params.update(dynamic_stabilization=0.0, alpha_ds=0.02)
    ok &= expect_error(CLFTS, params, "dynamic_stabilization together with its alias alpha_ds")
    for value in (0, -1, 1.5, "fast"):
        params = base_params(); params["stress_interval"] = value
        ok &= expect_error(SCFT, params, f"stress_interval {value!r}")

    if ok:
        print("Test passed, parameter handling")
        return 0
    print("Test failed, parameter handling")
    return 1


if __name__ == "__main__":
    sys.exit(main())
