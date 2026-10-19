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
    return "cpu-mkl" if "cpu-mkl" in avail else [p for p in avail if p.startswith("cpu")][0]


def base_params():
    return {
        "nx": [8, 8, 8],
        "lx": [4.0, 4.0, 4.0],
        "chain_model": "continuous",
        "ds": 1/10,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15.0},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}],
        }],
        "platform": cpu_platform(),
        "box_is_altering": False,
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
    }


def lfts_params():
    params = base_params()
    params.update({
        "langevin": {"nbar": 10000, "dt": 8.0, "max_step": 10},
        "compressor": {"name": "am", "max_hist": 20, "start_error": 1e-1, "mix_min": 0.1, "mix_init": 0.1},
        "saddle": {"max_iter": 100, "tolerance": 1e-4},
        "recording": {"dir": "data", "recording_period": 10, "sf_computing_period": 1, "sf_recording_period": 10},
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
    ok = True

    # ---- SCFT defaults
    calc = build(SCFT, base_params())
    checks = {
        "max_iter": calc.max_iter == 2000,
        "tolerance": calc.tolerance == 1e-8,
        "stress_interval": calc.stress_interval == 1,
        "scale_stress": calc.scale_stress == 1,
        "fts_consistent_energy": calc.fts_consistent_energy is False,
        "zeta_n": calc.zeta_n is None,
        "space_group": calc.sg is None,
        "monomer_types": calc.monomer_types == ["A", "B"],
        "chi_n": calc.chi_n == {"A,B": 15.0},
        "blocks_input": calc.distinct_polymers[0]["blocks_input"] == [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]],
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
    params.update({"max_iter": 10, "tolerance": 1e-5, "stress_interval": "adaptive", "scale_stress": 2.0,
                   "fts_consistent_energy": True, "zeta_n": 100.0, "numerical_method": "rk2",
                   "crystal_system": "Cubic", "box_is_altering": True})
    calc = build(SCFT, params)
    if not (calc.max_iter == 10 and calc.tolerance == 1e-5 and calc.stress_interval == "adaptive"
            and calc.scale_stress == 2.0 and calc.fts_consistent_energy is True and calc.zeta_n == 100.0
            and calc.prop_solver.numerical_method == "rk2"
            and calc.lx_reduced_indices == [0] and calc.lx_full_indices == [0, 0, 0]):
        print("FAILED: SCFT explicit options")
        ok = False

    # ---- chi N completion and sorted monomer types (through LFTS: a missing pair makes
    #      the 3x3 chi matrix singular, which SCFT cannot invert)
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
    params["distinct_polymers"] = [{
        "volume_fraction": 1.0,
        "blocks": [{"type": "R", "length": 1.0, "fraction": {"A": 0.5, "B": 0.5}}],
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
    params = base_params(); params["distinct_polymers"][0]["volume_fraction"] = 0.5
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

    if ok:
        print("Test passed, parameter handling")
        return 0
    print("Test failed, parameter handling")
    return 1


if __name__ == "__main__":
    sys.exit(main())
