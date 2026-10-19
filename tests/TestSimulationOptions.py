#!/usr/bin/env python3
"""Unit test for the method-specific option dataclasses.

Checks that SCFT, L-FTS and CL-FTS options are read with the documented
defaults, that missing required keys are reported as ValidationError, and
that the options object does not alias the user's dictionaries.
"""

import sys

from polymerfts import ScftOptions, LftsOptions, ClftsOptions, ValidationError


def expect_error(cls, params, what):
    try:
        cls.from_params(params)
    except ValidationError:
        return
    print(f"FAILED: {what} should raise ValidationError")
    sys.exit(1)


def main():
    # ---- SCFT: defaults and required keys
    scft = ScftOptions.from_params({
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "box_is_altering": True,
    })
    assert scft.max_iter == 2000 and scft.tolerance == 1e-8, "SCFT defaults"
    assert scft.stress_interval == 1 and scft.scale_stress == 1, "SCFT stress defaults"
    assert scft.fts_consistent_energy is False and scft.space_group is None, "SCFT defaults"
    assert scft.crystal_system is None and scft.smearing is None, "SCFT defaults"
    expect_error(ScftOptions, {"box_is_altering": False}, "missing optimizer")
    expect_error(ScftOptions, {"optimizer": {"name": "am"}}, "missing box_is_altering")

    opt = {"name": "adam", "lr": 1e-2, "gamma": 0.9}
    scft = ScftOptions.from_params({
        "optimizer": opt, "box_is_altering": False, "max_iter": 10, "tolerance": 1e-5,
        "stress_interval": "adaptive", "scale_stress": 2.0, "fts_consistent_energy": True,
        "space_group": {"symbol": "Im-3m", "number": 529}, "crystal_system": "Cubic",
    })
    assert scft.max_iter == 10 and scft.tolerance == 1e-5 and scft.stress_interval == "adaptive"
    assert scft.scale_stress == 2.0 and scft.fts_consistent_energy is True
    assert scft.space_group == {"symbol": "Im-3m", "number": 529} and scft.crystal_system == "Cubic"
    opt["lr"] = 1.0
    assert scft.optimizer["lr"] == 1e-2, "options must copy the user's dictionaries"

    # ---- L-FTS
    lfts_params = {
        "langevin": {"nbar": 10000, "dt": 8.0, "max_step": 100},
        "compressor": {"name": "am", "max_hist": 20, "start_error": 1e-1, "mix_min": 0.1, "mix_init": 0.1},
        "saddle": {"max_iter": 100, "tolerance": 1e-4},
        "recording": {"dir": "data", "recording_period": 10, "sf_computing_period": 1, "sf_recording_period": 10},
        "verbose_level": 1,
    }
    lfts = LftsOptions.from_params(lfts_params)
    assert lfts.wtmd is None and lfts.validation_runtime is None and lfts.smearing is None, "LFTS defaults"
    assert lfts.langevin["nbar"] == 10000 and lfts.verbose_level == 1
    for key in ("langevin", "compressor", "saddle", "recording", "verbose_level"):
        p = dict(lfts_params); del p[key]
        expect_error(LftsOptions, p, f"missing {key}")

    # ---- CL-FTS: alpha_ds alias
    clfts_params = {
        "langevin": {"nbar": 10000, "dt": 0.01, "max_step": 100},
        "recording": {"dir": "data", "recording_period": 10},
        "verbose_level": 1,
    }
    clfts = ClftsOptions.from_params(clfts_params)
    assert clfts.alpha_ds == 0.0 and clfts.smearing is None, "CLFTS defaults"
    assert ClftsOptions.from_params({**clfts_params, "dynamic_stabilization": 0.05}).alpha_ds == 0.05
    assert ClftsOptions.from_params({**clfts_params, "alpha_ds": 0.02}).alpha_ds == 0.02
    expect_error(ClftsOptions, {"recording": {}, "verbose_level": 1}, "missing langevin")

    print("TestSimulationOptions: all checks passed")


if __name__ == "__main__":
    main()
