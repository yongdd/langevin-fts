#!/usr/bin/env python3
"""Unit test for the shared configuration layer (polymerfts.parse_system).

Checks that the parameter dictionary shared by SCFT, LFTS and CLFTS is
interpreted and validated in one place, without building any solver.
"""

import sys
import copy
import numpy as np

from polymerfts import parse_system, SystemSpec, ValidationError, PlatformSelector


def base_params():
    return {
        "nx": [16, 16, 16],
        "lx": [4.0, 4.0, 4.0],
        "chain_model": "continuous",
        "ds": 1/50,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15.0},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}],
        }],
    }


def expect_error(params, what):
    try:
        parse_system(params)
    except ValidationError:
        return True
    print(f"FAILED: {what} did not raise ValidationError")
    return False


def main():
    ok = True

    # Minimal diblock
    spec = parse_system(base_params())
    ok &= isinstance(spec, SystemSpec)
    ok &= spec.grid.nx == (16, 16, 16) and spec.grid.dim == 3 and spec.grid.total_grid == 4096
    ok &= spec.grid.bc == ("periodic",) * 6 and spec.grid.angles is None
    ok &= spec.chain.model == "continuous" and spec.chain.numerical_method is None
    ok &= spec.monomer_types == ("A", "B") and spec.chi_n == {"A,B": 15.0}
    ok &= spec.zeta_n is None and spec.reduce_memory is False
    ok &= spec.polymers[0]["blocks_input"] == [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]]
    ok &= spec.platform in PlatformSelector.avail_platforms()
    if not ok:
        print("FAILED: minimal diblock spec", spec)

    # Explicit options are passed through
    params = base_params()
    params.update({"bc": ["reflecting"] * 6, "angles": [90.0, 90.0, 120.0], "zeta_n": 100.0,
                   "reduce_memory": True, "numerical_method": "rk2",
                   "aggregate_propagator_computation": False})
    spec = parse_system(params)
    ok2 = (spec.grid.bc == ("reflecting",) * 6 and spec.grid.angles == (90.0, 90.0, 120.0)
           and spec.zeta_n == 100.0 and spec.reduce_memory is True
           and spec.chain.numerical_method == "rk2" and spec.aggregate_propagator_computation is False)
    if not ok2:
        print("FAILED: explicit options", spec)
    ok &= ok2

    # Missing chi_n pairs are filled with zero and keys are sorted
    params = base_params()
    params["segment_lengths"] = {"C": 1.0, "A": 1.0, "B": 1.0}
    params["chi_n"] = {"B,A": 15.0, "A,C": 5.0}
    spec = parse_system(params)
    ok3 = spec.monomer_types == ("A", "B", "C") and spec.chi_n == {"A,B": 15.0, "A,C": 5.0, "B,C": 0.0}
    if not ok3:
        print("FAILED: chi_n completion", spec.chi_n)
    ok &= ok3

    # Random copolymer gets an effective segment length and its fractions
    params = base_params()
    params["segment_lengths"] = {"A": 1.0, "B": 2.0}
    params["distinct_polymers"] = [{
        "volume_fraction": 1.0,
        "blocks": [{"type": "R", "length": 1.0, "fraction": {"A": 0.5, "B": 0.5}}],
    }]
    spec = parse_system(params)
    ok4 = (np.isclose(spec.chain.segment_lengths["R"], np.sqrt(0.5 * 1.0 + 0.5 * 4.0))
           and spec.random_fraction == {"R": {"A": 0.5, "B": 0.5}}
           and spec.polymers[0]["block_monomer_types"] == ["R"]
           and spec.monomer_types == ("A", "B"))
    if not ok4:
        print("FAILED: random copolymer", spec)
    ok &= ok4

    # The input dictionary is not modified
    params = base_params()
    snapshot = copy.deepcopy(params)
    parse_system(params)
    if params != snapshot:
        print("FAILED: parse_system modified its input")
        ok = False

    # Invalid inputs raise ValidationError
    params = base_params(); params["distinct_polymers"][0]["volume_fraction"] = 0.5
    ok &= expect_error(params, "volume fractions not summing to 1")
    params = base_params(); params["chi_n"] = {"A,Z": 1.0}
    ok &= expect_error(params, "chi_n with unknown monomer")
    params = base_params(); del params["chain_model"]
    ok &= expect_error(params, "missing chain_model")
    params = base_params(); params["lx"] = [4.0, 4.0]
    ok &= expect_error(params, "nx/lx length mismatch")
    params = base_params(); params["ds"] = -0.1
    ok &= expect_error(params, "negative ds")

    if ok:
        print("Test passed, parse_system()")
        return 0
    print("Test failed, parse_system()")
    return 1


if __name__ == "__main__":
    sys.exit(main())
