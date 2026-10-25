#!/usr/bin/env python3
"""Input checks of the bindings, reduced-basis single steps and Anderson mixing
of non-finite histories.

- Monomer names ending with an underscore give the same results as plain names.
- NaN box lengths and angles are rejected.
- advance_propagator_single_segment works in the reduced basis when the
  computation has a space group, like compute_propagators.
- Anderson mixing of a history whose residual products overflow uses no
  history and calls no LAPACK routine, which would print DLASCL warnings.
"""

import subprocess
import sys

import numpy as np
from polymerfts import _core


def rejects(call):
    try:
        call()
    except (RuntimeError, ValueError):
        return
    raise AssertionError("Invalid input was accepted")


def underscore_names(platform, chain_model, names):
    a, b = names
    fac = _core.PlatformSelector.create_factory(platform, False, "real")
    mol = fac.create_molecules_information(chain_model, 0.1, {a: 1.0, b: 1.5})
    mol.add_polymer(0.5, [[a, 0.5, 0, 1], [b, 0.5, 1, 2]])
    # A star with two identical arms: its keys are aggregated
    mol.add_polymer(0.5, [[a, 0.4, 0, 1], [a, 0.4, 0, 2], [b, 0.3, 0, 3]])
    opt = fac.create_propagator_computation_optimizer(mol, True)
    cb = fac.create_computation_box([16], [4.0])
    pc = fac.create_propagator_computation(cb, mol, opt, "rqm4")
    w = np.random.default_rng(41).normal(0, 5, (2, 16))
    pc.compute_propagators({a: w[0], b: w[1]})
    pc.compute_concentrations()
    assert pc.check_total_partition()
    return np.array([pc.get_total_partition(p) for p in range(2)] +
                    [x for m in names for x in pc.get_total_concentration(m)])


def test_underscore_names(platform):
    for chain_model in ("continuous", "discrete"):
        plain = underscore_names(platform, chain_model, ("A", "B"))
        for names in (("A_", "B_"), ("A_", "_")):
            np.testing.assert_allclose(underscore_names(platform, chain_model, names), plain,
                                       rtol=1e-12, atol=0)


def test_nan_box(platform):
    fac = _core.PlatformSelector.create_factory(platform, False, "real")
    nan = float("nan")
    rejects(lambda: fac.create_computation_box([16], [nan]))
    rejects(lambda: fac.create_computation_box([8]*3, [2.0, nan, 2.0]))
    rejects(lambda: fac.create_computation_box([8]*3, [2.0]*3, [90.0, 90.0, nan]))
    cb = fac.create_computation_box([16], [4.0])
    rejects(lambda: cb.set_lx([nan]))
    assert cb.get_lx() == [4.0]


def test_single_segment_reduced_basis(platform):
    fac = _core.PlatformSelector.create_factory(platform, False, "real")
    nx = [8]*3
    sg = _core.SpaceGroup(nx, "Im-3m", 529)
    rng = np.random.default_rng(42)
    w = {m: sg.symmetrize(rng.normal(0, 5, 512)) for m in "AB"}
    q_in = sg.symmetrize(rng.normal(0, 5, 512))
    q_out = []
    for group in (None, sg):
        basis = (lambda x: x) if group is None else sg.to_reduced_basis
        cb = fac.create_computation_box(nx, [3.0]*3)
        mol = fac.create_molecules_information("continuous", 0.05, {"A": 1.0, "B": 1.0})
        mol.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
        opt = fac.create_propagator_computation_optimizer(mol, False)
        pc = fac.create_propagator_computation(cb, mol, opt, "rqm4", group)
        pc.compute_propagators({m: basis(w[m]) for m in "AB"})
        out = pc.advance_propagator_single_segment(basis(q_in), 0, 0, 1)
        assert out.shape == (cb.get_n_basis(),)
        q_out.append(out)
    # With a space group the full grid is rejected, like in compute_propagators
    rejects(lambda: pc.advance_propagator_single_segment(q_in, 0, 0, 1))
    np.testing.assert_allclose(q_out[1], sg.to_reduced_basis(q_out[0]), rtol=1e-11, atol=1e-11)


# Run in a child process, whose stdout and stderr capture what LAPACK prints
ANDERSON_CHILD = r"""
import sys
import numpy as np
from polymerfts import _core

n = 16
fac = _core.PlatformSelector.create_factory(sys.argv[1], False, "real")
am = fac.create_anderson_mixing(n, 5, 1e3, 0.1, 0.1)
w = np.linspace(0.1, 1.0, n)
# The squared norm of the oldest residual overflows, so in the third step the
# least-squares matrix of the two-entry history has one infinite element
residuals = [np.full(n, 1e200), np.linspace(-1.0, 1.0, n), np.cos(np.arange(n))]
for d in residuals:
    w_new = am.calculate_new_fields(w, d, 1.0, 0.5)
# No usable history: the new field is the newest field plus its residual
assert np.array_equal(w_new, w + residuals[-1])
"""


def test_anderson_non_finite(platform):
    child = subprocess.run([sys.executable, "-c", ANDERSON_CHILD, platform],
                           capture_output=True, text=True, timeout=600)
    output = child.stdout + child.stderr
    assert child.returncode == 0, output
    assert "DLASCL" not in output and "illegal value" not in output, output


def main():
    for platform in _core.PlatformSelector.avail_platforms():
        test_underscore_names(platform)
        test_nan_box(platform)
        test_single_segment_reduced_basis(platform)
        test_anderson_non_finite(platform)
        print("Input checks:", platform, "OK")


if __name__ == "__main__":
    main()
