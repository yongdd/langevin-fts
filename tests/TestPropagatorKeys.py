#!/usr/bin/env python3
"""
Propagator keys and masked solvents.

1. Grafting labels are opaque names: labels with digits ("G1", "g_2", "1", "L10")
   give the same Q and phi as the label "G" for linear and branched chains with
   grafted ends. Until 2026-10 the digits inside {label} were read as segment
   numbers (map::at on any chain of two or more blocks). Labels with characters
   of the key format are rejected.
2. Aggregation with blocks of different local ds: a comb whose side blocks are
   not multiples of ds gives the same Q and phi with aggregation on and off.
   Until 2026-10 the mixed keys of the aggregated propagators took the ds index
   of the outer block for every dependency (map::at).
3. A solvent in a masked box: its concentration vanishes inside the mask and
   its volume average is its volume fraction. Until 2026-10 it was nonzero
   inside the obstacles (0.23 in the case below).
4. A mask with a space group: for a mask with the symmetry of the group, the
   space group (irreducible and m3 bases) gives the result of the full grid,
   including the solvent partition function and volume averages; until 2026-10
   the reduced-basis integrals had no mask weights (Q_solvent 0.9955 instead of
   0.9694 here). A mask without the symmetry is rejected (it used to be replaced
   silently: an off-center obstacle disappeared, Q off by 4.5 %).
"""

import io
import sys
import contextlib

import numpy as np
from polymerfts import _core
from polymerfts.propagator_solver import PropagatorSolver

n_pass = 0
n_fail = 0


def check(cond, msg):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"[PASS] {msg}")
    else:
        n_fail += 1
        print(f"[FAIL] {msg}")


def cpu_platform():
    cpu = [p for p in _core.PlatformSelector.avail_platforms() if p.startswith("cpu")]
    return cpu[0] if cpu else "cuda"


def solver(nx, lx, ds, bonds, aggregate=True, mask=None):
    return PropagatorSolver(nx=nx, lx=lx, ds=ds, bond_lengths=bonds, bc=["periodic"] * (2 * len(nx)),
                            chain_model="continuous", numerical_method="rqm4", platform=cpu_platform(),
                            aggregate_propagator_computation=aggregate, mask=mask)


def test_grafting_labels():
    rng = np.random.default_rng(2)
    nx, lx = [24], [2.0]
    w = {"A": rng.normal(0.0, 2.0, 24), "B": rng.normal(0.0, 2.0, 24)}
    q0 = np.exp(-((np.arange(24) - 6.0) ** 2) / 4.0)
    architectures = {
        "linear": [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]],
        "branched": [["A", 0.4, 0, 1], ["B", 0.3, 1, 2], ["B", 0.3, 1, 3]],
    }
    for name, blocks in architectures.items():
        results = {}
        for label in ("G", "G1", "g_2", "1", "L10"):
            with contextlib.redirect_stdout(io.StringIO()):
                s = solver(nx, lx, 0.05, {"A": 1.0, "B": 1.2})
                s.add_polymer(1.0, blocks, grafting_points={0: label})
                s.compute_propagators(w, q_init={label: q0})
                s.compute_concentrations()
            results[label] = (s.get_partition_function(0), np.array(s.get_concentration("A")))
        q_ref, phi_ref = results["G"]
        err = max(max(abs(q - q_ref) / abs(q_ref), np.max(np.abs(phi - phi_ref))) for q, phi in results.values())
        check(err < 1e-14, f"{name} chain: grafting labels with digits give the result of 'G' ({err:.1e})")

    for label in ("G(1)", "a+b", "x,y", ""):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                s = solver(nx, lx, 0.05, {"A": 1.0, "B": 1.2})
                s.add_polymer(1.0, architectures["linear"], grafting_points={0: label})
                s.compute_propagators(w, q_init={label: q0})
            rejected = False
        except Exception as e:
            rejected = "invalid initial condition label" in str(e)
        check(rejected, f"grafting label '{label}' is rejected")


def comb(backbone, side, arm):
    """Backbone of three blocks; a side block at nodes 1, 2, 3, each ending in two arms."""
    blocks = [["C", backbone, 0, 1], ["C", backbone, 1, 2], ["C", backbone, 2, 3]]
    v = 4
    for node in (1, 2, 3):
        blocks.append(["B", side, node, v])
        blocks += [["A", arm, v, v + 1], ["A", arm, v, v + 2]]
        v += 3
    return blocks


def test_aggregation_mixed_ds():
    rng = np.random.default_rng(1)
    nx, lx = [32], [3.0]
    w = {m: rng.normal(0.0, 2.0, 32) for m in "ABC"}
    for side, arm in ((0.35, 0.3), (0.33, 0.27), (0.3, 0.3)):
        out = {}
        for aggregate in (False, True):
            with contextlib.redirect_stdout(io.StringIO()):
                s = solver(nx, lx, 0.1, {"A": 1.0, "B": 1.1, "C": 0.9}, aggregate=aggregate)
                s.add_polymer(0.7, comb(0.2, side, arm))
                s.add_polymer(0.3, [["A", 0.45, 0, 1], ["B", 0.35, 1, 2]])
                s.compute_propagators(w)
                s.compute_concentrations()
            out[aggregate] = ([s.get_partition_function(p) for p in range(2)],
                              {m: np.array(s.get_concentration(m)) for m in "ABC"})
        dq = max(abs(a - b) / abs(b) for a, b in zip(out[True][0], out[False][0]))
        dphi = max(np.max(np.abs(out[True][1][m] - out[False][1][m])) for m in "ABC")
        check(dq < 1e-12 and dphi < 1e-12,
              f"comb with side {side}, arms {arm} (ds 0.1): aggregation on = off (Q {dq:.1e}, phi {dphi:.1e})")


def test_masked_solvent():
    nx = [16, 16, 16]
    x = (np.arange(16) + 0.5) / 16
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    mask = (((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) > 0.15 ** 2).astype(float).ravel()
    rng = np.random.default_rng(0)
    with contextlib.redirect_stdout(io.StringIO()):
        s = solver(nx, [3.0] * 3, 0.05, {"A": 1.0, "S": 1.0}, mask=mask)
        s.add_polymer(0.8, [["A", 1.0, 0, 1]])
        s._molecules.add_solvent(0.2, "S")
        s.compute_propagators({"A": rng.normal(0, 1, 4096), "S": rng.normal(0, 1, 4096)})
        s.compute_concentrations()
    phi_a = np.array(s.get_concentration("A"))
    phi_s = np.array(s.get_concentration("S"))
    inside = mask < 0.5
    check(np.max(np.abs(phi_s[inside])) == 0.0 and np.max(np.abs(phi_a[inside])) == 0.0,
          "masked box: no polymer and no solvent inside the obstacle")
    check(abs(np.mean(phi_s) - 0.2) < 1e-12 and abs(np.mean(phi_a) - 0.8) < 1e-12,
          f"masked box: volume averages are the volume fractions (solvent {np.mean(phi_s):.15f})")


def test_mask_space_group():
    nx, lx = [16, 16, 16], [3.0, 3.0, 3.0]
    x = (np.arange(16) + 0.5) / 16
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    def dist2(center):
        d = np.abs(np.stack([X, Y, Z]) - center)
        d = np.minimum(d, 1.0 - d)
        return np.sum(d * d, axis=0)

    # Cavities at the corner and the body center (Im-3m), a symmetric field
    mask = np.ones(nx)
    mask[dist2(0.0) < 0.15 ** 2] = 0.0
    mask[dist2(0.5) < 0.15 ** 2] = 0.0
    mask = mask.ravel()
    with contextlib.redirect_stdout(io.StringIO()):
        sg0 = _core.SpaceGroup(nx, "Im-3m", 529)
    rng = np.random.default_rng(0)
    w_full = sg0.from_reduced_basis(sg0.to_reduced_basis(rng.normal(0.0, 1.0, (2, 4096))))
    w_full *= 3.0 / np.max(np.abs(w_full))

    def run(basis):
        with contextlib.redirect_stdout(io.StringIO()):
            sg = None
            if basis is not None:
                sg = _core.SpaceGroup(nx, "Im-3m", 529)
                if basis == "m3":
                    sg.enable_m3_physical_basis()
            s = PropagatorSolver(nx=nx, lx=lx, ds=0.05, bond_lengths={"A": 1.0, "S": 1.0}, bc=["periodic"] * 6,
                                 chain_model="continuous", platform=cpu_platform(), mask=mask, space_group=sg)
            s.add_polymer(0.8, [["A", 1.0, 0, 1]])
            s._molecules.add_solvent(0.2, "S")
            convert = (lambda f: f) if sg is None else (lambda f: sg.to_reduced_basis(f[None, :])[0])
            w = {"A": convert(w_full[0]), "S": convert(w_full[1])}
            s.compute_propagators(w)
            s.compute_concentrations()
            expand = (lambda f: f) if sg is None else (lambda f: sg.from_reduced_basis(np.asarray(f)[None, :])[0])
            return {"Q": s.get_partition_function(0), "Q_solvent": s.propagator_computation.get_solvent_partition(0),
                    "phi_S": expand(s.get_concentration("S")), "mean_w": s.mean(w["A"])}

    reference = run(None)
    for basis in ("irreducible", "m3"):
        r = run(basis)
        diff = max(abs(r["Q"] - reference["Q"]) / reference["Q"],
                   abs(r["Q_solvent"] - reference["Q_solvent"]) / reference["Q_solvent"],
                   np.max(np.abs(r["phi_S"] - reference["phi_S"])), abs(r["mean_w"] - reference["mean_w"]))
        check(diff < 1e-12, f"symmetric mask, Im-3m {basis} basis: Q, Q_solvent, phi_S, mean(w) = full grid ({diff:.1e})")

    # An obstacle off the symmetry positions
    asymmetric = (((X - 0.3) ** 2 + (Y - 0.6) ** 2 + (Z - 0.45) ** 2) > 0.12 ** 2).astype(float).ravel()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sg = _core.SpaceGroup(nx, "Im-3m", 529)
            s = PropagatorSolver(nx=nx, lx=lx, ds=0.05, bond_lengths={"A": 1.0}, bc=["periodic"] * 6,
                                 chain_model="continuous", platform=cpu_platform(), mask=asymmetric, space_group=sg)
            s.add_polymer(1.0, [["A", 1.0, 0, 1]])
            s.compute_propagators({"A": sg.to_reduced_basis(w_full[:1])[0]})
        rejected = False
    except Exception as e:
        rejected = "mask is not invariant under the space group" in str(e)
    check(rejected, "a mask without the symmetry of the space group is rejected")


def main():
    test_grafting_labels()
    test_aggregation_mixed_ds()
    test_masked_solvent()
    test_mask_space_group()
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
