#!/usr/bin/env python3
"""Gibbs-ensemble CL-FTS (gibbs_ensemble.py): two boxes exchange chains, point ions and volume.

Two homopolymers A and B (discrete chains, N = 20) on 4x4 grids:
1. Input: grand-canonical polymers, volume fractions that do not sum to 1, a species absent from a
   box, a charged box that is not neutral, skip_steps >= cycle_steps, a zero sample interval,
   point ions of continuous chains without n_segments and a negative seed are rejected.
2. Mean-field limit (nbar 1e28) of the symmetric blend at chi N = 3: the boxes reach the
   Flory-Huggins binodal ln(phi/(1 - phi)) = chi N (2 phi - 1).
3. Mean-field limit with charged A chains (one charge per chain), Cl- counterions and Na+ Cl- salt
   as canonical point ions. The coexistence of the Flory-Huggins free energy with ideal ions,
   sum_b [sum_k n_k (ln(n_k/V_b) - 1) + V_b chi N phi_A phi_B], is solved independently for the
   neutral transfers (B, NaCl, A with its Cl). Started there, the exchange potentials of the
   boxes agree in every neutral combination and the amounts stay; started from distant boxes,
   the boxes stay neutral and approach it.
4. Mean-field limit with charged A chains and grand-canonical Na+ Cl- salt (no canonical ions):
   the Donnan potential of each box, sinh(psi) = rho_A/(2 e^mu), lets the boxes carry their net
   charge. The coexistence of the same free energy with these potentials is solved independently;
   started there, every exchange potential agrees and the amounts stay, and from distant boxes
   the amounts approach it.
5. Fluctuating blend (nbar 1e4): three cycles continued by continue_run() reproduce six
   uninterrupted ones bit for bit (history and fields), and a second run() starts again from the
   boxes of the input; a continuation that lists the point ions in another order is refused.
"""

import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from scipy.optimize import brentq, fsolve
from polymerfts import GibbsEnsemble, PlatformSelector, ValidationError

N, CHI = 20, 3.0


def cpu_platform():
    return next(p for p in PlatformSelector.avail_platforms() if p.startswith("cpu"))


def parameters(directory, cycles, nbar=1e28, fractions=((0.7, 0.3), (0.3, 0.7)), charge=0.0, ions=None,
               volumes=(1.0, 1.0)):
    params = {"nx": [4, 4], "lx": [1.0, 1.0], "ds": 1 / N, "chain_model": "discrete",
              "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": CHI}, "zeta_n": 50.0,
              "polymers": [{"volume_fraction": 0.5, "blocks": [{"type": "A", "length": 1.0}]},
                           {"volume_fraction": 0.5, "blocks": [{"type": "B", "length": 1.0}]}],
              "platform": cpu_platform(), "langevin": {"nbar": nbar, "dt": 0.5, "max_step": 1},
              "recording": {"dir": str(directory), "checkpoint_interval": 1, "sample_interval": 1,
                            "statistics_save_interval": 1},
              "verbose_level": 0,
              "gibbs_ensemble": {"boxes": [{"lx": [np.sqrt(volume)] * 2, "volume_fractions": list(phi)}
                                           for volume, phi in zip(volumes, fractions)],
                                 "cycles": cycles, "cycle_steps": 20, "skip_steps": 10, "exchange_rate": 1.0}}
    if charge:
        params["smearing"] = {"type": "gaussian", "a_int": 0.1}
        params["electrostatics"] = {"charges": {"A": charge}, "coupling": 100.0, "ions": ions}
    return params


def quiet(make):
    with contextlib.redirect_stdout(io.StringIO()):
        return make()


def expect_validation_error(make, message):
    try:
        quiet(make)
    except ValidationError as error:
        assert message in str(error), f"expected '{message}', got '{error}'"
        return
    raise AssertionError(f"expected a ValidationError with '{message}'")


def salt(concentration=0.01):
    return {"Na": {"valence": 1.0, "concentration": concentration, "smearing_length": 0.1},
            "Cl": {"valence": -1.0, "concentration": concentration, "smearing_length": 0.1}}


def inputs(root):
    params = parameters(root / "inputs", 2)
    params["polymers"][1] = {"chemical_potential": 0.0, "blocks": [{"type": "B", "length": 1.0}]}
    expect_validation_error(lambda: GibbsEnsemble(params), "grand-canonical")
    params = parameters(root / "inputs", 2, fractions=((0.7, 0.4), (0.3, 0.7)))
    expect_validation_error(lambda: GibbsEnsemble(params), "must sum to 1")
    params = parameters(root / "inputs", 2)
    params["gibbs_ensemble"]["skip_steps"] = 20
    expect_validation_error(lambda: GibbsEnsemble(params), "smaller than 'cycle_steps'")
    params = parameters(root / "inputs", 2, charge=0.05, ions=salt())
    params["gibbs_ensemble"]["boxes"][0]["ion_concentrations"] = {"Na": 0.01, "Cl": 0.01}
    expect_validation_error(lambda: GibbsEnsemble(params), "not electroneutral")
    params = parameters(root / "inputs", 2, fractions=((1.0, 0.0), (0.3, 0.7)))
    expect_validation_error(lambda: GibbsEnsemble(params), "positive amount in both boxes")
    params = parameters(root / "inputs", 2)
    params["recording"]["sample_interval"] = 0
    expect_validation_error(lambda: GibbsEnsemble(params), "'recording.sample_interval' must be an integer")
    params = parameters(root / "inputs", 2, charge=0.05, ions=salt())
    params["chain_model"] = "continuous"
    expect_validation_error(lambda: GibbsEnsemble(params), "n_segments")
    expect_validation_error(lambda: GibbsEnsemble(parameters(root / "inputs", 2), random_seed=-5), "non-negative integer")


def blend(root):
    gibbs = quiet(lambda: GibbsEnsemble(parameters(root / "blend", 20), random_seed=1))
    quiet(gibbs.run)
    last = gibbs.history["cycles"][-1]["boxes"]
    binodal = brentq(lambda p: np.log(p / (1 - p)) - CHI * (2 * p - 1), 0.51, 0.999, xtol=1e-14)
    for b, expected in ((0, binodal), (1, 1 - binodal)):
        phi = last[b]["volume_fractions"][0]
        assert abs(phi - expected) < 2e-6, f"box {b}: phi_A {phi:.8f} against the binodal {expected:.8f}"
    assert abs(last[0]["volume"] - 1.0) < 1e-6 and abs(last[0]["volume"] + last[1]["volume"] - 2.0) < 1e-12, "volumes"


def coexistence():
    """Box I amounts (A, B, Na, Cl) per sqrt(Nbar) at the mean-field coexistence of 1 A, 1 B, 0.4 Na and
    1.4 Cl in total (A carries one charge), from the exchange potentials of the free energy."""
    total = np.array([1.0, 1.0, 0.4, 1.4])

    def potentials(n):
        volume = n[0] + n[1]
        phi_a, phi_b = n[0] / volume, n[1] / volume
        pressure = np.sum(n) / volume + CHI * phi_a * phi_b
        mu = np.log(n / volume)
        return mu + np.array([CHI * phi_b - pressure, CHI * phi_a - pressure, 0.0, 0.0])

    def neutral_differences(x):
        n = np.array([x[0], x[1], x[2], x[2] + x[0]])
        d = potentials(n) - potentials(total - n)
        return [d[1], d[2] + d[3], d[0] + d[3]]
    x = fsolve(neutral_differences, [0.82, 0.15, 0.12], xtol=1e-14)
    assert np.max(np.abs(neutral_differences(x))) < 1e-12
    box = np.array([x[0], x[1], x[2], x[2] + x[0]])
    return np.array([box, total - box])


def neutral_combinations(potentials):
    """Exchange potentials of the neutral transfers B, NaCl and A + Cl (A carries one charge)."""
    p = np.asarray(potentials)
    return np.array([p[1], p[2] + p[3], p[0] + p[3]])


def charged(root):
    """A chains carry one charge (z_A = 0.05 per segment, N = 20); amounts per sqrt(Nbar): n_k^b."""
    expected = coexistence()
    volumes = expected[:, 0] + expected[:, 1]
    fractions = [tuple(n[:2] / v) for n, v in zip(expected, volumes)]
    params = parameters(root / "charged_start", 3, charge=0.05, ions=salt(), fractions=fractions, volumes=volumes)
    for b in range(2):
        params["gibbs_ensemble"]["boxes"][b]["ion_concentrations"] = {
            "Na": expected[b, 2] / (N * volumes[b]), "Cl": expected[b, 3] / (N * volumes[b])}
    gibbs = quiet(lambda: GibbsEnsemble(params, random_seed=1))
    quiet(gibbs.run)
    for cycle in gibbs.history["cycles"]:
        boxes = cycle["boxes"]
        difference = neutral_combinations(boxes[0]["exchange_potentials"]) - \
            neutral_combinations(boxes[1]["exchange_potentials"])
        assert np.max(np.abs(difference)) < 1e-6, f"cycle {cycle['cycle']}: neutral exchange differences {difference}"
    assert np.max(np.abs(gibbs.amounts - expected)) < 1e-6, "amounts moved away from the coexistence"

    # From distant boxes: neutral at every cycle, approaching the coexistence
    gibbs = quiet(lambda: GibbsEnsemble(parameters(root / "charged", 30, charge=0.05, ions=salt()), random_seed=1))
    quiet(gibbs.run)
    charges = np.array([1.0, 0.0, 1.0, -1.0])
    distances = []
    for cycle in gibbs.history["cycles"]:
        amounts = np.array(cycle["amounts_after"])
        assert np.all(np.abs(amounts @ charges) < 1e-12), "boxes are not neutral"
        distances.append(np.max(np.abs(amounts - expected)))
    assert distances[-1] < 0.05 * distances[0] and all(b <= a * 1.001 for a, b in zip(distances, distances[1:])), \
        f"distance to the coexistence: {distances[0]:.2e} -> {distances[-1]:.2e}"


def donnan_coexistence(salt_density):
    """Box I amounts (A, B) per sqrt(Nbar) at the mean-field coexistence of 1 A and 1 B in total with
    grand-canonical salt of number density e^mu = salt_density in its reservoir (A carries one charge).
    The ions in a box follow its Donnan potential psi (per charge): densities e^(mu -+ psi), with
    sinh(psi) = rho_A/(2 e^mu) for neutrality; A gains psi in its chemical potential."""
    total = np.array([1.0, 1.0])

    def potentials(n):
        volume = n[0] + n[1]
        rho_a, rho_b = n / volume
        psi = np.arcsinh(rho_a / (2 * salt_density))
        pressure = rho_a + rho_b + 2 * salt_density * np.cosh(psi) + CHI * rho_a * rho_b
        return np.array([np.log(rho_a) + CHI * rho_b + psi - pressure, np.log(rho_b) + CHI * rho_a - pressure])
    x = fsolve(lambda x: potentials(x) - potentials(total - x), [0.85, 0.15], xtol=1e-13)
    assert np.max(np.abs(potentials(x) - potentials(total - x))) < 1e-12
    return np.array([x, total - x])


def donnan(root):
    """Charged A chains (one charge per chain) with grand-canonical Na+ Cl- of reservoir density 0.5."""
    ions = {"Na": {"valence": 1.0, "chemical_potential": np.log(0.5), "smearing_length": 0.1},
            "Cl": {"valence": -1.0, "chemical_potential": np.log(0.5), "smearing_length": 0.1}}
    expected = donnan_coexistence(0.5)
    volumes = expected.sum(axis=1)
    fractions = [tuple(n / v) for n, v in zip(expected, volumes)]
    gibbs = quiet(lambda: GibbsEnsemble(parameters(root / "donnan_start", 3, charge=0.05, ions=ions,
                                                   fractions=fractions, volumes=volumes), random_seed=1))
    quiet(gibbs.run)
    for cycle in gibbs.history["cycles"]:
        boxes = cycle["boxes"]
        difference = np.subtract(boxes[0]["exchange_potentials"], boxes[1]["exchange_potentials"])
        assert np.max(np.abs(difference)) < 1e-6, f"cycle {cycle['cycle']}: exchange differences {difference}"
    assert np.max(np.abs(gibbs.amounts - expected)) < 1e-6, "amounts moved away from the Donnan coexistence"

    # From distant boxes
    gibbs = quiet(lambda: GibbsEnsemble(parameters(root / "donnan", 20, charge=0.05, ions=ions), random_seed=1))
    quiet(gibbs.run)
    distances = [np.max(np.abs(np.array(cycle["amounts_after"]) - expected)) for cycle in gibbs.history["cycles"]]
    assert distances[-1] < 1e-3 * distances[0], f"distance to the coexistence: {distances[0]:.2e} -> {distances[-1]:.2e}"


def restart(root):
    def make(directory, cycles):
        return quiet(lambda: GibbsEnsemble(parameters(directory, cycles, nbar=1e4), random_seed=3))
    full = make(root / "full", 6)
    quiet(full.run)
    part = make(root / "part", 3)
    quiet(part.run)
    part = make(root / "part", 6)
    quiet(part.continue_run)
    a = json.loads((root / "full" / "gibbs_history.json").read_text())["cycles"]
    b = json.loads((root / "part" / "gibbs_history.json").read_text())["cycles"]
    assert a == b, "continued history differs from the uninterrupted one"
    for box in ("box0", "box1"):
        x = loadmat(root / "full" / box / "fields_000120.mat", squeeze_me=True)
        y = loadmat(root / "part" / box / "fields_000120.mat", squeeze_me=True)
        keys = [k for k in x if k.startswith("w_")]
        assert keys and all(np.array_equal(x[k], y[k]) for k in keys), f"{box}: continued fields differ"
    # A second run() starts again from the boxes of the input and replaces the history
    quiet(part.run)
    c = json.loads((root / "part" / "gibbs_history.json").read_text())["cycles"]
    assert c == a, "a second run() differs from the first"

    # The amounts and chemical potentials follow the order of the point ions, which a continuation
    # must keep, grand-canonical ones included
    ions = dict(salt(), K={"valence": 1.0, "chemical_potential": np.log(N * 0.005), "smearing_length": 0.1})
    def charged(cycles, ions):
        params = parameters(root / "ions", cycles, charge=0.05, ions=ions)
        params["electrostatics"]["screening_length"] = 0.5
        return quiet(lambda: GibbsEnsemble(params, random_seed=3))
    quiet(charged(1, ions).run)
    for order in (("Cl", "Na", "K"), ("K", "Na", "Cl")):
        expect_validation_error(lambda: charged(2, {name: ions[name] for name in order}).continue_run(), "another order")
    expect_validation_error(lambda: charged(2, salt()).continue_run(), "input differs")


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, check in (("inputs", inputs), ("blend binodal", blend), ("charged coexistence", charged),
                            ("Donnan coexistence", donnan), ("restart", restart)):
            check(root)
            print(f"[PASS] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
