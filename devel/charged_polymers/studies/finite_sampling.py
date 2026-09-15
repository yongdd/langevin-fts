"""Consolidated finite sampling workflows. Run through workflow.py."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from numba import njit
from studies.statistics import statistics
from studies.exact import C, CHARGES, COUNTS, TYPES, cosine_operator, enumerate_particles

# Original workflow: sample_positive_1d.py

BEAD_TYPES = np.repeat(np.arange(5), COUNTS)


def setup(sites, coupling, field_setup):
    saved = np.load(field_setup)
    attractive = np.ascontiguousarray(saved["mapping"][:, 0])
    coefficient = float(saved["coeff"][0])
    assert coefficient > 0 and saved["signs"][0] == 1
    k2 = (np.pi * np.arange(sites)) ** 2
    bond = cosine_operator(sites, np.exp(-k2 * .25 / 6))
    smear = cosine_operator(sites, np.exp(-.1**2 * k2 / 2))
    inv = np.zeros(sites)
    inv[1:] = 1 / k2[1:]
    green = cosine_operator(sites, inv)
    overlap = smear.T @ smear
    electro = smear.T @ green @ smear
    contact = np.full((5, 5), 4.)
    contact[0, 4] += 2
    contact[4, 0] += 2
    prefactor = C * sites / 81
    physical = prefactor * (contact[:, :, None, None] * overlap
        + coupling * np.outer(CHARGES, CHARGES)[:, :, None, None] * electro)
    repulsive = physical + prefactor * np.outer(attractive, attractive)[:, :, None, None] * (
        overlap - 1 / sites) / (2 * coefficient)
    assert bond.min() > 0 and np.max(abs(bond.sum(axis=1) - 1)) < 1e-13
    return bond, smear, green, physical, repulsive, attractive, coefficient


@njit(cache=True)
def particle_energy(positions, pair):
    energy = 0.
    for i in range(9):
        for j in range(9):
            energy += .5 * pair[BEAD_TYPES[i], BEAD_TYPES[j], positions[i], positions[j]]
    return energy


@njit(cache=True)
def conditional(positions, index, pair, log_bond, field_potential):
    sites = len(log_bond)
    t = BEAD_TYPES[index]
    exponent = np.empty(sites)
    for r in range(sites):
        energy = .5 * pair[t, t, r, r] + field_potential[t, r]
        for j in range(9):
            if j != index:
                energy += pair[t, BEAD_TYPES[j], r, positions[j]]
        exponent[r] = -energy
        if index < 4:
            exponent[r] += log_bond[positions[index - 1], r]
            if index < 3:
                exponent[r] += log_bond[r, positions[index + 1]]
    weight = np.exp(exponent - exponent.max())
    return weight / weight.sum()


@njit(cache=True)
def sample(sweeps, burn, stride, rng, bond, smear, pair, attractive, coefficient, joint):
    sites = len(bond)
    positions = np.zeros(9, np.int64)
    log_bond = np.log(bond)
    x = np.zeros(sites)
    fields = np.zeros((5, sites))
    n = (sweeps - burn) // stride
    obs = np.empty((n, 3 + 7 * sites))
    z = (np.arange(sites) + .5) / sites
    mode = np.sqrt(2 / sites) * np.cos(np.pi * z)
    count = 0
    field_variance = sites / (2 * C * coefficient)
    for sweep in range(sweeps):
        if joint:
            smooth_x = smear @ x
            for t in range(5):
                fields[t] = C / 9 * attractive[t] * smooth_x
        for index in rng.permutation(8) + 1:
            weights = conditional(positions, index, pair, log_bond, fields)
            draw = rng.random()
            total = 0.
            for r in range(sites):
                total += weights[r]
                if draw < total or r == sites - 1:
                    positions[index] = r
                    break
        phi = np.zeros((5, sites))
        for i in range(9):
            phi[BEAD_TYPES[i], positions[i]] += sites / 9
        linear = attractive @ (phi @ smear.T)
        linear -= linear.mean()
        center = -linear / (2 * coefficient)
        if joint:
            x = center + np.sqrt(field_variance) * rng.normal(0., 1., sites)
            x -= x.mean()
        if sweep >= burn and (sweep - burn) % stride == 0 and count < n:
            obs[count, 0] = (phi[0] @ z) / (sites * 4 / 9)
            obs[count, 1:1 + 5 * sites] = phi.ravel()
            # Rao-Blackwellized local field moments; actual first-mode moments for joint sampling.
            obs[count, 1 + 5 * sites:1 + 6 * sites] = center
            obs[count, 1 + 6 * sites:1 + 7 * sites] = center**2 + field_variance * (1 - 1 / sites)
            amplitude = (x if joint else center) @ mode
            obs[count, -2] = amplitude
            obs[count, -1] = amplitude**2 + (0. if joint else field_variance)
            count += 1
    return obs[:count], positions, x


def deterministic_checks(config, coupling):
    bond, smear, green, physical, repulsive, attractive, coefficient = config
    sites = len(bond)
    rng = np.random.default_rng(5517)
    maximum = 0.
    conditional_error = 0.
    for _ in range(32):
        p = rng.integers(sites, size=9); p[0] = 0
        x = rng.normal(0, 5, sites); x -= x.mean()

        def direct_energy(positions, with_field):
            phi = np.array([(positions[BEAD_TYPES == t, None] == np.arange(sites)).sum(axis=0)
                            for t in range(5)]) * sites / 9
            smooth = phi @ smear.T
            charge = CHARGES @ smooth
            energy = C * (2 * np.mean(smooth[0] * smooth[4]) + 2 * np.mean((smooth.sum(axis=0) - 1)**2)
                          + coupling / 2 * np.mean(charge * (green @ charge)))
            linear = attractive @ smooth; linear -= linear.mean()
            if with_field:
                energy += C * np.mean(linear**2 / (4 * coefficient) + linear * x)
            return energy, linear

        energy, linear = direct_energy(p, False)
        maximum = max(maximum, abs(particle_energy(p, physical) - 2 * C - energy),
            abs(particle_energy(p, repulsive) - C * np.mean(linear**2) / (4 * coefficient) - 2 * C - energy))
        field = C / 9 * attractive[:, None] * (smear @ x)[None, :]
        for index in [1, 3, 4, 6, 8]:
            log_weight = []
            for r in range(sites):
                trial = p.copy(); trial[index] = r
                e, _ = direct_energy(trial, True)
                log_weight.append(-e + sum(np.log(bond[trial[j], trial[j + 1]]) for j in range(3)))
            reference = np.exp(np.array(log_weight) - max(log_weight)); reference /= reference.sum()
            conditional_error = max(conditional_error, float(np.max(abs(
                conditional(p, index, repulsive, np.log(bond), field) - reference))))
    assert maximum < 1e-11, maximum
    assert conditional_error < 1e-11, conditional_error
    return dict(gaussian_elimination_energy_error=float(maximum),
                heatbath_conditional_probability_error=conditional_error, random_field_std=5.)


def run():
    ap = argparse.ArgumentParser(description='Positive particle/real-field Gibbs sampler for the existing 1D finite action.\n\nIntegrate the repulsive auxiliary fields and psi analytically, retain the\nattractive real field, and update particles and that field conditionally.\nThis avoids exhaustive enumeration in production sampling. It is not CL and\nnot the PRL model. A fully integrated particle Gibbs sampler is a control.')
    ap.add_argument("--sites", type=int, required=True)
    ap.add_argument("--coupling", type=float, required=True)
    ap.add_argument("--mode", choices=["joint", "particles"], default="joint")
    ap.add_argument("--sweeps", type=int, default=2000000)
    ap.add_argument("--burn", type=int, default=200000)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--field-setup", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.sites < 2 or args.coupling < 0 or not 0 <= args.burn < args.sweeps or args.stride < 1:
        ap.error("Require sites>=2, E>=0, sweeps>burn>=0, stride>=1")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = setup(args.sites, args.coupling, args.field_setup)
    checks = deterministic_checks(config, args.coupling)
    bond, smear, _, physical, repulsive, attractive, coefficient = config
    rng = np.random.default_rng(args.seed)
    values, positions, x = sample(args.sweeps, args.burn, args.stride, rng, bond, smear,
        repulsive if args.mode == "joint" else physical, attractive, coefficient, args.mode == "joint")
    labels = ["height"] + [f"phi_{t}_cell{i}" for t in TYPES for i in range(args.sites)]
    labels += [f"field_mean_cell{i}" for i in range(args.sites)] + [f"field_second_cell{i}" for i in range(args.sites)]
    labels += ["field_mode1_mean", "field_mode1_second"]
    stats = statistics(values, args.stride)
    mass_error = float(np.max(abs(values[:, 1:1 + 5 * args.sites].reshape(-1, 5, args.sites).mean(axis=2) - COUNTS / 9)))
    assert mass_error < 1e-12
    result = dict(status="completed", sites=args.sites, coupling=args.coupling, mode=args.mode,
        seed=args.seed, sweeps=args.sweeps, burn=args.burn, stride=args.stride,
        labels=labels, statistics=stats, deterministic_checks=checks, max_species_mass_error=mass_error,
        field_moment_estimator="Local moments conditional on particles; first-mode moments use actual field in joint mode",
        final_positions=positions.tolist(), final_real_field=x.tolist(), final_rng_state=rng.bit_generator.state,
        note="Same 9-bead finite action; particle/real-field Gibbs, not CL or PRL.")
    if args.sites <= 4:
        exact = enumerate_particles(args.sites, args.coupling)
        logw = exact["log_bond"] - C * (exact["contact"] + exact["coulomb"])
        weights = np.exp(logw - max(logw)); weights /= weights.sum()
        linear = (exact["phi"] @ smear.T).transpose(0, 2, 1) @ attractive
        linear -= linear.mean(axis=1, keepdims=True)
        centers = -linear / (2 * coefficient)
        target = np.r_[exact["height"], exact["mean"].ravel(), weights @ centers,
                       weights @ centers**2 + (args.sites - 1) / (2 * C * coefficient)]
        mode = np.sqrt(2 / args.sites) * np.cos(np.pi * (np.arange(args.sites) + .5) / args.sites)
        amplitudes = centers @ mode
        target = np.r_[target, weights @ amplitudes,
                       weights @ amplitudes**2 + args.sites / (2 * C * coefficient)]
        zscore = abs(np.array(stats["mean"]) - target) / np.maximum(stats["se"], 1e-300)
        result.update(exact=target.tolist(), max_standardized_discrepancy=float(zscore.max()))
    result["wall_seconds"] = time.perf_counter() - started
    np.savez_compressed(args.out / "samples.npz", observables=values,
        sweeps=args.burn + np.arange(len(values)) * args.stride)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["statistics", "labels", "exact"]}, indent=2))
