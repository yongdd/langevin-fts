"""Finite canonical brush: native CL versus exhaustive positive particle weights.

One grafted four-bead P chain, two C(+1), SP(+3), SM(-1), and S:
nine beads, C*V=9/4, N_ref=4, V=1. The first P bead is fixed at cell 0.
The reflecting bond, smearing, and Coulomb matrices are constructed explicitly
from cosine modes, independently of the propagator and electrostatics code.
This is a finite lattice identity check, not a continuum or PRL benchmark.
Submit each parameter configuration as a separate SLURM job for long runs.
"""
import argparse
import contextlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from brush_clfts import BrushChargedCLFTS
from validate_brush_sampling import saddle

TYPES = ["P", "C", "SP", "SM", "S"]
COUNTS = np.array([4, 2, 1, 1, 1])
CHARGES = np.array([-1., 1., 3., -1., 0.])
C = 9/4


def parameters(args):
    return dict(nx=[args.sites], lx=[1.], ds=.25, chain_model="discrete",
        segment_lengths={t: 1. for t in TYPES}, chi_n={"P,S": 2.}, zeta_n=4.,
        distinct_polymers=[dict(volume_fraction=n/9,
            blocks=[dict(type=t, length=1. if t == "P" else .25)],
            **({"grafting_points": {0: "G"}} if t == "P" else {}))
            for t, n in zip(TYPES, COUNTS)],
        charges=dict(zip(TYPES, CHARGES)), radiuses={t: .1 for t in TYPES},
        bjerrum_e=args.coupling, bc=["reflecting"]*2,
        graft_width=0., graft_center=0., chain_backend="spectral",
        langevin=dict(dt=args.dt, nbar=C*C, max_step=round(args.duration/args.dt)),
        alpha_ds=args.stabilization,
        recording=dict(dir=str(args.out), recording_period=round(args.duration/args.dt),
            sf_computing_period=10**9, sf_recording_period=10**9),
        platform=args.platform, verbose_level=0)


def cosine_operator(sites, eigenvalues):
    basis = np.cos(np.pi*(np.arange(sites)[:, None]+.5)*np.arange(sites)/sites)
    weights = np.full(sites, 2/sites)
    weights[0] = 1/sites
    return (basis*(weights*eigenvalues))@basis.T


def enumerate_particles(sites, coupling):
    k2 = (np.pi*np.arange(sites))**2
    bond = cosine_operator(sites, np.exp(-k2*.25/6))
    smear = cosine_operator(sites, np.exp(-.1**2*k2/2))
    inverse = np.zeros(sites)
    inverse[1:] = 1/k2[1:]
    green = cosine_operator(sites, inverse)
    assert bond.min() > 0 and np.max(abs(bond.sum(axis=1)-1)) < 1e-14
    positions = np.column_stack([np.zeros(sites**8, dtype=int),
                                np.array(list(itertools.product(range(sites), repeat=8)))])
    phi = np.zeros((len(positions), len(TYPES), sites))
    start = 0
    for t, count in enumerate(COUNTS):
        phi[:, t] = (positions[:, start:start+count, None] == np.arange(sites)).sum(axis=1)*sites/9
        start += count
    log_bond = sum(np.log(bond[positions[:, j], positions[:, j+1]]) for j in range(3))
    smooth = phi@smear.T
    rho = smooth.sum(axis=1)
    charge = np.einsum("cts,t->cs", smooth, CHARGES)
    contact = 2*np.mean(smooth[:, 0]*smooth[:, 4], axis=1)+2*np.mean((rho-1)**2, axis=1)
    coulomb = .5*coupling*np.mean(charge*(charge@green.T), axis=1)
    log_weight = log_bond-C*(contact+coulomb)
    weight = np.exp(log_weight-logsumexp(log_weight))
    mean = np.einsum("c,cts->ts", weight, phi)
    z = (np.arange(sites)+.5)/sites
    height = (mean[0]@z)/mean[0].sum()
    return dict(phi=phi, log_bond=log_bond, mean=mean, height=height,
                bond=bond, smear=smear, green=green, z=z,
                contact=contact, coulomb=coulomb)


def input_fields(sim, w):
    mon = sim.mpt.to_monomer_fields(w)
    return np.array([sim.electro.smear(t, mon[sim.monomer_types.index(t)]+
                     sim.electro.charges[t]*sim.psi) for t in TYPES])


def partition_cancellation(fields, bond):
    """|Q(complex W)|/Q(Re W); unaffected by uniform field shifts."""
    ratios = []
    for t, v in zip(TYPES, fields):
        ew = np.exp(-.25*(v-v.real.min()))
        real_ew = abs(ew)
        if t == "P":
            src = np.zeros(len(v)); src[0] = len(v)
            q, qr = src*ew, src*real_ew
            for _ in range(3):
                q, qr = ew*(bond@q), real_ew*(bond@qr)
        else:
            q, qr = ew, real_ew
        ratios.append(float(abs(q.sum())/qr.sum()))
    return ratios


def deterministic_checks(sim, exact):
    rng = np.random.default_rng(5517)
    shape = (len(TYPES), len(exact["z"]))
    w = rng.normal(0, 5, shape)+1j*rng.normal(0, 5, shape)
    w[sim._frozen_aux] = 0
    sim.psi = rng.normal(0, 5, shape[1])+1j*rng.normal(0, 5, shape[1])
    sim.psi -= sim.psi.mean()
    phi, _ = sim.compute_concentrations(w)
    fields = input_fields(sim, w)
    exponent = exact["log_bond"]-C*np.einsum("cts,ts->c", exact["phi"], fields)/shape[1]
    weight = np.exp(exponent-exponent.real.max())
    reference = np.einsum("c,cts->ts", weight, exact["phi"])/weight.sum()
    error = float(np.max(abs(np.array([phi[t] for t in TYPES])-reference)))
    assert error < 1e-11, error
    assert sim.solver.check_total_partition()
    # Gaussian elimination of the contact and electrostatic fields must give
    # the same particle energy used in the positive reference measure.
    elimination_error = 0.
    for index in np.linspace(0, len(exact["phi"])-1, 11, dtype=int):
        ph = exact["phi"][index]@exact["smear"].T
        wm = np.broadcast_to(4*(ph.sum(axis=0)-1), shape).copy()
        wm[0] += 2*ph[4]; wm[4] += 2*ph[0]
        wa = sim.mpt.to_aux_fields(np.array([wm[TYPES.index(t)] for t in sim.monomer_types]))
        h = sim.mpt.compute_hamiltonian(sim.molecules, wa, [1.]*len(TYPES), sim.cb, True)
        h += np.mean(np.sum(ph*wm, axis=0))
        c = CHARGES@ph
        psi = sim.electro.e_coupling*(exact["green"]@c)
        h += .5*np.mean(c*psi)
        elimination_error = max(elimination_error, abs(h-exact["contact"][index]-exact["coulomb"][index]))
    assert elimination_error < 1e-11, elimination_error
    return dict(fixed_complex_field_density_error=error, gaussian_elimination_error=float(elimination_error))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sites", type=int, choices=[2, 4], default=2)
    ap.add_argument("--coupling", type=float, default=25.)
    ap.add_argument("--dt", type=float, default=.01)
    ap.add_argument("--duration", type=float, default=2000.)
    ap.add_argument("--equil", type=float, default=200.)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--stabilization", type=float, default=0.)
    ap.add_argument("--platform", default="cpu-mkl")
    ap.add_argument("--deterministic-only", action="store_true")
    args = ap.parse_args()
    if args.dt <= 0 or args.duration <= args.equil or args.equil < 0:
        ap.error("Require dt > 0 and duration > equil >= 0")
    args.out.mkdir(parents=True, exist_ok=True)
    p = parameters(args)
    exact = enumerate_particles(args.sites, args.coupling)
    result = dict(params=p, species=TYPES, configurations=len(exact["phi"]),
        exact_profiles=exact["mean"].tolist(), exact_height=float(exact["height"]),
        note="Finite canonical lattice benchmark; not PRL or continuum validation.")
    samples, drift, ratios, times = [], [], [], []
    last = {}
    start = time.perf_counter()
    try:
        with open(args.out/"simulation.log", "w") as log, contextlib.redirect_stdout(log):
            sim = BrushChargedCLFTS(p, random_seed=args.seed)
            result["deterministic_checks"] = deterministic_checks(sim, exact)
            if not args.deterministic_only:
                w, _, error = saddle(sim)
                result["saddle_residual"] = error
                compute = sim.compute_concentrations
                steps = p["langevin"]["max_step"]
                equil = round(args.equil/args.dt)
                stride = max(1, round(.1/args.dt))
                call, max_mass = 0, 0.

                def sample(w_aux):
                    nonlocal call, max_mass
                    call += 1
                    last.update(w_aux=w_aux.copy(), psi=sim.psi.copy(), step=call)
                    ph, elapsed = compute(w_aux)
                    mass = max(abs(ph[t].mean()-n/9) for t, n in zip(TYPES, COUNTS))
                    if not np.isfinite(mass) or mass > 1e-10:
                        raise FloatingPointError(f"Species mass error {mass} at step {call}")
                    max_mass = max(max_mass, mass)
                    if call <= steps and call % 2000 == 0 and not sim.solver.check_total_partition():
                        raise FloatingPointError(f"Partition mismatch at step {call}")
                    if equil < call <= steps and call % stride == 0:
                        f = sim.mpt.compute_func_deriv(w_aux, sim.smearing.apply_to_dict(ph), sim._active_aux)
                        fp, _ = sim._psi_force(ph)
                        used = f.copy()
                        for j, i in enumerate(sim._active_aux):
                            if i in sim.mpt.aux_fields_real_idx:
                                used[j] += 1j*sim.alpha_ds*w_aux[i].imag
                        drift.append([max(float(abs(f).max()), float(abs(fp).max())),
                                      max(float(abs(used).max()), float(abs(fp).max()))])
                        samples.append(np.array([ph[t] for t in TYPES]))
                        ratios.append(partition_cancellation(input_fields(sim, w_aux), exact["bond"]))
                        times.append(call*args.dt)
                    return ph, elapsed

                sim.compute_concentrations = sample
                mon = sim.mpt.to_monomer_fields(w)
                sim.run(dict(zip(sim.monomer_types, mon)))
                result.update(max_species_mass_error=float(max_mass), n_samples=len(samples))
        result["status"] = "completed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      failure_step=last.get("step"))
        if last:
            np.savez_compressed(args.out/"failure_state.npz", **last)
    result["wall_seconds"] = time.perf_counter()-start
    np.savez_compressed(args.out/"samples.npz", profiles=samples, drift=drift,
        partition_cancellation=ratios, times=times, z=exact["z"])
    (args.out/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "params"}, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
