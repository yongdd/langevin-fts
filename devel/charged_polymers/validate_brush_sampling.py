"""Same-model MF/CL benchmark, NOT a PRL reproduction.

Uses the small, finite-compressibility, canonical model in validate_clfts.
Run one SLURM job per dt/nbar/seed, e.g.:
    python validate_brush_sampling.py --out /shared/run --dt .05 --nbar 1e4
All species block profiles, drift samples and final RNG checkpoints are kept.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import root

from brush_clfts import BrushChargedCLFTS
from validate_clfts import parameters


def saddle(sim):
    """Solve all active field equations including their means; fix only psi's gauge."""
    active = sim._active_aux
    nz = sim.cb.get_nx()[-1]
    M, ng = len(sim.monomer_types), sim.cb.get_total_grid()

    def unpack(x):
        x = x.reshape(len(active)+1, nz)
        w = np.zeros((M, ng), dtype=complex)
        w[active] = np.tile(x[:-1], (1, ng//nz))
        sim.psi = np.tile(x[-1], ng//nz).astype(complex)
        return w

    def residual(x):
        w = unpack(x)
        phi, _ = sim.compute_concentrations(w)
        f = sim.mpt.compute_func_deriv(w, sim.smearing.apply_to_dict(phi), active)
        fp, _ = sim._psi_force(phi)
        fp = fp.reshape(-1, nz).mean(axis=0).real.copy()
        fp[0] = sim.psi.real.mean()
        return np.concatenate([f.reshape(len(active), -1, nz).mean(axis=1).real.ravel(), fp])

    result = root(residual, np.zeros((len(active)+1)*nz), tol=1e-10)
    error = float(np.max(abs(residual(result.x))))
    if error > 1e-8:
        raise RuntimeError(f"Same-model saddle failed: {result.message}, residual={error}")
    w = unpack(result.x)
    phi, _ = sim.compute_concentrations(w)
    return w, phi, error


def observables(profiles, z, mask, bulk):
    height = np.sum(z*profiles["P"])/np.sum(profiles["P"])
    excess = profiles["C"]-np.mean(profiles["C"][bulk]) + 3*(
        profiles["SP"]-np.mean(profiles["SP"][bulk]))
    gamma = 1-np.sum(excess[mask])/np.sum(profiles["P"][mask])
    return np.array([height, gamma])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dt", type=float, default=.05)
    ap.add_argument("--nbar", type=float, default=1e4)
    ap.add_argument("--duration", type=float, default=500.)
    ap.add_argument("--equil", type=float, default=100.)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--platform", default="cpu-mkl")
    ap.add_argument("--refinement", type=int, default=1)
    ap.add_argument("--saddle-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p = parameters(args.platform)
    if args.refinement < 1:
        raise ValueError("refinement must be a positive integer.")
    p["nx"] = [n*args.refinement for n in p["nx"]]
    steps = round(args.duration/args.dt)
    equil = round(args.equil/args.dt)
    if steps <= equil or equil < 0 or args.dt <= 0:
        raise ValueError("Require duration > equil >= 0 and dt > 0.")
    p["langevin"].update(dt=args.dt, nbar=args.nbar, max_step=steps)
    p["recording"].update(dir=str(args.out), recording_period=steps,
                         sf_computing_period=steps, sf_recording_period=steps)
    sim = BrushChargedCLFTS(p, random_seed=args.seed)
    w, phi, error = saddle(sim)
    nz = p["nx"][-1]
    z = (np.arange(nz)+.5)*p["lx"][-1]/nz
    prof = {t: v.reshape(-1, nz).mean(axis=0) for t,v in phi.items()}
    h = np.sum(z*prof["P"])/np.sum(prof["P"])
    mask, bulk = z < 2*h.real, z > .75*p["lx"][-1]
    mf = observables(prof, z, mask, bulk)
    result = dict(params=p, saddle_residual=error, mf_observables=mf.real.tolist(),
                  observable_names=["mean_z", "valence_weighted_gamma_fixed_mask"],
                  note="2D canonical finite-compressibility validation model; not PRL.")
    np.savez_compressed(args.out/"saddle.npz", w_aux=w, psi=sim.psi,
                        z=z, **{f"phi_{t}": v for t,v in prof.items()})
    if args.saddle_only:
        (args.out/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k != "params"}, indent=2))
        return
    nblock, ntypes = 20, len(sim.monomer_types)
    sums = np.zeros((nblock, ntypes, nz), dtype=complex)
    counts = np.zeros(nblock, dtype=int)
    samples, drift, sample_steps = [], [], []
    stride = max(1, round(.25/args.dt))
    compute = sim.compute_concentrations
    call, mass_error, partition_error = 0, 0., 0.
    fractions = {v["blocks"][0]["type"]: v["volume_fraction"] for v in p["distinct_polymers"]}

    def sample(w_aux):
        nonlocal call, mass_error, partition_error
        ph, elapsed = compute(w_aux)
        call += 1
        mass_error = max(mass_error, max(abs(ph[t].mean()-fractions[t]) for t in ph))
        partition_error = max(partition_error, max(c.partition_error for c in sim.solver.chains))
        if equil < call <= steps and call % stride == 0:
            pp = {t:v.reshape(-1,nz).mean(axis=0) for t,v in ph.items()}
            b = min(nblock-1, (call-equil-1)*nblock//(steps-equil))
            sums[b] += np.array([pp[t] for t in sim.monomer_types])
            counts[b] += 1
            samples.append(observables(pp,z,mask,bulk))
            f = sim.mpt.compute_func_deriv(w_aux, sim.smearing.apply_to_dict(ph), sim._active_aux)
            fp, _ = sim._psi_force(ph)
            drift.append(max(float(np.max(abs(f))), float(np.max(abs(fp)))))
            sample_steps.append(call)
        return ph, elapsed

    sim.compute_concentrations = sample
    mon = sim.mpt.to_monomer_fields(w)
    start = time.time()
    sim.run({t:mon[i] for i,t in enumerate(sim.monomer_types)})
    profiles = sums/counts[:,None,None]
    block_obs = np.array([observables(dict(zip(sim.monomer_types, a)),z,mask,bulk) for a in profiles])
    result.update(wall_seconds=time.time()-start, n_samples=len(samples),
        block_duration=(args.duration-args.equil)/nblock,
        cl_observables=block_obs.real.mean(axis=0).tolist(),
        block_standard_error=(block_obs.real.std(axis=0,ddof=1)/np.sqrt(nblock)).tolist(),
        cl_imaginary_mean=block_obs.imag.mean(axis=0).tolist(),
        imaginary_block_error=(block_obs.imag.std(axis=0,ddof=1)/np.sqrt(nblock)).tolist(),
        max_species_mass_error=float(mass_error), max_partition_error=partition_error,
        drift_quantiles=np.quantile(drift,[.5,.9,.99,.999,1.]).tolist(),
        block_error_note="Check serial correlation and seed consistency; block errors alone are not proof of CL correctness.")
    np.savez_compressed(args.out/"samples.npz", z=z, mask=mask, bulk=bulk,
        monomer_types=sim.monomer_types, block_profiles=profiles, block_counts=counts,
        block_observables=block_obs, observables=samples, steps=sample_steps, drift_max=drift)
    (args.out/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "params"},indent=2))


if __name__ == "__main__":
    main()
