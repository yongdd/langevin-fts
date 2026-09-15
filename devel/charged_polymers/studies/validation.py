"""Consolidated validation workflows. Run through workflow.py."""
import argparse
import contextlib
import copy
import json
import tempfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from models.brush_clfts import BrushChargedCLFTS
from models.clfts_charged import ChargedCLFTS
from models.stable_grafted_chain import ScaledGraftedChain
import time
from scipy.optimize import root

# Original workflow: validate_clfts.py

def parameters(platform="cpu-mkl"):
    fractions = {"P": .15, "C": .15, "SP": .02, "SM": .06, "S": .62}
    return dict(
        nx=[8, 32], lx=[1.2, 2.], ds=.1, chain_model="discrete",
        segment_lengths={t: 1. for t in fractions}, chi_n={"P,S": 10.}, zeta_n=20.,
        distinct_polymers=[dict(volume_fraction=f,
            blocks=[dict(type=t, length=1. if t == "P" else .1)],
            **({"grafting_points": {0: "G"}} if t == "P" else {}))
            for t, f in fractions.items()],
        charges={"P": -1., "C": 1., "SP": 3., "SM": -1., "S": 0.},
        radiuses={t: .2 for t in fractions}, bjerrum_e=25.,
        langevin=dict(max_step=4, dt=.025, nbar=1e4), alpha_ds=0.,
        recording=dict(dir="unused", recording_period=2,
                       sf_computing_period=1, sf_recording_period=2),
        chain_backend="direct", grafting="quenched", graft_width=0.,
        bc=["reflecting"]*4, bond_boundaries=["reflecting", "hard-reflecting"],
        platform=platform, verbose_level=0)


def strong_field(platform):
    N, nz, L = 50, 704, 6.
    b = 1/np.sqrt(N)
    z = (np.arange(nz)+.5)*L/nz
    src = np.exp(-.5*((z-b)/.2)**2)
    src /= src.sum()*L/nz
    chain = ScaledGraftedChain([nz], [L], N, b, ["absorbing"], platform=platform)
    v = np.where(z < 3., 100., 0.)/N
    phi = chain.compute(v, src).copy()
    logq = chain.log_partition
    h = (z*phi).sum()/phi.sum()
    # Independent direct image-sum result from the pre-fix audit.
    assert abs(logq.real + 62.93191845413595) < 1e-10
    assert abs(h-3.07601922390051) < 1e-10
    assert phi.real.min() >= 0
    assert chain.partition_error < 1e-11
    shifted = chain.compute(v+1000.+20j, src)
    assert np.max(abs(shifted-phi)) < 1e-9
    assert abs((chain.log_partition-logq).real + N*1000.) < 1e-9
    return dict(log_q_real=logq.real, height=float(h.real),
                mass_error=float(abs(phi.mean()-1)),
                large_uniform_shift_density_error=float(np.max(abs(shifted-phi))))


def forces_and_grafts(platform):
    p = parameters(platform)
    sim = BrushChargedCLFTS(p, random_seed=17)
    rng = np.random.default_rng(9183)
    M, ng = len(sim.monomer_types), sim.cb.get_total_grid()
    w = rng.normal(0., 5., (M, ng))+1j*rng.normal(0., 5., (M, ng))
    w[sim._frozen_aux] = 0.
    psi = rng.normal(0., 5., ng)+1j*rng.normal(0., 5., ng)
    psi -= psi.mean()
    sim.psi = psi.copy()
    phi, _ = sim.compute_concentrations(w)
    assert sim.solver.check_total_partition()
    mass_error = max(abs(phi[p0["blocks"][0]["type"]].mean()-p0["volume_fraction"])
                     for p0 in p["distinct_polymers"])
    anchor = sim.solver.chains[0].anchor_density.mean(axis=-1)
    anchor_error = float(np.max(abs(anchor-1)))
    assert mass_error < 1e-12 and anchor_error < 1e-12
    f = sim.mpt.compute_func_deriv(w, sim.smearing.apply_to_dict(phi), sim._active_aux)
    fp, _ = sim._psi_force(phi)

    def H(wv, pv):
        sim.psi = pv.copy()
        sim.compute_concentrations(wv)
        q = [sim.solver.get_total_partition(i) for i in range(M)]
        psif, c = sim._psi_force(sim.solver.phi)
        return sim._chain_hamiltonian(wv, q)+.5*np.mean(pv*(psif-c))

    errors, eps = [], 1e-4
    for j, i in enumerate(sim._active_aux):
        dw = np.zeros_like(w)
        dw[i] = rng.normal(size=ng)+1j*rng.normal(size=ng)
        expected = np.mean(f[j]*dw[i])*(-1 if i in sim.mpt.aux_fields_imag_idx else 1)
        actual = (H(w+eps*dw, psi)-H(w-eps*dw, psi))/(2*eps)
        errors.append(float(abs(actual-expected)))
    dp = rng.normal(size=ng)+1j*rng.normal(size=ng)
    dp -= dp.mean()
    errors.append(float(abs((H(w, psi+eps*dp)-H(w, psi-eps*dp))/(2*eps)-np.mean(fp*dp))))
    assert max(errors) < 1e-8, errors
    # A lateral field must distinguish the annealed and quenched ensembles.
    z = (np.arange(p["nx"][-1])+.5)*p["lx"][-1]/p["nx"][-1]
    x = (np.arange(p["nx"][0])+.5)*p["lx"][0]/p["nx"][0]
    lateral = np.broadcast_to(.5*np.cos(np.pi*x[:, None]/p["lx"][0]), p["nx"])
    src = np.broadcast_to((z == z[0]).astype(float), p["nx"])
    annealed = ScaledGraftedChain(p["nx"], p["lx"], 10, np.sqrt(.1),
                                 p["bond_boundaries"], "annealed", platform)
    annealed.compute(lateral, src)
    annealed_cv = float(annealed.anchor_density.mean(axis=-1).real.std())
    assert annealed_cv > .5
    return dict(species_mass_error=float(mass_error), anchor_column_error=anchor_error,
                derivative_errors=errors, annealed_anchor_cv=annealed_cv), phi


def checkpoint_and_monitoring(platform, root):
    p = parameters(platform)
    initial = {t: np.full(np.prod(p["nx"]), 5.+2j) for t in p["segment_lengths"]}
    p["recording"]["dir"] = str(root/"full")
    full = BrushChargedCLFTS(copy.deepcopy(p), random_seed=51)
    full.run(initial)
    p["recording"]["dir"] = str(root/"split")
    p["langevin"]["max_step"] = 2
    first = BrushChargedCLFTS(copy.deepcopy(p), random_seed=51)
    first.run(initial)
    p["langevin"]["max_step"] = 4
    resumed = BrushChargedCLFTS(copy.deepcopy(p), random_seed=9999)
    resumed.continue_run(str(root/"split"/"fields_000002.mat"))
    a = loadmat(root/"full"/"fields_000004.mat", squeeze_me=True)
    b = loadmat(root/"split"/"fields_000004.mat", squeeze_me=True)
    keys = [k for k in a if k.startswith(("w_", "phi_", "psi_real", "psi_imag"))
            and k != "w_shifts_json"]
    error = max(np.max(abs(a[k]-b[k])) for k in keys)
    assert error < 1e-12, error
    saved_params = json.loads(str(a["initial_params_json"]))
    assert saved_params["distinct_polymers"][0]["grafting_points"] == {"0": "G"}
    assert full._random_psi.bit_generator.state == resumed._random_psi.bit_generator.state
    # Saved densities must correspond to the saved (post-update) fields.
    wmon = np.array([a[f"w_{t}_real"]+1j*a[f"w_{t}_imag"] for t in full.monomer_types])
    phi, _ = full.compute_concentrations(full.mpt.to_aux_fields(wmon))
    saved_error = max(np.max(abs(phi[t]-a[f"phi_{t}_real"]-1j*a[f"phi_{t}_imag"])) for t in phi)
    assert saved_error < 1e-12
    # First recorded H must use the initial w, psi and Q at the same time.
    fresh = BrushChargedCLFTS(parameters(platform), random_seed=51)
    waux = fresh.mpt.to_aux_fields(np.array([initial[t] for t in fresh.monomer_types]))
    waux[fresh._frozen_aux] = 0.
    fresh.compute_concentrations(waux)
    h0 = fresh._chain_hamiltonian(waux, [1.]*len(fresh.monomer_types))
    hfile = loadmat(root/"full"/"dH_000002.mat", squeeze_me=True)
    recorded = hfile["H_history_real"][0]+1j*hfile["H_history_imag"][0]
    assert abs(h0-recorded) < 1e-12
    # Zero coupling is a defined, finite neutral electrostatic sector.
    p["bjerrum_e"] = 0.
    p["recording"]["dir"] = str(root/"zero")
    zero = BrushChargedCLFTS(p, random_seed=13)
    zero.psi[:] = 5.+2j  # E=0 must discard a previously nonzero psi state.
    zero.run(initial)
    assert not np.any(zero.psi)
    for name in ("dH_000002.mat", "fields_000004.mat"):
        data = loadmat(root/"zero"/name, squeeze_me=True)
        assert all(np.all(np.isfinite(v)) for k,v in data.items()
                   if k.startswith(("H_history", "psi_real", "psi_imag", "phi_")))
    p["recording"]["dir"] = str(root/"zero_periodic")
    periodic_zero = ChargedCLFTS(p, random_seed=13)
    periodic_zero.psi[:] = 5.+2j
    periodic_zero.run(initial)
    data = loadmat(root/"zero_periodic"/"dH_000002.mat", squeeze_me=True)
    assert not np.any(periodic_zero.psi)
    assert np.all(np.isfinite(data["H_history_real"]))
    assert np.all(np.isfinite(data["H_history_imag"]))
    return dict(restart_max_error=float(error), saved_density_error=float(saved_error),
                h_time_level_error=float(abs(h0-recorded)), zero_coupling_finite=True)


def donnan_check():
    from models.prl_scft import PRLBrushSCFT
    sim = PRLBrushSCFT(N=100, L=10., Nz=128, zplus=3, Ib_molar=.3, correlations=False)
    rho = np.full(sim.Nz, .1)
    c, p, m = sim._solve_pb(rho)
    charge = sim.zP*rho+c+sim.zp*p-m
    error = float(np.max(abs(charge)))
    assert error < 1e-10 and np.max(abs(sim.psi)) > .01
    donnan = float(sim.psi.mean())
    # A deliberately frozen, incorrect self-energy must remain unconverged
    # after the iteration cap, irrespective of the W-only stopping metric.
    sim.sigma = .01
    sim.uP[:] = 1.
    sim.solve(max_iter=1, lam_u=0., lam_w=0., psi_exact=True)
    assert sim.residuals["self_energy"] == 1.
    assert sim.err >= 1. and not sim.converged
    return dict(max_charge_error=error, donnan_potential=donnan,
                stale_self_energy_rejected=True)


def check_implementation():
    parser = argparse.ArgumentParser(description='Deterministic regression checks for charged brush CL; writes JSON evidence.\n\nRun from an installed polymerfts environment with an idle GPU selected:\n    python devel/charged_polymers/validate_clfts.py --output /path/results.json\nNo calibrated test parameters or pre-existing result files are changed.')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platforms", nargs="+", default=["cpu-mkl", "cuda"])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results, reference = {}, None
    with args.output.with_suffix(".log").open("w") as log, contextlib.redirect_stdout(log):
        for platform in args.platforms:
            with tempfile.TemporaryDirectory() as tmp:
                force_result, phi = forces_and_grafts(platform)
                results[platform] = dict(strong_field=strong_field(platform), forces=force_result,
                    checkpoint=checkpoint_and_monitoring(platform, Path(tmp)))
                if reference is not None:
                    difference = max(np.max(abs(phi[t]-reference[t])) for t in phi)
                    assert difference < 1e-12
                    results["cpu_cuda_phi_error"] = float(difference)
                reference = phi
        results["donnan"] = donnan_check()
    args.output.write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


# Original workflow: validate_brush_sampling.py

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


def run_brush():
    ap = argparse.ArgumentParser(description='Same-model MF/CL benchmark, NOT a PRL reproduction.\n\nUses the small, finite-compressibility, canonical model in validate_clfts.\nRun one SLURM job per dt/nbar/seed, e.g.:\n    python validate_brush_sampling.py --out /shared/run --dt .05 --nbar 1e4\nAll species block profiles, drift samples and final RNG checkpoints are kept.')
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
