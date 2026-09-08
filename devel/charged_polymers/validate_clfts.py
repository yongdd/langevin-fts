"""Deterministic regression checks for charged brush CL; writes JSON evidence.

Run from an installed polymerfts environment with an idle GPU selected:
    python devel/charged_polymers/validate_clfts.py --output /path/results.json
No calibrated test parameters or pre-existing result files are changed.
"""
import argparse
import contextlib
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from brush_clfts import BrushChargedCLFTS
from clfts_charged import ChargedCLFTS
from stable_grafted_chain import ScaledGraftedChain


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
    from prl_scft import PRLBrushSCFT
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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


if __name__ == "__main__":
    main()
