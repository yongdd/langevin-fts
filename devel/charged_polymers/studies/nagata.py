"""Drift-tail criterion and convergent regimes of the two-cell CL fixture. Run through workflow.py.

The fixture is the two-cell canonical model of studies/exact.py with chi_PS N,
zeta N and E as parameters. The CL dynamics is that of studies/reduced.py (four
complex half-difference amplitudes: three active auxiliary fields and psi),
vectorized over independent walkers in NumPy. Every step adds the drift
magnitude max_i |f_i| to log-spaced bins: Nagata, Nishimura & Shimasaki,
PRD 94, 114515 (2016), require its distribution to fall off exponentially or
faster. The exact reference sums all 2^8 particle configurations.
"""
import argparse
import contextlib
import io
import itertools
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp

from studies.exact import C, CHARGES, COUNTS, TYPES, cosine_operator, parameters
from studies.validation import saddle
from models.brush_clfts import BrushChargedCLFTS

SMEAR_RADIUS = 0.1
OBSERVABLES = ["C_cell0", "P_cell0", "SP_cell0", "height", "field_2"]
BINS = np.logspace(-3, 8, 221)


def exact_reference(chi, zeta, coupling, sites=2):
    """Positive particle sum of the finite action (studies/exact.py with chi and zeta)."""
    k2 = (np.pi*np.arange(sites))**2
    bond = cosine_operator(sites, np.exp(-k2*.25/6))
    smear = cosine_operator(sites, np.exp(-SMEAR_RADIUS**2*k2/2))
    inverse = np.zeros(sites)
    inverse[1:] = 1/k2[1:]
    green = cosine_operator(sites, inverse)
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
    contact = chi*np.mean(smooth[:, 0]*smooth[:, 4], axis=1) + zeta/2*np.mean((rho-1)**2, axis=1)
    coulomb = .5*coupling*np.mean(charge*(charge@green.T), axis=1)
    log_weight = log_bond - C*(contact+coulomb)
    weight = np.exp(log_weight - logsumexp(log_weight))
    return dict(phi=phi, weight=weight, bond=bond, mean=np.einsum("c,cts->ts", weight, phi))


def build(chi, zeta, coupling):
    """Native model, reduced coefficients (as studies/reduced.py) and exact reference."""
    args = SimpleNamespace(sites=2, coupling=coupling, dt=1e-3, duration=1., stabilization=0.,
                           out=Path("unused"), platform="cpu-mkl")
    params = parameters(args)
    params["chi_n"] = {"P,S": chi}
    params["zeta_n"] = zeta
    with contextlib.redirect_stdout(io.StringIO()):
        sim = BrushChargedCLFTS(params, random_seed=1)
    active = sim._active_aux
    assert len(active) == 3, active
    exact = exact_reference(chi, zeta, coupling)
    weights = np.zeros(4)
    for remaining in itertools.product(range(2), repeat=3):
        pos = (0,)+remaining
        weights[pos.count(0)-1] += np.prod([exact["bond"][pos[j], pos[j+1]] for j in range(3)])
    config = dict(mapping=np.ascontiguousarray(
                      sim.mpt.matrix_a[[sim.monomer_types.index(t) for t in TYPES]][:, active]),
                  smear=float(np.exp(-SMEAR_RADIUS**2*np.pi**2/2)), chain_weights=weights,
                  coeff=sim.mpt.h_coef_mu2[active].copy(),
                  signs=np.array([1. if i in sim.mpt.aux_fields_real_idx else -1. for i in active]),
                  mobility=np.r_[sim.dt_scaling[active], sim.psi_dt_scaling], coupling=float(coupling))
    return sim, config, exact


def drift(x, cfg):
    """Reduced drift for walkers x (W, 4); also the cell-0 half-differences of the densities."""
    fields = cfg["smear"]*(x[:, :3]@cfg["mapping"].T + CHARGES[None, :]*x[:, 3:4])
    j = np.arange(4)
    exponent = -.25*fields[:, :1]*(2*j-2)[None, :]
    exponent = exponent - exponent.real.max(axis=1, keepdims=True)
    weight = cfg["chain_weights"][None, :]*np.exp(exponent)
    delta = np.empty_like(fields)
    delta[:, 0] = (weight*(2*j-2)[None, :]).sum(axis=1)/(9*weight.sum(axis=1))
    delta[:, 1:] = -COUNTS[None, 1:]/9*np.tanh(.25*fields[:, 1:])
    force = np.empty_like(x)
    derivative = 2*cfg["coeff"][None, :]*x[:, :3] + cfg["smear"]*(delta@cfg["mapping"])
    force[:, :3] = -(cfg["mobility"][:3]*cfg["signs"])[None, :]*derivative
    force[:, 3] = 0
    if cfg["coupling"] > 0:
        force[:, 3] = cfg["mobility"][3]*(-np.pi**2/cfg["coupling"]*x[:, 3] + cfg["smear"]*(delta@CHARGES))
    return force, delta


def check_against_native(sim, cfg, samples=32):
    """Reduced densities and drift against the native solver at random complex fields (std 5)."""
    rng = np.random.default_rng(5517)
    active = sim._active_aux
    errors = dict(density=0., drift=0.)
    for _ in range(samples):
        w = rng.normal(0, 5, (5, 2)) + 1j*rng.normal(0, 5, (5, 2))
        w[sim._frozen_aux] = 0.
        sim.psi = rng.normal(0, 5, 2) + 1j*rng.normal(0, 5, 2)
        sim.psi -= sim.psi.mean()
        if sim.electro.e_coupling == 0:
            sim.psi[:] = 0
        x = np.r_[(w[active, 0]-w[active, 1])/2, sim.psi[0]][None, :]
        f, d = drift(x, cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            ph, _ = sim.compute_concentrations(w)
        expected = np.array([ph[t] for t in TYPES])
        errors["density"] = max(errors["density"], float(np.max(abs(expected[:, 0]-COUNTS/9-d[0]))))
        derivatives = sim.mpt.compute_func_deriv(w, sim.smearing.apply_to_dict(ph), active)
        fp, _ = sim._psi_force(ph)
        expected_f = np.r_[-sim.dt_scaling[active]*(derivatives[:, 0]-derivatives[:, 1])/2,
                           sim.psi_dt_scaling*(fp[0]-fp[1])/2]
        errors["drift"] = max(errors["drift"], float(np.max(abs(f[0]-expected_f))))
    return errors


def observables(delta, x):
    return np.column_stack([COUNTS[1]/9 + delta[:, 1], COUNTS[0]/9 + delta[:, 0],
                            COUNTS[2]/9 + delta[:, 2], .5 - 9/16*delta[:, 0], x[:, 2]])


def exact_observables(exact, cfg):
    mean = exact["mean"]
    dphi = (exact["phi"][:, :, 0]-exact["phi"][:, :, 1])/2
    centers = -cfg["smear"]*(dphi@cfg["mapping"])/(2*cfg["coeff"])
    return [float(mean[1, 0]), float(mean[0, 0]), float(mean[2, 0]),
            float((mean[0]@np.array([.25, .75]))/mean[0].sum()), float(exact["weight"]@centers[:, 2])]


def simulate(cfg, x0, walkers, dt, equil, duration, seed, method):
    """Independent walkers from the saddle; drift magnitudes at every step after equilibration."""
    rng = np.random.default_rng(seed)
    x = np.tile(x0.astype(complex), (walkers, 1))
    previous = np.zeros_like(x)
    alive = np.ones(walkers, bool)
    scale = np.sqrt(2*dt/C*cfg["mobility"])
    imaginary = np.r_[cfg["signs"] < 0, True]
    sums = np.zeros((walkers, len(OBSERVABLES)), complex)
    counts = np.zeros(len(BINS)+1, np.int64)
    max_drift, measured = 0., 0
    for n in range(round(duration/dt)):
        f, delta = drift(x, cfg)
        magnitude = np.abs(f).max(axis=1)
        alive &= np.isfinite(magnitude) & np.isfinite(x).all(axis=1)
        if n >= round(equil/dt):
            counts += np.bincount(np.searchsorted(BINS, magnitude[alive]), minlength=len(BINS)+1)
            max_drift = max(max_drift, float(magnitude[alive].max()))
            sums[alive] += observables(delta[alive], x[alive])
            measured += 1
        noise = scale[None, :]*rng.normal(size=x.shape)
        noise = np.where(imaginary[None, :], 1j*noise, noise)
        if cfg["coupling"] == 0:
            noise[:, 3] = 0
        f = np.where(alive[:, None], f, 0)
        if method == "heun":  # additive-noise Heun, the same increment in both stages
            corrected, _ = drift(x + dt*f + noise, cfg)
            x = x + .5*dt*(f + np.where(alive[:, None], corrected, 0)) + noise
        else:  # production LM for the auxiliary fields, ETD1 for psi (studies/reduced.py)
            new = x.copy()
            new[:, :3] = x[:, :3] + dt*f[:, :3] + .5*(previous[:, :3] + noise[:, :3])
            if cfg["coupling"] > 0:
                gamma = np.pi**2/cfg["coupling"]*cfg["mobility"][3]
                decay, factor = np.exp(-gamma*dt), -np.expm1(-gamma*dt)/gamma
                amplitude = np.sqrt(-np.expm1(-2*gamma*dt)/(2*gamma*dt))
                new[:, 3] = decay*x[:, 3] + factor*(f[:, 3] + gamma*x[:, 3]) + amplitude*noise[:, 3]
            x, previous = new, noise
        x = np.where(alive[:, None], x, x0[None, :])
    return sums[alive]/measured, counts, max_drift, int(alive.sum()), measured


def run():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chi", type=float, default=2.)
    ap.add_argument("--zeta", type=float, default=4.)
    ap.add_argument("--coupling", type=float, default=100.)
    ap.add_argument("--dt", type=float, default=.000625)
    ap.add_argument("--walkers", type=int, default=2048)
    ap.add_argument("--equil", type=float, default=20.)
    ap.add_argument("--duration", type=float, default=120.)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--method", choices=["heun", "lm-etd"], default="heun")
    args = ap.parse_args()
    if args.dt <= 0 or args.walkers < 2 or args.equil < 0 or args.duration <= args.equil:
        ap.error("Require dt > 0, walkers >= 2 and duration > equil >= 0")
    started = time.perf_counter()
    sim, cfg, exact = build(args.chi, args.zeta, args.coupling)
    checks = check_against_native(sim, cfg)
    assert max(checks.values()) < 1e-10, checks
    with contextlib.redirect_stdout(io.StringIO()):
        w, _, residual = saddle(sim)
    x0 = np.r_[(w[sim._active_aux, 0]-w[sim._active_aux, 1])/2, sim.psi[0]]
    means, counts, max_drift, alive, measured = simulate(
        cfg, x0, args.walkers, args.dt, args.equil, args.duration, args.seed, args.method)
    estimate, error = means.mean(axis=0), means.std(axis=0, ddof=1)/np.sqrt(len(means))
    reference = exact_observables(exact, cfg)
    ideal = exact_observables(exact_reference(0., 0., 0.), cfg)[:4]
    summary = dict({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                   native_checks=checks, saddle_residual=residual, alive=alive, measured_steps=measured,
                   max_drift=max_drift, wall_seconds=time.perf_counter()-started, observables={})
    for k, name in enumerate(OBSERVABLES):
        summary["observables"][name] = dict(
            exact=reference[k], ideal=ideal[k] if k < 4 else None, cl=float(estimate[k].real),
            cl_imag=float(estimate[k].imag), se=float(error[k].real),
            z=float((estimate[k].real-reference[k])/error[k].real))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    np.savez(args.out/"drift_histogram.npz", bins=BINS, counts=counts)
    print(json.dumps({k: summary[k] for k in ("chi", "zeta", "coupling", "dt", "method", "alive",
                                              "max_drift", "observables")}, indent=1))


def tail_statistics(bins, counts):
    """Complementary CDF S(u) = P(max|f| > u) and its log-log slopes per half decade of u."""
    total = counts.sum()
    beyond = total - np.cumsum(counts[:-1])
    survival = beyond/total
    median = float(bins[np.searchsorted(1-survival, .5)])
    edges = [2, 3.16, 10, 31.6, 100, 316, 1000]
    slopes = {}
    for a, b in zip(edges[:-1], edges[1:]):
        ia, ib = np.searchsorted(bins, a), np.searchsorted(bins, b)
        slopes[f"{a:g}-{b:g}"] = (float(np.log(survival[ib]/survival[ia])/np.log(bins[ib]/bins[ia]))
                                  if beyond[ib] >= 20 else None)
    at = {f"S({k}x median)": float(survival[min(np.searchsorted(bins, k*median), len(survival)-1)])
          for k in (10, 100)}
    return survival, dict(samples=int(total), median=median, slopes=slopes, **at)


def analyze():
    ap = argparse.ArgumentParser(description="Tables and figures of `cl nagata` runs")
    ap.add_argument("campaign", type=Path, help="Directory with one subdirectory per run")
    ap.add_argument("--figure", type=Path, default=None, help="Figure path (.pdf/.png); default in campaign")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = []
    for path in sorted(args.campaign.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        histogram = np.load(path.parent/"drift_histogram.npz")
        survival, stats = tail_statistics(histogram["bins"], histogram["counts"])
        records.append(dict(name=path.parent.name, chi=summary["chi"], zeta=summary["zeta"],
                            coupling=summary["coupling"], dt=summary["dt"], method=summary["method"],
                            walkers=summary["walkers"], alive=summary["alive"], max_drift=summary["max_drift"],
                            observables=summary["observables"], tail=stats,
                            max_abs_z=max(abs(v["z"]) for v in summary["observables"].values()),
                            _survival=survival, _bins=histogram["bins"]))
    figure = args.figure or args.campaign/"nagata_drift_tails.pdf"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # Heun at the common step; solid: CL differs from the exact sum (max|z| > 3), dashed: agrees
    main = sorted((r for r in records if r["method"] == "heun" and r["dt"] == .000625),
                  key=lambda r: (r["coupling"], r["zeta"]))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for k, record in enumerate(main):
        s, b = record["_survival"], record["_bins"]
        ax = axes[0] if record["coupling"] == 0 else axes[1]
        ax.loglog(b[s > 0], s[s > 0], "-" if record["max_abs_z"] > 3 else "--", color=colors[k % 20],
                  label=f"ζN={record['zeta']:g}, E={record['coupling']:g} (max|z|={record['max_abs_z']:.1f})")
    u = np.logspace(1, 3.5, 8)
    for ax, title in zip(axes[:2], ["E = 0", "E > 0"]):
        ax.loglog(u, 3e-3*u**-2., "k:", lw=1, label=r"$\propto u^{-2}$")
        ax.set(xlabel=r"$u=\max_i|f_i|$", ylabel=r"$P(\max_i|f_i|>u)$", title=title,
               xlim=(.3, 3e4), ylim=(1e-10, 1.5))
        ax.grid(alpha=.3)
        ax.legend(fontsize=6.5)
    for record in records:
        marker = {"heun": "o", "lm-etd": "s"}[record["method"]] if record["dt"] == .000625 else "^"
        axes[2].semilogx(record["tail"]["S(10x median)"], record["max_abs_z"], marker,
                         color="C3" if record["coupling"] > 0 else "C0", ms=6, alpha=.8)
    axes[2].axhline(2, color="k", ls=":", lw=1)
    axes[2].set(xlabel="fraction of steps with drift > 10 × median", ylabel="max |CL − exact| / SE",
                title="bias against tail weight (blue E = 0, red E > 0)")
    axes[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(figure)
    fig.savefig(Path(figure).with_suffix(".png"), dpi=130)
    table = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    (args.campaign/"nagata_summary.json").write_text(json.dumps(table, indent=1)+"\n")
    print(f"{'run':16s} {'max|z|':>6s} {'S(10x)':>9s} {'S(100x)':>9s}  slopes per half decade of u")
    for r in table:
        slopes = " ".join("   -  " if v is None else f"{v:6.2f}" for v in r["tail"]["slopes"].values())
        print(f"{r['name']:16s} {r['max_abs_z']:6.2f} {r['tail']['S(10x median)']:9.2e} "
              f"{r['tail']['S(100x median)']:9.2e}  {slopes}")
