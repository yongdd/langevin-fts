"""Consolidated reduced workflows. Run through workflow.py."""
import argparse
import contextlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from numba import njit
from scipy.io import loadmat
from studies.exact import C, CHARGES, COUNTS, TYPES, enumerate_particles, parameters
from models.brush_clfts import BrushChargedCLFTS
from studies.validation import saddle

# Original workflow: diagnose_clfts_reduced.py

@njit(cache=True, error_model="numpy")
def densities(x, mapping, smear, chain_weights):
    """Return half-differences phi(0)-f, smeared field amplitudes, Q ratios."""
    fields = np.empty(5, np.complex128)
    delta = np.empty(5, np.complex128)
    ratios = np.empty(5)
    for t in range(5):
        fields[t] = smear*(mapping[t, 0]*x[0]+mapping[t, 1]*x[1]+
                           mapping[t, 2]*x[2]+CHARGES[t]*x[3])
    exponent = np.empty(4, np.complex128)
    for j in range(4):
        exponent[j] = -.25*fields[0]*(2*j-2)
    scale = np.max(exponent.real)
    den, num, norm = 0j, 0j, 0.
    for j in range(4):
        weight = chain_weights[j]*np.exp(exponent[j]-scale)
        den += weight
        num += (2*j-2)*weight
        norm += abs(weight)
    delta[0] = num/(9*den)
    ratios[0] = abs(den)/norm
    for t in range(1, 5):
        v = .25*fields[t]
        delta[t] = -COUNTS[t]/9*np.tanh(v)
        # Stable evaluation of |cosh(v)| / cosh(Re v).
        a, b = np.exp(v-abs(v.real)), np.exp(-v-abs(v.real))
        ratios[t] = abs(a+b)/(abs(a)+abs(b))
    return delta, fields, ratios


@njit(cache=True)
def drift(x, mapping, smear, chain_weights, coeff, signs, mobility, coupling):
    delta, fields, ratios = densities(x, mapping, smear, chain_weights)
    force = np.empty(4, np.complex128)
    for i in range(3):
        deriv = 2*coeff[i]*x[i]
        for t in range(5):
            deriv += mapping[t, i]*smear*delta[t]
        force[i] = -mobility[i]*signs[i]*deriv
    force[3] = 0j
    if coupling > 0:
        force[3] = -np.pi**2/coupling*x[3]
        for t in range(5):
            force[3] += CHARGES[t]*smear*delta[t]
        force[3] *= mobility[3]
    return force, delta, fields, ratios


@njit(cache=True)
def step(x, previous, noise, dt, method, mapping, smear, chain_weights,
         coeff, signs, mobility, coupling):
    f, _, _, _ = drift(x, mapping, smear, chain_weights, coeff, signs, mobility, coupling)
    if method == 0:  # Production LM for contact fields, ETD1 for psi.
        out = x.copy()
        for i in range(3):
            out[i] += dt*f[i]+.5*(previous[i]+noise[i])
        if coupling > 0:
            gamma = np.pi**2/coupling*mobility[3]
            decay = np.exp(-gamma*dt)
            factor = -np.expm1(-gamma*dt)/gamma
            amp = np.sqrt(-np.expm1(-2*gamma*dt)/(2*gamma*dt))
            out[3] = decay*x[3]+factor*(f[3]+gamma*x[3])+amp*noise[3]
        return out
    if method == 1:  # Additive-noise Euler-Maruyama.
        return x+dt*f+noise
    # Additive-noise stochastic Heun with the SAME increment in predictor/corrector.
    predicted = x+dt*f+noise
    corrected, _, _, _ = drift(predicted, mapping, smear, chain_weights,
                               coeff, signs, mobility, coupling)
    return x+.5*dt*(f+corrected)+noise


@njit(cache=True)
def trajectory(x, dt, steps, equil, stride, method, rng, mapping, smear,
               chain_weights, coeff, signs, mobility, coupling):
    size = (steps-equil)//stride
    profiles = np.empty((size, 5, 2), np.complex128)
    states = np.empty((size, 4), np.complex128)
    drifts = np.empty((size, 2))
    cancellations = np.empty((size, 5))
    saved_fields = np.empty((size, 5), np.complex128)
    times = np.empty(size)
    previous = np.zeros(4, np.complex128)
    noise = np.zeros(4, np.complex128)
    scale = np.sqrt(2*dt/C*mobility)
    max_drift, minimum_ratio = 0., 1.
    worst_state = x.copy()
    sample, completed = 0, 0
    for n in range(steps):
        f, delta, fields, ratios = drift(x, mapping, smear, chain_weights,
                                         coeff, signs, mobility, coupling)
        maximum = np.max(np.abs(f))
        minimum_ratio = min(minimum_ratio, np.min(ratios))
        if not np.isfinite(maximum):
            break
        if maximum > max_drift:
            max_drift = maximum
            worst_state = x.copy()
        if n >= equil and (n-equil) % stride == 0 and sample < size:
            for t in range(5):
                profiles[sample, t, 0] = COUNTS[t]/9+delta[t]
                profiles[sample, t, 1] = COUNTS[t]/9-delta[t]
            states[sample] = x
            saved_fields[sample] = fields
            times[sample] = n*dt
            drifts[sample, 0] = maximum
            drifts[sample, 1] = maximum
            cancellations[sample] = ratios
            sample += 1
        for i in range(4):
            value = scale[i]*rng.normal()
            noise[i] = value if i < 3 and signs[i] > 0 else 1j*value
        if coupling == 0:
            noise[3] = 0j
        x = step(x, previous, noise, dt, method, mapping, smear, chain_weights,
                 coeff, signs, mobility, coupling)
        previous = noise.copy()
        completed = n+1
    return (profiles[:sample], states[:sample], drifts[:sample],
            cancellations[:sample], saved_fields[:sample], times[:sample], x,
            previous, worst_state, max_drift, minimum_ratio, completed, completed*dt)


@njit(cache=True)
def adaptive_trajectory(x, dt, steps, equil, stride, rng, mapping, smear,
                        chain_weights, coeff, signs, mobility, coupling, target):
    """Euler with state-dependent dt; observations at fixed physical times.

    The step is chosen BEFORE drawing noise. No proposal rejection, clipping,
    or step-count weighting is used. dt*max|drift| <= dt_max*target.
    """
    size = (steps-equil)//stride
    profiles = np.empty((size,5,2),np.complex128)
    states = np.empty((size,4),np.complex128)
    drifts = np.empty((size,2))
    cancellations = np.empty((size,5))
    saved_fields = np.empty((size,5),np.complex128)
    times = np.empty(size)
    noise = np.zeros(4,np.complex128)
    worst_state = x.copy()
    max_drift, minimum_ratio, current_time = 0.,1.,0.
    sample, completed = 0,0
    duration = steps*dt
    while current_time < duration:
        f,delta,fields,ratios = drift(x,mapping,smear,chain_weights,coeff,signs,mobility,coupling)
        maximum = np.max(np.abs(f))
        if not np.isfinite(maximum):
            break
        minimum_ratio = min(minimum_ratio,np.min(ratios))
        if maximum > max_drift:
            max_drift,worst_state = maximum,x.copy()
        if sample < size and current_time >= (equil+sample*stride)*dt:
            for t in range(5):
                profiles[sample,t,0] = COUNTS[t]/9+delta[t]
                profiles[sample,t,1] = COUNTS[t]/9-delta[t]
            states[sample],saved_fields[sample] = x,fields
            times[sample] = current_time
            drifts[sample,0],drifts[sample,1] = maximum,maximum
            cancellations[sample] = ratios
            sample += 1
        next_time = (equil+sample*stride)*dt if sample < size else duration
        actual_dt = min(dt*min(1.,target/max(maximum,1e-300)),next_time-current_time)
        if actual_dt <= 0 or current_time+actual_dt == current_time:
            break
        for i in range(4):
            value = np.sqrt(2*actual_dt/C*mobility[i])*rng.normal()
            noise[i] = value if i < 3 and signs[i] > 0 else 1j*value
        if coupling == 0:
            noise[3] = 0j
        x += actual_dt*f+noise
        current_time += actual_dt
        completed += 1
    return (profiles[:sample],states[:sample],drifts[:sample],cancellations[:sample],
            saved_fields[:sample],times[:sample],x,noise,worst_state,max_drift,
            minimum_ratio,completed,current_time)


def setup(sim, exact):
    active = sim._active_aux
    assert len(active) == 3
    mapping = np.ascontiguousarray(sim.mpt.matrix_a[
        [sim.monomer_types.index(t) for t in TYPES]][:, active])
    coeff = sim.mpt.h_coef_mu2[active].copy()
    signs = np.array([1. if i in sim.mpt.aux_fields_real_idx else -1. for i in active])
    mobility = np.r_[sim.dt_scaling[active], sim.psi_dt_scaling]
    weights = np.zeros(4)
    for remaining in itertools.product(range(2), repeat=3):
        pos = (0,)+remaining
        count0 = pos.count(0)
        weights[count0-1] += np.prod([exact["bond"][pos[j],pos[j+1]] for j in range(3)])
    return (mapping, np.exp(-.1**2*np.pi**2/2), weights, coeff, signs,
            mobility, sim.electro.e_coupling)


def field_reference(exact, config):
    """Entire field moments from Gaussian integration for each particle state."""
    mapping, smear, _, coeff, _, _, coupling = config
    dphi = (exact["phi"][:,:,0]-exact["phi"][:,:,1])/2
    centers = np.column_stack([-smear*(dphi@mapping)/(2*coeff),
                               coupling/np.pi**2*smear*(dphi@CHARGES)])
    covariance = np.r_[1/(2*C*coeff),-coupling/(C*np.pi**2)]
    exponent = exact["log_bond"]-C*(exact["contact"]+exact["coulomb"])
    weight = np.exp(exponent-exponent.max())
    weight /= weight.sum()
    return dict(mean=(weight@centers).tolist(),
                second=(weight@(centers**2)+covariance).tolist())


def checks(sim, config, exact):
    """Random complex fields (std=5) and an actual production one-step update."""
    rng = np.random.default_rng(5517)
    errors = {"density": 0., "drift": 0., "cancellation": 0.}
    errors["projected_noise_variance"] = abs(sim.langevin["sigma"]**2/2-2*sim.langevin["dt"]/C)
    assert errors["projected_noise_variance"] < 1e-14, errors
    active = sim._active_aux
    for _ in range(64):
        w = rng.normal(0,5,(5,2))+1j*rng.normal(0,5,(5,2))
        w[sim._frozen_aux] = 0.
        sim.psi = rng.normal(0,5,2)+1j*rng.normal(0,5,2)
        sim.psi -= sim.psi.mean()
        if sim.electro.e_coupling == 0:
            sim.psi[:] = 0
        x = np.r_[(w[active,0]-w[active,1])/2, sim.psi[0]]
        f, d, fields, ratios = drift(x, *config)
        ph, _ = sim.compute_concentrations(w)
        expected = np.array([ph[t] for t in TYPES])
        errors["density"] = max(errors["density"], float(np.max(abs(expected[:,0]-COUNTS/9-d))))
        derivatives = sim.mpt.compute_func_deriv(w,sim.smearing.apply_to_dict(ph),active)
        fp, _ = sim._psi_force(ph)
        expected_f = np.r_[-sim.dt_scaling[active]*(derivatives[:,0]-derivatives[:,1])/2,
                           sim.psi_dt_scaling*(fp[0]-fp[1])/2]
        errors["drift"] = max(errors["drift"], float(np.max(abs(f-expected_f))))
        from studies.exact import partition_cancellation
        expected_ratio = partition_cancellation(np.column_stack([fields,-fields]),exact["bond"])
        errors["cancellation"] = max(errors["cancellation"],float(np.max(abs(ratios-expected_ratio))))
    assert max(errors.values()) < 1e-10, errors
    # A complete BrushChargedCLFTS.run step, including its saved post-update fields.
    previous = rng.normal(0,sim.langevin["sigma"],(5,2))
    noise = rng.normal(0,sim.langevin["sigma"],(5,2))
    xi = rng.normal(0,sim.langevin["sigma"],2)

    class FixedNoise:
        def __init__(self, values, generator):
            self.values, self.bit_generator = values, generator.bit_generator

        def normal(self, *args):
            return self.values.copy()

    sim.random = FixedNoise(noise, sim.random)
    sim._random_psi = FixedNoise(xi, sim._random_psi)
    rotation = np.array([1 if i in sim.mpt.aux_fields_real_idx else 1j for i in active])
    projected = np.r_[rotation*np.sqrt(sim.dt_scaling[active])*(noise[active,0]-noise[active,1])/2,
                      1j*np.sqrt(sim.psi_dt_scaling)*(xi[0]-xi[1])/2]
    prev_projected = np.r_[rotation*np.sqrt(sim.dt_scaling[active])*(previous[active,0]-previous[active,1])/2,0j]
    if sim.electro.e_coupling == 0:
        projected[3] = 0j
    reference_step = step(x,prev_projected,projected,sim.langevin["dt"],0,*config)
    sim.langevin["max_step"] = 1
    sim.recording["recording_period"] = 1
    mon = sim.mpt.to_monomer_fields(w)
    sim.run(dict(zip(sim.monomer_types,mon)),normal_noise_prev=previous)
    data = loadmat(Path(sim.recording["dir"])/"fields_000001.mat",simplify_cells=True)
    w_after = np.array([data[f"w_{t}_real"]+1j*data[f"w_{t}_imag"] for t in sim.monomer_types])
    wa = sim.mpt.to_aux_fields(w_after)
    actual = np.r_[(wa[active,0]-wa[active,1])/2,sim.psi[0]]
    errors["native_full_step"] = float(np.max(abs(actual-reference_step)))
    assert errors["native_full_step"] < 1e-11, errors
    return errors


def run():
    ap = argparse.ArgumentParser(description='Four complex amplitudes for the unchanged two-cell canonical CL benchmark.\n\nOnly uniform fields are eliminated: canonical conditional densities depend on\nfield differences, and uniform modes have decoupled Gaussian dynamics. The\nthree interacting auxiliary differences and electrostatic difference remain.\nThis is a diagnostic implementation, not a production solver or FFT change.\nRequires numba (the campaign installs it into a separate evidence directory).')
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--coupling",type=float,default=100.)
    ap.add_argument("--dt",type=float,default=.0025)
    ap.add_argument("--duration",type=float,default=10000.)
    ap.add_argument("--equil",type=float,default=1000.)
    ap.add_argument("--seed",type=int,default=301)
    ap.add_argument("--method",choices=["lm-etd","euler","heun","adaptive-euler"],default="heun")
    ap.add_argument("--drift-target",type=float,default=10.)
    ap.add_argument("--psi-mobility",type=float,default=1.)
    ap.add_argument("--checks-only",action="store_true")
    args = ap.parse_args()
    if args.dt <= 0 or args.equil < 0 or args.duration <= args.equil or args.psi_mobility <= 0 or args.drift_target <= 0:
        ap.error("Require dt, mobility > 0 and duration > equil >= 0")
    args.out.mkdir(parents=True,exist_ok=True)
    args.sites,args.stabilization,args.platform = 2,0.,"cpu-mkl"
    params = parameters(args)
    params["psi_dt_scaling"] = args.psi_mobility
    exact = enumerate_particles(2,args.coupling)
    started = time.perf_counter()
    with open(args.out/"native_checks.log","w") as log,contextlib.redirect_stdout(log):
        sim = BrushChargedCLFTS(params,random_seed=args.seed)
        config = setup(sim,exact)
        validation = checks(sim,config,exact)
        w,_,residual = saddle(sim)
        x = np.r_[(w[sim._active_aux,0]-w[sim._active_aux,1])/2,sim.psi[0]]
    summary = dict(status="completed",params=params,species=TYPES,
        exact_profiles=exact["mean"].tolist(),exact_height=float(exact["height"]),
        exact_field_moments=field_reference(exact,config),
        deterministic_checks=validation,saddle_residual=residual,seed=args.seed,
        method=args.method,duration=args.duration,equil=args.equil,
        adaptive_drift_target=args.drift_target if args.method=="adaptive-euler" else None,
        note="Unchanged two-cell action; uniform Gaussian modes analytically decoupled.")
    np.savez(args.out/"setup.npz",mapping=config[0],smear=config[1],chain_weights=config[2],
             coeff=config[3],signs=config[4],mobility=config[5],coupling=config[6],initial=x)
    if not args.checks_only:
        steps,equil = round(args.duration/args.dt),round(args.equil/args.dt)
        stride = max(1,round(.1/args.dt))
        rng = np.random.default_rng(args.seed)
        if args.method == "adaptive-euler":
            result = adaptive_trajectory(x,args.dt,steps,equil,stride,rng,*config,args.drift_target)
        else:
            result = trajectory(x,args.dt,steps,equil,stride,
                ["lm-etd","euler","heun"].index(args.method),rng,*config)
        ph,states,drifts,ratios,fields,times,final,previous,worst,maximum,minimum,completed,reached_time = result
        np.savez_compressed(args.out/"samples.npz",profiles=ph,states=states,drift=drifts,
            partition_cancellation=ratios,fields=fields,times=times,z=exact["z"])
        np.savez(args.out/"final_state.npz",state=final,previous=previous,worst_state=worst)
        summary.update(n_samples=len(ph),completed_steps=completed,reached_time=reached_time,
            max_species_mass_error=float(np.max(abs(ph.mean(axis=2)-COUNTS/9))) if len(ph) else None,
            max_drift_every_step=float(maximum),min_cancellation_every_step=float(minimum),
            final_rng_state=rng.bit_generator.state)
        if reached_time < steps*args.dt-1e-9 or not np.isfinite(final).all():
            summary["status"] = "failed"
            summary["error"] = "Nonfinite reduced trajectory"
    summary["wall_seconds"] = time.perf_counter()-started
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "params"},indent=2))
    if summary["status"] != "completed":
        raise SystemExit(1)
