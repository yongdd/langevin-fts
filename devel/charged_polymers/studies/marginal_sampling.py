"""Consolidated marginal sampling workflows. Run through workflow.py."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from numba import njit
from scipy.integrate import quad
from scipy.special import logsumexp
from studies.statistics import statistics
from studies.reduced import field_reference
from studies.exact import C, CHARGES, COUNTS, TYPES, enumerate_particles

# Original workflow: validate_clfts_positive_marginal.py

def marginal_setup(config, exact):
    mapping,smear,_,coeff,signs,_,coupling=config
    assert signs.tolist()==[1.,-1.,-1.]
    dphi=(exact["phi"][:,:,0]-exact["phi"][:,:,1])/2
    linear=smear*(dphi@mapping)
    charge=smear*(dphi@CHARGES)
    log_repulsive=exact["log_bond"]+C*np.sum(linear[:,1:]**2/(4*coeff[1:]),axis=1)
    log_repulsive-=C*coupling/(2*np.pi**2)*charge**2
    # Integer occupancy key, not rounded field/interaction values.
    groups=np.rint((exact["phi"][:,0,0]-exact["phi"][:,4,0])*9/2).astype(int)
    unique=np.unique(groups)
    assert len(unique)==5
    centers=np.column_stack([-linear/(2*coeff),coupling/np.pi**2*charge])
    covariance=np.r_[1/(2*C*coeff),-coupling/(C*np.pi**2)]
    h=exact["phi"][:,0]@exact["z"]/(8/9)
    observables=np.column_stack([h,exact["phi"].reshape(-1,10),centers,centers**2+covariance])
    log_a,ell,averages=[],[],[]
    for group in unique:
        mask=groups==group
        coefficient=float(linear[mask,0].mean())
        assert np.max(abs(linear[mask,0]-coefficient))<1e-13
        norm=logsumexp(log_repulsive[mask])
        weight=np.exp(log_repulsive[mask]-norm)
        log_a.append(norm)
        ell.append(coefficient)
        averages.append(weight@observables[mask])
    log_a,ell,averages=np.array(log_a),np.array(ell),np.array(averages)
    # Integrating the last Gaussian must recover the ORIGINAL particle weights.
    integrated=log_repulsive+C*linear[:,0]**2/(4*coeff[0])
    reference=exact["log_bond"]-C*(exact["contact"]+exact["coulomb"])
    weight_error=float(np.max(abs(np.exp(integrated-logsumexp(integrated))-
                                      np.exp(reference-logsumexp(reference)))))
    assert weight_error<1e-13,weight_error
    fm=field_reference(exact,config)
    target=np.r_[exact["height"],exact["mean"].ravel(),fm["mean"],fm["second"]]
    return log_a,ell,averages,float(coeff[0]),target,weight_error


@njit(cache=True)
def evaluate(x,log_a,ell,averages,coeff):
    exponent=log_a-C*ell*x
    weights=np.exp(exponent-np.max(exponent))
    weights/=weights.sum()
    force=-2*coeff*x-np.dot(weights,ell)
    observable=weights@averages
    # The retained real field is observed directly; the others are integrated.
    observable[11]=x
    observable[15]=x*x
    return force,observable


def log_density(x,log_a,ell,coeff):
    return -C*coeff*x*x+logsumexp(log_a-C*ell*x)


def deterministic_checks(log_a,ell,averages,coeff,target):
    sigma=np.sqrt(1/(2*C*coeff))
    mean=-ell/(2*coeff)
    group_log_weight=log_a+C*ell**2/(4*coeff)
    group_weight=np.exp(group_log_weight-logsumexp(group_log_weight))
    # Integrate actual conditional observables with numerical real quadrature.
    def integrand(x,i):
        mixture=np.sum(group_weight*np.exp(-.5*((x-mean)/sigma)**2))/(np.sqrt(2*np.pi)*sigma)
        return mixture*evaluate(x,log_a,ell,averages,coeff)[1][i]
    numerical=np.array([quad(integrand,-np.inf,np.inf,args=(i,),epsabs=1e-11,epsrel=1e-11)[0]
                        for i in range(len(target))])
    error=float(np.max(abs(numerical-target)))
    assert error<1e-10,error
    rng=np.random.default_rng(5517)
    force_error=0.
    for x in rng.normal(0,5,64):
        dx=1e-4
        # Fourth-order central derivative to separate truncation from roundoff.
        derivative=(log_density(x-2*dx,log_a,ell,coeff)-8*log_density(x-dx,log_a,ell,coeff)+
                    8*log_density(x+dx,log_a,ell,coeff)-log_density(x+2*dx,log_a,ell,coeff))/(12*dx*C)
        force_error=max(force_error,abs(evaluate(x,log_a,ell,averages,coeff)[0]-derivative))
    assert force_error<1e-8,force_error
    return dict(real_quadrature_error=error,force_derivative_error=float(force_error))


@njit(cache=True)
def trajectory(dt,steps,equil,stride,rng,log_a,ell,averages,coeff):
    size=(steps-equil)//stride
    values=np.empty((size,averages.shape[1]))
    times=np.empty(size)
    x=0.
    sample=0
    maximum=0.
    for n in range(steps):
        f,obs=evaluate(x,log_a,ell,averages,coeff)
        maximum=max(maximum,abs(f))
        if n>=equil and (n-equil)%stride==0 and sample<size:
            values[sample]=obs
            times[sample]=n*dt
            sample+=1
        noise=np.sqrt(2*dt/C)*rng.normal()
        predict=x+dt*f+noise
        corrected,_=evaluate(predict,log_a,ell,averages,coeff)
        x+=.5*dt*(f+corrected)+noise
        if not np.isfinite(x):
            break
    return values[:sample],times[:sample],x,maximum,n+1


def run():
    ap=argparse.ArgumentParser(description='Exactly integrate repulsive fields of the two-cell fixture; sample one real field.\n\nThe physical finite action is unchanged. Gaussian integration over the two\nrepulsive auxiliary amplitudes and psi leaves a strictly positive, five-term\nmarginal of the attractive amplitude. This is a finite-model diagnostic,\nnot a scalable replacement for the production CL solver.')
    ap.add_argument("--setup",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--dt",type=float,default=.005)
    ap.add_argument("--duration",type=float,default=20000.)
    ap.add_argument("--equil",type=float,default=2000.)
    ap.add_argument("--seed",type=int,default=901)
    ap.add_argument("--checks-only",action="store_true")
    args=ap.parse_args()
    if args.dt<=0 or args.equil<0 or args.duration<=args.equil:
        ap.error("Require dt>0 and duration>equil>=0")
    args.out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    data=np.load(args.setup)
    config=tuple(data[k] for k in ["mapping","smear","chain_weights","coeff","signs","mobility","coupling"])
    exact=enumerate_particles(2,float(config[-1]))
    log_a,ell,averages,coeff,target,weight_error=marginal_setup(config,exact)
    checks=deterministic_checks(log_a,ell,averages,coeff,target)
    checks["particle_weight_error"]=weight_error
    labels=["height"]+[f"phi_{t}_cell{i}" for t in TYPES for i in range(2)]
    labels += [f"field_{i}" for i in range(4)]+[f"field2_{i}" for i in range(4)]
    result=dict(status="completed",coupling=float(config[-1]),dt=args.dt,duration=args.duration,
        equil=args.equil,seed=args.seed,labels=labels,exact=target.tolist(),checks=checks,
        method="real Heun after exact Gaussian marginalization",
        note="Same finite particle action; changed sampling representation, not production CL.")
    np.savez(args.out/"marginal.npz",log_a=log_a,ell=ell,averages=averages,coeff=coeff)
    if not args.checks_only:
        steps,equil=round(args.duration/args.dt),round(args.equil/args.dt)
        stride=max(1,round(.1/args.dt))
        rng=np.random.default_rng(args.seed)
        values,times,final,maximum,completed=trajectory(args.dt,steps,equil,stride,rng,log_a,ell,averages,coeff)
        np.savez_compressed(args.out/"samples.npz",observables=values,times=times)
        stats=statistics(values,args.dt*stride)
        z=abs(np.array(stats["mean"])-target)/np.maximum(np.array(stats["se"]),1e-300)
        mass=float(np.max(abs(values[:,1:11].reshape(-1,5,2).mean(axis=2)-COUNTS/9)))
        assert mass<1e-12,mass
        result.update(statistics=stats,standardized_discrepancy=z.tolist(),max_species_mass_error=mass,
            completed_steps=completed,final_field=final,max_drift=maximum,n_samples=len(values),
            final_rng_state=rng.bit_generator.state)
        if completed!=steps or not np.isfinite(values).all():
            result["status"]="failed"
    result["wall_seconds"]=time.perf_counter()-started
    (args.out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ["statistics","labels","exact"]},indent=2))
    if result["status"]!="completed":
        raise SystemExit(1)
