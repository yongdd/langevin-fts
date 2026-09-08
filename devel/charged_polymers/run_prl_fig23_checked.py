"""One salt concentration per SLURM job, using the checked Fig. 1 equations.

Canonical monovalent counterions, no added monovalent reservoir, equal radii.
The explicitly defined equimolar slab estimate does not establish the paper's
unspecified Gibbs-height prescription. Production SCFT/FFT sources are unchanged.
"""
import argparse
import json
from pathlib import Path
import time
from types import MethodType

import numpy as np
from scipy.special import erfcx, logsumexp
from scipy.linalg import solve_banded
from scipy.optimize import NoConvergence

from prl_scft import PRLBrushSCFT
from run_prl_fig1_checked import coupled_newton

MOLAR_TO_NM3 = 0.6022


def extended_chain(s, potential):
    """Scaled positive chains with an extended exponent range and one Q.

    Research runner only. The double-precision legacy bead slice can hit its
    1e-300 normalization floor at low salt. No clipping/floor or independent
    bead normalization is used here; contour consistency is checked explicitly.
    """
    ld=np.longdouble
    potential=np.asarray(potential,dtype=ld)
    half=int(np.ceil(8*s.b/s.dz))
    offsets=np.arange(-half,half+1,dtype=ld)*ld(s.dz)
    kernel=ld(s.dz)*np.sqrt(ld(3)/(2*ld(np.pi)*ld(s.b)**2))*np.exp(-ld(1.5)*(offsets/ld(s.b))**2)
    def bond(q):
        return np.convolve(np.r_[np.zeros(half,dtype=ld),q,q[-1:-half-1:-1]],kernel,mode='valid')
    shift=potential.min()
    ew=np.exp(-(potential-shift))
    back=np.empty((s.N,s.Nz),dtype=ld);scales=np.empty(s.N,dtype=ld)
    back[-1]=ew/ew.max();scales[-1]=np.log(ew.max())
    for n in range(s.N-2,-1,-1):
        q=ew*bond(back[n+1]);scale=q.max()
        if scale<=0 or not np.isfinite(scale):raise FloatingPointError('Invalid extended propagator')
        back[n]=q/scale;scales[n]=scales[n+1]+np.log(scale)
    src=np.asarray(s.q_init['G'],dtype=ld)
    denominator=np.mean(src*back[0])
    if denominator<=0:raise FloatingPointError('Extended partition vanished')
    log_q=np.log(denominator)+scales[0]-s.N*shift
    a=src/src.max();la=np.log(src.max())
    density=np.zeros(s.Nz,dtype=ld);error=0.
    for n in range(s.N):
        sl=a*back[n]*np.exp(la+scales[n]-scales[0])/denominator
        error=max(error,float(abs(sl.mean()-1)))
        density+=sl/s.N
        if n<s.N-1:
            a=bond(ew*a);scale=a.max();a/=scale;la+=np.log(scale)
    if not np.isfinite(density).all() or error>1e-10:
        raise FloatingPointError(f'Extended contour consistency failed: {error}')
    s.log_partition=float(log_q)
    s.extended_partition_error=error
    return np.asarray(density*(s.sigma*s.N/s.L),dtype=float)


def log_chain_reference(s):
    """Positive Gaussian-chain reference entirely in logarithmic arithmetic.

    Each contour slice uses the same partition function. The Gaussian bond
    quadrature is evaluated directly, without FFTs or beadwise mass fitting.
    This also handles the extreme graft-to-free-end scale separation at low salt.
    """
    potential=s.W/s.N
    half=int(np.ceil(8*s.b/s.dz))
    offsets=np.arange(-half,half+1)*s.dz
    log_kernel=np.log(s.dz)+.5*np.log(3/(2*np.pi*s.b**2))-1.5*(offsets/s.b)**2

    def bond(q):
        padded=np.r_[np.full(half,-np.inf),q,q[-1:-half-1:-1]]
        windows=np.lib.stride_tricks.sliding_window_view(padded,len(log_kernel))
        return logsumexp(windows+log_kernel,axis=1)

    backward=np.empty((s.N,s.Nz))
    backward[-1]=-potential
    for n in range(s.N-2,-1,-1):backward[n]=-potential+bond(backward[n+1])
    src=s.q_init['G']
    incoming=np.full(s.Nz,-np.inf)
    np.log(src,out=incoming,where=src>0)
    log_q=logsumexp(incoming+backward[0])-np.log(s.Nz)
    density=np.zeros(s.Nz)
    partition_error=0.
    for n in range(s.N):
        sl=np.exp(incoming+backward[n]-log_q)
        partition_error=max(partition_error,float(abs(sl.mean()-1)))
        density+=sl/s.N
        if n<s.N-1:incoming=bond(incoming-potential)
    return density*(s.sigma*s.N/s.L),float(log_q),partition_error


def canonical_pb_newton(s, rhoP, nC_total=None, n_newton=200):
    """Same canonical PB equations with their full normalization derivative.

    J = A + u v^T, where A is tridiagonal and the rank-one term comes from
    differentiating n_C exp(-psi-u_C)/integral exp(-psi-u_C). At exactly zero
    salt the potential gauge is fixed at the last cell; the omitted Neumann
    equation is checked again before returning.
    """
    assert s.with_counterions and s.c1 == 0 and not getattr(s, 'reservoir', 0.)
    alpha=4*np.pi*s.lB
    amount=abs(s.zP)*s.sigma*s.N
    invdz2=1/s.dz**2
    zero_salt=s.rho_b==0
    psi=s.psi.copy()
    if zero_salt:
        psi-=psi[-1]

    def evaluate(p):
        bc=np.exp(-np.clip(p+s.um,-300,300))
        c=amount*bc/(bc.sum()*s.dz) if amount else np.zeros_like(p)
        plus=s.rho_b*np.exp(-np.clip(s.zp*p+s.up,-300,300))
        minus=s.zp*s.rho_b*np.exp(-np.clip(-p+s.um,-300,300))
        lap=np.diff(np.pad(p,1,mode='edge'),2)*invdz2
        return lap+alpha*(s.zP*rhoP+c+s.zp*plus-minus),c,plus,minus

    tol=1e-9*max(1.,alpha*(abs(rhoP).max()+s.zp*s.rho_b+amount/s.L))
    if getattr(s,'pb_absolute_tolerance',None) is not None:
        tol=min(tol,s.pb_absolute_tolerance)
    for iteration in range(n_newton):
        full,c,plus,minus=evaluate(psi)
        if abs(full).max()<tol:
            s.psi=psi
            return c,plus,minus
        f=full.copy()
        if zero_salt:f[-1]=psi[-1]
        diagonal=alpha*(c+s.zp**2*plus+minus)
        ab=np.zeros((3,s.Nz))
        ab[0,1:]=invdz2;ab[2,:-1]=invdz2
        ab[1,:]=-2*invdz2-diagonal
        ab[1,[0,-1]]+=invdz2
        u=alpha*c
        v=c*s.dz/amount if amount else np.zeros_like(c)
        if zero_salt:
            ab[1,-1]=1.;ab[2,-2]=0.;u[-1]=0.
        yz=solve_banded((1,1),ab,np.column_stack([-f,u]))
        y,q=yz[:,0],yz[:,1]
        denominator=1+v@q
        if abs(denominator)<1e-13:
            raise RuntimeError('Canonical PB Jacobian is numerically singular')
        step=y-q*(v@y)/denominator
        norm=np.linalg.norm(f)
        fraction=1.
        for _ in range(40):
            candidate=psi+fraction*step
            trial=evaluate(candidate)[0]
            if zero_salt:trial[-1]=candidate[-1]
            if np.linalg.norm(trial)<norm:
                psi=candidate
                break
            fraction*=.5
        else:
            raise RuntimeError(f'Canonical PB line search failed: {abs(full).max():.3e}')
    raise RuntimeError(f'Canonical PB did not converge: {abs(evaluate(psi)[0]).max():.3e}')


def analytic_curves(rho_molar, valence, radius=.25, bjerrum=.7):
    rho = np.asarray(rho_molar) * MOLAR_TO_NM3
    kappa = np.sqrt(4*np.pi*bjerrum*valence*(valence+1)*rho)
    x = radius*kappa
    du = -(1+2*x*x/np.pi)*erfcx(x/np.sqrt(np.pi)) + 2*x/np.pi
    rhs = -valence*(valence+1)*kappa*bjerrum/4*du
    gamma = (1-rhs)/(valence+1-rhs)
    # Eq. 9 divided by sigma*N*v/h; v=1 nm^3 here.
    potential = -((valence**2-1)*gamma+1)/(valence**2*(valence+1)*rho)
    return gamma, potential


def observables(s):
    rho, dz, z = s.rhoP, s.dz, s.z
    mass = float(rho.sum()*dz)
    h_moment = float(2*(z*rho).sum()*dz/mass)
    # Integral [rho(z)-rho_ref*Theta(h-z)] dz = 0, with rho_ref=max rho.
    # The reference density is an explicit choice, not inferred from PRL data.
    h_equimolar = float(mass/rho.max())
    result = dict(height_moment_nm=h_moment, height_equimolar_peak_nm=h_equimolar,
                  equimolar_reference_density=float(rho.max()),
                  phi_peak=float(s.v*rho.max()), mass=mass,
                  counterion_mass=float(s.rhoC.sum()*dz),
                  psi_wall=float(s.psi[0]), psi_far=float(s.psi[-1]),
                  psi_wall_relative=float(s.psi[0]-s.psi[-1]),
                  far_polymer_density=float(rho[-1]),
                  far_counterion_density=float(s.rhoC[-1]))
    for label, h in [('moment',h_moment), ('equimolar_peak',h_equimolar)]:
        scale = h/(s.sigma*s.N*s.v)
        result['psi_normalized_'+label] = float(s.psi[0]*scale)
        result['psi_relative_normalized_'+label] = float((s.psi[0]-s.psi[-1])*scale)
    for fraction in [.5,.8]:
        mask = rho > fraction*rho.max()
        g = 1-s.zp*(s.rhop[mask]-s.rho_b)/rho[mask]
        weights = rho[mask]/rho[mask].sum()
        result[f'gamma_core_{fraction:g}'] = float(weights@g)
        result[f'canonical_fraction_core_{fraction:g}'] = float(
            s.rhoC[mask].sum()/rho[mask].sum())
    if s.rho_b > 0:
        g, p = analytic_curves(s.rho_b/MOLAR_TO_NM3,s.zp)
        result.update(gamma_eq7=float(g), psi_normalized_eq9=float(p),
                      kappa_bulk_inv_nm=float(1/s.kappa_b))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zplus',type=int,choices=[1,2,3],required=True)
    p.add_argument('--rho-molar',type=float,required=True)
    p.add_argument('--closure',choices=['mf','lda','nonlocal'],default='lda')
    p.add_argument('--neutral',action='store_true')
    p.add_argument('--L',type=float,default=200.)
    p.add_argument('--nz',type=int,default=1600)
    p.add_argument('--warm',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--direction',default='independent')
    p.add_argument('--pb-solver',choices=['existing','rank1'],default='rank1')
    p.add_argument('--pb-tolerance',type=float,default=None,
                   help='Optional stricter absolute PB residual for high-salt validation')
    p.add_argument('--reference',choices=['log','dense'],default='log')
    p.add_argument('--chain-solver',choices=['existing','extended'],default='extended')
    p.add_argument('--newton-cycles',type=int,default=1)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    result=dict(params={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},status='running')
    s=PRLBrushSCFT(zplus=args.zplus,zP=0. if args.neutral else -1.,
        correlations=args.closure!='mf' and not args.neutral,
        counterions=True,L=args.L,Nz=args.nz,Ib_molar=.3)
    s.set_bulk(args.rho_molar*MOLAR_TO_NM3)
    if args.pb_solver=='rank1':
        s._solve_pb=MethodType(canonical_pb_newton,s)
        s.pb_absolute_tolerance=args.pb_tolerance
    if args.chain_solver=='extended':
        s._chain_density=MethodType(extended_chain,s)
    if args.closure=='nonlocal':
        s.selfenergy='nonlocal'
    warm=np.load(args.warm)
    for attr in ['W','psi','uP','up','um']:
        if attr in warm and (s.corr or attr in ['W','psi']):
            setattr(s,attr,np.interp(s.z,warm['z'],warm[attr]))
    try:
        for cycle in range(args.newton_cycles):
            try:
                coupled_newton(s)
                break
            except NoConvergence as exc:
                values=np.asarray(exc.args[0])
                s.W=s.N*values[:s.Nz].copy();s.um=values[s.Nz:].copy()
                s.uP=s.zP**2*s.um;s.up=s.zp**2*s.um
                if cycle+1==args.newton_cycles:raise
                print(json.dumps(dict(restarted_newton_cycle=cycle+2)),flush=True)
        s.solve(max_iter=12,tol=1e-7,lam_w=.001,lam_u=.05,
                psi_exact=True,use_am=False,dW_cap=.05)
        result['residuals']=s.residuals
        if not s.converged:
            raise RuntimeError(f'Final residual failed: {s.residuals}')
        rho=s.rhoP.copy()
        if args.reference=='log':
            direct,log_partition,partition_error=log_chain_reference(s)
        else:
            s.chain_backend='direct'
            direct=PRLBrushSCFT._chain_density(s,s.W/s.N)
            log_partition=float(s.log_partition)
            partition_error=s._direct_chain.partition_error
        chain_error=float(abs(direct-rho).max())
        if chain_error>1e-10 or partition_error>1e-10:
            raise RuntimeError(f'Independent chain density failed: {chain_error}')
        mass=float(s.rhoP.sum()*s.dz)
        cmass=float(s.rhoC.sum()*s.dz)
        assert abs(mass-s.sigma*s.N)<1e-10
        assert abs(cmass-abs(s.zP)*s.sigma*s.N)<1e-10
        exponents=[s.W/s.N,s.psi+s.um,s.zp*s.psi+s.up,-s.psi+s.um]
        if max(abs(v).max() for v in exponents)>=290:
            raise RuntimeError('State approaches existing exponential clipping')
        obs=observables(s)
        if not all(np.isfinite(v) for v in obs.values()):
            raise RuntimeError('Nonfinite observable')
        result.update(status='completed',observables=obs,independent_chain_error=chain_error,
                      contour_partition_error=partition_error,
                      log_partition=log_partition)
        np.savez_compressed(args.out/'profiles.npz',z=s.z,rhoP=rho,phiP=s.v*rho,
            rhoC=s.rhoC,rhop=s.rhop,rhom=s.rhom,W=s.W,psi=s.psi,uP=s.uP,up=s.up,um=s.um,
            rhoP_direct=direct)
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc)[:1500])
        raise
    finally:
        np.savez_compressed(args.out/'checkpoint.npz',z=s.z,W=s.W,psi=s.psi,uP=s.uP,up=s.up,um=s.um)
        result['wall_seconds']=time.perf_counter()-start
        (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
