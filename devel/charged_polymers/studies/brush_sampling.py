"""Consolidated brush sampling workflows. Run through workflow.py."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from models.positive_brush_1d import (
    Parameters,
    initial_state,
    advance,
    check_state,
    energy,
    potential,
    activity_log,
    _delta_e,
    _global_delta,
)
from studies.statistics import statistics
import itertools
from scipy.special import gammaln, logsumexp

# Original workflow: run_positive_brush.py

def observe(p,s):
    n=s['counts'];height=2*(n[0]@s['z'])/n[0].sum() if n[0].sum() else 0.
    return np.r_[height,n[2].sum(),energy(p,s)[0],potential(p,s)[0],
                 n[:,0],s['x'][0],s['x'][0]**2]


def save_state(path,p,s):
    np.savez_compressed(path,chain=s['chain'],counts=s['counts'],q=s['q'],x=s['x'],
                        params=json.dumps(asdict(p)),rng=json.dumps(s['rng'].bit_generator.state))


def calibrate(p,s,target,rounds=30,round_sweeps=4000):
    """Tune only during warmup; production uses a frozen formula-unit activity."""
    target_number=.6022*target*p.area*p.length
    history=[]
    for i in range(rounds):
        trace=[]
        for _ in range(100):
            advance(p,s,round_sweeps//100);trace.append(s['counts'][2].sum())
        mean=float(np.mean(trace));var=float(np.var(trace,ddof=1))
        # d<N>/d(log formula-unit activity) = Var(N). The floor regularizes
        # a short warmup estimate; this adaptation is never used in production.
        delta=float(np.clip((target_number-mean)/max(var,mean/(p.valence+1),.1),-1.,1.))
        if i>=rounds//2:delta*=.5
        history.append(dict(round=i,shift=p.log_salt_shift,mean=mean,variance=var,delta=delta))
        p.log_salt_shift+=delta
        if i%5==0:print('calibration',history[-1],flush=True)
    check_state(p,s)
    return history


def run():
    ap=argparse.ArgumentParser(description='Run actual Monte Carlo sampling, with a separately calibrated salt bath.\n\nLong runs belong in SLURM. All uncertainties include autocorrelation/blocking;\ncompletion alone is not an equilibration or continuum-convergence claim.')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--valence',type=int,default=3);ap.add_argument('--rho',type=float,default=.01)
    ap.add_argument('--sites',type=int,default=400);ap.add_argument('--length',type=float,default=200.)
    ap.add_argument('--area',type=float,default=40.);ap.add_argument('--chains',type=int,default=4)
    ap.add_argument('--mode',choices=['joint','particles'],default='joint')
    ap.add_argument('--initial',choices=['extended','compact'],default='extended')
    ap.add_argument('--bulk',action='store_true');ap.add_argument('--tiny',action='store_true')
    ap.add_argument('--bath',type=Path,help='Frozen activity from bulk summary.json')
    ap.add_argument('--burn',type=int,default=200000);ap.add_argument('--sweeps',type=int,default=1000000)
    ap.add_argument('--stride',type=int,default=100);ap.add_argument('--profile-every',type=int,default=10)
    ap.add_argument('--calibration-rounds',type=int,default=30)
    ap.add_argument('--resume',type=Path,help='Own trusted checkpoint; burn is applied again as specified')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Refusing to overwrite a completed run')
    p=Parameters(sites=args.sites,length=args.length,area=args.area,chains=args.chains,
                 valence=args.valence,activity_molar=args.rho,joint=args.mode=='joint')
    if args.bulk:p.chains=0;p.periodic=True;p.joint=False
    if args.tiny:p=replace(p,sites=2,length=2,area=2,chains=1,beads=2,activity_molar=.2)
    bath=None
    if args.bath:
        bath=json.loads(args.bath.read_text());bp=bath['params']
        for name in ['area','sites','length','valence','born_radius','bjerrum_length']:
            if getattr(p,name)!=bp[name]:raise ValueError(f'Bath {name} must match brush')
        if p.activity_molar!=bp['activity_molar']:raise ValueError('Bath activity reference must match')
        if not bp['periodic'] or bp['chains']!=0:raise ValueError('Expected pure periodic salt bath')
        p.log_salt_shift=bp['log_salt_shift']
    elif not args.bulk and not args.tiny and p.activity_molar:
        raise ValueError('Nonzero-salt brush requires separately measured --bath')
    if args.resume:
        d=np.load(args.resume);saved=Parameters(**json.loads(str(d['params'])))
        if asdict(saved)!=asdict(p):raise ValueError('Resume parameters differ')
        s=initial_state(p,args.seed,args.initial)
        for key in ['chain','counts','q','x']:s[key][:]=d[key]
        s['rng'].bit_generator.state=json.loads(str(d['rng']))
    else:s=initial_state(p,args.seed,args.initial)
    start=time.monotonic();history=[]
    source_root=Path(__file__).resolve().parents[1]
    code_hashes={f:hashlib.sha256((source_root/f).read_bytes()).hexdigest()
                 for f in ['models/positive_brush_1d.py','studies/brush_sampling.py',
                           'studies/statistics.py','workflow.py']}
    # Compile/warm the kernel before timing the campaign.
    advance(p,s,1)
    if args.bulk and not args.resume:history=calibrate(p,s,args.rho,args.calibration_rounds)
    acceptance=np.zeros(8,np.int64);max_error=check_state(p,s)
    burn_trace=[]
    burn_chunk=max(args.stride,args.burn//100)
    remaining=args.burn
    while remaining:
        amount=min(remaining,burn_chunk);advance(p,s,amount);remaining-=amount
        burn_trace.append(np.r_[args.burn-remaining,observe(p,s)])
    max_error=max(max_error,check_state(p,s));save_state(out/'after_burn.npz',p,s)
    n=args.sweeps//args.stride
    if n<100 or args.sweeps%args.stride:raise ValueError('Use >=100 whole sample intervals')
    trace=np.empty((n,len(LABELS)));profiles=[];psisteps=[];steps=[]
    for i in range(n):
        acceptance+=advance(p,s,args.stride)
        trace[i]=observe(p,s)
        if (i+1)%args.profile_every==0:
            profiles.append(s['counts'].copy());psisteps.append(potential(p,s));steps.append((i+1)*args.stride)
        if (i+1)%max(1,n//10)==0:
            max_error=max(max_error,check_state(p,s));save_state(out/'checkpoint.npz',p,s)
            progress=dict(fraction=(i+1)/n,height=float(trace[:i+1,0].mean()),salt=float(trace[:i+1,1].mean()),elapsed=time.monotonic()-start)
            (out/'progress.json').write_text(json.dumps(progress)+'\n');print(progress,flush=True)
    max_error=max(max_error,check_state(p,s))
    stats=statistics(trace,args.stride)
    halves=[statistics(t,args.stride) for t in np.array_split(trace,2)]
    diff=abs(np.array(halves[0]['mean'])-halves[1]['mean'])
    denom=np.hypot(halves[0]['se'],halves[1]['se'])
    split=np.divide(diff,denom,out=np.zeros_like(diff),where=denom>0)
    profile=np.array(profiles);cell_volume=p.area*p.length/p.sites
    np.savez_compressed(out/'samples.npz',trace=trace,burn_trace=np.array(burn_trace),
        counts=profile,psi=np.array(psisteps),profile_steps=steps,z=s['z'])
    result=dict(status='completed',params=p.metadata(),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        seed=args.seed,labels=LABELS,statistics=stats,split_half_sigma=split.tolist(),half_statistics=halves,
        profile_mean_nm3=(profile.mean(axis=0)/cell_volume).tolist(),
        max_incremental_charge_error=max_error,exact_occupancy=True,exact_neutrality=True,
        acceptance_counters=acceptance.tolist(),elapsed_seconds=time.monotonic()-start,
        calibration=history,bulk_reference=str(args.bath) if args.bath else None,
        code_sha256=code_hashes)
    if args.bulk:
        result['measured_bulk_molar']=stats['mean'][1]/(.6022*p.area*p.length)
        result['measured_bulk_molar_se']=stats['se'][1]/(.6022*p.area*p.length)
        result['target_bulk_molar']=args.rho
    if bath:
        result['measured_bulk_molar']=bath['measured_bulk_molar']
        result['measured_bulk_molar_se']=bath['measured_bulk_molar_se']
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','statistics','split_half_sigma','elapsed_seconds']},indent=2))


# Original workflow: check_positive_brush.py

LABELS = ['height_nm','salt_units','electrostatic_energy','psi_wall',
          'P_cell0','C_cell0','plus_cell0','minus_cell0','x_cell0','x_cell0_squared']


def log_weight(p,s,joint=False):
    """Recompute full weight, independently of every local acceptance formula."""
    n=s['counts'];q=s['f']@(-n[0]+n[1]+p.valence*n[2]-n[3])
    value=-s['coefficient']*np.sum((q-q.mean() if p.periodic else q)**2)
    value-=gammaln(p.capacity-n[0]+1).sum()+gammaln(n[1:]+1).sum()
    if n[2].sum():value+=n[2].sum()*activity_log(p)
    for chain in s['chain']:
        value+=s['loggraft'][chain[0]]+s['logbond'][chain[:-1],chain[1:]].sum()
    if joint:
        value+=n[0]@s['x']-p.capacity/(4*p.chi)*np.sum(s['x']**2)
    else:value-=p.chi/p.capacity*np.sum(n[0]*(p.capacity-n[0]))
    return float(value)


def exact_two_cells(p,nmax=16):
    """Enumerate every labeled polymer path, C occupancy, and neutral salt state.

    The positive tail bound drops Ue>=0 and sums the ideal neutral-salt tail.
    It is a bound on omitted partition weight, not just the final shell size.
    """
    assert p.sites==2 and p.chains==1 and p.beads==2
    s=initial_state(p);K=p.capacity;nC=p.chains*p.beads
    cell_volume=p.area*p.length/2
    all_logs=[];all_obs=[];all_seconds=[];base_logs=[]
    for path in itertools.product(range(2),repeat=p.beads):
        pn=np.bincount(path,minlength=2)
        if pn.max()>K:continue
        s['chain'][0]=path;s['counts'][0]=pn
        logbase=s['loggraft'][path[0]]+s['logbond'][path[0],path[1]]
        logbase-=gammaln(K-pn+1).sum()+p.chi/K*np.sum(pn*(K-pn))
        base_logs.append(logbase)
        for c0 in range(nC+1):
            cn=np.array([c0,nC-c0])
            for n in range(nmax+1):
                a,b=np.meshgrid(np.arange(n+1),np.arange(p.valence*n+1),indexing='ij')
                a=a.ravel();b=b.ravel()
                charges=np.stack([-pn[0]+cn[0]+p.valence*a-b,
                                  -pn[1]+cn[1]+p.valence*(n-a)-(p.valence*n-b)],axis=1)
                q=charges@s['f'].T
                if p.periodic:q-=q.mean(axis=1)[:,None]
                ue=s['coefficient']*np.sum(q*q,axis=1)
                logw=logbase-gammaln(cn+1).sum()-ue
                logw-=gammaln(a+1)+gammaln(n-a+1)+gammaln(b+1)+gammaln(p.valence*n-b+1)
                if n:logw+=n*activity_log(p)
                xmean=2*p.chi/K*(pn[0]-pn.mean())
                obs=np.empty((len(a),len(LABELS)))
                obs[:,0]=2*(pn@s['z'])/nC;obs[:,1]=n;obs[:,2]=ue
                obs[:,3]=4*np.pi*p.bjerrum_length/p.area*(p.length/2)*q[:,0]
                obs[:,4]=pn[0];obs[:,5]=cn[0];obs[:,6]=a;obs[:,7]=b
                obs[:,8]=xmean;obs[:,9]=xmean*xmean+p.chi/K
                second=obs**2
                second[:,8]=obs[:,9]
                second[:,9]=xmean**4+6*xmean*xmean*p.chi/K+3*(p.chi/K)**2
                all_logs.extend(logw);all_obs.extend(obs);all_seconds.extend(second)
    norm=logsumexp(all_logs);weights=np.exp(np.array(all_logs)-norm)
    obs=np.array(all_obs)
    # log W_n = n log(a+ a-^z) + (z+1)n log(M) - log(n! (zn)!).
    ns=np.arange(nmax+1,nmax+1001)
    tail=ns*(activity_log(p)+(p.valence+1)*np.log(2))-gammaln(ns+1)-gammaln(p.valence*ns+1)
    # Remaining ideal shells form a series with decreasing term ratio.
    r=np.exp(activity_log(p)+(p.valence+1)*np.log(2)-np.log(ns[-1]+1)
             -np.log(np.arange(p.valence*ns[-1]+1,p.valence*(ns[-1]+1)+1)).sum())
    logtail=logsumexp(tail)-np.log1p(-r)
    bound=np.exp(logsumexp(base_logs)+nC*np.log(2)-gammaln(nC+1)+logtail-norm)
    if bound>1e-12:raise RuntimeError(f'Exact salt tail too large: {bound}')
    return dict(labels=LABELS,mean=(weights@obs).tolist(),
                variance=(weights@np.array(all_seconds)-(weights@obs)**2).tolist(),
                relative_partition_tail_bound=float(bound),states=len(weights),log_partition=float(norm))


def deterministic_checks():
    rng=np.random.default_rng(9141);errors={'energy_delta':0.,'polymer_balance':0.,'joint_balance':0.,'ion_balance':0.,'salt_balance':0.,'global_salt_balance':0.,'gaussian_identity':0.}
    for periodic in [False,True]:
        for valence in [1,2,3]:
            p=Parameters(sites=8,length=4,area=8,chains=1,beads=3,valence=valence,activity_molar=.4,periodic=periodic)
            s=initial_state(p)
            s['counts'][1]=rng.multinomial(3,np.ones(8)/8)
            s['counts'][2]=rng.multinomial(10,np.ones(8)/8)
            s['counts'][3]=rng.multinomial(10*valence,np.ones(8)/8)
            s['x']=rng.normal(0,5,size=8);s['x']-=s['x'].mean()
            for trial in range(100):
                counts=s['counts'];charge=-counts[0]+counts[1]+valence*counts[2]-counts[3]
                s['q'][:]=s['f']@charge
                a,b=rng.choice(8,2,replace=False);transfer=rng.integers(-6,7)
                delta=_delta_e(s['q'],s['q'].sum(),s['f'],s['coefficient'],periodic,a,b,float(transfer),s['radius'])[0]
                old=energy(p,s)[0];qnew=s['q']+transfer*(s['f'][:,b]-s['f'][:,a])
                new=s['coefficient']*np.sum((qnew-qnew.mean() if periodic else qnew)**2)
                errors['energy_delta']=max(errors['energy_delta'],abs(delta-new+old))
                # Polymer/solvent exchange with symmetric proposal.
                bead=rng.integers(p.beads);a=s['chain'][0,bead]
                if a!=b and counts[0,b]<p.capacity:
                    oldw=log_weight(p,s);oldjoint=log_weight(p,s,True)
                    oldcontact=energy(p,s)[1]
                    de=_delta_e(s['q'],s['q'].sum(),s['f'],s['coefficient'],periodic,a,b,-1.,s['radius'])[0]
                    ratio=-de+np.log((p.capacity-counts[0,b])/(p.capacity-counts[0,a]+1))
                    bond=0.
                    if bead==0:bond+=s['loggraft'][b]-s['loggraft'][a]
                    else:bond+=s['logbond'][s['chain'][0,bead-1],b]-s['logbond'][s['chain'][0,bead-1],a]
                    if bead<p.beads-1:bond+=s['logbond'][b,s['chain'][0,bead+1]]-s['logbond'][a,s['chain'][0,bead+1]]
                    counts[0,a]-=1;counts[0,b]+=1;s['chain'][0,bead]=b
                    errors['polymer_balance']=max(errors['polymer_balance'],abs(log_weight(p,s)-oldw-(ratio+bond-energy(p,s)[1]+oldcontact)))
                    errors['joint_balance']=max(errors['joint_balance'],abs(log_weight(p,s,True)-oldjoint-(ratio+bond+s['x'][b]-s['x'][a])))
                    counts[0,a]+=1;counts[0,b]-=1;s['chain'][0,bead]=a
                # Ion independence proposal: Binomial conditional proposal ratio.
                t=rng.integers(1,4);a,b=rng.choice(8,2,replace=False);total=counts[t,a]+counts[t,b]
                newa=rng.binomial(total,.5);olda=counts[t,a];change=olda-newa
                val=1 if t==1 else (valence if t==2 else -1)
                de=_delta_e(s['q'],s['q'].sum(),s['f'],s['coefficient'],periodic,a,b,float(val*change),s['radius'])[0]
                oldw=log_weight(p,s)
                proposal=gammaln(newa+1)+gammaln(total-newa+1)-gammaln(olda+1)-gammaln(total-olda+1)
                counts[t,a]-=change;counts[t,b]+=change
                errors['ion_balance']=max(errors['ion_balance'],abs(log_weight(p,s)-oldw+proposal+de))
                counts[t,a]+=change;counts[t,b]-=change
                # Site-uniform birth versus cation-uniform death proposal.
                a=rng.integers(8);nplus=counts[2].sum();oldplus=counts[2,a];oldminus=counts[3,a]
                expected=activity_log(p)+np.log(8)-np.log(nplus+1)-np.log(np.arange(oldminus+1,oldminus+valence+1)).sum()
                oldw=log_weight(p,s);counts[2,a]+=1;counts[3,a]+=valence
                proposal=np.log((oldplus+1)/(nplus+1))+np.log(8)
                errors['salt_balance']=max(errors['salt_balance'],abs(log_weight(p,s)-oldw+proposal-expected))
                counts[2,a]-=1;counts[3,a]-=valence
                cells=rng.integers(0,p.sites,size=valence+1)
                oldw=log_weight(p,s);oldminus=counts[3].copy();oldplus=counts[2,cells[0]]
                de=_global_delta(s['q'],s['q'].sum(),s['f'],s['coefficient'],periodic,cells,valence,1.)[0]
                expected=activity_log(p)+(valence+1)*np.log(p.sites)-np.log(nplus+1)
                expected-=np.log(np.arange(valence*nplus+1,valence*(nplus+1)+1)).sum()
                counts[2,cells[0]]+=1
                for cell in cells[1:]:counts[3,cell]+=1
                # Ordered-position multiplicities cancel in both directions.
                proposal=np.log((oldplus+1)/(nplus+1))+(valence+1)*np.log(p.sites)
                proposal+=np.sum(gammaln(counts[3]+1)-gammaln(oldminus+1))
                proposal-=np.log(np.arange(valence*nplus+1,valence*(nplus+1)+1)).sum()
                errors['global_salt_balance']=max(errors['global_salt_balance'],abs(log_weight(p,s)-oldw+proposal-expected+de))
                counts[2,cells[0]]-=1;counts[3]=oldminus
            # Integrated real-field exponent equals contact exponent up to a
            # constant depending only on total P. Check independent occupancies.
            for _ in range(30):
                n=rng.multinomial(3,np.ones(8)/8)
                value=p.chi/p.capacity*np.sum((n-n.mean())**2)+p.chi/p.capacity*np.sum(n*(p.capacity-n))
                expected=p.chi*3-p.chi/p.capacity*9/8
                errors['gaussian_identity']=max(errors['gaussian_identity'],abs(value-expected))
    if max(errors.values())>1e-10:raise AssertionError(errors)
    # Unlike the two/eight-cell checks, these grids actually exercise the
    # compact CDF-tail update window used by the production sampler.
    error_large=0.
    for periodic in [False,True]:
        for sites in [400,800]:
            p=Parameters(sites=sites,periodic=periodic);s=initial_state(p)
            charges=rng.integers(-4,5,size=sites);charges[-1]-=charges.sum()
            s['q'][:]=s['f']@charges;old=energy(p,s)[0]
            for trial in range(100):
                a=int(rng.integers(0,sites));b=int(np.clip(a+rng.integers(-8,9),0,sites-1)) if trial%2 else int(rng.integers(0,sites))
                charge=float(rng.integers(-4,5))
                de=_delta_e(s['q'],s['q'].sum(),s['f'],s['coefficient'],periodic,a,b,charge,s['radius'])[0]
                qnew=s['q']+charge*(s['f'][:,b]-s['f'][:,a])
                new=s['coefficient']*np.sum((qnew-qnew.mean() if periodic else qnew)**2)
                error_large=max(error_large,abs(new-old-de)/max(1.,abs(new),abs(old)))
    if error_large>1e-12:raise AssertionError(error_large)
    errors['large_grid_energy_scaled_error']=error_large
    return errors


def check():
    ap=argparse.ArgumentParser(description='Independent finite-state and detailed-balance checks for positive_brush_1d.');ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    results={'deterministic':deterministic_checks(),'exact':{}}
    for z in [1,2,3]:
        p=Parameters(sites=2,length=2,area=2,chains=1,beads=2,valence=z,activity_molar=.2)
        results['exact'][str(z)]=exact_two_cells(p)
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
