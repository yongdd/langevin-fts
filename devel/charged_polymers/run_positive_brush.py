"""Run actual Monte Carlo sampling, with a separately calibrated salt bath.

Long runs belong in SLURM. All uncertainties include autocorrelation/blocking;
completion alone is not an equilibration or continuum-convergence claim.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from positive_brush_1d import Parameters, initial_state, advance, check_state, energy, potential
from check_positive_brush import LABELS
from analyze_clfts_exact import statistics


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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
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
    code_hashes={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()
                 for f in ['positive_brush_1d.py','run_positive_brush.py']}
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


if __name__=='__main__':main()
