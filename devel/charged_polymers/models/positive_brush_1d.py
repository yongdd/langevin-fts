"""Positive particle/real-field sampling of an explicit planar finite action.

P/S occupancies are exactly incompressible; monovalent C is canonical;
z:1 salt is grand canonical and charge neutral. The electrostatic Gaussian
field is integrated out, not replaced by a PB saddle or an LDA self-energy.
This retains z fluctuations only, not the PRL's three-dimensional measure.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np
from numba import njit
from scipy.special import ndtr


@dataclass
class Parameters:
    sites: int = 400
    length: float = 200.
    area: float = 40.
    beads: int = 100
    chains: int = 4
    segment_length: float = 1.
    monomer_volume: float = 1.
    chi: float = .5
    bjerrum_length: float = .7
    born_radius: float = .25
    valence: int = 3
    graft_width: float = .5
    activity_molar: float = .01
    log_salt_shift: float = 0.
    periodic: bool = False
    joint: bool = True

    def metadata(self):
        d=asdict(self)
        d.update(dz=self.length/self.sites,cell_capacity=self.capacity,
                 grafting_density=self.chains/self.area,
                 salt_axis='ideal-reference activity; not calibrated bulk concentration')
        return d

    @property
    def capacity(self):
        value=self.area*self.length/(self.sites*self.monomer_volume)
        if not np.isclose(value,round(value),rtol=0,atol=1e-10) or value<1:
            raise ValueError('Exact finite-cell P/S occupancy requires integer A*dz/v >= 1')
        return int(round(value))


def operators(p):
    """Reflecting/periodic Gaussian charge shapes and discrete Gaussian bonds."""
    m=p.sites;dz=p.length/m;z=(np.arange(m)+.5)*dz
    width=p.born_radius/np.sqrt(np.pi)
    edges=np.arange(m+1)*dz
    if width<=0:raise ValueError('A positive charge radius is required')
    cumulative=np.zeros((m+1,m))
    for image in [-1,0,1]:
        centers=z+2*image*p.length if not p.periodic else z+image*p.length
        cumulative+=ndtr((edges[:,None]-centers)/width)-ndtr(-centers/width)
        if not p.periodic:
            centers=-z+2*image*p.length
            cumulative+=ndtr((edges[:,None]-centers)/width)-ndtr(-centers/width)
    cumulative/=cumulative[-1]
    cumulative[0]=0.;cumulative[-1]=1.
    if np.min(np.diff(cumulative,axis=0)) < -1e-14:
        raise RuntimeError('Charge shape lost positivity')
    f=np.ascontiguousarray(cumulative[1:])
    distance=z[:,None]-z
    logbond=-1.5*(distance/p.segment_length)**2
    if not p.periodic:
        reflected=-1.5*((z[:,None]+z-2*p.length)/p.segment_length)**2
        logbond=np.logaddexp(logbond,reflected)
    else:
        for shift in [-p.length,p.length]:
            logbond=np.logaddexp(logbond,-1.5*((distance+shift)/p.segment_length)**2)
    loggraft=-.5*(z/p.graft_width)**2
    coefficient=2*np.pi*p.bjerrum_length*dz/p.area
    # At this radius the omitted normal-CDF tails are below floating precision.
    radius=int(np.ceil(10*width/dz))+2
    return z,f,np.ascontiguousarray(logbond),loggraft,coefficient,radius


def initial_state(p,seed=1,initial='extended'):
    if p.valence not in [1,2,3] or p.chi<0 or p.activity_molar<0:
        raise ValueError('Require valence 1/2/3, chi>=0, activity>=0')
    if p.chains*p.beads>=p.capacity*p.sites:
        raise ValueError('The box must contain solvent')
    rng=np.random.default_rng(seed)
    z,f,logbond,loggraft,coefficient,radius=operators(p)
    chain=np.zeros((p.chains,p.beads),dtype=np.int64)
    counts=np.zeros((4,p.sites),dtype=np.int64)  # P, C, salt +, salt -
    step=.5*p.segment_length if initial=='extended' else max(.12*p.segment_length,p.chains*p.monomer_volume/(.7*p.area))
    for c in range(p.chains):
        for n in range(p.beads):
            desired=min(p.sites-1,round((n+.5)*step/(p.length/p.sites)-.5))
            candidates=np.argsort(abs(np.arange(p.sites)-desired),kind='stable')
            target=next(int(j) for j in candidates if counts[0,j]<p.capacity)
            chain[c,n]=target;counts[0,target]+=1
    counts[1]=counts[0]  # a neutral initial cloud, subsequently sampled
    units=round(.6022*p.activity_molar*p.area*p.length)
    if units:
        cells=rng.multinomial(units,np.full(p.sites,1/p.sites))
        counts[2]=cells;counts[3]=p.valence*cells
    charge=-counts[0]+counts[1]+p.valence*counts[2]-counts[3]
    q=f@charge
    x=2*p.chi/p.capacity*(counts[0]-counts[0].mean())
    return dict(chain=chain,counts=counts,q=q,x=x,rng=rng,z=z,f=f,
                logbond=logbond,loggraft=loggraft,coefficient=coefficient,radius=radius)


@njit(cache=True)
def _delta_e(q,qsum,f,coefficient,periodic,source,target,charge,radius):
    m=len(q)
    lo=0 if periodic else max(0,min(source,target)-radius)
    hi=m if periodic else min(m,max(source,target)+radius+1)
    delta=0.;total=0.
    for j in range(lo,hi):
        dq=charge*(f[j,target]-f[j,source])
        delta+=2*q[j]*dq+dq*dq;total+=dq
    if periodic:delta-=(2*qsum*total+total*total)/m
    return coefficient*delta,total,lo,hi


@njit(cache=True)
def _move_q(q,f,source,target,charge,lo,hi):
    for j in range(lo,hi):q[j]+=charge*(f[j,target]-f[j,source])


@njit(cache=True)
def _global_delta(q,qsum,f,coefficient,periodic,cells,valence,direction):
    dq=np.zeros(len(q));change=0.;total=0.
    for j in range(len(q)):
        v=valence*f[j,cells[0]]
        for k in range(1,valence+1):v-=f[j,cells[k]]
        v*=direction;dq[j]=v;total+=v;change+=2*q[j]*v+v*v
    if periodic:change-=(2*qsum*total+total*total)/len(q)
    return coefficient*change,dq,total


@njit(cache=True)
def _pick_particle(rng,counts,total):
    pick=rng.integers(0,total);cumulative=0
    for j in range(len(counts)):
        cumulative+=counts[j]
        if pick<cumulative:return j
    return len(counts)-1


@njit(cache=True)
def batch(sweeps,rng,chain,counts,q,x,f,logbond,loggraft,coefficient,radius,
          capacity,chi,valence,periodic,joint,log_activity,move_width,
          grand_attempts,long_ion_attempts):
    """Every subkernel leaves the stated finite weight invariant.

    Salt births add one +z ion and z anions at the same cell. Death selects a
    cation uniformly and requires z anions there. Their proposal asymmetry
    yields the explicit total-cation factors below. Ion redistribution then
    explores separated configurations; no permanently bonded ion groups exist.
    """
    m=len(q);nc,nb=chain.shape;qsum=q.sum()
    totals=np.array([counts[t].sum() for t in range(4)])
    accepted=np.zeros(8,np.int64)
    for sweep in range(sweeps):
        for attempt in range(nc*nb):
            c=rng.integers(0,nc);n=rng.integers(0,nb)
            source=chain[c,n]
            target=source+int(round(rng.normal(0.,move_width)))
            if periodic:target%=m
            if target<0 or target>=m or source==target:continue
            accepted[1]+=1
            if counts[0,target]>=capacity:continue
            de,dqsum,lo,hi=_delta_e(q,qsum,f,coefficient,periodic,source,target,-1.,radius)
            ratio=-de+math.log((capacity-counts[0,target])/(capacity-counts[0,source]+1.))
            if joint:
                ratio+=x[target]-x[source]
            else:
                pa=counts[0,source];pb=counts[0,target]
                de_contact=chi/capacity*((pa-1)*(capacity-pa+1)+(pb+1)*(capacity-pb-1)-pa*(capacity-pa)-pb*(capacity-pb))
                ratio-=de_contact
            if n==0:ratio+=loggraft[target]-loggraft[source]
            else:ratio+=logbond[chain[c,n-1],target]-logbond[chain[c,n-1],source]
            if n<nb-1:ratio+=logbond[target,chain[c,n+1]]-logbond[source,chain[c,n+1]]
            if math.log(rng.random())<min(0.,ratio):
                chain[c,n]=target;counts[0,source]-=1;counts[0,target]+=1
                _move_q(q,f,source,target,-1.,lo,hi);qsum+=dqsum;accepted[0]+=1
        # Independence proposals from the exact ideal-gas conditional binomial.
        # Its factorial/proposal factors cancel, leaving only electrostatic dU.
        for t in range(1,4):
            if totals[t]==0:continue
            charge=1. if t==1 else (float(valence) if t==2 else -1.)
            for attempt in range(m+long_ion_attempts):
                left=rng.integers(0,m if periodic else m-1)
                if attempt<m:right=(left+1)%m
                else:right=rng.integers(0,m)
                if left==right:continue
                total=counts[t,left]+counts[t,right]
                if total==0:continue
                newleft=rng.binomial(total,.5)
                change=counts[t,left]-newleft
                if change==0:continue
                accepted[3]+=1
                de,dqsum,lo,hi=_delta_e(q,qsum,f,coefficient,periodic,left,right,charge*change,radius)
                if math.log(rng.random())<min(0.,-de):
                    counts[t,left]-=change;counts[t,right]+=change
                    _move_q(q,f,left,right,charge*change,lo,hi);qsum+=dqsum;accepted[2]+=1
        for attempt in range(grand_attempts):
            # Independent insertion positions / uniformly selected particles
            # avoid the rare same-cell coincidences of dilute multivalent salt.
            # The anion selections on deletion are ordered without replacement.
            if attempt<4:
                birth=rng.random()<.5
                if not birth and totals[2]==0:continue
                cells=np.empty(valence+1,np.int64)
                if birth:
                    for k in range(valence+1):cells[k]=rng.integers(0,m)
                    ratio=log_activity+(valence+1)*math.log(m)-math.log(totals[2]+1.)
                    for k in range(1,valence+1):ratio-=math.log(totals[3]+k)
                    direction=1.;accepted[5]+=1
                else:
                    cells[0]=_pick_particle(rng,counts[2],totals[2])
                    for k in range(1,valence+1):
                        cells[k]=_pick_particle(rng,counts[3],totals[3]-k+1)
                        counts[3,cells[k]]-=1
                    for k in range(1,valence+1):counts[3,cells[k]]+=1
                    ratio=-log_activity-(valence+1)*math.log(m)+math.log(totals[2])
                    for k in range(valence):ratio+=math.log(totals[3]-k)
                    direction=-1.;accepted[7]+=1
                de,dq,dqsum=_global_delta(q,qsum,f,coefficient,periodic,cells,valence,direction)
                if math.log(rng.random())<min(0.,ratio-de):
                    step=1 if birth else -1
                    counts[2,cells[0]]+=step
                    for k in range(1,valence+1):counts[3,cells[k]]+=step
                    totals[2]+=step;totals[3]+=valence*step
                    q+=dq;qsum+=dqsum
                    if birth:accepted[4]+=1
                    else:accepted[6]+=1
                continue
            if rng.random()<.5:
                cell=rng.integers(0,m)
                ratio=log_activity+math.log(m)-math.log(totals[2]+1.)
                for j in range(1,valence+1):ratio-=math.log(counts[3,cell]+j)
                accepted[5]+=1
                if math.log(rng.random())<min(0.,ratio):
                    counts[2,cell]+=1;counts[3,cell]+=valence
                    totals[2]+=1;totals[3]+=valence;accepted[4]+=1
            elif totals[2]>0:
                pick=rng.integers(0,totals[2]);cumulative=0;cell=0
                for j in range(m):
                    cumulative+=counts[2,j]
                    if pick<cumulative:cell=j;break
                accepted[7]+=1
                if counts[3,cell]<valence:continue
                ratio=math.log(totals[2])-math.log(m)-log_activity
                for j in range(valence):ratio+=math.log(counts[3,cell]-j)
                if math.log(rng.random())<min(0.,ratio):
                    counts[2,cell]-=1;counts[3,cell]-=valence
                    totals[2]-=1;totals[3]-=valence;accepted[6]+=1
        if joint and chi>0:
            mean=0.
            for j in range(m):
                x[j]=2*chi/capacity*counts[0,j]+math.sqrt(2*chi/capacity)*rng.normal(0.,1.)
                mean+=x[j]/m
            for j in range(m):x[j]-=mean
    return accepted


def activity_log(p):
    if p.activity_molar==0:return -np.inf
    cell_volume=p.area*p.length/p.sites
    a=.6022*p.activity_molar*cell_volume
    return math.log(a)+p.valence*math.log(p.valence*a)+p.log_salt_shift


def energy(p,state):
    q=state['q']
    electro=state['coefficient']*np.sum((q-q.mean() if p.periodic else q)**2)
    n=state['counts'][0]
    contact=p.chi/p.capacity*np.sum(n*(p.capacity-n))
    return float(electro),float(contact)


def check_state(p,s):
    counts=s['counts'];np_beads=p.chains*p.beads
    assert counts.dtype==np.int64 and (counts>=0).all()
    assert counts[0].sum()==np_beads and counts[1].sum()==np_beads
    assert counts[3].sum()==p.valence*counts[2].sum()
    assert counts[0].max()<=p.capacity
    assert np.array_equal(np.bincount(s['chain'].ravel(),minlength=p.sites),counts[0])
    charge=-counts[0]+counts[1]+p.valence*counts[2]-counts[3]
    direct=s['f']@charge
    error=float(np.max(abs(direct-s['q'])))
    if error>1e-8:raise RuntimeError(f'Incremental electrostatic charge error: {error}')
    s['q'][:]=direct
    return error


def advance(p,s,sweeps,grand_attempts=None,long_ion_attempts=8):
    return batch(sweeps,s['rng'],s['chain'],s['counts'],s['q'],s['x'],s['f'],s['logbond'],s['loggraft'],
                 s['coefficient'],s['radius'],p.capacity,p.chi,p.valence,p.periodic,p.joint,
                 activity_log(p),.6*p.segment_length/(p.length/p.sites),
                 p.sites if grand_attempts is None else grand_attempts,long_ion_attempts)


def potential(p,s):
    q=s['q']-s['q'].mean() if p.periodic else s['q']
    psi=np.r_[0.,-4*np.pi*p.bjerrum_length/p.area*(p.length/p.sites)*np.cumsum(q[:-1])]
    return psi-psi[-1]
