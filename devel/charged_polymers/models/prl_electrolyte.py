"""Electrolyte limits of the PRL correlation-augmented SCFT closure.

Sources: Agrawal & Wang, PRL 129, 228001 (2022), Eqs. 2, 6 and SI II;
JCTC 18, 6271 (2022), Eqs. 12--24. This is a deterministic electrolyte
extension, not the grafted-chain solver and not CL sampling. Equal Born radii
and equal ion/solvent volumes are required by the bulk free-energy integral.
No polymer connectivity or dipolar dielectric response is included.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, root, least_squares, newton_krylov, NoConvergence
from scipy.special import erfcx, logsumexp, expit
from numpy.polynomial.legendre import leggauss


def born_self_energy(kappa, radius, bjerrum, valence=1.):
    """Absolute u/kBT, including the constant Born term; PRL Eq. 6.

    The existing brush `_u_fn` uses the algebraically identical exp*erfc.
    erfcx avoids overflow when testing the concentrated electrolyte limit.
    """
    x = radius*np.asarray(kappa)
    return valence**2*bjerrum/(2*radius)*(1-x*erfcx(x/np.sqrt(np.pi)))


def planar_green_diagonal(kappa_squared, wave_numbers, dz):
    """Diagonal of [-d_z^2 + q^2 + kappa(z)^2]^{-1}, divided by dz.

    The endpoints continue into infinite uniform media with their respective
    endpoint screening lengths. Permittivity factors are applied by the caller.
    Continued fractions evaluate the inverse in O(Nq*Nz) memory and time.
    """
    k2=np.asarray(kappa_squared); qs=np.asarray(wave_numbers)
    if k2.ndim!=1 or len(k2)<2 or qs.ndim!=1 or dz<=0 or np.min(k2)<0 or np.min(qs)<=0:
        raise ValueError('Invalid planar Green-function grid')
    s=qs[:,None]**2+k2[None,:]; diagonal=2/dz**2+s; b2=dz**-4
    def exterior(t):return 2/(2/dz**2+t+np.sqrt(t*(t+4/dz**2)))
    outside_l=exterior(s[:,0]); outside_r=exterior(s[:,-1])
    gl=np.empty_like(diagonal);gr=np.empty_like(diagonal)
    gl[:,0]=1/(diagonal[:,0]-b2*outside_l)
    gr[:,-1]=1/(diagonal[:,-1]-b2*outside_r)
    for j in range(1,len(k2)):gl[:,j]=1/(diagonal[:,j]-b2*gl[:,j-1])
    for j in range(len(k2)-2,-1,-1):gr[:,j]=1/(diagonal[:,j]-b2*gr[:,j+1])
    inv=diagonal.copy()
    inv[:,1:]-=b2*gl[:,:-1];inv[:,:-1]-=b2*gr[:,1:]
    inv[:,0]-=b2*outside_l;inv[:,-1]-=b2*outside_r
    return 1/inv/dz


@dataclass
class EqualSizeBulk:
    """Electroneutral salt, in units of a (Born radius) and kBT.

    n = (c+ + c-) a^3; g = q+ q- lB/a; v = molecular volume/a^3.
    Ion and solvent volumes are equal. Constant Born and ideal reference
    terms cancel from coexistence and are omitted from f and chemical mu.
    """
    g: float
    qp: float = 2.
    qm: float = 2.
    volume: float = 4*np.pi/3

    def __post_init__(self):
        if min(self.g, self.qp, self.qm, self.volume) <= 0:
            raise ValueError('Positive coupling, valences and volume required')
        self._nodes, self._weights = leggauss(96)

    def correlation_mu(self, n):
        x = np.sqrt(4*np.pi*self.g*np.asarray(n))
        return -.5*self.g*x*erfcx(x/np.sqrt(np.pi))

    def correlation_free_energy(self, n):
        # Integrate the chemical potential from zero density. Transform to
        # kappa*a first to remove the sqrt(n) endpoint singularity.
        n = np.asarray(n)
        x = np.sqrt(4*np.pi*self.g*n)
        t = x[..., None]*(self._nodes+1)/2
        return -x/(8*np.pi)*np.sum(self._weights*t*t*erfcx(t/np.sqrt(np.pi)), axis=-1)

    def mu(self, n):
        n = np.asarray(n)
        return np.log(n*self.volume)-np.log1p(-n*self.volume)+self.correlation_mu(n)

    def free_energy(self, n):
        n = np.asarray(n)
        return n*np.log(n*self.volume)+(1/self.volume-n)*np.log1p(-n*self.volume)+self.correlation_free_energy(n)

    def pressure(self, n):
        n = np.asarray(n)
        return -np.log1p(-self.volume*n)/self.volume+n*self.correlation_mu(n)-self.correlation_free_energy(n)

    def stability(self, n):
        """n*dmu/dn; positive on a locally stable homogeneous branch."""
        n = np.asarray(n); x = np.sqrt(4*np.pi*self.g*n)
        e = erfcx(x/np.sqrt(np.pi))
        a = (1+2*x*x/np.pi)*e-2*x/np.pi
        return 1/(1-self.volume*n)-self.g*x*a/4

    @classmethod
    def critical(cls, volume=4*np.pi/3):
        def residual(y):
            n,g = np.exp(y); x = np.sqrt(4*np.pi*g*n)
            e = erfcx(x/np.sqrt(np.pi)); ep = 2*x/np.pi*e-2/np.pi
            a = (1+2*x*x/np.pi)*e-2*x/np.pi
            ap = 4*x/np.pi*e+(1+2*x*x/np.pi)*ep-2/np.pi
            return [1/(1-volume*n)-g*x*a/4,
                    volume*n/(1-volume*n)**2-g*x*(a+x*ap)/8]
        answer = root(residual, np.log([.005,13.]), tol=1e-11)
        if not answer.success or max(abs(np.asarray(residual(answer.x))))>1e-9:
            raise RuntimeError('Critical-point calculation failed')
        n,g = np.exp(answer.x)
        return dict(n=float(n), c_star=float(8*n), g=float(g),
                    T_star=float(2/g), residual=float(max(abs(np.asarray(residual(answer.x))))))

    def coexistence(self):
        grid = np.geomspace(1e-15, (1-1e-9)/self.volume, 1000)
        values = self.stability(grid)
        brackets = np.flatnonzero(values[:-1]*values[1:]<0)
        if len(brackets)!=2:
            raise ValueError('No two spinodals: temperature must be below critical')
        sv,sl = [brentq(self.stability, grid[i],grid[i+1],xtol=1e-15) for i in brackets]
        lower,upper = float(self.mu(sl)),float(self.mu(sv))

        def concentrations(mu):
            # Log density resolves the very dilute phase, and the spinodals
            # keep the two roots separate (avoid the trivial n_v=n_l root).
            nv = np.exp(brentq(lambda logn:self.mu(np.exp(logn))-mu,-700.,np.log(sv),xtol=1e-12))
            nl = brentq(lambda n:self.mu(n)-mu,sl,(1-1e-12)/self.volume,xtol=1e-14)
            return nv,nl

        def pressure_difference(mu):
            nv,nl = concentrations(mu)
            return self.pressure(nl)-self.pressure(nv)
        mu = brentq(pressure_difference,lower+1e-11,upper-1e-11,xtol=1e-12)
        nv,nl = concentrations(mu)
        delta_u = (self.correlation_mu(nl)-self.correlation_mu(nv))/(self.qp*self.qm)
        # Signed q+ and -q- in the Boltzmann factors; psi_v=0.
        psi_g = -(self.qp-self.qm)*delta_u
        residual = max(abs(self.mu(nv)-self.mu(nl)), abs(self.pressure(nv)-self.pressure(nl)))
        # Check the common tangent lies below all intermediate homogeneous
        # states, in addition to equality of mu and pressure.
        sample = np.geomspace(nv,nl,501)
        tangent = self.free_energy(nv)+mu*(sample-nv)
        tangent_gap = float(np.min(self.free_energy(sample)-tangent))
        if residual>1e-8 or tangent_gap < -1e-9 or min(self.stability(nv),self.stability(nl))<=0:
            raise RuntimeError('Coexistence acceptance checks failed')
        return dict(n_v=float(nv),n_l=float(nl),c_star_v=float(8*nv),c_star_l=float(8*nl),
                    mu=float(mu),pressure=float(self.pressure(nv)),psi_g=float(psi_g),
                    residual=float(residual),tangent_min_gap=tangent_gap,
                    stability_v=float(self.stability(nv)),stability_l=float(self.stability(nl)),
                    spinodal_v=float(sv),spinodal_l=float(sl))


def solve_wall(sigma=-.01, cb_molar=.2, cells=400, length=10., correlations=True,
               lB=.7, born=.16, excluded_radius=.25, initial=None):
    """2:1 salt at a charged wall, JCTC dilute-branch first attempt.

    Physical units nm and kBT. Potential is defined on nodes including the
    wall. The far boundary has psi=0; ions are grand canonical. Nonlinear
    algebraic density equations and finite-volume Poisson are solved together.
    This does not implement the paper's separate condensed-layer variational
    construction and cannot certify the equilibrium condensed branch.
    """
    if cells<4 or min(cb_molar,length,lB,born,excluded_radius)<=0:
        raise ValueError('Wall calculation requires positive scales, salt and at least four cells')
    z=np.linspace(0,length,cells+1); dz=z[1]-z[0]; m=len(z)
    volume=4*np.pi*excluded_radius**3/3
    bulk=np.array([1.,2.])*cb_molar*.6022
    if volume*sum(bulk)>=1:raise ValueError('Bulk packing fraction must be below one')
    charges=np.array([2.,-1.])
    kb=np.sqrt(4*np.pi*lB*np.dot(charges**2,bulk))
    ub=born_self_energy(kb,born,lB,charges) if correlations else np.zeros(2)
    log_fugacity=np.log(bulk*volume)-np.log1p(-volume*sum(bulk))+ub
    from scipy.sparse import bmat, diags
    from scipy.sparse.linalg import spsolve

    def unpack(y):
        psi=y[:m]; odds=y[m:].reshape(2,m)
        logden=logsumexp(np.vstack([np.zeros(m),odds]),axis=0)
        density=np.exp(odds-logden)/volume
        return psi,odds,density

    def residual(y):
        psi,odds,c=unpack(y)
        kappa=np.sqrt(4*np.pi*lB*np.sum(charges[:,None]**2*c,axis=0))
        u=born_self_energy(kappa[None,:],born,lB,charges[:,None]) if correlations else np.zeros_like(c)
        chemistry=odds+charges[:,None]*psi+u-log_fugacity[:,None]
        charge=np.sum(charges[:,None]*c,axis=0)
        # Scale Poisson by dz^2. Neumann flux appears at the half wall cell;
        # outer endpoint is Dirichlet. Geometry is recorded in the summary.
        poisson=np.empty(m)
        poisson[1:-1]=-(psi[2:]-2*psi[1:-1]+psi[:-2])-4*np.pi*lB*dz*dz*charge[1:-1]
        poisson[0]=-(psi[1]-psi[0])-4*np.pi*lB*dz*sigma-2*np.pi*lB*dz*dz*charge[0]
        poisson[-1]=psi[-1]
        return np.r_[chemistry.ravel(),poisson]

    def jacobian(y):
        psi,odds,c=unpack(y)
        strength=np.sum(charges[:,None]**2*c,axis=0)
        charge=np.sum(charges[:,None]*c,axis=0)
        kappa=np.sqrt(4*np.pi*lB*strength); x=born*kappa
        e=erfcx(x/np.sqrt(np.pi))
        ep=2*x/np.pi*e-2/np.pi
        du=-charges[:,None]**2*lB/2*(e+x*ep) if correlations else np.zeros_like(c)
        dkap=2*np.pi*lB/kappa*c*(charges[:,None]**2-volume*strength)
        cj=[[diags((1. if i==j else 0.)+du[i]*dkap[j]) for j in range(2)] for i in range(2)]
        main=np.full(m,2.);main[0]=main[-1]=1.
        lower=-np.ones(m-1);lower[-1]=0.
        a=diags([lower,main,-np.ones(m-1)],[-1,0,1])
        factors=np.full(m,-4*np.pi*lB*dz*dz);factors[0]*=.5;factors[-1]=0.
        pc=[diags(factors*c[j]*(charges[j]-volume*charge)) for j in range(2)]
        return bmat([[diags(np.full(m,charges[0])),*cj[0]],
                     [diags(np.full(m,charges[1])),*cj[1]], [a,*pc]],format='csc')

    if initial is None:
        psi=4*np.pi*lB*sigma/kb*np.exp(-kb*z)
        odds=np.log(bulk[:,None]*volume)-np.log1p(-volume*sum(bulk))-charges[:,None]*psi
        y=np.r_[psi,odds.ravel()]
    else:
        y=np.r_[np.interp(z,initial['z'],initial['psi']),
                np.vstack([np.interp(z,initial['z'],u) for u in initial['odds']]).ravel()]
    direction=np.random.default_rng(20260914).normal(size=len(y))
    eps=2e-6
    finite_difference=(residual(y+eps*direction)-residual(y-eps*direction))/(2*eps)
    jacobian_error=float(np.max(abs(jacobian(y)@direction-finite_difference))/max(1.,np.max(abs(finite_difference))))
    if jacobian_error>1e-7:raise RuntimeError('Wall Jacobian directional check failed')
    result=least_squares(residual,y,jac=jacobian,method='trf',ftol=1e-11,
                         xtol=1e-11,gtol=1e-11,max_nfev=500,x_scale='jac',
                         tr_options={'atol':1e-12,'btol':1e-12,'maxiter':1000})
    # The least-squares gradient can be tiny while the unscaled Poisson
    # residual is still too large on fine grids. Polish the same equations
    # with sparse Newton steps; do not relax the physical acceptance limits.
    y=result.x.copy();polish=0
    for polish in range(20):
        r=residual(y)
        if abs(r).max()<1e-13:break
        step=spsolve(jacobian(y),-r)
        merit=np.linalg.norm(r)
        for fraction in 2.**(-np.arange(16)):
            trial=y+fraction*step
            if np.linalg.norm(residual(trial))<merit:
                y=trial;break
        else:break
    psi,odds,c=unpack(y); err=residual(y)
    chem_error=float(abs(err[:2*m]).max())
    # Physical Poisson residual, not just the grid-scaled solver residual.
    poisson_error=float(abs(err[2*m:-1]).max()/dz**2)
    charge=np.dot(charges,c)
    integrated_charge=float(np.trapezoid(charge,z))
    surface_derivative=-4*np.pi*lB*sigma
    outer_derivative=(psi[-1]-psi[-2])/dz-2*np.pi*lB*dz*charge[-1]
    gauss_error=float(abs(sigma+integrated_charge+outer_derivative/(4*np.pi*lB)))
    accepted=bool(chem_error<1e-7 and poisson_error<1e-6 and gauss_error<1e-6)
    return dict(z=z,psi=psi,c=c,odds=odds,summary=dict(
        accepted=accepted,solver_success=bool(result.success),nfev=int(result.nfev),newton_polish_steps=polish,
        jacobian_directional_error=jacobian_error,
        chemistry_residual=chem_error,poisson_residual=poisson_error,
        gauss_error=gauss_error,integrated_mobile_charge=integrated_charge,
        psi_wall=float(psi[0]),psi_max=float(psi.max()),
        max_packing=float(volume*np.max(c.sum(axis=0))),
        surface_derivative=surface_derivative,outer_derivative=float(outer_derivative),
        effective_model=dict(qp=2.,qm=1.,born_radius_nm=born,excluded_radius_nm=excluded_radius,
                             volume_nm3=volume,bjerrum_nm=lB,cb_molar=cb_molar,
                             length_nm=length,cells=cells,correlations=bool(correlations)),
        branch='dilute initial state; condensed-layer minimization not implemented'))


def solve_symmetric_interface(temperature=.8, cells=400, length=80., nq=128,
                              qmax=100., initial=None):
    """Exploratory PRL-2022 G_s+G_l planar interface, symmetric equal-size salt.

    a=1 units, full 3D fluctuations through a transverse wave-number integral.
    Infinite bulk extensions supply discrete transparent boundary conditions.
    Subtract the homogeneous Green function on the SAME discrete operator;
    its uniform-density value cancels exactly before the continuum limit.
    A phase condition pins translation; its auxiliary force must tend to zero
    for a solution of the unforced equations to be accepted.
    """
    if not 0<temperature<1 or cells<4 or length<=0 or nq<8 or qmax<=1e-5:
        raise ValueError('Invalid subcritical interface parameters')
    critical=EqualSizeBulk.critical()
    bulk=EqualSizeBulk(critical['g']/temperature)
    coexist=bulk.coexistence(); nv,nl=coexist['n_v'],coexist['n_l']
    z=np.linspace(-length/2,length/2,cells+1); dz=z[1]-z[0]; m=len(z)
    qnodes,qweights=leggauss(nq)
    lo,hi=np.log(1e-5),np.log(qmax)
    qs=np.exp((qnodes+1)*(hi-lo)/2+lo)
    weights=qweights*(hi-lo)/2*qs**2  # q dq = q^2 d(log q)
    def nonlocal_mu(n):
        s=qs[:,None]**2+4*np.pi*bulk.g*n[None,:]
        actual=planar_green_diagonal(4*np.pi*bulk.g*n,qs,dz)
        local=1/np.sqrt(s*(s+4/dz**2))/dz
        return bulk.g*np.sum(weights[:,None]*(actual-local),axis=0)

    def logodds(n):return np.log(n*bulk.volume)-np.log1p(-n*bulk.volume)
    start=nv+(nl-nv)*(.5+.5*np.tanh(z/4))
    if initial is not None:start=np.interp(z,initial['z'],initial['n'])
    pin_reference=nv+(nl-nv)*(z>0)+.5*(nl-nv)*(z==0)
    pin_weight=1/np.cosh(z/8)**2
    calls=0

    def unpack(y):
        n=np.r_[nv,expit(y[:-1])/bulk.volume,nl]
        return n,y[-1]
    def residual(y):
        nonlocal calls
        calls+=1
        n,force=unpack(y)
        chemistry=logodds(n)+bulk.correlation_mu(n)+nonlocal_mu(n)-coexist['mu']
        phase=np.trapezoid(n-pin_reference,z)/(nl-nv)
        return np.r_[chemistry[1:-1]+force*pin_weight[1:-1],phase]
    y0=np.r_[logodds(start[1:-1]),0.]
    success=True
    try:
        y=newton_krylov(residual,y0,f_tol=2e-8,maxiter=100,inner_maxiter=50,
                         rdiff=1e-5,line_search='armijo')
    except NoConvergence as exc:
        success=False;y=exc.args[0]
    if not success:
        # An explicit finite-difference Jacobian is a slower independent
        # fallback for the nearly singular interface-translation mode.
        fallback=root(residual,y,method='hybr',options={'xtol':1e-9,'eps':1e-9,'maxfev':8000})
        y=fallback.x
        success=bool(fallback.success or np.max(abs(residual(y)))<2e-8)
    n,force=unpack(y)
    correction=nonlocal_mu(n)
    raw=logodds(n)+bulk.correlation_mu(n)+correction-coexist['mu']
    error=float(abs(raw[1:-1]).max())
    endpoints=float(max(abs(raw[0]),abs(raw[-1])))
    accepted=bool(success and error<1e-6 and abs(force)<1e-6 and endpoints<1e-5)
    return dict(z=z,n=n,local_mu=bulk.correlation_mu(n),nonlocal_mu=correction,
                chemical_residual=raw,summary=dict(accepted=accepted,
                critical=critical,coexistence=coexist,nonlinear_solver_success=success,
                interior_residual=error,endpoint_residual=endpoints,
                pinning_force=float(force),residual_calls=calls,nq=nq,qmax=qmax,
                phase_condition=float(np.trapezoid(n-pin_reference,z)/(nl-nv)),
                min_density=float(n.min()),max_density=float(n.max()),
                effective_model=dict(qp=2.,qm=2.,volume_over_a3=bulk.volume,g=bulk.g,
                                     length_over_a=length,cells=cells),
                scope='Exploratory symmetric nonlocal interface; grid, box and wave-number convergence required'))
