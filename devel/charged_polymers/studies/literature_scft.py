"""Deterministic PRL-electrolyte literature campaign; one case per job."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad, solve_bvp
from scipy.optimize import brentq

from models.prl_electrolyte import EqualSizeBulk, born_self_energy, solve_wall, solve_symmetric_interface, planar_green_diagonal
from models.prl_scft import PRLBrushSCFT


def foundation_check(out):
    """Independent Fourier quadrature and salt-free PB boundary-value check."""
    rows=[]
    for x in [0.,1e-4,.01,.1,1.,3.,10.]:
        a=.16; lb=.7; q=2.; kappa=x/a
        integrand=lambda k:np.exp(-(a*k)**2/np.pi)*(1. if kappa==0 else k*k/(k*k+kappa*kappa))
        value,error=quad(integrand,0,np.inf,epsabs=1e-11,epsrel=1e-11)
        numeric=q*q*lb/np.pi*value
        analytic=float(born_self_energy(kappa,a,lb,q))
        legacy=q*q*lb/(2*a)*float(PRLBrushSCFT._u_fn(np.asarray(x)))
        rows.append(dict(a_kappa=x,analytic=analytic,fourier=numeric,
                         legacy=legacy,quadrature_error=error,
                         relative_error=abs(numeric-analytic)/max(abs(analytic),1e-14)))
    # Exercise the original brush closure in its uniform, polymer-free limit.
    # No propagator is needed, so construct only its documented closure state.
    s=PRLBrushSCFT.__new__(PRLBrushSCFT)
    s.corr=True;s.Nz=32;s.dz=.05;s.aP=s.aplus=s.aion=.16
    s.lB=.7;s.zP=-1.;s.zp=2.;s.kappa_b=.4
    zero=np.zeros(s.Nz); cp=np.full(s.Nz,.2); cm=2*cp
    actual=s._self_energies(zero,zero,cp,cm)
    kap=np.sqrt(4*np.pi*s.lB*(4*.2+.4))
    expected=[born_self_energy(kap,.16,.7,q)-born_self_energy(.4,.16,.7,q) for q in [1.,2.,1.]]
    closure_error=max(float(np.max(abs(a-b))) for a,b in zip(actual,expected))

    length=40.
    z=np.linspace(0,length,801)
    def ode(z,y,p):return np.vstack([y[1],-2*p[0]*np.exp(-y[0])])
    def bc(a,b,p):return np.array([a[0],a[1]-2,b[1]])
    sol=solve_bvp(ode,bc,z,np.vstack([2*np.log1p(z),2/(1+z)]),p=[1.],tol=1e-9,max_nodes=15000)
    if not sol.success:raise RuntimeError(sol.message)
    density=sol.p[0]*np.exp(-sol.sol(z)[0])
    k=brentq(lambda k:k*np.tan(k*length)-1,1e-9,np.pi/(2*length)*(1-1e-12),xtol=1e-15)
    exact=k*k/np.cos(k*(length-z))**2
    pb_error=float(np.max(abs(density-exact)/exact))
    mass=quad(lambda t:float(sol.p[0]*np.exp(-sol.sol(t)[0])),0,length,epsabs=1e-10)[0]
    np.savez_compressed(out/'profiles.npz',z=z,density=density,finite_box_exact=exact,
                        infinite_wall_pb=1/(1+z)**2)
    # Independent differentiation and adaptive quadrature check the bulk f.
    model=EqualSizeBulk(16.)
    n=.025; step=1e-6
    fd=(model.free_energy(n-2*step)-8*model.free_energy(n-step)+8*model.free_energy(n+step)-model.free_energy(n+2*step))/(12*step)
    fc=quad(lambda density:float(model.correlation_mu(density)),0,n,epsabs=1e-12,epsrel=1e-12)[0]
    thermo_error=float(abs(fd-model.mu(n)))
    charging_error=float(abs(fc-model.correlation_free_energy(n)))
    # Green recurrences versus a separately constructed dense matrix inverse.
    dz=.2;k2=np.linspace(.25,4.,11);qs=np.array([.03,.7,5.])
    green=planar_green_diagonal(k2,qs,dz);dense_error=0.
    for j,q in enumerate(qs):
        s=q*q+k2;a=np.diag(2/dz**2+s)+np.diag(np.full(10,-1/dz**2),1)+np.diag(np.full(10,-1/dz**2),-1)
        for endpoint in [0,-1]:
            d=2/dz**2+s[endpoint]
            root_decay=(d-np.sqrt(d*d-4/dz**4))/(2/dz**2)
            a[endpoint,endpoint]-=root_decay/dz**2
        expected=np.diag(np.linalg.inv(a))/dz
        dense_error=max(dense_error,float(np.max(abs(green[j]-expected))))
    # Independent continuum two-half-space Green function checks the
    # transparent boundaries, q measure and continuum limit together.
    from numpy.polynomial.legendre import leggauss
    nodes,weights=leggauss(160)
    lo,hi=np.log(1e-5),np.log(100.)
    wave=np.exp((nodes+1)*(hi-lo)/2+lo);weights=weights*(hi-lo)/2*wave**2
    step_checks=[]
    for cells in [400,800,1600]:
        dz=40/cells;coords=-20+(np.arange(cells)+.5)*dz
        k2=np.where(coords<0,.25,4.)
        s=wave[:,None]**2+k2
        difference=planar_green_diagonal(k2,wave,dz)-1/(dz*np.sqrt(s*(s+4/dz**2)))
        u=np.sum(weights[:,None]*difference,axis=0)
        errors=[]
        for location in [-1.,1.]:
            i=int(np.argmin(abs(coords-location)));zz=coords[i]
            def exact_integrand(q):
                pv,pl=np.sqrt(q*q+.25),np.sqrt(q*q+4.)
                reflect=(pv-pl)/(pv+pl)
                return q*(reflect*np.exp(2*pv*zz)/(2*pv) if zz<0 else -reflect*np.exp(-2*pl*zz)/(2*pl))
            exact=quad(exact_integrand,0,np.inf,epsabs=1e-12,epsrel=1e-11)[0]
            errors.append(float(abs(u[i]/exact-1)))
        step_checks.append(dict(cells=cells,dz=dz,relative_errors=errors))
    accepted=(max(r['relative_error'] for r in rows)<1e-9 and closure_error<1e-12
              and pb_error<1e-7 and abs(mass-1)<1e-8 and thermo_error<1e-8 and charging_error<1e-10
              and dense_error<1e-12 and max(step_checks[-1]['relative_errors'])<1e-3
              and max(step_checks[-1]['relative_errors'])<max(step_checks[0]['relative_errors'])/8)
    return dict(accepted=bool(accepted),self_energy=rows,legacy_uniform_closure_error=closure_error,
                pb_relative_error=pb_error,pb_mobile_charge=float(mass),pb_nodes=int(len(sol.x)),
                pb_scaled_box_length=length,bulk_mu_derivative_error=thermo_error,
                bulk_charging_integral_error=charging_error,
                green_dense_inverse_error=dense_error,green_step_interface_checks=step_checks,
                scope='Deterministic electrolyte subproblems; not a CL or strong-coupling certification')


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['foundation','bulk','wall','interface'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--temperature',type=float,default=.8,help='T/Tc of this closure')
    p.add_argument('--qp',type=float,default=2.,help='Bulk mode only; wall is fixed 2:1, interface 2:2')
    p.add_argument('--qm',type=float,default=2.,help='Bulk mode only; wall is fixed 2:1, interface 2:2')
    p.add_argument('--volume',type=float,default=4*np.pi/3,help='Equal ion and solvent volume / a^3')
    p.add_argument('--sigma',type=float,default=-.01)
    p.add_argument('--cb',type=float,default=.2,help='Salt formula-unit molarity')
    p.add_argument('--cells',type=int,default=400)
    p.add_argument('--length',type=float,default=10.,help='Box length: nm for wall, Born-radius units for interface')
    p.add_argument('--meanfield',action='store_true')
    p.add_argument('--nq',type=int,default=128)
    p.add_argument('--qmax',type=float,default=100.)
    p.add_argument('--warm',type=Path)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'summary.json').exists():raise FileExistsError('Use a fresh result directory')
    start=time.perf_counter()
    result=dict(params={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},status='running')
    root=Path(__file__).resolve().parents[1]
    result['source_hashes']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                             [Path(__file__),root/'models/prl_electrolyte.py',root/'models/prl_scft.py']}
    try:
        if args.case=='foundation':result.update(foundation_check(args.out))
        elif args.case=='bulk':
            crit=EqualSizeBulk.critical(args.volume)
            model=EqualSizeBulk(crit['g']/args.temperature,args.qp,args.qm,args.volume)
            result.update(critical=crit,g=model.g,**model.coexistence(),accepted=True)
        elif args.case=='wall':
            initial=dict(np.load(args.warm)) if args.warm else None
            state=solve_wall(args.sigma,args.cb,args.cells,args.length,not args.meanfield,initial=initial)
            result.update(state.pop('summary'))
            np.savez_compressed(args.out/'profiles.npz',**state)
        else:
            initial=dict(np.load(args.warm)) if args.warm else None
            state=solve_symmetric_interface(args.temperature,args.cells,args.length,args.nq,args.qmax,initial)
            result.update(state.pop('summary'))
            np.savez_compressed(args.out/'profiles.npz',**state)
        result['status']='completed' if result['accepted'] else 'unconverged'
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),accepted=False)
        raise
    finally:
        result['wall_seconds']=time.perf_counter()-start
        (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    if not result['accepted']:raise RuntimeError('Numerical acceptance checks failed; see summary.json')


def analyze():
    p=argparse.ArgumentParser(description='Compare the deterministic electrolyte campaign with original vector figure curves.')
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    import re
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pymupdf as fitz
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    root=Path(__file__).resolve().parents[1]
    cases={f.parent.name:json.loads(f.read_text()) for f in args.campaign.glob('*/summary.json')}
    previews=args.campaign/'preview';previews.mkdir(exist_ok=True)
    def profile(name):return dict(np.load(args.campaign/name/'profiles.npz'))
    def save(fig,name):
        fig.savefig(args.out/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(previews/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)

    # Calibration is read from the axis ticks in the authors' EPS files;
    # all coordinates and files are preserved, with no parameter fitting.
    def red_path(name):
        text=(args.campaign/'sources'/name).read_text()
        block=text.split('1.000 0.000 0.000 setrgbcolor',1)[1].split('stroke',1)[0]
        return np.array([[float(x),float(y)] for x,y in re.findall(r'([-0-9.]+) ([-0-9.]+) [ml]\n',block)])
    xy=red_path('figure_1.eps')
    curve=np.c_[(xy[:,0]-99.8229)/((441.901323-99.8229)/.6),
                .5+(xy[:,1]-60.359615)/((295.55976-60.359615)/.5)]
    xy=red_path('figure_3a.eps')
    galvani=np.c_[.5+(xy[:,0]-98.612307)/((441.901322-98.612307)/.4),
                  (xy[:,1]-57.562740)/31.609259]
    np.savetxt(args.out/'paper_figure1.csv',curve,delimiter=',',header='c_star,T_over_Tc',comments='')
    np.savetxt(args.out/'paper_figure3a.csv',galvani,delimiter=',',header='T_over_Tc,psi_G',comments='')
    bulk_rows=[];galvani_rows=[]
    for name,d in cases.items():
        if not d.get('accepted'):continue
        if name.startswith('bulk_sym_vertex_'):
            index=int(name.rsplit('_',1)[1]);t=curve[index,1]
            same=curve[abs(curve[:,1]-t)<1e-7,0]
            pv,pl=float(min(same)),float(max(same))
            bulk_rows.append(dict(T_over_Tc=float(t),vapor=d['c_star_v'],liquid=d['c_star_l'],
                                  paper_vapor=pv,paper_liquid=pl,
                                  vapor_relative_error=float(d['c_star_v']/pv-1),
                                  liquid_relative_error=float(d['c_star_l']/pl-1),residual=d['residual']))
        elif name.startswith('bulk_asym_vertex_'):
            index=int(name.rsplit('_',1)[1]);t,value=galvani[index]
            galvani_rows.append(dict(T_over_Tc=float(t),psi_G=d['psi_g'],paper_psi_G=float(value),
                                     absolute_error=float(d['psi_g']-value),residual=d['residual']))
    bulk_rows.sort(key=lambda d:d['T_over_Tc']);galvani_rows.sort(key=lambda d:d['T_over_Tc'])
    if not bulk_rows or not galvani_rows:raise RuntimeError('Accepted bulk comparison runs missing')

    fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for a in ax:
        mask=(curve[:,1]>=.5)&(curve[:,1]<=1.001)
        a.plot(curve[mask,0],curve[mask,1],color='0.45',ls='--',lw=1.5,label='Paper theory (original vector curve)')
        for branch in ['vapor','liquid']:
            a.plot([d[branch] for d in bulk_rows],[d['T_over_Tc'] for d in bulk_rows],
                   'o',ms=4,mfc='none',color='#087F8C',label='Calculated coexistence' if branch=='vapor' else None)
        a.set(xlabel=r'Total ion concentration $c(2a)^3$',ylabel=r'$T/T_c$',ylim=(.49,1.015));a.grid(alpha=.2)
    ax[0].set_xlim(-.005,.62);ax[0].legend(fontsize=8,loc='upper right')
    ax[1].set_xscale('log');ax[1].set_xlim(1e-6,.7)
    ax[0].set_title('PRL 2022 Fig. 1: linear density axis');ax[1].set_title('Dilute branch resolved on a log axis')
    save(fig,'bulk_coexistence')

    fig,ax=plt.subplots(figsize=(6.6,4),layout='constrained')
    ax.plot(galvani[:,0],galvani[:,1],'--',color='0.4',label='Paper theory (original vector curve)')
    ax.plot([d['T_over_Tc'] for d in galvani_rows],[d['psi_G'] for d in galvani_rows],
            'o',mfc='none',color='#B65734',label='Calculated bulk equilibrium')
    ax.set(xlabel=r'$T/T_c$',ylabel=r'$\Delta\psi_G=e(\Phi_l-\Phi_v)/(k_BT)$',
           xlim=(.49,.91),ylim=(1.5,6.4),title='PRL 2022 Fig. 3(a): 2:1 salt');ax.grid(alpha=.2);ax.legend(fontsize=9)
    save(fig,'bulk_galvani')

    foundation_name='foundation_green' if 'foundation_green' in cases else 'foundation'
    data=profile(foundation_name);f=cases[foundation_name]
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    rows=f['self_energy'];xs=np.geomspace(1e-4,10,300)
    ax[0].semilogx(xs,born_self_energy(xs/.16,.16,.7,2),color='#087F8C',label='PRL self-energy')
    ax[0].plot([r['a_kappa'] for r in rows[1:]],[r['fourier'] for r in rows[1:]],'o',mfc='none',color='#B65734',label='Independent Fourier integral')
    ax[0].set(xlabel=r'$a\kappa$',ylabel=r'$u/(k_BT)$',title='Homogeneous self-energy');ax[0].legend(fontsize=8)
    ax[1].semilogy(data['z'],data['finite_box_exact'],color='0.4',label='Finite-cell analytical PB')
    ax[1].plot(data['z'][::30],data['density'][::30],'o',mfc='none',color='#087F8C',label='Numerical boundary-value solution')
    ax[1].plot(data['z'],data['infinite_wall_pb'],'--',color='#B65734',label='Infinite single-wall PB limit')
    ax[1].set(xlabel=r'$z/\mu$',ylabel=r'$\widetilde n$',title='Counterion-only charged wall');ax[1].legend(fontsize=8)
    for a in ax:a.grid(alpha=.2)
    save(fig,'foundation_checks')

    # Original PDF vector paths, not a fit to the numerical results.
    wall_pdf=root/'refs/Agrawal_2022_Ion_Condensation_and_Charge_Inversion.pdf'
    colors={(0.75,0.,0.75):.01,(0.,.75,.75):.0244,(0.,.5,0.):.0245}
    paper_wall={}
    with fitz.open(wall_pdf) as doc:
        for path in doc[15].get_drawings():
            if path['color'] in colors and len(path['items'])>4:
                points=[path['items'][0][1]]+[item[2] for item in path['items'] if item[0]=='l']
                points=np.asarray([[v.x,v.y] for v in points])
                coords=np.c_[(points[:,0]-215.38803100585938)/75.23446655273438*.25,
                              (142.9552764892578-points[:,1])/531.7352294921875]
                paper_wall[colors[path['color']]]=coords
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained');wall_rows=[]
    for sig,ax in zip([.01,.0244,.0245],axes):
        name=f'wall_wkb_{sig}_{1600 if sig!=.0245 else 800}_polished'
        d=cases[name]
        if not d['accepted']:raise RuntimeError('Fine-grid wall run did not pass')
        state=profile(name); target=paper_wall[sig]
        mask=(target[:,0]>=0)&(target[:,0]<=.75)
        sampled=np.interp(target[mask,0],state['z'],state['psi'])
        err=float(np.max(abs(sampled-target[mask,1])))
        wall_rows.append(dict(sigma=-sig,case=name,psi_wall=d['psi_wall'],
                              paper_curve_max_absolute_error=err,gauss_error=d['gauss_error'],
                              chemistry_residual=d['chemistry_residual'],poisson_residual=d['poisson_residual']))
        ax.plot(target[:,0]/.25,target[:,1],'--',color='0.4',label='Paper figure')
        ax.plot(state['z']/.25,state['psi'],color='#087F8C',label='Dilute-branch WKB solve')
        mf=f'wall_mf_{sig}_200'
        if mf in cases:
            state_mf=profile(mf);ax.plot(state_mf['z']/.25,state_mf['psi'],':',color='#B65734',label='Mean-field control')
        ax.set(xlabel=r'$z/a_0$',ylabel=r'$\psi$',xlim=(0,3),title=rf'$\sigma=-{sig}\ e/\mathrm{{nm}}^2$');ax.grid(alpha=.2)
        ax.legend(fontsize=7)
        np.savetxt(args.out/f'paper_wall_{sig}.csv',target,delimiter=',',header='z_nm,psi',comments='')
    save(fig,'wall_comparison')

    interface=None
    interface_name='interface_090_1600_L80'
    if interface_name in cases and cases[interface_name].get('accepted'):
        source_file=args.campaign/'sources/figure_2a.eps'
        text=source_file.read_text().split('0.000 0.392 0.000 setrgbcolor',1)[1].split('stroke',1)[0]
        xy=np.array([[float(x),float(y)] for x,y in re.findall(r'([-0-9.]+) ([-0-9.]+) [ml]\n',text)])
        paper=np.c_[(xy[:,0]-293.540148)/56.837236,(xy[:,1]-68.6972525)/1684.66255]
        state=profile(interface_name);zz=state['z']/2;density=4*state['n']
        # Fig. 1 uses total ion density; Fig. 2(a)'s symmetric profiles
        # have the scale of one species, c+=c-=n/2. Keep this factor explicit.
        paper_mid=float(np.interp((paper[:,1].min()+paper[:,1].max())/2,paper[:,1],paper[:,0]))
        computed_mid=float(np.interp((density[0]+density[-1])/2,density,zz))
        direct_error=float(np.max(abs(np.interp(paper[:,0],zz,density)-paper[:,1])))
        aligned_error=float(np.max(abs(np.interp(paper[:,0]-paper_mid+computed_mid,zz,density)-paper[:,1])))
        span=float(density[-1]-density[0])
        coarse=profile('interface_090_400_L80');medium=profile('interface_090_800_L80')
        comparisons={}
        for name in ['interface_090_800_L160','interface_090_400_L80_nq256','interface_090_400_L80_qmax200','interface_090_400_helper_check']:
            other=profile(name)
            common=(other['z']>=coarse['z'][0])&(other['z']<=coarse['z'][-1])
            comparisons[name]=float(np.max(abs(other['n'][common]-np.interp(other['z'][common],coarse['z'],coarse['n']))))
        fine_difference=float(np.max(abs(state['n']-np.interp(state['z'],medium['z'],medium['n']))))
        interface=dict(case=interface_name,T_over_Tc=.9,direct_GDS_frame_max_error=direct_error,
                       midpoint_aligned_max_error=aligned_error,midpoint_aligned_span_error=aligned_error/span,
                       translation_in_2a_units=paper_mid-computed_mid,
                       paper_displayed_GDS_in_original_2a_units=(293.540148-85.515865)/56.837236,
                       fine_grid_max_total_density_change=fine_difference,controls_max_total_density_changes=comparisons,
                       scope='One symmetric profile at the displayed T/Tc=0.9; midpoint alignment removes only translation. Figure-frame/GDS discrepancy remains. No surface-tension or asymmetric-interface reproduction.')
        fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for a,shift_p,shift_c in [(ax[0],0.,0.),(ax[1],paper_mid,computed_mid)]:
            a.plot(paper[:,0]-shift_p,paper[:,1],'--',color='0.4',label='Paper Fig. 2(a), label 0.9')
            a.plot(zz-shift_c,density,color='#087F8C',label='Calculated nonlocal interface')
            a.set(xlim=(-4,4),ylabel=r'$c_+(2a)^3=c_-(2a)^3$');a.grid(alpha=.2);a.legend(fontsize=8)
        ax[0].set(title='Displayed GDS frames',xlabel=r'$(z-z_{\rm GDS})/(2a)$')
        ax[1].set(title='Independent midpoint alignment',xlabel=r'$(z-z_{1/2})/(2a)$')
        save(fig,'nonlocal_interface')
        np.savetxt(args.out/'paper_figure2a.csv',paper,delimiter=',',header='z_minus_displayed_GDS_over_2a,c_species_star',comments='')

    report=dict(campaign=str(args.campaign),foundation=f,bulk=bulk_rows,galvani=galvani_rows,
                wall=wall_rows,interface=interface,cases=cases,
                max_coexistence_relative_error=max(abs(d[k]) for d in bulk_rows for k in ['vapor_relative_error','liquid_relative_error']),
                max_galvani_absolute_error=max(abs(d['absolute_error']) for d in galvani_rows),
                source_notes='EPS curves from public arXiv 2206.02030v2 source; JCTC PDF page 16 vector paths. Axis calibrations are explicit in literature_scft.py. Digitization errors are not statistical confidence intervals.',
                model_assumption='Equal ion/solvent volume 4*pi*a^3/3 in bulk. The SI gives general volume formulas but no complete numerical parameter table. No output-dependent parameter fitting.',
                source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in
                    [args.campaign/'sources/figure_1.eps',args.campaign/'sources/figure_3a.eps',wall_pdf,
                     root/'refs/Agrawal_2022_Vapor_Liquid_Interface_in_Ionic_Fluids_SI.pdf',
                     args.campaign/'sources/figure_2a.eps']})
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['max_coexistence_relative_error','max_galvani_absolute_error','wall']},indent=2))


if __name__=='__main__':run()
