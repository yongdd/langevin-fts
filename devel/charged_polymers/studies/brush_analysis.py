"""Consolidated brush analysis workflows. Run through workflow.py."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from reports import write_analysis_note
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Original workflow: analyze_positive_brush.py



def derived(density,psi,z,p,rho):
    """Observables of ensemble means, not the mean of nonlinear single states."""
    polymer=density[0];h=2*(polymer@z)/polymer.sum()
    output=[h]
    for c in [.5,.8]:
        core=polymer>c*polymer.max()
        output.append(1-p['valence']*np.sum(density[2,core]-.6022*rho)/polymer[core].sum())
    output.append(psi[0]*h/(p['grafting_density']*p['beads']*p['monomer_volume']))
    # Net physical core charge includes C and anions; Gamma alone does not.
    core=polymer>.5*polymer.max()
    output.append(np.sum((-polymer+density[1]+p['valence']*density[2]-density[3])[core])/polymer[core].sum())
    return np.array(output)


DERIVED=['height_nm','gamma_core_0.5','gamma_core_0.8','psi_normalized','net_core_charge_per_P']


def one_run(path):
    d=json.loads(path.read_text());data=np.load(path.parent/'samples.npz');p=d['params']
    counts=data['counts'];psi=data['psi'];z=data['z'];rho=d.get('measured_bulk_molar',0.)
    density=counts/(p['area']*p['dz'])
    center=derived(density.mean(axis=0),psi.mean(axis=0),z,p,rho)
    # Delete one temporal block at a time. Recompute the core mask too.
    jack_errors={}
    for nblocks in [10,20,40]:
        if len(counts)<nblocks*5:continue
        pieces=np.array_split(np.arange(len(counts)),nblocks)
        values=[]
        for indices in pieces:
            nleft=len(counts)-len(indices)
            a=(density.sum(axis=0)-density[indices].sum(axis=0))/nleft
            b=(psi.sum(axis=0)-psi[indices].sum(axis=0))/nleft
            values.append(derived(a,b,z,p,rho))
        values=np.array(values)
        jack_errors[str(nblocks)]=np.sqrt((nblocks-1)/nblocks*np.sum((values-values.mean(axis=0))**2,axis=0))
    se=np.maximum.reduce(list(jack_errors.values()))
    # Dense scalar trace gives the more precise height estimator / temporal SE.
    center[0]=d['statistics']['mean'][0];se[0]=max(se[0],d['statistics']['se'][0])
    scalar=data['trace'][:,[0,3]]
    normalization=p['grafting_density']*p['beads']*p['monomer_volume']
    center[3]=np.prod(scalar.mean(axis=0))/normalization
    scalar_errors=[]
    for nblocks in [10,20,40]:
        values=[]
        for part in np.array_split(scalar,nblocks):
            mean=(scalar.sum(axis=0)-part.sum(axis=0))/(len(scalar)-len(part))
            values.append(np.prod(mean)/normalization)
        values=np.array(values)
        scalar_errors.append(np.sqrt((nblocks-1)/nblocks*np.sum((values-values.mean())**2)))
    # Include a delta-method bound with unknown covariance, in addition to
    # block jackknife; this avoids underestimating a slow potential mode.
    scalar_errors.append((abs(scalar[:,1].mean())*d['statistics']['se'][0]
                          +abs(scalar[:,0].mean())*d['statistics']['se'][3])/normalization)
    se[3]=max(scalar_errors)
    bulk_se=d.get('measured_bulk_molar_se',0.)
    shift=np.zeros_like(center)
    if bulk_se:
        shift=derived(density.mean(axis=0),psi.mean(axis=0),z,p,rho+bulk_se)-derived(density.mean(axis=0),psi.mean(axis=0),z,p,rho)
        se=np.hypot(se,shift)
    return dict(run=path.parent.name,params=p,seed=d['seed'],rho=rho,rho_se=bulk_se,
        labels=DERIVED,mean=center.tolist(),se=se.tolist(),
        shared_bulk_error=shift.tolist(),
        jackknife_se={k:v.tolist() for k,v in jack_errors.items()},
        height_ess=d['statistics']['effective_samples'][0],height_tau_sweeps=d['statistics']['tau_physical'][0],
        split_half_sigma=d['split_half_sigma'][:4],elapsed_seconds=d['elapsed_seconds'],
        max_charge_update_error=d['max_incremental_charge_error']),data['trace'][:,0]


def split_rhat(chains):
    n=min(map(len,chains))//2
    x=np.array([c[:n] for c in chains]+[c[-n:] for c in chains])
    w=np.var(x,axis=1,ddof=1).mean();b=n*np.var(x.mean(axis=1),ddof=1)
    return float(np.sqrt(((n-1)*w/n+b/n)/w)) if w else 1.


def analyze(root):
    exact=json.loads((root/'exact.json').read_text());tiny=[]
    for path in sorted(root.glob('exact_v2_*/summary.json')):
        d=json.loads(path.read_text());e=exact['exact'][str(d['params']['valence'])]
        n=10 if d['params']['joint'] else 8
        sigma=np.abs(np.array(d['statistics']['mean'][:n])-e['mean'][:n])/d['statistics']['se'][:n]
        tiny.append(dict(run=path.parent.name,worst_sigma=float(max(sigma)),
                         mean=d['statistics']['mean'],se=d['statistics']['se']))
    bulk=[]
    for path in sorted(root.glob('bath*/summary.json')):
        d=json.loads(path.read_text())
        bulk.append(dict(run=path.parent.name,params=d['params'],rho=d['measured_bulk_molar'],
            se=d['measured_bulk_molar_se'],target=d['target_bulk_molar'],
            number_ess=d['statistics']['effective_samples'][1],split_half_sigma=d['split_half_sigma'][1]))
    runs=[];traces={};grouped=defaultdict(list)
    for pattern in ['brush_*/summary.json','check_*/summary.json']:
        for path in sorted(root.glob(pattern)):
            continuation=root/('continued_'+path.parent.name)/'summary.json'
            if continuation.exists():path=continuation
            row,t=one_run(path);runs.append(row);traces[row['run']]=t
            p=row['params']
            if pattern.startswith('brush') and p['joint']:
                grouped[(p['valence'],p['activity_molar'],p['sites'],p['area'])].append(row)
    groups=[]
    for key,rows in sorted(grouped.items()):
        means=np.array([r['mean'] for r in rows]);ses=np.array([r['se'] for r in rows]);n=len(rows)
        shared=np.array([r['shared_bulk_error'] for r in rows])
        within=np.sqrt(np.sum(np.maximum(0.,ses**2-shared**2),axis=0))/n
        between=np.std(means,axis=0,ddof=1)/np.sqrt(n) if n>1 else within
        denom=np.hypot(ses,ses[0])
        pair_sigma=np.max(np.divide(abs(means-means[0]),denom,out=np.zeros_like(means),where=denom>0),axis=0)
        groups.append(dict(valence=key[0],target_rho=key[1],sites=key[2],area=key[3],
            rho=rows[0]['rho'],rho_se=rows[0]['rho_se'],runs=[r['run'] for r in rows],
            mean=means.mean(axis=0).tolist(),se=np.hypot(np.maximum(within,between),shared.mean(axis=0)).tolist(),
            pair_sigma=pair_sigma.tolist(),height_split_rhat=split_rhat([traces[r['run']] for r in rows]),
            min_height_ess=min(r['height_ess'] for r in rows)))
    extension_complete=all((root/f'continued_brush_z2_r0.01_joint_s{s}/summary.json').exists() for s in [4101,4201])
    result=dict(definitions=DERIVED,deterministic=exact['deterministic'],tiny=tiny,bulk=bulk,runs=runs,groups=groups,
        extensions_complete=extension_complete,
        status='finite planar model; stochastic errors do not establish grid/area/3D PRL convergence')
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def plots(root,data,prl_root):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    groups=data['groups'];zero=next((g for g in groups if g['target_rho']==0),None)
    if not zero:return
    colors={1:'#2454aa',2:'#dd8a12',3:'#ba3869'}
    ref=json.loads((prl_root/'paper_fig23_digitized.json').read_text()) if prl_root and (prl_root/'paper_fig23_digitized.json').exists() else None
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for z,ax in zip([1,2,3],axes):
        rows=[g for g in groups if g['valence']==z and g['target_rho']>0]
        x=np.array([r['rho'] for r in rows]);h=np.array([r['mean'][0] for r in rows]);se=np.array([r['se'][0] for r in rows])
        y=np.log10(h/zero['mean'][0]);err=np.hypot(se/h,zero['se'][0]/zero['mean'][0])/np.log(10)
        ax.errorbar(x,y,yerr=err,xerr=[r['rho_se'] for r in rows],fmt='o-',color=colors[z],capsize=3,label='1D positive MC (1 SE)')
        if ref:
            pts=[r for r in ref['fig2'] if r['zplus']==z and .0007<=r['rho_molar']<=15]
            if pts:ax.plot([r['rho_molar'] for r in pts],[r['log10_h_over_h0'] for r in pts],'s',mfc='none',color='.4',label='PRL theory curve')
        ax.set(xscale='log',xlabel=r'Measured salt bath $\rho_b$ [M]',title=f'$z_+={z}$')
        ax.axhline(0,color='.8',lw=.8)
    axes[0].set_ylabel(r'$\log_{10}(h/h_0)$');axes[0].legend(fontsize=8)
    fig.suptitle('Actual stochastic samples — finite planar model, N=100, A=40 nm²')
    fig.savefig(root/'sampling_fig2.pdf');fig.savefig(root/'sampling_fig2.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6.1),layout='constrained')
    for col,z in enumerate([1,2,3]):
        rows=[g for g in groups if g['valence']==z and g['target_rho']>0]
        for row,idx in [(0,1),(1,3)]:
            ax=axes[row,col]
            ax.errorbar([r['rho'] for r in rows],[r['mean'][idx] for r in rows],
                yerr=[r['se'][idx] for r in rows],fmt='o-',color=colors[z],capsize=3)
            if row==0:ax.plot([r['rho'] for r in rows],[r['mean'][2] for r in rows],'--',color=colors[z],alpha=.6,label='core threshold 0.8')
            if ref:
                kind='gamma' if row==0 else 'potential'
                pts=[r for r in ref['fig3'] if r['zplus']==z and r['observable']==kind and .0007<=r['rho_molar']<=15]
                if pts:ax.plot([r['rho_molar'] for r in pts],[r['value'] for r in pts],'s',mfc='none',color='.4',label='PRL theory curve')
            ax.axhline(0,color='.5',lw=.7);ax.set_xscale('log')
        axes[0,col].set_title(f'$z_+={z}$');axes[1,col].set_xlabel(r'Measured $\rho_b$ [M]')
    axes[0,0].set_ylabel(r'Effective-charge estimator $\Gamma_{0.5}$');axes[0,0].legend(fontsize=8)
    axes[1,0].set_ylabel(r'$(\langle\psi_0\rangle-\langle\psi_L\rangle)h/(\sigma Nv)$')
    fig.suptitle('Fig. 3 observables measured from the sampled ensemble; error bars = 1 SE')
    fig.savefig(root/'sampling_fig3.pdf');fig.savefig(root/'sampling_fig3.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6.3),layout='constrained')
    for rho,col in [(0,0),(.01,1)]:
        z=1 if rho==0 else 3
        for r in data['runs']:
            p=r['params']
            if p['valence']!=z or p['activity_molar']!=rho or p['sites']!=400 or p['area']!=40:continue
            a=np.load(root/r['run']/'samples.npz');t=a['trace'][:,0]
            stride=100;blocks=np.array_split(t,100)
            steps=np.cumsum([len(b) for b in blocks])*stride
            axes[0,col].plot(steps,[b.mean() for b in blocks],lw=.8,label=f"{('joint' if p['joint'] else 'particles')}, {r['seed']}")
            prof=a['counts'][:,0].mean(axis=0)/(p['area']*p['dz'])
            axes[1,col].plot(a['z'],prof,lw=1,label=str(r['seed']))
        axes[0,col].set(title=f'z+={z}, target salt={rho:g} M',xlabel='Production sweeps',ylabel='Block mean height [nm]')
        axes[0,col].legend(fontsize=7);axes[1,col].set(xlim=(0,90),xlabel='z [nm]',ylabel='Mean polymer density [nm⁻³]')
    fig.suptitle('Independent starts and sampled density profiles')
    fig.savefig(root/'sampling_diagnostics.pdf');fig.savefig(root/'sampling_diagnostics.png',dpi=160);plt.close(fig)


def write_report(root, data):
    lines = ['## Run counts', '', '| Category | Count |', '|---|---:|']
    lines += [f'| {key} | {len(data[key])} |' for key in ['tiny', 'bulk', 'runs', 'groups']]
    lines += ['', f"Recorded extension intervals complete: {data['extensions_complete']}.", '',
              '## Sample averages', '',
              '| z+ | Target salt [M] | Measured bulk [M] | h [nm] | Γ0.5 | Ψ |',
              '|---:|---:|---:|---:|---:|---:|']
    for g in data['groups']:
        values = [f"{g['mean'][i]:.4f} ± {g['se'][i]:.4f}" for i in [0, 1, 3]]
        lines.append(f"| {g['valence']} | {g['target_rho']:g} | {g['rho']:.6g} ± {g['rho_se']:.2g} | " + ' | '.join(values) + ' |')
    lines += ['', '## Sampling diagnostics', '',
              '| z+ | Target salt [M] | Height split Rhat | Minimum height ESS |',
              '|---:|---:|---:|---:|']
    for g in data['groups']:
        lines.append(f"| {g['valence']} | {g['target_rho']:g} | {g['height_split_rhat']:.5f} | {g['min_height_ess']:.2f} |")
    lines += ['', '## Grid/cross-sectional-area checks', '',
              '| Run | h [nm] | Height ESS |', '|---|---:|---:|']
    for r in data['runs']:
        if r['run'].startswith('check_'):
            lines.append(f"| {r['run']} | {r['mean'][0]:.4f} ± {r['se'][0]:.4f} | {r['height_ess']:.1f} |")
    lines += ['', 'Per-run settings, detailed-balance checks, exact-sum comparisons and',
              'charge-conservation errors are retained in analysis.json.',
              '[Height](sampling_fig2.pdf) · [Charge/potential](sampling_fig3.pdf) · '
              '[Diagnostics](sampling_diagnostics.pdf)']
    write_analysis_note(root, 'POSITIVE_BRUSH_RESULTS.md',
                        'Planar brush sampling: statistics', lines, 'analysis.json')


def run_analysis():
    ap=argparse.ArgumentParser(description='Analyze the positive sampling campaign; never label SCFT as samples.');ap.add_argument('root',type=Path)
    ap.add_argument('--prl-root',type=Path);args=ap.parse_args()
    data=analyze(args.root);plots(args.root,data,args.prl_root)
    print(json.dumps(dict(tiny=len(data['tiny']),bulk=len(data['bulk']),runs=len(data['runs']),groups=len(data['groups']),
        worst_tiny_sigma=max([r['worst_sigma'] for r in data['tiny']],default=None)),indent=2))
    for g in data['groups']:print(g['valence'],g['target_rho'],'h',g['mean'][0],'+/-',g['se'][0],
                                'Rhat',g['height_split_rhat'],'ESS',g['min_height_ess'])
    write_report(args.root, data)
