"""Compare actual planar MC with concentration-matched LDA/MF SCFT.

Reads completed runs only. SCFT profiles are conservatively cell-averaged
onto the MC grid; potentials are evaluated at the same cell-center positions.
These are different statistical models, not independent implementations of
one finite action. No simulation or production solver is changed here.
"""
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prl_scft import PRLBrushSCFT

OBS=['height_nm','gamma_core_0.5','gamma_core_0.8','psi_normalized']
STYLE={'mc':dict(color='#2454aa',label='Positive 1D MC'),
       'lda':dict(color='#c54b39',label='Smoothed-LDA SCFT'),
       'mf':dict(color='#228269',label='Mean-field SCFT')}


def checked(path):
    d=json.loads(path.read_text());p=d['params']
    assert d['status']=='completed',path
    assert p['chain_solver']=='extended' and p['pb_solver']=='rank1' and p['reference']=='log'
    assert p['L']==200 and p['nz']==1600 and not p['neutral']
    assert max(d['residuals'].values())<1e-7
    assert d['independent_chain_error']<1e-10 and d['contour_partition_error']<1e-10
    assert abs(d['observables']['mass']-10)<1e-10 and abs(d['observables']['counterion_mass']-10)<1e-10
    d['source']=str(path)
    return d


def mc_profile(root,group):
    means=[];ses=[]
    for run in group['runs']:
        d=np.load(root/run/'samples.npz');values=d['counts']/(40*.5)
        errors=[]
        for k in [10,20,40]:
            blocks=np.array([x.mean(axis=0) for x in np.array_split(values,k)])
            errors.append(blocks.std(axis=0,ddof=1)/np.sqrt(k))
        means.append(values.mean(axis=0));ses.append(np.maximum.reduce(errors));z=d['z']
    means=np.array(means);ses=np.array(ses)
    mean=means.mean(axis=0)
    within=np.sqrt(np.sum(ses**2,axis=0))/len(ses)
    between=means.std(axis=0,ddof=1)/np.sqrt(len(means))
    assert abs(mean[0].sum()*.5-10)<1e-10
    return z,mean,np.maximum(within,between)


def scft_profile(record,z,rho):
    d=np.load(Path(record['source']).parent/'profiles.npz')
    dz=np.diff(d['z'])[0];coarse=np.diff(z)[0];ratio=int(round(coarse/dz))
    assert ratio==4 and len(d['z'])==ratio*len(z)
    assert np.allclose(d['z'].reshape(-1,ratio).mean(axis=1),z,rtol=0,atol=1e-12)
    fine=np.array([d[k] for k in ['rhoP','rhoC','rhop','rhom']])
    density=fine.reshape(4,len(z),ratio).mean(axis=2)
    error=float(np.max(abs(density.sum(axis=1)*coarse-fine.sum(axis=1)*dz)))
    assert error<1e-8
    psi=np.interp(z,d['z'],d['psi']);psi-=psi[-1]
    h=record['observables']['height_moment_nm']
    values=[h]
    for threshold in [.5,.8]:
        core=density[0]>threshold*density[0].max()
        values.append(1-record['params']['zplus']*np.sum(density[2,core]-.6022*rho)/density[0,core].sum())
    values.append(psi[0]*h/10)
    return density,psi,np.array(values),error


def build_data(root,mc_root,scft_root):
    mc=json.loads((mc_root/'analysis.json').read_text())
    campaign=json.loads((root/'cases.json').read_text())
    old=json.loads((scft_root/'comparison.json').read_text())
    signature=inspect.signature(PRLBrushSCFT.__init__).parameters
    correspondence={'N':'beads','b':'segment_length','v':'monomer_volume','sigma':'grafting_density',
                    'chi':'chi','lB':'bjerrum_length','aP':'born_radius','aion':'born_radius'}
    for run in mc['runs']:
        if run['params']['area']!=40 or run['params']['sites']!=400:continue
        for key,other in correspondence.items():assert signature[key].default==run['params'][other]
    rows=[];profiles={};all_scft=[]
    for c in campaign:
        group=next(g for g in mc['groups'] if g['valence']==c['valence'] and g['target_rho']==c['target_rho'])
        assert c['mc_runs']==group['runs'] and c['measured_rho']==group['rho']
        z,mean,se=mc_profile(mc_root,group)
        key=f"z{c['valence']}_r{c['target_rho']:g}"
        profiles[key+'_z']=z;profiles[key+'_mc']=mean;profiles[key+'_mc_se']=se
        row=dict(**c,mc_mean=group['mean'][:4],mc_se=group['se'][:4],
                 mc_height_rhat=group['height_split_rhat'],scft={})
        for closure in ['lda','mf']:
            record=checked(Path(c[closure]));p=record['params']
            assert p['rho_molar']==c['measured_rho'] and p['closure']==closure and p['zplus']==c['valence']
            density,psi,values,mass_error=scft_profile(record,z,c['measured_rho'])
            original=record['observables']
            native=[original['height_moment_nm'],original['gamma_core_0.5'],original['gamma_core_0.8'],original['psi_relative_normalized_moment']]
            row['scft'][closure]=dict(mean=values.tolist(),native_grid_mean=native,
                height_mc_minus_scft=float(group['mean'][0]-values[0]),
                height_relative_difference_percent=float(100*(group['mean'][0]/values[0]-1)),
                polymer_profile_relative_L1=float(np.sum(abs(mean[0]-density[0]))*.5/10),
                polymer_profile_max_difference=float(np.max(abs(mean[0]-density[0]))),
                rebin_mass_error=mass_error,residuals=record['residuals'],
                independent_chain_error=record['independent_chain_error'],contour_partition_error=record['contour_partition_error'])
            profiles[key+'_'+closure]=density;profiles[key+'_'+closure+'_psi']=psi
            all_scft.append(record)
        nominal=checked(Path(c['nominal_lda']))
        row['lda_height_bath_matching_shift']=row['scft']['lda']['mean'][0]-nominal['observables']['height_moment_nm']
        rows.append(row)
    assert len(rows)==16
    zero=next(r for r in rows if r['target_rho']==0)
    data=dict(labels=OBS,rows=rows,
        height_reference={k:zero['mc_mean'][0] if k=='mc' else zero['scft'][k]['mean'][0] for k in STYLE},
        height_reference_mc_se=zero['mc_se'][0],
        validation=dict(max_scft_residual=max(max(r['residuals'].values()) for r in all_scft),
            max_independent_chain_error=max(r['independent_chain_error'] for r in all_scft),
            max_contour_partition_error=max(r['contour_partition_error'] for r in all_scft),
            max_bath_matching_height_shift=max(abs(r['lda_height_bath_matching_shift']) for r in rows),
            max_rebin_mass_error=max(r['scft'][k]['rebin_mass_error'] for r in rows for k in ['lda','mf'])),
        primary_lda_curve=[dict(z=r['params']['zplus'],rho=r['params']['rho_molar'],
                              height=r['observables']['height_moment_nm']) for r in old['primary']],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [mc_root/'analysis.json',scft_root/'comparison.json',root/'cases.json']},
        interpretation='Concentration-matched comparison of different models, not an exact-action MC versus SCFT identity test.')
    np.savez_compressed(root/'comparison_profiles.npz',**profiles)
    (root/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    flat=[]
    for r in rows:
        row={k:r[k] for k in ['valence','target_rho','measured_rho','measured_rho_se']}
        for i,obs in enumerate(OBS):
            row['mc_'+obs]=r['mc_mean'][i];row['mc_'+obs+'_se']=r['mc_se'][i]
            for k in ['lda','mf']:row[k+'_'+obs]=r['scft'][k]['mean'][i]
        flat.append(row)
    with (root/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    return data,profiles


def save(fig,root,name):
    fig.savefig(root/(name+'.pdf'));fig.savefig(root/(name+'.png'),dpi=170);plt.close(fig)


def plot_all(root,scft_root,data,profiles):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    paper=json.loads((scft_root/'paper_fig23_digitized.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(12,6.3),layout='constrained')
    for col,z in enumerate([1,2,3]):
        rows=[r for r in data['rows'] if r['valence']==z and r['target_rho']>0]
        x=np.array([r['measured_rho'] for r in rows]);h=np.array([r['mc_mean'][0] for r in rows]);se=np.array([r['mc_se'][0] for r in rows])
        for index in [0,1]:
            ax=axes[index,col]
            y=h if index==0 else np.log10(h/data['height_reference']['mc'])
            err=se if index==0 else np.hypot(se/h,data['height_reference_mc_se']/data['height_reference']['mc'])/np.log(10)
            ax.errorbar(x,y,yerr=err,xerr=[r['measured_rho_se'] for r in rows],fmt='o-',capsize=3,**STYLE['mc'])
            vals=np.array([r['scft']['mf']['mean'][0] for r in rows])
            ax.plot(x,vals if index==0 else np.log10(vals/data['height_reference']['mf']),'--',**STYLE['mf'])
            curve=sorted([r for r in data['primary_lda_curve'] if r['z']==z and .0003<=r['rho']<=32],key=lambda r:r['rho'])
            heights=np.array([r['height'] for r in curve])
            ax.plot([r['rho'] for r in curve],heights if index==0 else np.log10(heights/data['height_reference']['lda']),'-',**STYLE['lda'])
            vals=np.array([r['scft']['lda']['mean'][0] for r in rows])
            ax.plot(x,vals if index==0 else np.log10(vals/data['height_reference']['lda']),'.',color=STYLE['lda']['color'])
            ax.set(xscale='log',xlim=(.0007,14))
        pts=[p for p in paper['fig2'] if p['zplus']==z and .0007<=p['rho_molar']<=14]
        axes[1,col].plot([p['rho_molar'] for p in pts],[p['log10_h_over_h0'] for p in pts],'s',mfc='none',color='.4',label='PRL published theory')
        axes[0,col].set_title(f'$z_+={z}$');axes[1,col].set_xlabel(r'Bulk salt $\rho_b$ [M]')
    axes[0,0].set_ylabel('Height h [nm]');axes[1,0].set_ylabel(r"$\log_{10}(h/h_0)$ (each model's $h_0$)")
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
    fig.suptitle('1D MC vs concentration-matched SCFT — error bars: MC 1 SE')
    save(fig,root,'mc_scft_height')
    fig,axes=plt.subplots(2,3,figsize=(12,6.2),layout='constrained')
    for col,z in enumerate([1,2,3]):
        rows=[r for r in data['rows'] if r['valence']==z and r['target_rho']>0]
        x=[r['measured_rho'] for r in rows]
        for j,idx in enumerate([1,3]):
            ax=axes[j,col]
            ax.errorbar(x,[r['mc_mean'][idx] for r in rows],yerr=[r['mc_se'][idx] for r in rows],fmt='o-',capsize=3,**STYLE['mc'])
            for k,ls in [('lda','-'),('mf','--')]:
                ax.plot(x,[r['scft'][k]['mean'][idx] for r in rows],ls,**STYLE[k])
            kind='gamma' if j==0 else 'potential'
            pts=[p for p in paper['fig3'] if p['zplus']==z and p['observable']==kind]
            ax.plot([p['rho_molar'] for p in pts],[p['value'] for p in pts],'s',mfc='none',color='.5',label='PRL published theory')
            ax.axhline(0,color='.7',lw=.8);ax.set(xscale='log',xlim=(.0007,14))
        axes[0,col].set_title(f'$z_+={z}$');axes[1,col].set_xlabel(r'Bulk salt $\rho_b$ [M]')
    axes[0,0].set_ylabel(r'Effective charge $\Gamma_{0.5}$');axes[1,0].set_ylabel(r'$(\psi_0-\psi_L)h/(\sigma Nv)$')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Same core definition / same potential endpoints; different physical models')
    save(fig,root,'mc_scft_charge')
    fig,axes=plt.subplots(2,3,figsize=(12,6.2),layout='constrained')
    for col,z in enumerate([1,2,3]):
        key=f'z{z}_r0.01';x=profiles[key+'_z']
        for row,idx in [(0,0),(1,2)]:
            ax=axes[row,col]
            y=profiles[key+'_mc'][idx];se=profiles[key+'_mc_se'][idx]
            ax.plot(x,y,**STYLE['mc']);ax.fill_between(x,y-se,y+se,color=STYLE['mc']['color'],alpha=.18)
            for k,ls in [('lda','-'),('mf','--')]:ax.plot(x,profiles[key+'_'+k][idx],ls,**STYLE[k])
            ax.set(xlim=(0,90),xlabel='z [nm]')
        axes[0,col].set_title(f'$z_+={z}$, bath ≈ 0.01 M')
    axes[0,0].set_ylabel(r'Polymer $\rho_P$ [nm$^{-3}$]');axes[1,0].set_ylabel(r'Salt cation $\rho_+$ [nm$^{-3}$]')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Profiles: conservative SCFT cell averages on the MC grid; MC bands = 1 SE')
    save(fig,root,'mc_scft_profiles')


def report(root,data):
    lines=['# Direct comparison of 1D sampling and PRL-type SCFT','',
        'The samples are closer to mean-field SCFT with correlations disabled than to smoothed-LDA SCFT with correlations.',
        'This is not a comparison of sampling methods for the same action.','',
        '## Comparison method','',
        '- MC: the completed finite planar sampler with N=100, A=40 nm², L=200 nm, M=400.',
        '- LDA: the existing smoothed-LDA SCFT using PRL parameters. Equivalence to the full nonlocal theory in the paper has not been validated.',
        '- MF: the same SCFT with correlation self-energies disabled (uK=0), retaining the mean Coulomb potential and canonical C.',
        '- SCFT was recomputed at the measured MC bulk concentrations for 15 nonzero salt points. The MF salt-free reference was solved separately.',
        '- Retained parameters: N=100, b=1 nm, σ=.1 nm⁻², v=1 nm³, χ=.5, ℓB=.7 nm, a=.25 nm, and canonical C with Nσ=10 nm⁻².',
        '- Concentration means bulk cation molarity, distinct from ionic strength. 1 M = .6022 nm⁻³.',
        '- Height is h=2∫zρP dz/(σN). Normalized curves use each model\'s independently computed h0, without matching to a shared h0.',
        '- SCFT M=1600 densities were averaged over four cells onto MC M=400. Γ uses the same .5 peak core definition.',
        '- SCFT potentials were interpolated to z=.25/199.75 nm and compared as differences at the same locations as MC, not as absolute gauges.',
        '- MC error bars are 1 SE and exclude model bias and SCFT approximation errors.','',
        '## Height [nm]','',
        '| z+ | Target ρb [M] | Matched ρb [M] | MC ± SE | LDA SCFT | MF SCFT | MC−LDA | MC−MF |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['rows']:
        h=r['mc_mean'][0];a=r['scft']['lda']['mean'][0];b=r['scft']['mf']['mean'][0]
        lines.append(f"| {r['valence']} | {r['target_rho']:g} | {r['measured_rho']:.7g} | {h:.3f} ± {r['mc_se'][0]:.3f} | {a:.3f} | {b:.3f} | {h-a:+.3f} | {h-b:+.3f} |")
    lines+=['','## Charge and potential: trivalent salt','',
        '| Target ρb [M] | MC Γ ± SE | LDA Γ | MF Γ | MC Ψ ± SE | LDA Ψ | MF Ψ |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['rows']:
        if r['valence']!=3:continue
        cols=[f"{r['target_rho']:g}"]
        for idx in [1,3]:cols += [f"{r['mc_mean'][idx]:.4f} ± {r['mc_se'][idx]:.4f}"]+[f"{r['scft'][k]['mean'][idx]:.4f}" for k in ['lda','mf']]
        lines.append('| '+' | '.join(cols)+' |')
    vals=data['validation']
    lines+=['','## Interpretation','',
        'The large multivalent contraction and re-expansion in LDA-SCFT are absent from MC. MC heights and densities are much closer to MF.',
        'MC and MF nevertheless differ: finite A, solvent entropy from integer cell occupancy, grid, charge smearing, and actual 1D fluctuations all matter.',
        'A 1D mean density ρ(z) differs from restricting charge fluctuations to 1D. The PRL correlation theory includes a 3D Green function.',
        'Current MC samples a planar Coulomb action retaining only transverse wave number q_parallel=0. More iterations cannot restore excluded transverse correlation modes.',
        'These results motivate investigating model differences first; this comparison does not establish an MC algorithm error or failure of the original CL.',
        'The paper neglects uP^ex but retains the individual-charge term uP^0. That omission alone is not an established cause of the discrepancy.',
        '[Paper and Eqs. 3–5](https://arxiv.org/html/2404.09103v3), [saved SI](../../references/2025_prl_duan_si.pdf).',
        'PRL squares in the plots are digitized theory curves. The paper uses a Gibbs dividing-surface height; our height is a first moment.','',
        '## Numerical checks','',
        f"- Maximum full SCFT residual {vals['max_scft_residual']:.3e}, independent log-chain density error {vals['max_independent_chain_error']:.3e}, contour Q error {vals['max_contour_partition_error']:.3e}.",
        f"- Matching nominal concentrations to the actual MC bath changes LDA height by at most {vals['max_bath_matching_height_shift']:.6f} nm.",
        f"- Maximum profile rebinning integral error {vals['max_rebin_mass_error']:.3e}. Polymer/C integrals were each checked against 10 nm⁻².",
        '- The failed initial MF salt-free attempt is preserved. A retry from the existing smooth LDA solution converged without changing residual thresholds or equations.','',
        f'Raw data: `{root}`. `cases.json` records every MC–SCFT mapping.',
        '`comparison.json`, `comparison.csv`, and `comparison_profiles.npz` contain numbers and full profiles. `jobs.jsonl` and each `run.sh` preserve execution details.',
        '[Height plot](prl_comparison_figs/sampling_scft_20260907/mc_scft_height.pdf) · [Charge/potential plot](prl_comparison_figs/sampling_scft_20260907/mc_scft_charge.pdf) · [Density profiles](prl_comparison_figs/sampling_scft_20260907/mc_scft_profiles.pdf)','']
    Path(__file__).with_name('SAMPLING_SCFT_COMPARISON.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path)
    ap.add_argument('--mc-root',type=Path);ap.add_argument('--scft-root',type=Path)
    args=ap.parse_args();mc=args.mc_root or args.root.parent/'positive_brush_20260907';scft=args.scft_root or args.root.parent/'fig23_prl_20260907'
    data,profiles=build_data(args.root,mc,scft);plot_all(args.root,scft,data,profiles);report(args.root,data)
    print(json.dumps(dict(validation=data['validation'],saltfree=data['height_reference']),indent=2))
    for r in data['rows']:
        print(r['valence'],r['target_rho'],'MC',r['mc_mean'][0],'LDA',r['scft']['lda']['mean'][0],'MF',r['scft']['mf']['mean'][0])


if __name__=='__main__':main()
