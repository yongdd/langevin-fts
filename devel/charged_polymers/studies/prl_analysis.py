"""Consolidated prl analysis workflows. Run through workflow.py."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from reports import write_analysis_note
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from studies.prl_runs import analytic_curves
import inspect
from models.prl_scft import PRLBrushSCFT

# Original workflow: analyze_prl_fig23.py



HEIGHTS=['height_moment_nm','height_equimolar_peak_nm']


COLORS={1:'#c83c32',2:'#3167bc',3:'#348649'}


def gather(root):
    records=[]
    for path in sorted(root.glob('*/summary.json'),key=lambda p:p.stat().st_mtime):
        record=json.loads(path.read_text());record['run']=path.parent.name
        records.append(record)
    return records


def checked_scan(record):
    p=record['params']
    return (record['status']=='completed' and p.get('chain_solver')=='extended'
            and p.get('pb_solver')=='rank1' and p.get('reference')=='log')


def pick(records,zp,rho,L=200,nz=1600,direction=None):
    matches=[r for r in records if checked_scan(r) and not r['params']['neutral']
             and r['params']['closure']=='lda' and r['params']['zplus']==zp
             and r['params']['L']==L and r['params']['nz']==nz
             and np.isclose(r['params']['rho_molar'],rho,rtol=1e-10,atol=0)
             and (direction is None or r['params']['direction']==direction)]
    if not matches:return None
    if direction is None:
        # Prefer the increasing-concentration branch, while recording exactly
        # which seed and direction produced every selected state.
        return max(enumerate(matches),key=lambda ir:({'up':2,'down':1}.get(ir[1]['params']['direction'],0),ir[0]))[1]
    return matches[-1]


def save_plot(fig,root,name):
    fig.savefig(root/(name+'.pdf'))
    fig.savefig(root/(name+'.png'),dpi=160)
    plt.close(fig)


def write_report(root, data):
    lines = ['## Run counts', '', '| Category | Count |', '|---|---:|']
    lines += [f'| {key} | {value} |' for key, value in data['counts'].items()]
    lines += ['', '## Validation maxima', '', '| Check | Maximum error |', '|---|---:|']
    lines += [f'| {key} | {value:.4e} |' for key, value in data['validation_maxima'].items()]
    lines += ['', '## Accepted scan points', '',
              '[Run names, concentrations and observables](fig23_primary.csv).',
              'Missing points, branch comparisons, grid/box controls and digitized-paper',
              'comparisons are retained in comparison.json.', '',
              '[Fig. 2](fig2_checked.pdf) · [Fig. 3](fig3_checked.pdf) · '
              '[Potential units check](fig3_units_check.pdf)']
    write_analysis_note(root, 'FIG23_PRL_REPRODUCTION.md',
                        'PRL Figs. 2–3: scan statistics', lines, 'comparison.json')


def analyze(root,paper):
    records=gather(root)
    byname={r['run']:r for r in records}
    grid=np.asarray(json.loads((root/'rho_grid.json').read_text()))
    zero=byname['saltfree_extended'];neutral=byname['neutral_extended']
    assert checked_scan(zero) and checked_scan(neutral)
    primary=[];missing=[];pairs=[];gridchecks=[]
    for zp in [1,2,3]:
        for rho in grid:
            point=pick(records,zp,rho)
            if point:primary.append(point)
            else:missing.append(dict(zplus=zp,rho_molar=float(rho)))
            up=pick(records,zp,rho,direction='up');down=pick(records,zp,rho,direction='down')
            if up and down:
                with np.load(root/up['run']/'profiles.npz') as a,np.load(root/down['run']/'profiles.npz') as b:
                    assert np.array_equal(a['z'],b['z'])
                    density_difference=float(np.max(abs(a['rhoP']-b['rhoP'])))
                    potential_difference=float(np.max(abs(a['psi']-b['psi'])))
                pairs.append(dict(zplus=zp,rho_molar=float(rho),up=up['run'],down=down['run'],
                    density_difference=density_difference,potential_profile_difference=potential_difference,
                    height_difference_nm=abs(up['observables'][HEIGHTS[0]]-down['observables'][HEIGHTS[0]]),
                    gamma_difference=abs(up['observables']['gamma_core_0.5']-down['observables']['gamma_core_0.5']),
                    psi_difference=abs(up['observables']['psi_wall_relative']-down['observables']['psi_wall_relative'])))
    for point in records:
        if not checked_scan(point) or not point['run'].startswith('check_'):continue
        p=point['params'];base=pick(records,p['zplus'],p['rho_molar'])
        if base is None:continue
        row=dict(run=point['run'],base=base['run'],zplus=p['zplus'],rho_molar=p['rho_molar'],L=p['L'],nz=p['nz'])
        for key in HEIGHTS+['gamma_core_0.5','psi_wall','psi_wall_relative','psi_normalized_moment','psi_relative_normalized_moment']:
            row[key]=point['observables'][key]
            row['delta_'+key]=point['observables'][key]-base['observables'][key]
        gridchecks.append(row)

    comparison2=[]
    for p in paper['fig2']:
        point=pick(records,p['zplus'],p['rho_molar'])
        if point is None:continue  # no interpolation disguised as a calculation
        for key in HEIGHTS:
            value=np.log10(point['observables'][key]/zero['observables'][key])
            comparison2.append(dict(**p,height_definition=key,run=point['run'],computed=float(value),
                                    difference=float(value-p['log10_h_over_h0'])))
    comparison3=[]
    for p in paper['fig3']:
        g,psi=analytic_curves(p['rho_molar'],p['zplus'])
        calculated=float(g if p['observable']=='gamma' else psi)
        row=dict(**p,equation_value=calculated,difference=calculated-p['value'])
        if p['observable']=='potential':
            row['equation_value_times_molar_conversion']=calculated*.6022
        comparison3.append(row)

    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained',sharey=True)
    for ax,key,title in zip(axes,HEIGHTS,[r'Moment height $h=2\langle z\rangle$',r'Equimolar slab estimate $h=\int\rho_P dz/\max\rho_P$']):
        for zp in [1,2,3]:
            for direction,style in [('up','-'),('down','--')]:
                values=[]
                for rho in grid:
                    point=pick(records,zp,rho,direction=direction)
                    values.append(np.log10(point['observables'][key]/zero['observables'][key]) if point else np.nan)
                ax.semilogx(grid,values,style,color=COLORS[zp],lw=1.6 if direction=='up' else .9,
                            marker='o' if direction=='up' else None,ms=2.5,label=fr'$z_+={zp}$' if direction=='up' else None)
            extra=[s for s in primary if s['params']['zplus']==zp
                   and s['params']['direction'] not in ['up','down']]
            ax.scatter([s['params']['rho_molar'] for s in extra],
                       [np.log10(s['observables'][key]/zero['observables'][key]) for s in extra],
                       color=COLORS[zp],marker='D',s=18)
            points=[p for p in paper['fig2'] if p['zplus']==zp]
            ax.scatter([p['rho_molar'] for p in points],[p['log10_h_over_h0'] for p in points],
                       facecolors='none',edgecolors=COLORS[zp],marker='s',s=32,lw=1)
        ax.axhline(np.log10(neutral['observables'][key]/zero['observables'][key]),color='.3',ls=':',label='Computed neutral')
        ax.set(title=title,xlabel=r'$\rho_b$ [M]',xlim=(7e-8,70),ylim=(-.72,.025))
        ax.grid(alpha=.15)
    axes[0].set_ylabel(r'$\log_{10}(h/h_0)$; $h_0$ from zero salt')
    axes[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Fig. 2: solid = increasing salt, dashed = decreasing salt; squares = published; diamonds = other seeds',fontsize=9)
    save_plot(fig,root,'fig2_checked')

    dense=np.logspace(-3,np.log10(20),1200)
    fig,axes=plt.subplots(2,3,figsize=(10,6.2),layout='constrained',sharex=True)
    for col,zp in enumerate([1,2,3]):
        g,psi=analytic_curves(dense,zp)
        axes[0,col].semilogx(dense,g,'--',color='.35',label='Eq. (7)')
        axes[1,col].semilogx(dense,psi,'--',color='.35',label='Eq. (9), M converted')
        for direction,style in [('up','-'),('down',':')]:
            rho=grid[grid>=.001]
            states=[pick(records,zp,r,direction=direction) for r in rho]
            obs=lambda key:np.array([s['observables'][key] if s else np.nan for s in states])
            axes[0,col].semilogx(rho,obs('gamma_core_0.5'),style,color='#2454aa',lw=1.3,label='SCFT core' if direction=='up' else None)
            axes[1,col].semilogx(rho,obs('psi_relative_normalized_moment'),style,color='#2454aa',lw=1.3,label=r'SCFT $(\psi_0-\psi_L)h/(\sigma Nv)$' if direction=='up' else None)
            if direction=='up':
                axes[0,col].semilogx(rho,obs('gamma_core_0.8'),color='#2454aa',ls='-.',lw=.9,label='SCFT core threshold 0.8')
                axes[1,col].semilogx(rho,obs('psi_normalized_moment'),color='#7d78a0',ls='-.',lw=.9,label=r'SCFT $\psi_0 h/(\sigma Nv)$')
        for row,kind in enumerate(['gamma','potential']):
            extra=[s for s in primary if s['params']['zplus']==zp and s['params']['rho_molar']>=.001
                   and s['params']['direction'] not in ['up','down']]
            key='gamma_core_0.5' if row==0 else 'psi_relative_normalized_moment'
            axes[row,col].scatter([s['params']['rho_molar'] for s in extra],
                                 [s['observables'][key] for s in extra],color='#2454aa',marker='D',s=18)
            pts=[p for p in paper['fig3'] if p['zplus']==zp and p['observable']==kind]
            axes[row,col].scatter([p['rho_molar'] for p in pts],[p['value'] for p in pts],marker='s',s=22,
                                  facecolors='none',edgecolors='#c25d1d',label='Published raster')
            axes[row,col].axhline(0,color='.7',lw=.6)
            axes[row,col].set(xlim=(.001,20),ylim=(-1.6,1.3) if row==0 else (-4,5))
            axes[row,col].grid(alpha=.15)
        axes[0,col].set_title(fr'$z_+={zp}$')
        axes[1,col].set_xlabel(r'$\rho_b$ [M]')
    axes[0,0].set_ylabel(r'Effective charge $\Gamma$')
    axes[1,0].set_ylabel('Normalized surface potential')
    axes[0,0].legend(fontsize=6.5,loc='lower left')
    axes[1,0].legend(fontsize=6.2,loc='upper left')
    save_plot(fig,root,'fig3_checked')

    # This diagnostic multiplier is shown explicitly; it is not applied to the
    # physical curves or silently fitted into the SCFT calculation.
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained',sharey=True)
    for ax,zp in zip(axes,[1,2,3]):
        g,psi=analytic_curves(dense,zp)
        ax.semilogx(dense,psi,color='#2454aa',label='Eq. (9), density in nm$^{-3}$')
        ax.semilogx(dense,.6022*psi,'--',color='#c25d1d',label='Eq. (9) multiplied by 0.6022')
        pts=[p for p in paper['fig3'] if p['zplus']==zp and p['observable']=='potential']
        ax.scatter([p['rho_molar'] for p in pts],[p['value'] for p in pts],color='black',s=15,label='Published raster')
        ax.set(title=fr'$z_+={zp}$',xlabel=r'$\rho_b$ [M]',xlim=(.003,20),ylim=(-4,4.5))
        ax.grid(alpha=.15);ax.axhline(0,color='.7',lw=.6)
    axes[0].set_ylabel(r'$\psi_S/(\sigma Nv/h)$');axes[0].legend(fontsize=6.3,loc='upper left')
    save_plot(fig,root,'fig3_units_check')

    minima=[]
    for zp in [1,2,3]:
        states=[s for s in primary if s['params']['zplus']==zp]
        low=min(states,key=lambda s:s['observables'][HEIGHTS[0]])
        minima.append(dict(zplus=zp,rho_molar=low['params']['rho_molar'],run=low['run'],
                           height_nm=low['observables'][HEIGHTS[0]],
                           log_height_ratio=float(np.log10(low['observables'][HEIGHTS[0]]/zero['observables'][HEIGHTS[0]]))))
    good=[r for r in records if checked_scan(r)]
    analytic_minima=[]
    fine=np.logspace(-7,np.log10(50),100001)
    for zp in [1,2,3]:
        g,p=analytic_curves(fine,zp);i=g.argmin();j=p.argmax()
        analytic_minima.append(dict(zplus=zp,gamma_min=float(g[i]),gamma_rho=float(fine[i]),
                                    potential_max=float(p[j]),potential_rho=float(fine[j])))
    summary=dict(model='Canonical monovalent C + GC z:1 salt, smoothed LDA, c1=0; not full PRL SI.',
        height_definition_warning='Moment and peak-reference equimolar slab heights are explicitly defined diagnostics; the published Gibbs-height extraction is unspecified.',
        saltfree=zero,neutral=neutral,records=records,primary=primary,missing=missing,
        up_down_pairs=pairs,grid_box_checks=gridchecks,fig2_comparison=comparison2,
        fig3_equation_comparison=comparison3,sampled_minima=minima,analytic_extrema=analytic_minima,
        counts=dict(completed=sum(r['status']=='completed' for r in records),failed=sum(r['status']=='failed' for r in records),
                    extended_checked=len(good),primary=len(primary),paired=len(pairs)),
        validation_maxima=dict(full_residual=max(max(r['residuals'].values()) for r in good),
            chain_error=max(r['independent_chain_error'] for r in good),
            contour_error=max(r['contour_partition_error'] for r in good),
            mass_error=max(abs(r['observables']['mass']-10) for r in good),
            counterion_mass_error=max(abs(r['observables']['counterion_mass']-(0 if r['params']['neutral'] else 10)) for r in good)))
    repo=Path(__file__).resolve().parents[3]
    protected=json.loads((root/'protected_source_sha256.json').read_text())
    from studies import audit_protected_sources
    audit=audit_protected_sources(repo,protected)
    summary['protected_source_audit']=audit
    summary['protected_sources_unchanged']=not audit['moved_sources'] and not audit['unverified_sources']
    if audit['unverified_sources']:raise RuntimeError(f"Unverified protected sources: {audit['unverified_sources']}")
    (root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    rows=[]
    for s in primary:
        rows.append(dict(run=s['run'],zplus=s['params']['zplus'],rho_molar=s['params']['rho_molar'],
                         direction=s['params']['direction'],**s['observables']))
    with (root/'fig23_primary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(counts=summary['counts'],missing=missing,minima=minima,
                         validation=summary['validation_maxima']),indent=2))
    write_report(root,summary)
    return summary


def clusters(mask,x,top,bottom):
    ys=np.flatnonzero(mask[top:bottom,x-3:x+4].any(axis=1))+top
    return [g for g in np.split(ys,np.flatnonzero(np.diff(ys)>2)+1) if len(g)]


def digitize_fig23(root):
    paths=[root/'paper_page2_xref54.png',root/'paper_page4_xref106.jpeg']
    rgb=np.asarray(Image.open(paths[0])).astype(float)
    r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    masks={1:(r>180)&(r-g>80)&(r-b>80),2:(b>150)&(b-r>70)&(b-g>70),
           3:(g>70)&(g-r>35)&(g-b>20)}
    # Pixel guides refer only to the visible THEORY lines. They distinguish
    # them from experimental symbols, legend strokes and annotations.
    guides={1:[86,86,87,94,115,270,575,662],
            2:[112,168,336,674,761,773,676,665],
            3:[190,508,882,985,974,881,724,666]}
    rows2=[]
    for zp in [1,2,3]:
        positions=[-6.2,-5,-4,-3,-2,-1,0,1] if zp==3 else range(-6,2)
        for log_r,hint in zip(positions,guides[zp]):
            x=round(267+(log_r+7)*(1495-267)/8)
            groups=clusters(masks[zp],x,12,1070)
            group=min(groups,key=lambda g:abs(g.mean()-hint))
            y=float(group.mean())
            if abs(y-hint)>70:raise RuntimeError((zp,log_r,y,hint))
            rows2.append(dict(zplus=zp,rho_molar=10.**log_r,log10_h_over_h0=(85-y)/1530,
                              x_pixel=x,y_pixel=y,reading_uncertainty=.01))
    rgb3=np.asarray(Image.open(paths[1])).astype(float)
    r,g,b=rgb3[:,:,0],rgb3[:,:,1],rgb3[:,:,2]
    masks3=dict(gamma=(b>130)&(b-r>50)&(g-r>20),
                potential=(r>120)&(g>40)&(g<155)&(b<110)&(r>1.3*g)&(g>1.3*b))
    anchors={1:(355,547),2:(975,1168),3:(1586,1779)}
    rows3=[];omitted=[]
    for zp in [1,2,3]:
        left,right=anchors[zp]
        for rho in [.003,.01,.03,.1,.3,1.,3.,10.]:
            x=round(left+(np.log10(rho)+1)*(right-left)/2)
            for key,mask in masks3.items():
                if key=='potential' and rho<=.01:
                    continue  # clipped/very steep parts are unsuitable for pointwise digitization
                groups=clusters(mask,x,35,695)
                if len(groups)!=1:
                    omitted.append(dict(zplus=zp,rho_molar=rho,observable=key,
                                        reason='clipped or interrupted by the horizontal reference line'))
                    continue
                y=float(groups[0].mean())
                value=(367-y)/(213.125 if key=='gamma' else 85.25)
                rows3.append(dict(zplus=zp,rho_molar=rho,observable=key,value=value,
                                  x_pixel=x,y_pixel=y,reading_uncertainty=.02 if key=='gamma' else .05))
    metadata=dict(source='Duan et al., PRL 134, 048101 (2025), published PDF, embedded rasters xref 54 and 106',
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  method='Color-separated pixel groups in +/-3 pixel windows. Fig. 2 groups selected using recorded image-only pixel guides; experimental symbols excluded.',
                  fig2_pixel_guides=guides,fig2_axes=dict(logrho_minus7=267,logrho_plus1=1495,logh_zero=85,pixels_per_logh=1530),
                  fig3_axes=dict(logrho_minus1_plus1=anchors,y_zero=367,pixels_per_gamma=213.125,pixels_per_potential=85.25),
                  limitation='Selected interior raster points; uncertainties are approximate reading ranges, not statistical errors or author data.',
                  fig2=rows2,fig3=rows3,fig3_omitted=omitted)
    (root/'paper_fig23_digitized.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for number,pixels,points in [(2,rgb,rows2),(3,rgb3,rows3)]:
        fig,ax=plt.subplots(figsize=(11,7),layout='constrained')
        ax.imshow(pixels.astype(np.uint8))
        ax.scatter([p['x_pixel'] for p in points],[p['y_pixel'] for p in points],facecolors='none',edgecolors='magenta',s=45,lw=1.2)
        ax.set_axis_off();fig.savefig(root/f'paper_fig{number}_digitization_check.png',dpi=150);plt.close(fig)
    return metadata


def analyze_scan():
    p=argparse.ArgumentParser(description='Digitize PRL Figs. 2-3 and summarize separately checked salt-scan runs.');p.add_argument('root',type=Path)
    args=p.parse_args()
    d=digitize_fig23(args.root)
    print(json.dumps(dict(fig2_points=len(d['fig2']),fig3_points=len(d['fig3'])),indent=2))
    analyze(args.root,d)


# Original workflow: compare_oned_prl.py



def digitize_fig1(root):
    path = root / "paper_fig1.jpeg"
    rgb = np.asarray(Image.open(path)).astype(float)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    masks = dict(red=(r > 170) & (r - g > 70) & (r - b > 70),
                 blue=(b > 150) & (b - r > 70) & (b - g > 70),
                 green=(g > 65) & (g - r > 35) & (g - b > 25),
                 black=np.max(rgb, axis=2) < 90)
    # Axis centers, inspected on the embedded 1612x958 image: z/b=0..60, phi=0..1.
    x0, x1, top, bottom = 213, 1579, 24, 789
    rows = []
    for color, zplus in [("black", 0), ("red", 1), ("blue", 2), ("green", 3)]:
        for z in [2, 5, 8, 10, 12]:
            x = round(x0 + (x1 - x0) * z / 60)
            ys = np.flatnonzero(masks[color][top + 4:bottom - 4, x - 4:x + 5].any(axis=1)) + top + 4
            clusters = np.split(ys, np.flatnonzero(np.diff(ys) > 2) + 1)
            values = [(bottom - group.mean()) / (bottom - top) for group in clusters if len(group)]
            assert len(values) == (1 if zplus == 0 else 2), (color, z, values)
            # In this inspected interval, solid curves are above the same-color dashed curves.
            for closure, value in zip(["mf"] if zplus == 0 else ["corr", "mf"], values):
                rows.append(dict(zplus=zplus, paper_curve=closure, z_over_b=z, phi=float(value),
                                 digitization_uncertainty_phi=.004))
    metadata = dict(source="Duan et al., PRL 134, 048101 (2025), Fig. 1, embedded image xref 53",
        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), image_size=[rgb.shape[1], rgb.shape[0]],
        axes=dict(x0=x0, x60=x1, phi1=top, phi0=bottom),
        method="Color masks, +/-4 pixel column window, separated row clusters; inspected solid/dashed ordering at z/b=2,5,8,10,12",
        limitation="Raster estimates of five interior points, not a full-profile digitization or original numerical data",
        points=rows)
    (root / "paper_fig1_digitized.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return rows


def analyze_fig1_finite():
    ap = argparse.ArgumentParser(description="Summarize the 1D positive sampler and compare PRL Fig. 1 plateau values.\n\nPRL values are digitized from the published raster, not authors' raw data.\nOnly five unambiguous interior positions are used for numerical comparison.")
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root
    points = digitize_fig1(root)
    positive = [dict(run=p.parent.name, **json.loads(p.read_text())) for p in sorted(root.glob("positive_*/summary.json"))]
    comparisons = []
    for s in positive:
        if s["mode"] != "joint":
            continue
        ref = next(t for t in positive if t["mode"] == "particles" and
                   t["sites"] == s["sites"] and t["coupling"] == s["coupling"])
        delta = np.array(s["statistics"]["mean"]) - ref["statistics"]["mean"]
        se = np.hypot(s["statistics"]["se"], ref["statistics"]["se"])
        z = np.divide(abs(delta), se, out=np.zeros_like(delta), where=se > 0)
        comparisons.append(dict(run=s["run"], reference=ref["run"],
            max_standardized_difference=float(z.max()), height_difference=float(delta[0]),
            height_combined_se=float(se[0]), n_correlated_observables=len(delta)))
    prl = []
    for p in sorted(root.glob("prl_*/summary.json")):
        s = json.loads(p.read_text())
        row = dict(run=p.parent.name, **s)
        if s["status"] == "completed":
            d = np.load(p.parent / "profiles.npz")
            curve = "mf" if s["params"]["closure"] == "mf" else "corr"
            selected = [t for t in points if t["zplus"] == s["params"]["zplus"] and t["paper_curve"] == curve]
            samples = []
            for t in selected:
                value = float(np.interp(t["z_over_b"], d["z"], d["phiP"]))
                samples.append(dict(**t, computed_phi=value, difference=value - t["phi"]))
            row.update(paper_comparison=samples,
                max_interior_phi_difference=max(abs(t["difference"]) for t in samples),
                interior_phi_rmse=float(np.sqrt(np.mean([t["difference"]**2 for t in samples]))))
        prl.append(row)
    result = dict(positive_runs=positive, sampler_comparisons=comparisons, prl_runs=prl)
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "prl_comparison.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "z_over_b", "paper_phi", "computed_phi", "difference", "raster_uncertainty"])
        for s in prl:
            for p in s.get("paper_comparison", []):
                writer.writerow([s["run"], p["z_over_b"], p["phi"], p["computed_phi"], p["difference"], .004])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {0: "#333333", 1: "#de493b", 2: "#376bc2", 3: "#31865a"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained", sharey=True)
    for ax, closure, title in zip(axes, ["mf", "lda"], ["Mean field", "Existing smoothed LDA closure"]):
        for zplus in [0, 1, 2, 3]:
            if zplus == 0:
                name = "prl_z0_mf_M3200"
            else:
                name = f"prl_z{zplus}_{closure}_M3200"
            paths = root / name / "profiles.npz"
            if not paths.exists():
                name = name.replace("M3200", "M1600")
                if zplus in [2, 3] and closure == "lda":
                    name = name.replace("lda_M1600", "lda_newton_M1600")
                paths = root / name / "profiles.npz"
            d = np.load(paths)
            ax.plot(d["z"], d["phiP"], color=colors[zplus], label="Neutral" if zplus == 0 else f"z+={zplus}")
            paper = [p for p in points if p["zplus"] == zplus and p["paper_curve"] == ("mf" if zplus == 0 or closure == "mf" else "corr")]
            ax.errorbar([p["z_over_b"] for p in paper], [p["phi"] for p in paper], yerr=.004,
                        color=colors[zplus], fmt="o", ms=4, mfc="white", capsize=2)
        ax.set(xlim=(0, 60), ylim=(0, .85), xlabel="z / b", title=title)
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Polymer volume fraction")
    axes[0].legend(fontsize=9)
    fig.suptitle("PRL Fig. 1 parameters: recomputed lines and digitized paper markers\n"
                 "Canonical counterions, L=200 nm, Gaussian graft width=0.5 b; no parameter fitting", fontsize=11)
    for suffix in ["png", "pdf"]:
        fig.savefig(root / f"prl_fig1_checked.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1), layout="constrained", sharey=True)
    for ax, valence in zip(axes, [1, 2, 3]):
        for name, label, color in [
            (f"prl_z{valence}_lda_M3200", "LDA, canonical C", "#376bc2"),
            (f"prl_z{valence}_nonlocal" + ("_continuation" if valence == 3 else "") + "_M1600",
             "Existing nonlocal, canonical C", "#b55643"),
            (f"prl_z{valence}_lda_gc_M1600", "LDA, GC salt only", "#31865a")]:
            path = root / name / "profiles.npz"
            if path.exists():
                d = np.load(path)
                ax.plot(d["z"], d["phiP"], color=color, label=label)
        selected = [p for p in points if p["zplus"] == valence and p["paper_curve"] == "corr"]
        ax.errorbar([p["z_over_b"] for p in selected], [p["phi"] for p in selected],
                    yerr=.004, color="black", fmt="o", ms=4, mfc="white", label="Paper raster")
        ax.set(xlim=(0, 50), ylim=(0, .85), xlabel="z / b", title=f"z+={valence}")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Polymer volume fraction")
    axes[0].legend(fontsize=8)
    fig.suptitle("Existing correlation closures and ensemble controls\n"
                 "L=200 nm; nonlocal backend retains documented SI/boundary approximations", fontsize=11)
    for suffix in ["png", "pdf"]:
        fig.savefig(root / f"prl_closure_comparison.{suffix}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for s in positive:
        if s["coupling"] not in [0, 100]:
            continue
        offset = {1101: -.2, 1201: 0., 2101: .2}[s["seed"]]
        color = "#b55643" if s["coupling"] else "#3269a7"
        x = [4, 16, 32].index(s["sites"]) + offset
        ax.errorbar(x, s["statistics"]["mean"][0], yerr=s["statistics"]["se"][0],
                    fmt="o" if s["mode"] == "joint" else "s", color=color, capsize=3)
    for E in [0, 100]:
        exact = next(s["exact"][0] for s in positive if s["sites"] == 4 and s["coupling"] == E)
        ax.hlines(exact, -.35, .35, color="#333333", linestyle="--")
    ax.set(xticks=[0, 1, 2], xticklabels=[4, 16, 32], xlabel="Number of cells in 1D", ylabel="Mean polymer height / L")
    ax.set_title("Same finite 9-bead action; positive sampling without enumeration\n"
                 "Circles: real-field/particle Gibbs; squares: particle Gibbs\n"
                 "Blue E=0; red E=100; dashes: exact four-cell values", fontsize=10)
    ax.grid(alpha=.15)
    for suffix in ["png", "pdf"]:
        fig.savefig(root / f"positive_1d_comparison.{suffix}", dpi=180)
    plt.close(fig)
    print(json.dumps(dict(positive_runs=len(positive), prl_completed=sum(s["status"] == "completed" for s in prl),
        max_exact_z=max(s.get("max_standardized_discrepancy", 0) for s in positive),
        max_joint_particle_z=max(s["max_standardized_difference"] for s in comparisons)), indent=2))


# Original workflow: compare_sampling_scft.py



OBS=['height_nm','gamma_core_0.5','gamma_core_0.8','psi_normalized']


STYLE={'mc':dict(color='#2454aa',label='Positive 1D MC'),
       'lda':dict(color='#c54b39',label='Smoothed-LDA SCFT'),
       'mf':dict(color='#228269',label='Mean-field SCFT')}


def checked_matched(path):
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
            record=checked_matched(Path(c[closure]));p=record['params']
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
        nominal=checked_matched(Path(c['nominal_lda']))
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


def report(root, data):
    lines = ['## Height [nm]', '',
             '| z+ | Target salt [M] | Matched salt [M] | MC ± SE | LDA SCFT | MF SCFT | MC−LDA | MC−MF |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['rows']:
        h = r['mc_mean'][0]; a = r['scft']['lda']['mean'][0]; b = r['scft']['mf']['mean'][0]
        lines.append(f"| {r['valence']} | {r['target_rho']:g} | {r['measured_rho']:.7g} | {h:.3f} ± {r['mc_se'][0]:.3f} | {a:.3f} | {b:.3f} | {h-a:+.3f} | {h-b:+.3f} |")
    lines += ['', '## Charge and potential: trivalent salt', '',
              '| Target salt [M] | MC Γ ± SE | LDA Γ | MF Γ | MC Ψ ± SE | LDA Ψ | MF Ψ |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['rows']:
        if r['valence'] != 3:
            continue
        cols = [f"{r['target_rho']:g}"]
        for idx in [1, 3]:
            cols += [f"{r['mc_mean'][idx]:.4f} ± {r['mc_se'][idx]:.4f}"]
            cols += [f"{r['scft'][k]['mean'][idx]:.4f}" for k in ['lda', 'mf']]
        lines.append('| ' + ' | '.join(cols) + ' |')
    lines += ['', '## Numerical checks', '', '| Check | Value |', '|---|---:|']
    lines += [f'| {key} | {value:.6e} |' for key, value in data['validation'].items()]
    lines += ['', '[Complete observable table](comparison.csv) · '
              '[Profiles](comparison_profiles.npz) · [Case mapping](cases.json)',
              '[Height](mc_scft_height.pdf) · [Charge/potential](mc_scft_charge.pdf) · '
              '[Density profiles](mc_scft_profiles.pdf)']
    write_analysis_note(root, 'SAMPLING_SCFT_COMPARISON.md',
                        'Sampling and concentration-matched SCFT: statistics', lines, 'comparison.json')


def analyze_matched():
    ap=argparse.ArgumentParser(description='Compare actual planar MC with concentration-matched LDA/MF SCFT.\n\nReads completed runs only. SCFT profiles are conservatively cell-averaged\nonto the MC grid; potentials are evaluated at the same cell-center positions.\nThese are different statistical models, not independent implementations of\none finite action. No simulation or production solver is changed here.');ap.add_argument('root',type=Path)
    ap.add_argument('--mc-root',type=Path);ap.add_argument('--scft-root',type=Path)
    args=ap.parse_args();mc=args.mc_root or args.root.parent/'positive_brush_20260907';scft=args.scft_root or args.root.parent/'fig23_prl_20260907'
    data,profiles=build_data(args.root,mc,scft);plot_all(args.root,scft,data,profiles);report(args.root,data)
    print(json.dumps(dict(validation=data['validation'],saltfree=data['height_reference']),indent=2))
    for r in data['rows']:
        print(r['valence'],r['target_rho'],'MC',r['mc_mean'][0],'LDA',r['scft']['lda']['mean'][0],'MF',r['scft']['mf']['mean'][0])
