"""Digitize PRL Figs. 2-3 and summarize separately checked salt-scan runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from run_prl_fig23_checked import analytic_curves

HEIGHTS=['height_moment_nm','height_equimolar_peak_nm']
COLORS={1:'#c83c32',2:'#3167bc',3:'#348649'}


def gather(root):
    records=[]
    for path in sorted(root.glob('*/summary.json'),key=lambda p:p.stat().st_mtime):
        record=json.loads(path.read_text());record['run']=path.parent.name
        records.append(record)
    return records


def checked(record):
    p=record['params']
    return (record['status']=='completed' and p.get('chain_solver')=='extended'
            and p.get('pb_solver')=='rank1' and p.get('reference')=='log')


def pick(records,zp,rho,L=200,nz=1600,direction=None):
    matches=[r for r in records if checked(r) and not r['params']['neutral']
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


def write_report(root,data):
    counts=data['counts'];valid=data['validation_maxima']
    text=f'''# PRL Figs. 2–3 recalculation at fixed parameters

2026-09-07. **Contraction and re-expansion occur, but quantitative reproduction remains incomplete.**
These results check smoothed-LDA SCFT, not the existing CL dynamics or the paper's full SI implementation.
See the later Fig. 2–3 SCFT section of [ONED_PRL_REPRODUCTION.pdf](ONED_PRL_REPRODUCTION.pdf);
the editable LaTeX supplement is [FIG23_PRL_REPRODUCTION.tex](FIG23_PRL_REPRODUCTION.tex).

## Model and scope

- N=100, b=1 nm, v=1 nm³, σ=0.1 nm⁻², χ=0.5, ℓB=0.7 nm, all Born radii=0.25 nm.
- Polymer charge −1, canonical monovalent C totaling 10 nm⁻², GC z:1 salt, point ions.
- **c1=0 fixed**. No valence-dependent background salt, reference height, or correlation coefficient was fitted.
- Standard grid L=200 nm, Nz=1600. Wall-centered half-Gaussian graft of width 0.5b,
  Gaussian bonds, hard bottom wall and reflecting top. This is not assumed identical to the paper's delta graft.
- For each z+=1,2,3, 19 standard points span rho_b=10⁻⁷…50 M. Validated solutions exist at **{counts['primary']}/57 points**.
  Every reported point was calculated; interpolation across failures was not used to fill results.
- Upward/downward continuation, smaller concentration steps near difficult points, and grid/box controls
  used **one SLURM job per parameter value**.
- The exact correspondence between current LDA and the paper's nonlocal Green function/I_ex treatment is unverified.
  The paper also neglects u_P^ex in brush calculations; that omission alone cannot explain the discrepancy.

## Fig. 2

Two heights were computed explicitly.

$$h_m=2\\int z\\rho_P(z)dz/(\\sigma N),\\qquad
h_{{eq}}=\\sigma N/\\max_z\\rho_P(z).$$

The second is an equimolar slab estimate using max rho_P as the reference density.
The paper/SI does not specify the Gibbs dividing-surface extraction procedure, so neither is assumed equal to its height.
Each h0 was obtained by solving **exactly rho_b=0**. The moment h0 is
{data['saltfree']['observables']['height_moment_nm']:.8f} nm,
and the equimolar h0 is {data['saltfree']['observables']['height_equimolar_peak_nm']:.8f} nm.
A rho_b=10⁻¹⁰ M control gives a moment height of about 50.910706 nm, consistent with the salt-free limit.

Minimum moment heights among the 19 standard points:

| z+ | rho_b [M] | h_m [nm] | log10(h_m/h0) |
|---|---:|---:|---:|
'''
    for p in data['sampled_minima']:
        text+=f"| {p['zplus']} | {p['rho_molar']:.6g} | {p['height_nm']:.8f} | {p['log_height_ratio']:.6f} |\n"
    text+='''
These are sampled minima, not exact minima over continuous concentration. The monovalent minimum is at the scan limit.
Multivalent contraction and re-expansion occur, but contraction starts at substantially different low-salt concentrations than in the paper.
Neither height definition yields simultaneous quantitative agreement for all three valences.

Twenty-four points were digitized from the paper's theory lines in the original raster, excluding experimental symbols.
Coordinates, color masks, line-selection guides, and image hashes are saved in `paper_fig23_digitized.json`.
The overlay in `paper_fig2_digitization_check.png` was checked visually against the original.
Digitization uncertainty is about ±0.01 in log10(h/h0), not a statistical error. Maximum differences at exactly matching concentrations:

| z+ | Maximum moment-height difference | Maximum equimolar-height difference |
|---|---:|---:|
'''
    for z in [1,2,3]:
        values=[max(abs(p['difference']) for p in data['fig2_comparison'] if p['zplus']==z and p['height_definition']==key) for key in HEIGHTS]
        text+=f'| {z} | {values[0]:.6f} | {values[1]:.6f} |\n'
    text+='''
## Fig. 3: analytical formulas, SCFT, and units

Differentiating the SI function u(x)=1−x erfcx(x/√π) gives Γ from Eq. (7) and potential from Eq. (9) directly.
The calculation uses κ²=4πℓB z+(z++1)rho_b and rho_b[nm⁻³]=0.6022 rho_b[M].
These analytical formulas are independent of SCFT. The SI S34 assumptions of rho_b≫rho_P and small density changes
do not automatically hold in low-salt SCFT.

In SCFT, Γ(z)=1−z+(rho_+(z)−rho_b)/rho_P(z) was averaged with rho_P weights over rho_P>c max(rho_P),
using c=0.5 and 0.8. The canonical C contribution in the core was saved separately.
Both psi(0) and psi(0)−psi(L) were recorded and normalized by h_m/(σNv).
Wall/far-end values use first/last cell centers (reference grid z=0.0625, 199.9375 nm).
Since psi(L)≠0 in the finite canonical box, the sign of psi(0) alone cannot establish charge inversion.

Among the 19 standard points, the divalent core Γ minimum is about −0.0140, but the relative potential remains negative.
The trivalent core Γ minimum is about −0.5314; the maximum normalized relative potential is about +0.7834.
This shows the qualitative separation of divalent overcharging from potential reversal,
and trivalent potential reversal. The trivalent potential peak nevertheless differs from the paper's value of about 2.5.
These are values at standard concentration points, not extrema over a continuous function.

**The monovalent potential in the paper's Fig. 3 differs from unit-consistent Eq. (9) by a factor of about 0.6022.**
For z+=1, the normalized potential is −1/(2v rho_b), independent of Γ, a, and ℓB.

| rho_b [M] | Eq. (9), M→nm⁻³ applied | Paper (approx.) | Eq. (9)×0.6022 |
|---|---:|---:|---:|
| 0.3 | −2.7676 | −1.6598 | −1.6667 |
| 1 | −0.8303 | −0.4985 | −0.5000 |
| 3 | −0.2768 | −0.1701 | −0.1667 |

The factor is consistent with the M→nm⁻³ conversion, but its cause is unconfirmed without the authors' raw data/plotting code.
Additional multivalent discrepancies remain. **Calculated curves were not rescaled to fit the paper.**
Only a separate diagnostic curve with this factor explicitly labeled was added to `fig3_units_check.pdf`.
The printed SI S42 lacks the rho_b denominator visible in main-text Eq. (9), so potential calculations use the main-text equation.

Forty Fig. 3 points were digitized. Uncertainties are about ±0.02 for Γ and ±0.05 for potential, not statistical errors.
Overly steep points and those obscured by axes/reference lines were excluded. Coordinate overlays were checked visually.

## Numerical reliability

Failures in initial pilot/independent runs were addressed only in the separate research runner.

1. Derivatives of canonical PB normalization were included. The Jacobian is tridiagonal plus rank one,
   solved with Sherman–Morrison. The equations were unchanged. For fields with std≈5, finite-difference
   Jacobian relative error was 3.77e−10, and the solution difference from the existing PB solver was 4.10e−11.
2. When a bead slice in the existing **real-space** float64 chain fell below its 1e−300 floor,
   densities could be wrong despite final normalization enforcing total mass. In a low-salt example,
   the SCFT residual was about 2e−10 but the independent log-space density difference was 0.209798 nm⁻³.
3. The research runner uses the np.longdouble exponent range and stepwise scaling.
   One Q is used at every contour position, without per-bead mass normalization.
   Independent all-log/log-sum-exp bond calculations validate it; the difference on the failing field was 6.99e−14.
4. When the relative inner high-salt PB tolerance failed the final absolute residual check,
   the problem was solved more tightly with `--pb-tolerance 1e-10`. Final acceptance criteria were not relaxed.

`prl_scft.py`, the Fig. 1 runner, existing CL drift, and production C++/CUDA/FFT were unchanged in this work.
SHA-256 hashes of the 33 protected files match those at the start.
The research runner does not validate a modification to the production path.

| Check | Acceptance threshold | Maximum error in accepted extended runs |
|---|---:|---:|
'''
    for label,key,tol in [('Full SCFT residual','full_residual',1e-7),('Independent chain density','chain_error',1e-10),('Contour Q','contour_error',1e-10),('Polymer integral','mass_error',1e-10),('C integral','counterion_mass_error',1e-10)]:
        text+=f'| {label} | {tol:.0e} | {valid[key]:.4e} |\n'
    text+=f'''
Preserved records include {counts['extended_checked']} validated extended runs, {counts['completed']} successful summaries,
and {counts['failed']} failed summaries. Successful summaries include initial float64 controls,
so their count differs from the extended-run count. Canceled/resubmitted jobs are tracked in SLURM records and `jobs.jsonl`.
For {counts['paired']} standard concentration pairs checked in both scan directions, the maximum moment-height difference is
{max(p['height_difference_nm'] for p in data['up_down_pairs']):.4e} nm.
For the same pairs, the largest full polymer density profile difference is {max(p['density_difference'] for p in data['up_down_pairs']):.4e} nm⁻³,
and the largest potential profile difference is {max(p['potential_profile_difference'] for p in data['up_down_pairs']):.4e}.
This agreement checks solution consistency, not a global free-energy minimum.

Selected grid/box checks: reference L=200, Nz=1600; refined grid L=200, Nz=3200;
larger box L=400, Nz=3200. Height changes are in nm.

| z+ | rho_b [M] | Finer grid Δh | Larger box Δh |
|---|---:|---:|---:|
'''
    for z,rho in [(1,0),(1,1e-7),(2,.1),(3,.01),(3,10)]:
        vals=[]
        for L in [200,400]:
            hits=[p for p in data['grid_box_checks'] if p['zplus']==z and abs(p['rho_molar']-rho)<1e-12 and p['L']==L]
            vals.append(f"{hits[-1]['delta_height_moment_nm']:.7f}" if hits else 'Incomplete')
        text+=f'| {z} | {rho:.5g} | {vals[0]} | {vals[1]} |\n'
    text+=f'''
At low salt, box dependence can be much greater for potential than height. For example, at zero salt, L=200→400
changes the wall–far-end potential difference by about −2.1744. These selected checks do not establish L→∞ convergence.

## Raw data and rerunning

Raw data: [{root}]({root}).

- `comparison.json`, `fig23_primary.csv`: actual run names, initial states, directions, and observables for every accepted point.
- Each run's `run.sh`, `slurm.out`, `summary.json`, `profiles.npz`, and failed checkpoints.
- `method_checks.json`, `low_salt_chain_diagnosis.json`, `protected_source_sha256.json`.
- `check_chain_gradient.py`, `chain_gradient_checks.json`: density versus directional derivatives of the common log Q.
- `paper_fig23_digitized.json`, digitization overlays, original PDF/SI.
- `source_v1/`, `source_final/`, `jobs.jsonl`, and submission helper `submit.py`.

Since early and final runs use different numerical algorithms, acceptance requires
`chain_solver=extended`, `pb_solver=rank1`, `reference=log`, and `status=completed`.
Failed or unvalidated results from the initial algorithm are excluded from the final scan.
Additional PB tolerance/iteration-cycle options are specified in each run's params and run.sh.

Reanalysis/LaTeX build (no new simulations):

```bash
source /home/yongdd/anaconda3/etc/profile.d/conda.sh
conda activate polymerfts
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python devel/charged_polymers/analyze_prl_fig23.py \\
  {root}
python devel/charged_polymers/build_prl_comparison_pdf.py \\
  --data-root {root.parent/'oned_prl_20260907'} \\
  --fig23-data-root {root}
```

To rerun a simulation, change `--out` in its run.sh to a new directory and submit through SLURM.
Preserve existing result directories. Tectonic compiles the PDFs, with equations typeset in TeX.
The current report includes this SCFT section; the combined PDF appends the preserved historical report.

Key remaining tasks: establish matching height definitions, ensembles, and boundaries; compare nonlocal Green function/I_ex treatments;
and confirm potential-figure normalization. Fig. 4 and the original CL correctness issue are not resolved here.

Source: [Duan et al., PRL 134, 048101 (2025)](https://doi.org/10.1103/PhysRevLett.134.048101),
main-text Figs. 2–3, Eqs. (7), (9), SI S33–S42. Comparisons use the saved published PDF.
'''
    (root/'FIG23_PRL_REPRODUCTION.md').write_text(text)


def analyze(root,paper):
    records=gather(root)
    byname={r['run']:r for r in records}
    grid=np.asarray(json.loads((root/'rho_grid.json').read_text()))
    zero=byname['saltfree_extended'];neutral=byname['neutral_extended']
    assert checked(zero) and checked(neutral)
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
        if not checked(point) or not point['run'].startswith('check_'):continue
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
    good=[r for r in records if checked(r)]
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
    repo=Path(__file__).resolve().parents[2]
    protected=json.loads((root/'protected_source_sha256.json').read_text())
    summary['protected_sources_unchanged']=all(hashlib.sha256((repo/k).read_bytes()).hexdigest()==v for k,v in protected.items())
    if not summary['protected_sources_unchanged']:raise RuntimeError('A protected source changed; review required.')
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


def digitize(root):
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


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    args=p.parse_args()
    d=digitize(args.root)
    print(json.dumps(dict(fig2_points=len(d['fig2']),fig3_points=len(d['fig3'])),indent=2))
    analyze(args.root,d)


if __name__=='__main__':main()
