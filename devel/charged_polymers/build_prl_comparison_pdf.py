"""Refresh numerical tables/figures and compile the two LaTeX reports.

Run in the polymerfts environment with Tectonic and pypdf installed::

    python devel/charged_polymers/build_prl_comparison_pdf.py --data-root RUN_DIR

The editable document is ONED_PRL_REPRODUCTION.tex. PRL_COMPARISON.tex appends
the preserved historical PDF. Existing PDFs are replaced only after both
LaTeX compilations and their text/font checks succeed. No simulation is run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from pypdf import PdfReader


def sampling_scft_assets(root,report):
    data=json.loads((root/'comparison.json').read_text())
    if len(data['rows'])!=16:raise RuntimeError('Incomplete MC/SCFT comparison')
    assets=report/'prl_comparison_figs'/'sampling_scft_20260907'
    assets.mkdir(parents=True,exist_ok=True)
    for name in ['mc_scft_height.pdf','mc_scft_charge.pdf','mc_scft_profiles.pdf']:
        shutil.copy2(root/name,assets/name)
    commands=[]
    def command(name,value):commands.append('\\newcommand{\\MCScft'+name+'}{'+str(value)+'}')
    def estimate(x,se,digits=2):return f'${x:.{digits}f}'+r'\pm'+f'{se:.{digits}f}$'
    for key,name in [('mc','MC'),('lda','LDA'),('mf','MF')]:
        command('Zero'+name,f"${data['height_reference'][key]:.3f}$")
    command('BathShift',f"${data['validation']['max_bath_matching_height_shift']:.3f}$")
    command('MaxMfHeightDifference',f"${max(abs(r['scft']['mf']['height_mc_minus_scft']) for r in data['rows']):.3f}$")
    rows=[]
    for z in [1,2,3]:
        row=next(r for r in data['rows'] if r['valence']==z and r['target_rho']==.01)
        cols=[f'${z}$',estimate(row['mc_mean'][0],row['mc_se'][0])]
        cols += [f"${row['scft'][k]['mean'][0]:.2f}$" for k in ['lda','mf']]
        rows.append(' & '.join(cols)+r' \\')
    command('HeightRows','%\n'+'\n'.join(rows)+'\n')
    rows=[]
    row=next(r for r in data['rows'] if r['valence']==3 and r['target_rho']==.1)
    for i,label in [(1,r'$\overline\Gamma_{0.5}$'),(3,r'$\Psi_{\rm rel}$')]:
        cols=[label,estimate(row['mc_mean'][i],row['mc_se'][i],3)]
        cols += [f"${row['scft'][k]['mean'][i]:.3f}$" for k in ['lda','mf']]
        rows.append(' & '.join(cols)+r' \\')
    command('ChargeRows','%\n'+'\n'.join(rows)+'\n')
    (assets/'numbers.tex').write_text('% Generated from concentration-matched comparison.json\n'+'\n'.join(commands)+'\n')
    return dict(root=str(root),sha256=hashlib.sha256((root/'comparison.json').read_bytes()).hexdigest())


def sampling_assets(root,report):
    data=json.loads((root/'analysis.json').read_text())
    if len(data['runs'])<42 or len(data['groups'])!=16 or len(data['tiny'])!=12 or not data['extensions_complete']:
        raise RuntimeError('Positive sampling campaign is incomplete; do not publish partial tables')
    assets=report/'prl_comparison_figs'/'positive_brush_20260907'
    assets.mkdir(parents=True,exist_ok=True)
    for name in ['sampling_fig2.pdf','sampling_fig3.pdf','sampling_diagnostics.pdf']:
        shutil.copy2(root/name,assets/name)
    commands=[]
    def command(name,value):commands.append('\\newcommand{\\Sampling'+name+'}{'+str(value)+'}')
    def number(x,digits=3):return f'{x:.{digits}g}'
    def estimate(mean,se):return f'${mean:.2f}'+r'\pm'+f'{se:.2f}$'
    zero=next(g for g in data['groups'] if g['target_rho']==0)
    command('ZeroHeight',estimate(zero['mean'][0],zero['se'][0]))
    command('TinySigma',number(max(r['worst_sigma'] for r in data['tiny'])))
    command('RhatMax',number(max(g['height_split_rhat'] for g in data['groups']),4))
    command('EssMin',number(min(r['height_ess'] for r in data['runs']),3))
    command('EssMax',number(max(r['height_ess'] for r in data['runs']),3))
    command('PairSigmaMax',number(max(max(g['pair_sigma'][:4]) for g in data['groups'])))
    command('MaxChargeError',f"${max(r['max_charge_update_error'] for r in data['runs']):.2e}$")
    command('BathRelativeSe',number(100*max(b['se']/b['rho'] for b in data['bulk'])))
    rows=[]
    for rho in [.001,.01,.1,1.,10.]:
        cols=[f'${rho:g}$']
        for z in [1,2,3]:
            g=next(g for g in data['groups'] if g['valence']==z and g['target_rho']==rho)
            cols.append(estimate(g['mean'][0],g['se'][0]))
        rows.append(' & '.join(cols)+r' \\')
    command('HeightRows','%\n'+'\n'.join(rows)+'\n')
    rows=[]
    for label,M,A,rho in [('Standard',400,40,.01),('Finer grid',800,40,.01),('Smaller area',400,20,.01),('Larger area',400,80,.01),('Salt-free, finer grid',800,40,0)]:
        matches=[r for r in data['runs'] if r['params']['sites']==M and r['params']['area']==A
                 and r['params']['activity_molar']==rho and r['params']['joint'] and r['params']['valence']==(3 if rho else 1)]
        means=[r['mean'][0] for r in matches];ses=[r['se'][0] for r in matches]
        mean=sum(means)/len(means);se=max(sum(s*s for s in ses)**.5/len(ses),abs(means[0]-means[-1])/2)
        rows.append(' & '.join([label,f'${M}$',f'${A}$',estimate(mean,se)])+r' \\')
    command('CheckRows','%\n'+'\n'.join(rows)+'\n')
    controls=[]
    for rho in [.01,1.]:
        joint=next(g for g in data['groups'] if g['valence']==3 and g['target_rho']==rho)
        particle=next(r for r in data['runs'] if not r['params']['joint'] and r['params']['activity_molar']==rho)
        sigma=abs(joint['mean'][0]-particle['mean'][0])/(joint['se'][0]**2+particle['se'][0]**2)**.5
        controls.append(sigma)
    command('ControlSigma',number(max(controls)))
    (assets/'numbers.tex').write_text('% Generated from positive sampling analysis.json\n'+'\n'.join(commands)+'\n')
    return dict(root=str(root),sha256=hashlib.sha256((root/'analysis.json').read_bytes()).hexdigest())


def fig23_assets(root, report):
    """Keep every displayed number tied to the checked campaign manifest."""
    data=json.loads((root/'comparison.json').read_text())
    assets=report/'prl_comparison_figs'/'fig23_prl_20260907'
    assets.mkdir(parents=True,exist_ok=True)
    for name in ['fig2_checked.pdf','fig3_checked.pdf','fig3_units_check.pdf']:
        shutil.copy2(root/name,assets/name)
    def number(x):
        if x is None:return r'---'
        if x==0:return '$0$'
        if abs(x)>=.001:return f'${x:.4g}$'
        m,e=f'{x:.2e}'.split('e')
        return f'${m}'+r'\times10^{'+str(int(e))+'}$'
    commands=[]
    def command(name,value):commands.append('\\newcommand{\\FigTwentyThree'+name+'}{'+str(value)+'}')
    for key in ['primary','paired','extended_checked','failed']:
        command(''.join(s.capitalize() for s in key.split('_'))+'Count',data['counts'][key])
    command('SaltfreeMoment',number(data['saltfree']['observables']['height_moment_nm']))
    command('SaltfreeEquimolar',number(data['saltfree']['observables']['height_equimolar_peak_nm']))
    command('NeutralMoment',number(data['neutral']['observables']['height_moment_nm']))
    for key,val in data['validation_maxima'].items():
        command(''.join(s.capitalize() for s in key.split('_')),number(val))
    for key in ['height_difference_nm','gamma_difference','psi_difference','density_difference','potential_profile_difference']:
        command('Pair'+''.join(s.capitalize() for s in key.split('_')),number(max(p[key] for p in data['up_down_pairs'])))
    rows=[]
    for row in data['sampled_minima']:
        rows.append(' & '.join([number(row['zplus']),number(row['rho_molar']),number(row['height_nm']),number(row['log_height_ratio'])])+r' \\')
    command('MinimaRows','%\n'+'\n'.join(rows)+'\n')
    rows=[]
    for z,rho in [(1,0),(1,1e-7),(2,.1),(3,.01),(3,10)]:
        def find(L):
            hits=[r for r in data['grid_box_checks'] if r['zplus']==z and abs(r['rho_molar']-rho)<1e-12 and r['L']==L]
            return hits[-1]['delta_height_moment_nm'] if hits else None
        rows.append(' & '.join([number(z),number(rho),number(find(200)),number(find(400))])+r' \\')
    command('GridRows','%\n'+'\n'.join(rows)+'\n')
    (assets/'numbers.tex').write_text('% Generated from the Fig. 2-3 comparison.json.\n'+'\n'.join(commands)+'\n')
    return dict(root=str(root),sha256=hashlib.sha256((root/'comparison.json').read_bytes()).hexdigest())


def table_rows(path, rows):
    command = path.stem.split("_")[0].capitalize() + "Rows"
    path.write_text("% Generated from comparison.json; rerun the PDF builder to refresh.\n"
                    + "\\newcommand{\\" + command + "}{%\n"
                    + "\n".join(" & ".join(row) + r" \\" for row in rows) + "\n}\n")


def compile_tex(engine, source, work):
    result = subprocess.run(
        [engine, "--keep-logs", "--keep-intermediates", "--synctex",
         "--outdir", str(work), source.name],
        cwd=source.parent, capture_output=True, text=True)
    (work / (source.stem + ".build.txt")).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"LaTeX compilation failed for {source.name}:\n"
                           + (result.stdout + result.stderr)[-6000:])
    log = (work / source.with_suffix(".log").name).read_text(errors="replace")
    for marker in ("Missing character:", "Overfull \\hbox", "Overfull \\vbox"):
        if marker in log:
            raise RuntimeError(f"{source.name}: {marker}; inspect {work / source.with_suffix('.log').name}")
    pdf = work / source.with_suffix(".pdf").name
    reader = PdfReader(pdf)
    if not reader.pages or any(not p.extract_text().strip() for p in reader.pages):
        raise RuntimeError(f"Empty page in {pdf}")
    return len(reader.pages)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--fig23-data-root",type=Path,
                    help="Checked Fig. 2-3 campaign; defaults to the sibling fig23_prl_20260907 directory")
    ap.add_argument('--sampling-data-root',type=Path,
                    help='Actual positive brush sampling campaign')
    ap.add_argument('--sampling-scft-root',type=Path,
                    help='Concentration-matched MC versus LDA/MF SCFT comparison')
    ap.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--engine", default="tectonic")
    args = ap.parse_args()
    root, report = args.data_root.resolve(), args.report_dir.resolve()
    fig23_root=args.fig23_data_root or root.parent/'fig23_prl_20260907'
    fig23_manifest=fig23_assets(fig23_root,report)
    sampling_manifest=sampling_assets(args.sampling_data_root or root.parent/'positive_brush_20260907',report)
    sampling_scft_manifest=sampling_scft_assets(args.sampling_scft_root or root.parent/'sampling_scft_20260907',report)
    engine = shutil.which(args.engine)
    if engine is None:
        raise RuntimeError("Tectonic is required; activate the polymerfts environment.")
    data_file = root / "comparison.json"
    data = json.loads(data_file.read_text())
    samplers = {s["run"]: s for s in data["positive_runs"]}
    prl = {s["run"]: s for s in data["prl_runs"]}
    assets = report / "prl_comparison_figs" / "oned_prl_20260907"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("positive_1d_comparison.pdf", "prl_fig1_checked.pdf", "prl_closure_comparison.pdf"):
        shutil.copy2(root / name, assets / name)

    def estimate(run, label, digits=6):
        s = samplers[run]
        i = s["labels"].index(label)
        return (f'${s["statistics"]["mean"][i]:.{digits}f}'
                + r" \pm " + f'{s["statistics"]["se"][i]:.{digits}f}$')

    density = [["Exact particle sum", "$0.47375505$"]]
    density += [[f"Real-field/particle Gibbs, seed {seed}", estimate(f"positive_M4_E25_joint_s{seed}", "phi_P_cell1", 8)]
                for seed in (1101, 1201)]
    density += [["Particle-only Gibbs, seed 2101", estimate("positive_M4_E25_particles_s2101", "phi_P_cell1", 8)],
                ["Historical native CL, seed 71", r"$0.45115654 \pm 0.00333992$"],
                ["Historical native CL, seed 72", r"$0.44947711 \pm 0.00337511$"]]
    table_rows(assets / "density_rows.tex", density)
    table_rows(assets / "height_rows.tex", [[f"${m}$"] + [
        estimate(f"positive_M{m}_E100_{mode}_s{seed}", "height")
        for mode, seed in (("joint", 1101), ("joint", 1201), ("particles", 2101))]
        for m in (4, 16, 32)])
    rows = []
    for z, closure in [(0, "mf")] + [(z, c) for c in ("mf", "lda") for z in (1, 2, 3)]:
        label = "Neutral" if z == 0 else f"$z_+={z}$, " + ("mean field" if closure == "mf" else "LDA correlations")
        point = next(t for t in prl[f"prl_z{z}_{closure}_M3200"]["paper_comparison"] if t["z_over_b"] == 5)
        rows.append([label, f'${point["phi"]:.4f}$', f'${point["computed_phi"]:.4f}$'])
    table_rows(assets / "prl_rows.tex", rows)
    rows = []
    for z in (1, 2, 3):
        nonlocal_name = f"prl_z{z}_nonlocal" + ("_continuation" if z == 3 else "") + "_M1600"
        names = [f"prl_z{z}_lda_M3200", f"prl_z{z}_lda_gc_M1600", nonlocal_name]
        rows.append([f"${z}$"] + [f'${prl[n]["max_interior_phi_difference"]:.4f}$' for n in names])
    table_rows(assets / "closure_rows.tex", rows)

    archive = report / "prl_comparison_figs/archive/PRL_COMPARISON_before_20260907_update.pdf"
    if not archive.is_file():
        raise FileNotFoundError(f"Preserved historical PDF is required: {archive}")
    # Keep failed build logs too. A TeX error must not replace the open PDFs.
    work = Path(tempfile.mkdtemp(prefix="latex_", dir=root))
    shutil.copytree(report / "prl_comparison_figs", work / "prl_comparison_figs")
    names = ("ONED_PRL_REPRODUCTION", "PRL_COMPARISON")
    for name in (*names,"FIG23_PRL_REPRODUCTION","POSITIVE_BRUSH_SAMPLING","SAMPLING_SCFT_COMPARISON"):
        shutil.copy2(report / (name + ".tex"), work / (name + ".tex"))
    # Read the editable sources in place so SyncTeX points to the repository,
    # while all compiler outputs still go to the isolated build directory.
    current_pages = compile_tex(engine, report / (names[0] + ".tex"), work)
    combined_pages = compile_tex(engine, report / (names[1] + ".tex"), work)
    old_pages = len(PdfReader(archive).pages)
    if combined_pages != current_pages + old_pages:
        raise RuntimeError("The combined PDF does not contain all report pages.")
    for name in names:
        for suffix in (".pdf", ".log", ".aux", ".out", ".synctex.gz"):
            source = work / (name + suffix)
            if source.exists():
                temporary = report / (name + suffix + ".tmp")
                shutil.copy2(source, temporary)
                temporary.replace(report / source.name)
    manifest = dict(engine=subprocess.check_output([engine, "--version"], text=True).strip(),
                    source="ONED_PRL_REPRODUCTION.tex", current_pages=current_pages,
                    positive_sampling=sampling_manifest,
                    sampling_scft=sampling_scft_manifest,
                    combined_pages=combined_pages, build_dir=str(work),
                    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    data_sha256=hashlib.sha256(data_file.read_bytes()).hexdigest())
    manifest['fig23']=fig23_manifest
    manifest['source_sha256']={name+'.tex':hashlib.sha256((report/(name+'.tex')).read_bytes()).hexdigest()
                               for name in (*names,'FIG23_PRL_REPRODUCTION','POSITIVE_BRUSH_SAMPLING','SAMPLING_SCFT_COMPARISON')}
    (root / "pdf_build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
