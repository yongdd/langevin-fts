"""One McCarty bulk parameter configuration, or analysis of saved results."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from models.mccarty import McCartyBulk, SEQUENCES, CLOSURES, scd, sequence_blocks


def run():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['critical', 'bulk'], required=True)
    p.add_argument('--sequence', choices=list(SEQUENCES), default='sv30')
    p.add_argument('--closure', choices=CLOSURES, default='electrostatic')
    p.add_argument('--bjerrum', type=float, default=.33, help='lB/b; ignored by critical search')
    p.add_argument('--excluded-volume', type=float, default=.0068, help='v/b^3; SI S3 benchmark value')
    p.add_argument('--smear', type=float, default=1/np.sqrt(6), help='Gaussian a/b')
    p.add_argument('--quadrature-points', type=int, default=384)
    p.add_argument('--cutoff-a', type=float, default=12.)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out/name).exists() for name in ['summary.json', 'curves.npz']):
        raise FileExistsError('Choose a fresh output directory')
    start = time.perf_counter()
    m = McCartyBulk(args.sequence, args.bjerrum, args.excluded_volume, args.smear,
                    args.closure, args.quadrature_points, args.cutoff_a)
    root = Path(__file__).resolve().parents[1]
    summary = dict(case=args.case, params=asdict(m), status='running',
        sequence_string=SEQUENCES[args.sequence], scd=scd(args.sequence),
        native_blocks=sequence_blocks(args.sequence),
        units=dict(length='b', density='b^-3', free_energy_density='kBT/b^3',
                   pressure='kBT/b^3', chemical_potential='kBT per chain'),
        scope='Salt-free, ideal-chain bulk Gaussian closure; no CL sampling or full Duan conformational self-consistency',
        excluded_volume_provenance='0.0068 is reported for SI Fig. S3; main Figs. 2--3 parameter correspondence remains to be confirmed',
        source_hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in [Path(__file__), root/'models/mccarty.py']})
    try:
        if args.case == 'critical':
            summary['critical'] = m.critical()
        else:
            summary['coexistence'] = m.coexistence()
            rho = np.geomspace(1e-7, 20., 321)
            k = np.geomspace(.001, 20., 200)
            sn, sq = m.structure_factors(1.47, k)
            np.savez_compressed(args.out/'curves.npz', rho=rho,
                free_energy=m.free_energy(rho), pressure=m.pressure(rho),
                excess_pressure=m.pressure(rho)-rho/m.n, mu_chain=m.mu_chain(rho),
                curvature=m.curvature(rho), k=k, s_number=sn, s_charge=sq,
                structure_density=1.47)
        summary['status'] = 'completed'
    except Exception as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        summary['seconds'] = time.perf_counter()-start
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


def analyze():
    p = argparse.ArgumentParser(description='Plot saved McCarty Gaussian results, with no new simulations')
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'savefig.bbox':'tight'})
    records = [(path, json.loads(path.read_text())) for path in sorted(args.campaign.glob('*/summary.json'))]
    if not records or any(r['status'] != 'completed' for _, r in records):
        raise RuntimeError('Campaign contains missing or failed results')
    manifest = args.campaign/'campaign.json'
    if manifest.exists():
        expected = {job['label'] for job in json.loads(manifest.read_text())['jobs']}
        if expected != {path.parent.name for path, _ in records}:
            raise RuntimeError('Completed directories do not match the submitted campaign')
    for path, r in records:
        for name, digest in r['source_hashes'].items():
            current = Path(__file__).resolve().parents[1]/name
            archived = args.campaign/'sources'/name
            source = archived if archived.is_file() else current
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Calculation source hash mismatch: {name}')
            # The response diagnostic below calls the model. Plot formatting
            # may change independently, but silently using a new model may not.
            if name.startswith('models/') and hashlib.sha256(current.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Use the archived numerical model to analyze this campaign: {name}')
    args.out.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(SEQUENCES, ['#bd24b8', '#198338', '#174fcc', '#dc3f27', '#222222']))
    styles = {'electrostatic':'-', 'paper-rpa':'--', 'coupled-rpa':':'}
    critical = [r for _, r in records if r['case'] == 'critical']
    bulk = [(path, r) for path, r in records if r['case'] == 'bulk']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    for closure in ['electrostatic', 'paper-rpa']:
        for seq in SEQUENCES:
            points = [r for _, r in bulk if r['params']['sequence'] == seq and r['params']['closure'] == closure
                      and r['coexistence']['two_phase']]
            points.sort(key=lambda r:r['params']['bjerrum'])
            cp = next((r['critical'] for r in critical if r['params']['sequence'] == seq and r['params']['closure'] == closure), None)
            if not points or cp is None:
                continue
            for phase in ['rho_dilute', 'rho_dense']:
                rho = [cp['rho']]+[r['coexistence'][phase] for r in points]
                lb = [cp['bjerrum']]+[r['params']['bjerrum'] for r in points]
                for ax in axes:
                    ax.plot(rho, lb, styles[closure], color=colors[seq], linewidth=1.3)
            axes[0].plot(cp['rho'], cp['bjerrum'], 'o', ms=3, color=colors[seq])
    for ax in axes:
        ax.set(xlabel=r'Monomer density $\rho b^3$', ylabel=r'$\ell_B/b$')
        ax.grid(alpha=.2)
    axes[0].set_xscale('log'); axes[0].set_xlim(1e-9, 15)
    axes[0].set_title('Both branches; dilute sensitivity visible')
    axes[1].set_xlim(0, 8); axes[1].set_title('Linear density view')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, label=s) for s, c in colors.items()]
    handles += [Line2D([], [], color='gray', label='Electrostatic Gaussian', linestyle='-'),
                Line2D([], [], color='gray', label='Paper RPA', linestyle='--')]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(.5, -.1), frameon=False)
    fig.suptitle('Computed salt-free bulk coexistence — ideal-chain Gaussian approximations')
    fig.tight_layout();fig.savefig(args.out/'mccarty_binodals.pdf');plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for row, seq in enumerate(['sv20', 'sv30']):
        for lb, color in [(.033, 'black'), (.33, '#d6322d')]:
            for closure in ['electrostatic', 'paper-rpa']:
                entry = next(((path, r) for path, r in bulk if r['params']['sequence'] == seq
                              and r['params']['closure'] == closure and abs(r['params']['bjerrum']-lb) < 1e-10), None)
                if entry is None:
                    raise RuntimeError(f'Missing EOS {seq}, {lb}, {closure}')
                values = np.load(entry[0].parent/'curves.npz')
                label = f'lB/b={lb:g}, {closure}'
                axes[row, 0].semilogx(values['rho'], values['excess_pressure'], styles[closure], color=color, label=label)
                axes[row, 1].semilogx(values['rho'], values['mu_chain'], styles[closure], color=color)
        axes[row, 0].set(xlim=(.1, 10), ylim=(-.2, .35), ylabel=r'$\beta b^3(\Pi-\rho k_BT/N)$', title=seq+' pressure')
        axes[row, 1].set(xlim=(1e-4, 10), ylabel=r'$\beta\mu_{chain}$', title=seq+' chemical potential')
        for ax in axes[row]:
            ax.set_xlabel(r'$\rho b^3$'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Computed equations of state — same axes/observables as McCarty Fig. 2')
    fig.tight_layout();fig.savefig(args.out/'mccarty_eos.pdf');plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for closure in CLOSURES:
        rows = sorted([r for r in critical if r['params']['closure'] == closure], key=lambda r:r['scd'])
        if rows:
            axes[0].plot([-r['scd'] for r in rows], [r['critical']['bjerrum'] for r in rows],
                         styles[closure]+'o', label=closure, ms=4)
    k = np.geomspace(.001, 20, 200)
    from models.mccarty import form_factors
    for seq, color in colors.items():
        axes[1].semilogx(k, form_factors(seq, k)[2], color=color, label=seq)
    axes[0].set(xlabel='−SCD (charge patchiness)', ylabel=r'Critical $\ell_B/b$')
    axes[1].set(xlabel=r'$kb$', ylabel=r'Ideal-chain $g_{qq}(k)$')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.out/'mccarty_sequence_response.pdf');plt.close(fig)
    summary = dict(campaign=str(args.campaign.resolve()), completed=len(records),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        critical=[dict(sequence=r['params']['sequence'], closure=r['params']['closure'], **r['critical']) for r in critical],
        max_delta_mu_chain=max(r['coexistence'].get('delta_mu_chain', 0) for _, r in bulk),
        max_delta_pressure=max(r['coexistence'].get('delta_pressure', 0) for _, r in bulk),
        scope='Computed bulk Gaussian curves; no sampled FTS/MD reproduction claim',
        records=[dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path, _ in records])
    (args.out/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (args.out/'critical.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sequence', 'closure', 'bjerrum', 'temperature',
                                                    'rho', 'scaled_curvature', 'scaled_third_derivative'])
        writer.writeheader();writer.writerows(summary['critical'])
    rows = [dict(sequence=r['params']['sequence'], closure=r['params']['closure'],
                 bjerrum=r['params']['bjerrum'], **r['coexistence']) for _, r in bulk]
    with (args.out/'coexistence.csv').open('w', newline='') as stream:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k != 'records'}, indent=2))
