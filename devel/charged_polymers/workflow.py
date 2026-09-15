"""Command-line entry point for charged-polymer research and report builds.

Run one parameter configuration per SLURM job for computations over one minute.
Use `workflow.py GROUP COMMAND --help` for the original, unchanged options.
"""
import argparse
import importlib
import json
from pathlib import Path
import runpy
import sys

ROUTES = {
    'prl': {
        'mccarty': ('mccarty', 'run', 'Sequence-resolved bulk Gaussian phase separation; not full Duan SCFT or CL'),
        'electrolyte': ('literature_scft', 'run', 'PRL electrolyte limits and literature reproduction attempts'),
        'fig1': ('prl_runs', 'run_fig1', 'Checked Fig. 1 at fixed ionic strength'),
        'salt': ('prl_runs', 'run_salt', 'One concentration for Figs. 2–3 or matched SCFT'),
        'meanfield': ('prl_runs', 'run_meanfield', 'Mean-field reference and charge continuation'),
    },
    'cl': {
        'brush': ('validation', 'run_brush', 'Same-model MF/CL benchmark'),
        'exact': ('exact', 'run', 'Native CL versus finite particle enumeration'),
        'reduced': ('reduced', 'run', 'Reduced CL integrator and adaptive-step study'),
    },
    'sample': {
        'brush': ('brush_sampling', 'run', 'N=100 planar brush, bulk bath, or tiny control'),
        'finite': ('finite_sampling', 'run', 'Finite 9-bead particle/real-field Gibbs sampler'),
        'marginal': ('marginal_sampling', 'run', 'Two-cell positive real-field marginal'),
    },
    'check': {
        'implementation': ('validation', 'check_implementation', 'Forces, grafts, checkpoints, and strong fields'),
        'brush': ('brush_sampling', 'check', 'Positive-brush detailed balance and exact sums'),
        'checkpoints': ('cl_diagnostics', 'check_checkpoints', 'Native densities at stored final fields'),
        'field-integral': ('cl_diagnostics', 'check_field_integral', 'Original-contour Gauss–Hermite quadrature'),
        'poles': ('cl_diagnostics', 'check_poles', 'Partition zeros and largest-drift states'),
        'identities': ('cl_diagnostics', 'check_identities', 'Generator identities and pole-cutoff contributions'),
    },
    'analyze': {
        'mccarty': ('mccarty', 'analyze', 'McCarty bulk coexistence and sequence comparisons'),
        'electrolyte': ('literature_scft', 'analyze', 'PRL electrolyte curves, controls and convergence evidence'),
        'cl-exact': ('cl_analysis', 'analyze_exact', 'Finite native CL statistics'),
        'cl-reduced': ('cl_analysis', 'analyze_reduced', 'Reduced-integrator comparisons'),
        'cl-brush': ('cl_analysis', 'analyze_brush', 'Same-model MF/CL summary'),
        'marginal': ('cl_analysis', 'analyze_marginal', 'Positive marginal and CL diagnostics'),
        'fig1-finite': ('prl_analysis', 'analyze_fig1_finite', 'Original mixed Fig. 1 / small-system data bundle'),
        'prl-scan': ('prl_analysis', 'analyze_scan', 'Figs. 2–3 digitization and checked scan'),
        'matched': ('prl_analysis', 'analyze_matched', 'MC versus concentration-matched LDA/MF SCFT'),
        'brush': ('brush_analysis', 'run_analysis', 'Planar MC statistics, profiles, and controls'),
    },
}

SCRIPT_ROUTES = {
    'test': {
        'mccarty': ('tests.test_mccarty', 'Sequence form factors, thermodynamics and coexistence'),
        'neutral-cl': ('tests.test_neutral_reduction_clfts', 'Charged CL versus neutral CL'),
    },
    'example': {
        'lamella': ('examples.Lamella', 'Charged-polymer CL demonstration'),
    },
}


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if args and args[0]=='legacy':
        parser=argparse.ArgumentParser(prog=f'{sys.argv[0]} legacy',description='Run a consolidated script using its historical filename and options.')
        parser.add_argument('script');parser.add_argument('options',nargs=argparse.REMAINDER)
        values=parser.parse_args(args[1:])
        mapping=json.loads((Path(__file__).parent/'studies/legacy_commands.json').read_text())
        target=mapping.get(Path(values.script).name)
        if target is None:parser.error('Unknown historical script; see studies/legacy_commands.json')
        sys.argv=[values.script,*values.options]
        return getattr(importlib.import_module(target['module']),target['function'])()
    if args and args[0]=='reports':
        sys.argv=[f'{sys.argv[0]} reports',*args[1:]]
        from reports.build_reports import main as build
        return build()
    parser=argparse.ArgumentParser(description=__doc__)
    groups=parser.add_subparsers(dest='group',required=True)
    for group,commands in ROUTES.items():
        sub=groups.add_parser(group,help=f'{group.capitalize()} workflows')
        choices=sub.add_subparsers(dest='command',required=True)
        for command,(_,_,description) in commands.items():
            item=choices.add_parser(command,help=description,add_help=False)
            item.add_argument('options',nargs=argparse.REMAINDER)
    for group,commands in SCRIPT_ROUTES.items():
        sub=groups.add_parser(group,help=f'{group.capitalize()} scripts')
        choices=sub.add_subparsers(dest='command',required=True)
        for command,(_,description) in commands.items():
            choices.add_parser(command,description=description,help=description)
    groups.add_parser('reports',help='Build the two consolidated PDFs')
    groups.add_parser('legacy',help='Use a historical script name with its original options')
    # Parse the two routing tokens only; each workflow owns its original parser.
    values=parser.parse_args(args if args and args[0] in SCRIPT_ROUTES else args[:2])
    if values.group in SCRIPT_ROUTES:
        module,_=SCRIPT_ROUTES[values.group][values.command]
        sys.argv=[f'{sys.argv[0]} {values.group} {values.command}']
        return runpy.run_module(module,run_name='__main__')
    module,function,_=ROUTES[values.group][values.command]
    sys.argv=[f'{sys.argv[0]} {values.group} {values.command}',*args[2:]]
    return getattr(importlib.import_module('studies.'+module),function)()


if __name__=='__main__':
    main()
