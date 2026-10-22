"""Execute imported scripts on allocated resources when explicitly enabled.

Set POLYMERFTS_SCRIPT_TEST_PLATFORM to cpu-mkl, cpu-fftw or cuda. PSCF_RPC
optionally also checks the source command using the real PSCF CPU executable.
"""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
TOOL = runpy.run_path(str(ROOT / 'scripts/pscf/pscf_converter.py'))
PSCF_TESTS = runpy.run_path(str(ROOT / 'tests/scripts/TestPscfIntegration.py'))


@pytest.mark.parametrize('case', ('1d', '2d', '3d', 'skewed', 'branched', 'mixture', 'relaxed'))
def test_imported_script_converges(tmp_path, case):
    platform = os.environ.get('POLYMERFTS_SCRIPT_TEST_PLATFORM')
    if not platform:
        pytest.skip('Set POLYMERFTS_SCRIPT_TEST_PLATFORM on allocated compute resources')
    assert platform in ('cpu-mkl', 'cpu-fftw', 'cuda')
    dim = int(case[0]) if case in ('1d', '2d', '3d') else 1 if case == 'relaxed' else 3
    nx = [32] + [4] * (dim - 1)
    p = PSCF_TESTS['base_params'](nx, [4.] + [3.] * (dim - 1), [90.] * 3)
    p['segment_lengths'] = {'long_A': 1., 'B_two': 1.}
    p['chi_n'] = {'long_A,B_two': 20.}
    for block in p['polymers'][0]['blocks']:
        if block['type'] == 'B2':
            block['type'] = 'B_two'
    if case == 'skewed':
        p.update(angles=[72., 83., 66.], crystal_system='Triclinic')
    if case == 'branched':
        p['polymers'][0]['blocks'] = [
            {'type': 'long_A', 'length': .25, 'v': 0, 'u': 1},
            {'type': 'long_A', 'length': .25, 'v': 1, 'u': 2},
            {'type': 'B_two', 'length': .5, 'v': 1, 'u': 3}]
    if case == 'mixture':
        p['polymers'][0]['volume_fraction'] = .8
        p['polymers'].append({'volume_fraction': .2, 'blocks': [{'type': 'long_A', 'length': 1.}]})
    if case == 'relaxed':
        p['relax_cell'] = True
    x = np.indices(nx, dtype=float)[0]
    w = np.cos(2 * np.pi * (x + .5) / nx[0])
    source = tmp_path / 'initial.json'
    source.write_text(json.dumps({'params': p, 'w_long_A': w.tolist(), 'w_B_two': (-w).tolist()}))
    pscf = TOOL['export_files'](source, tmp_path / 'pscf')
    if os.environ.get('PSCF_RPC'):
        PSCF_TESTS['run_pscf'](PSCF_TESTS['executable']('rpc'), pscf, dim)
    bundle = TOOL['import_script'](pscf / 'param', pscf / 'command', tmp_path / 'job',
                                 metadata=pscf / 'metadata.json', platform=platform)
    # Start outside the bundle; the generated script must resolve its own inputs.
    result = subprocess.run([sys.executable, str(bundle / 'simulation.py')], cwd=tmp_path,
                            capture_output=True, text=True, timeout=120)
    (tmp_path / 'polymerfts.log').write_text(result.stdout + '\n' + result.stderr)
    assert result.returncode == 0, (tmp_path / 'polymerfts.log').read_text()[-6000:]
    snapshots = [bundle / 'results/out' / (name + '.mat') for name in ('omega', 'rho', 'thermo')]
    assert all(path.is_file() for path in snapshots)
    fields = loadmat(snapshots[0], simplify_cells=True)
    assert json.loads(fields['simulation_metadata_json'])['execution_status'] == 'converged'
    assert np.isfinite(fields['free_energy'])
    total = sum(fields['phi_' + name] for name in p['segment_lengths'])
    np.testing.assert_allclose(total, 1., rtol=0, atol=1e-7)
    for name in p['segment_lengths']:
        expected = sum(polymer['volume_fraction'] *
                       sum(b['length'] for b in polymer['blocks'] if b['type'] == name) /
                       sum(b['length'] for b in polymer['blocks']) for polymer in p['polymers'])
        np.testing.assert_allclose(np.mean(fields['phi_' + name]), expected, rtol=0, atol=1e-8)
    if case != 'relaxed':
        np.testing.assert_allclose(np.atleast_1d(fields['lx']), p['lx'], rtol=0, atol=1e-10)
        np.testing.assert_allclose(fields['angles'], p['angles'], rtol=0, atol=1e-8)
    else:
        assert abs(float(np.asarray(fields['lx']).item()) - p['lx'][0]) > 1e-3
    assert (bundle / 'results/polymer_0.png').is_file()
    assert not (tmp_path / 'polymer_0.png').exists()
