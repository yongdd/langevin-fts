"""Optional interoperability checks against real PSCF periodic executables.

Set PSCF_RPC and/or PSCF_RPG to executable paths. Run on allocated compute
resources; each parametrized case defines one simulation configuration.
"""

import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT / 'scripts/pscf/pscf_converter.py'
TOOL = runpy.run_path(str(CONVERTER))

# PSCF cell parameters, independent of polymerfts's crystallographic angle input.
CELLS = {
    'lamellar': (1, 'lamellar', [4.]),
    'square': (2, 'square', [4.]),
    'rectangular': (2, 'rectangular', [4., 3.]),
    'hexagonal2d': (2, 'hexagonal', [4.]),
    'rhombic': (2, 'rhombic', [4., math.pi / 3]),
    'oblique': (2, 'oblique', [4., 3., math.pi / 3]),
    'cubic': (3, 'cubic', [4.]),
    'tetragonal': (3, 'tetragonal', [4., 3.]),
    'orthorhombic': (3, 'orthorhombic', [4., 3., 2.]),
    'hexagonal3d': (3, 'hexagonal', [4., 3.]),
    'rhombohedral': (3, 'rhombohedral', [4., math.pi / 3]),
    'monoclinic': (3, 'monoclinic', [4., 3., 2., math.pi / 3]),
    'triclinic': (3, 'triclinic', [4., 3., 2., .4, .3, 1.2]),
    'branched': (3, 'orthorhombic', [4., 3., 2.]),
    'mixture': (3, 'orthorhombic', [4., 3., 2.]),
}


def executable(backend):
    variable = 'PSCF_' + backend.upper()
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f'Set {variable} to run real PSCF integration tests')
    path = Path(value).expanduser().resolve()
    assert path.is_file() and os.access(path, os.X_OK), f'{variable} is not executable: {path}'
    return path


def base_params(nx, lx, angles):
    return {
        'nx': nx, 'lx': lx, 'angles': angles, 'chain_model': 'continuous',
        'ds': .025, 'segment_lengths': {'long_A': 1., 'B2': 1.2},
        'chi_n': {'long_A,B2': 20.}, 'relax_cell': False,
        'polymers': [{'volume_fraction': 1., 'blocks': [
            {'type': 'long_A', 'length': .5}, {'type': 'B2', 'length': .5}]}],
        'optimizer': {'name': 'am', 'max_hist': 20}, 'tolerance': 1e-8, 'max_iter': 5000,
    }


def run_pscf(binary, directory, dim, command='command'):
    result = subprocess.run([str(binary), '-d', str(dim), '-p', 'param', '-c', command],
                            cwd=directory, capture_output=True, text=True, encoding='utf-8', errors='replace',
                            timeout=90)
    (directory / 'pscf.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    assert result.returncode == 0, (directory / 'pscf.log').read_text(encoding='utf-8')[-6000:]
    # PSCF can exit successfully after a failed ITERATE and skip later commands.
    for name in ('omega', 'rho', 'thermo'):
        assert (directory / 'out' / name).is_file(), (directory / 'pscf.log').read_text(encoding='utf-8')[-6000:]


@pytest.mark.parametrize('backend', ('rpc', 'rpg'))
@pytest.mark.parametrize('case', CELLS)
def test_read_compute_write(tmp_path, case, backend):
    binary = executable(backend)
    dim, lattice, cell = CELLS[case]
    lx, angles = TOOL['lattice_geometry'](lattice, cell, dim)
    nx = {1: [16], 2: [8, 10], 3: [6, 8, 10]}[dim]
    p = base_params(nx, lx, angles)
    if case == 'branched':
        p['polymers'][0]['blocks'] = [
            {'type': 'long_A', 'length': .25, 'v': 19, 'u': 90},
            {'type': 'B2', 'length': .25, 'v': 90, 'u': 1234},
            {'type': 'long_A', 'length': .5, 'v': 90, 'u': 9999}]
    if case == 'mixture':
        p['polymers'][0]['volume_fraction'] = .6
        p['polymers'].append({'volume_fraction': .4, 'blocks': [{'type': 'long_A', 'length': .75}]})
    index = np.indices(nx, dtype=float)
    w = sum((5. / (axis + 1)) * np.cos(2 * np.pi * (index[axis] + .5) / n)
            for axis, n in enumerate(nx))
    source = tmp_path / 'initial.json'
    source.write_text(json.dumps({'params': p, 'w_long_A': w.tolist(), 'w_B2': (-w).tolist()}))
    directory = TOOL['export_files'](source, tmp_path / 'pscf', lattice=lattice)
    # COMPUTE isolates interoperability from convergence and cell optimization.
    (directory / 'compute').write_text(
        'READ_W_RGRID in/omega\nCOMPUTE\nWRITE_W_RGRID out/omega\n'
        'WRITE_C_RGRID out/rho\nWRITE_THERMO out/thermo\nFINISH\n')
    run_pscf(binary, directory, dim, 'compute')
    output = tmp_path / 'imported.mat'
    TOOL['import_files'](directory / 'out/omega', output, rho=directory / 'out/rho',
                         metadata=directory / 'metadata.json')
    imported = loadmat(output, simplify_cells=True)
    np.testing.assert_array_equal(np.atleast_1d(imported['nx']), nx)
    np.testing.assert_allclose(np.atleast_1d(imported['lx']), lx, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(imported['angles'], angles, rtol=0, atol=1e-8)
    np.testing.assert_allclose(imported['w_long_A'], w.ravel(), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(imported['w_B2'], -w.ravel(), rtol=1e-13, atol=1e-13)
    for name in ('long_A', 'B2'):
        # Each chain's integrated monomer amount follows its composition.
        expected = sum(polymer['volume_fraction'] *
                       sum(b['length'] for b in polymer['blocks'] if b['type'] == name) /
                       sum(b['length'] for b in polymer['blocks']) for polymer in p['polymers'])
        np.testing.assert_allclose(np.mean(imported['phi_' + name]), expected, rtol=0, atol=2e-10)


@pytest.mark.parametrize('backend', ('rpc', 'rpg'))
@pytest.mark.parametrize('case', ('1d', '2d', '3d'))
def test_generated_command_converges(tmp_path, case, backend):
    binary = executable(backend)
    dim = int(case[0])
    if dim == 3:
        # Exercise the documented example without modifying its input or commands.
        source = tmp_path / 'initial.mat'
        subprocess.run([sys.executable, str(ROOT / 'scripts/pscf/pscf_example.py'), '-o', str(source)],
                       check=True, capture_output=True, timeout=15)
    else:
        nx = [32] + [4] * (dim - 1)
        p = base_params(nx, [4.] + [3.] * (dim - 1), [90.] * 3)
        p['segment_lengths']['B2'] = 1.
        x = np.indices(nx, dtype=float)[0]
        w = np.cos(2 * np.pi * (x + .5) / nx[0])
        source = tmp_path / 'initial.json'
        source.write_text(json.dumps({'params': p, 'w_long_A': w.tolist(), 'w_B2': (-w).tolist()}))
    directory = TOOL['export_files'](source, tmp_path / 'pscf')
    run_pscf(binary, directory, dim)
    output = tmp_path / 'imported.mat'
    TOOL['import_files'](directory / 'out/omega', output, rho=directory / 'out/rho',
                         metadata=directory / 'metadata.json')
    imported = loadmat(output, simplify_cells=True)
    names = np.atleast_1d(imported['monomer_types'])
    total = sum(imported['phi_' + name] for name in names)
    assert np.isfinite(total).all()
    np.testing.assert_allclose(total, 1., rtol=0, atol=1e-7)
