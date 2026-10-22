"""Import PSCF models and commands; exercise generated scripts without a backend."""

import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT / 'scripts/pscf/pscf_converter.py'
TOOL = runpy.run_path(str(CONVERTER))
PARAM = '''# Compact input with arbitrary whitespace and comments
System{
  polymerModel thread
  Mixture{ nMonomer 2 monomers[ 1.0 1.2 ]
    nPolymer 1
    Polymer{ type linear nBlock 2 blocks[ 0 0.4 1 0.6 ] phi 1 }
    ds 1.0D-2
  }
  Interaction{ chi( 0 1 20 ) }
  Domain{ mesh 2 3 4 lattice orthorhombic groupName 'P_1' }
  AmIteratorGrid{ epsilon 1e-7 maxItr 1234 maxHist 20 isFlexible 0 scaleStress 2 }
}
'''
COMMAND = '''READ_W_RGRID in/omega
ITERATE
WRITE_W_RGRID out/omega
WRITE_C_RGRID out/rho
WRITE_PARAM out/data
WRITE_THERMO out/data
FINISH
'''


def source_files(tmp_path, param=PARAM, command=COMMAND):
    source = tmp_path / 'source'
    (source / 'in').mkdir(parents=True)
    (source / 'param').write_text(param)
    (source / 'command').write_text(command)
    fields = np.arange(48, dtype=float).reshape(2, 2, 3, 4) + .125
    TOOL['write_rgrid'](source / 'in/omega', [2, 3, 4], 'orthorhombic', [3., 4., 5.], fields)
    return source, fields


def imported(tmp_path, param=PARAM, command=COMMAND, **kwargs):
    source, fields = source_files(tmp_path, param, command)
    output = TOOL['import_script'](source / 'param', source / 'command', tmp_path / 'job', **kwargs)
    return output, fields


def recording_solver(monkeypatch, converged=True):
    calls = {'saved': []}

    class SCFT:
        def __init__(self, params):
            calls['params'] = copy.deepcopy(params)
            calls['cwd'] = Path.cwd()

        def run(self, initial_fields, return_result):
            calls['fields'] = initial_fields
            assert return_result is True
            return SimpleNamespace(converged=converged)

        def save_results(self, path):
            calls['saved'].append(Path(path))
            Path(path).write_text('snapshot')

    fake = ModuleType('polymerfts')
    fake.scft = SimpleNamespace(SCFT=SCFT)
    monkeypatch.setitem(sys.modules, 'polymerfts', fake)
    for key in ('OMP_NUM_THREADS', 'OMP_MAX_ACTIVE_LEVELS', 'OMP_STACKSIZE'):
        monkeypatch.setenv(key, os.environ.get(key, '1'))
    return calls


def test_import_bundle_executes_independently_and_preserves_model(tmp_path, monkeypatch):
    output, fields = imported(tmp_path, monomers=['long_A', 'B_two'], platform='cpu-fftw')
    manifest = json.loads((output / 'conversion.json').read_text())
    assert manifest['pscf_group_name'] == 'P_1'
    assert [item['target'] for item in manifest['outputs']] == [
        'out/omega.mat', 'out/rho.mat', 'out/data.mat', 'out/data.mat']
    # No source field dependency at runtime, even from a different working directory.
    (tmp_path / 'source/in/omega').unlink()
    monkeypatch.chdir(tmp_path)
    calls = recording_solver(monkeypatch)
    runpy.run_path(str(output / 'simulation.py'), run_name='__main__')
    assert calls['cwd'] == output / 'results'
    p = calls['params']
    assert p['platform'] == 'cpu-fftw'
    assert p['nx'] == [2, 3, 4] and p['lx'] == [3., 4., 5.]
    assert p['segment_lengths'] == {'long_A': 1., 'B_two': 1.2}
    assert p['chi_n'] == {'long_A,B_two': 20.}
    assert p['polymers'] == [{'volume_fraction': 1., 'blocks': [
        {'type': 'long_A', 'length': .4}, {'type': 'B_two', 'length': .6}]}]
    assert p['chain_model'] == 'continuous' and p['numerical_method'] == 'rqm4'
    assert p['ds'] == .01 and p['max_iter'] == 1234 and p['tolerance'] == 1e-7
    assert p['optimizer']['max_hist'] == 20 and p['stress_scale'] == 2
    assert p['relax_cell'] is False and 'space_group' not in p
    for i, name in enumerate(('long_A', 'B_two')):
        np.testing.assert_array_equal(calls['fields'][name], fields[i])
    assert calls['saved'] == [output / 'results/out' / name for name in ('omega.mat', 'rho.mat', 'data.mat')]
    before = list(calls['saved'])
    with pytest.raises(FileExistsError):
        runpy.run_path(str(output / 'simulation.py'), run_name='__main__')
    assert calls['saved'] == before


def test_unconverged_calculation_does_not_save(tmp_path, monkeypatch):
    output, _ = imported(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls = recording_solver(monkeypatch, converged=False)
    with pytest.raises(RuntimeError, match='did not converge'):
        runpy.run_path(str(output / 'simulation.py'), run_name='__main__')
    assert calls['saved'] == []


def test_missing_write_commands_produces_default_snapshot(tmp_path, monkeypatch):
    output, _ = imported(tmp_path, command='READ_W_RGRID in/omega ITERATE FINISH')
    assert json.loads((output / 'conversion.json').read_text())['default_output'] == 'fields.mat'
    monkeypatch.chdir(tmp_path)
    calls = recording_solver(monkeypatch)
    runpy.run_path(str(output / 'simulation.py'), run_name='__main__')
    assert calls['saved'] == [output / 'results/fields.mat']
    assert 'platform' not in calls['params']


@pytest.mark.parametrize('command', (
    'READ_W_BASIS omega ITERATE FINISH', 'READ_W_RGRID in/omega COMPUTE FINISH',
    'READ_W_RGRID in/omega ITERATE SWEEP FINISH',
    'READ_W_RGRID in/omega ITERATE ITERATE FINISH',
    'READ_W_RGRID in/omega ITERATE READ_W_RGRID in/omega FINISH',
    'READ_W_RGRID in/omega WRITE_THERMO out/data ITERATE FINISH',
    'READ_W_RGRID in/omega ITERATE WRITE_W_BASIS out/w FINISH',
    'READ_W_RGRID in/omega ITERATE WRITE_STRESS out/stress FINISH',
    'READ_W_RGRID in/omega SET_UNIT_CELL 4 5 6 ITERATE FINISH',
    'READ_W_RGRID in/omega ITERATE', 'READ_W_RGRID',
    'READ_W_RGRID in/omega ITERATE WRITE_THERMO',
    'READ_W_RGRID in/omega ITERATE FINISH ITERATE', '',
    'ITERATE FINISH', 'READ_W_RGRID in/omega ITERATE 1 FINISH',
))
def test_unsupported_or_malformed_commands_leave_no_bundle(tmp_path, command):
    source, _ = source_files(tmp_path, command=command)
    with pytest.raises(ValueError):
        TOOL['import_script'](source / 'param', source / 'command', tmp_path / 'job')
    assert not (tmp_path / 'job').exists()


@pytest.mark.parametrize('path', ('../outside', '/tmp/outside', '.', 'out/../omega', 'out/', 'out\\omega'))
def test_output_paths_cannot_escape_or_name_a_directory(tmp_path, path):
    with pytest.raises(ValueError, match='relative file path'):
        imported(tmp_path, command=f'READ_W_RGRID in/omega ITERATE WRITE_W_RGRID "{path}" FINISH')
    assert not (tmp_path / 'job').exists()


def test_conflicting_output_paths_are_rejected():
    with pytest.raises(ValueError, match='conflict'):
        TOOL['script_commands']('READ_W_RGRID w ITERATE WRITE_W_RGRID x WRITE_C_RGRID x.mat/y FINISH')


@pytest.mark.parametrize('old,new', (
    ('polymerModel thread', 'polymerModel bead'),
    ('Mixture{', 'Environment{} Mixture{'),
    ('ds 1.0D-2', 'ds .01 nSolvent 1'),
    ('ds 1.0D-2', 'ds .01 vMonomer 2'),
    ('phi 1', 'mu 0'),
    ('type linear', 'type random'),
    ('AmIteratorGrid{', 'AmIteratorBasis{'),
    ('epsilon 1e-7', 'epsilon 1e-7 errorType maxAbsResid'),
    ('maxHist 20', 'maxHist 20 lambda .5'),
    ('maxHist 20', 'maxHist 20 maxHist 30'),
    ('epsilon 1e-7', 'epsilon 0'),
    ('maxItr 1234', 'maxItr 1.5'),
    ('scaleStress 2', 'scaleStress nan'),
    ('isFlexible 0', 'isFlexible 2'),
    ('nMonomer 2', 'nMonomer 3'),
    ('monomers[ 1.0 1.2 ]', 'monomers[ 1.0 ]'),
    ('nPolymer 1', 'nPolymer 2'),
    ('nBlock 2', 'nBlock 3'),
    ('0 0.4 1 0.6', '0 0.4 2 0.6'),
    ('0 0.4 1 0.6', '0 -0.4 1 0.6'),
    ('phi 1', 'phi .8'),
    ('chi( 0 1 20 )', 'chi( 0 1 20 1 0 20 )'),
    ('chi( 0 1 20 )', 'chi( 0 0 20 )'),
    ('chi( 0 1 20 )', 'chi( 0 1 )'),
    ('mesh 2 3 4', 'mesh 4 3 2'),
    ('lattice orthorhombic', 'lattice cubic'),
    ('isFlexible 0', 'isFlexible 1 flexibleParams[ 1 0 1 ]'),
    ('isFlexible 0', 'isFlexible 1 flexibleParams[ 1 1 ]'),
    ('isFlexible 0', 'isFlexible 0 flexibleParams[ 0 0 0 ]'),
))
def test_unsupported_or_inconsistent_models_leave_no_bundle(tmp_path, old, new):
    with pytest.raises(ValueError):
        imported(tmp_path, param=PARAM.replace(old, new))
    assert not (tmp_path / 'job').exists()


@pytest.mark.parametrize('text', ('System{', 'System{ Mixture[ 1 } }', 'System{} trailing',
                                 'System{ ds }', 'System{ mesh[ [ 1 ] ] }'))
def test_malformed_parameter_syntax(text):
    with pytest.raises(ValueError):
        TOOL['parse_param'](text)


@pytest.mark.parametrize('flags,expected', (('1 1 1', True), ('0 0 0', False)))
def test_all_or_no_cell_parameters_may_relax(tmp_path, flags, expected):
    text = PARAM.replace('isFlexible 0', f'isFlexible 1 flexibleParams[ {flags} ]')
    output, _ = imported(tmp_path, param=text)
    assert TOOL['read_python_params'](output / 'simulation.py')['relax_cell'] is expected


def test_pscf_defaults_are_explicit_in_generated_script(tmp_path):
    text = PARAM.replace('maxItr 1234 maxHist 20 isFlexible 0 scaleStress 2', '')
    output, _ = imported(tmp_path, param=text)
    p = TOOL['read_python_params'](output / 'simulation.py')
    assert p['relax_cell'] is True
    assert p['max_iter'] == 200 and p['optimizer']['max_hist'] == 50 and p['stress_scale'] == 10.


def test_branched_tree_and_multiple_polymer_species(tmp_path):
    text = PARAM.replace('nPolymer 1', 'nPolymer 2').replace(
        'type linear nBlock 2 blocks[ 0 0.4 1 0.6 ] phi 1',
        'type branched nBlock 3 blocks[ 0 .2 0 1 1 .3 1 2 0 .5 1 3 ] phi .6')
    text = text.replace('ds 1.0D-2', 'Polymer{ nBlock 1 blocks[ 1 1.5 ] phi .4 } ds 1.0D-2')
    output, _ = imported(tmp_path, param=text)
    p = TOOL['read_python_params'](output / 'simulation.py')
    assert len(p['polymers']) == 2
    assert p['polymers'][0]['blocks'][2] == {'type': 'A', 'length': .5, 'v': 1, 'u': 3}
    assert p['polymers'][1]['blocks'] == [{'type': 'B', 'length': 1.5}]


@pytest.mark.parametrize('rows', ('0 .2 0 1 1 .3 1 2 0 .5 2 0', '0 .2 0 1 1 .3 0 1 0 .5 2 3',
                                  '0 .2 0 0 1 .3 1 2 0 .5 2 3'))
def test_invalid_branched_graph_is_rejected(tmp_path, rows):
    text = PARAM.replace('type linear nBlock 2 blocks[ 0 0.4 1 0.6 ]', f'type branched nBlock 3 blocks[ {rows} ]')
    with pytest.raises(ValueError, match='tree'):
        imported(tmp_path, param=text)


def test_input_working_directory_can_differ_from_parameter_location(tmp_path):
    source, fields = source_files(tmp_path)
    other = tmp_path / 'elsewhere'
    other.mkdir()
    (source / 'param').rename(other / 'param')
    output = TOOL['import_script'](other / 'param', source / 'command', tmp_path / 'job', input_dir=source)
    np.testing.assert_array_equal(loadmat(output / 'initial.mat')['w_A'].ravel(), fields[0].ravel())


def test_cli_and_column_metadata(tmp_path):
    source, fields = source_files(tmp_path)
    metadata = source / 'metadata.json'
    metadata.write_text(json.dumps(dict(format='polymerfts-pscf-rgrid-1', nx=[2, 3, 4],
                                        lattice='orthorhombic', monomer_types=['X', 'Y'], chain_model='continuous')))
    output = tmp_path / 'job'
    command = [sys.executable, str(CONVERTER), 'import-script', str(source / 'param'), '--command',
               str(source / 'command'), '--metadata', str(metadata), '-o', str(output), '--platform', 'cpu-mkl']
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(loadmat(output / 'initial.mat')['w_X'].ravel(), fields[0].ravel())
    marker = output / 'keep'
    marker.write_text('original')
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and 'Error:' in result.stderr and 'Traceback' not in result.stderr
    assert marker.read_text() == 'original'


@pytest.mark.parametrize('metadata', ({'chain_model': 'discrete'}, {'monomer_types': ['A', 'A']},
                                     {'monomer_types': ['A,B', 'C']}, {'monomer_types': ['A', 'B2']}))
def test_incompatible_metadata_is_rejected(tmp_path, metadata):
    source, _ = source_files(tmp_path)
    path = source / 'metadata.json'
    info = dict(format='polymerfts-pscf-rgrid-1', nx=[2, 3, 4], lattice='orthorhombic', monomer_types=['A', 'B'])
    info.update(metadata)
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError):
        TOOL['import_script'](source / 'param', source / 'command', tmp_path / 'job', metadata=path)
    assert not (tmp_path / 'job').exists()


CELLS = runpy.run_path(str(ROOT / 'tests/scripts/TestPscfIntegration.py'))['CELLS']


@pytest.mark.parametrize('case', [name for name in CELLS if name not in ('branched', 'mixture')])
@pytest.mark.parametrize('flexible', (False, True))
def test_cell_metric_and_representable_relaxation(tmp_path, case, flexible):
    dim, lattice, cell = CELLS[case]
    source, _ = source_files(tmp_path)
    nx = [4] * dim
    text = PARAM.replace('mesh 2 3 4', 'mesh ' + ' '.join(map(str, nx)))
    text = text.replace('lattice orthorhombic', f'lattice {lattice}')
    text = text.replace('isFlexible 0', f'isFlexible {int(flexible)}')
    (source / 'param').write_text(text)
    (source / 'in/omega').unlink()
    TOOL['write_rgrid'](source / 'in/omega', nx, lattice, cell, np.zeros((2, *nx)))
    if flexible and ((dim == 2 and lattice in ('square', 'rhombic')) or lattice == 'rhombohedral'):
        with pytest.raises(ValueError, match='cannot preserve their constraints'):
            TOOL['import_script'](source / 'param', source / 'command', tmp_path / 'job')
        assert not (tmp_path / 'job').exists()
    else:
        output = TOOL['import_script'](source / 'param', source / 'command', tmp_path / 'job')
        p = TOOL['read_python_params'](output / 'simulation.py')
        lattice_back, cell_back = TOOL['choose_lattice'](p, override=lattice)
        assert lattice_back == lattice and p['relax_cell'] is flexible
        np.testing.assert_allclose(cell_back, cell, rtol=0, atol=1e-12)
