"""z_inf and a_b/a of scripts/analysis/renormalization.py; no compiled backend or simulation."""

import math
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import savemat
from scipy.special import erf


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/analysis/renormalization.py'
TOOL = runpy.run_path(str(SCRIPT))
z_inf, bare_segment_ratio = TOOL['z_inf'], TOOL['bare_segment_ratio']


def bond_sum_z_inf(m, box, n, nbar, bonds=2000):
    """Beardsley and Matsen, J. Chem. Phys. 150, 174902 (2019): incompressible, contact.

    m and box are numbers (cubic box) or one value per axis.
    """
    dxs = [b / k for k, b in zip(np.broadcast_to(m, 3), np.broadcast_to(box, 3))]
    dv = math.prod(dxs)

    def p(i):
        return math.prod(dx * n ** .5 * (3 / (2 * math.pi * i)) ** .5 * erf(math.pi / dx / n ** .5 * (i / 6) ** .5)
                         for dx in dxs)
    tail = 2 / math.sqrt(.5 + bonds) * (3 * n / (2 * math.pi)) ** 1.5 * dv
    return 1 - (1 + 2 * (sum(p(i) for i in range(1, bonds + 1)) + tail)) / (math.sqrt(nbar) * n * dv)


def test_discrete_reduces_to_bond_sum():
    value = z_inf('discrete', [32] * 3, [4.5] * 3, 1e5, 90)
    assert value == pytest.approx(bond_sum_z_inf(32, 4.5, 90, 1e5), abs=5e-7)


@pytest.mark.parametrize('m, box, n, nbar', [(32, 4.5, 90, 1e5), (32, 4., 20, 1e4), (64, 4., 16, 1e4),
                                             (16, 4., 200, 1e4), ([48, 32, 24], [5., 3.5, 2.8], 50, 1e4)])
def test_singular_term_is_exact(m, box, n, nbar):
    # The 1/k^2 term is integrated exactly: the default quadrature reproduces the converged bond
    # sum, and a_b/a, whose integrand approaches 2/k^2, does not change under refinement
    nx, lx = [int(k) for k in np.broadcast_to(m, 3)], [float(b) for b in np.broadcast_to(box, 3)]
    assert z_inf('discrete', nx, lx, nbar, n) == pytest.approx(bond_sum_z_inf(m, box, n, nbar, bonds=20000), abs=1e-11)
    ratio = bare_segment_ratio('discrete', nx, lx, nbar, n)
    assert bare_segment_ratio('discrete', nx, lx, nbar, n, n_quad=128) == pytest.approx(ratio, abs=1e-12)


@pytest.mark.parametrize('m, z_read, ratio_read', [(32, .955, .9930), (40, .945, .9920), (48, .934, .9913)])
def test_paper_figure_2(m, z_read, ratio_read):
    # Matsen et al., J. Chem. Phys. 164, 014905 (2026), Fig. 2 at a_int = 0 (read from the figure):
    # N = 90, zeta = 1, Nbar = 1e5, L = 4.5 R0
    args = ('discrete', [m] * 3, [4.5] * 3, 1e5, 90, 90)
    assert z_inf(*args) == pytest.approx(z_read, abs=2e-3)
    assert bare_segment_ratio(*args) == pytest.approx(ratio_read, abs=5e-4)


@pytest.mark.parametrize('zeta_n', [None, 30.])
@pytest.mark.parametrize('smearing', [None, {'type': 'gaussian', 'a_int': .05}])
def test_continuous_is_limit_of_discrete(zeta_n, smearing):
    args = ([16] * 3, [4.] * 3, 1e4)
    for function in (z_inf, bare_segment_ratio):
        continuous = function('continuous', *args, None, zeta_n, smearing, n_quad=64)
        assert function('discrete', *args, 10 ** 8, zeta_n, smearing, n_quad=64) == pytest.approx(continuous, abs=1e-8)
        # Finite chains renormalize more: tanh(u) < u and sinh(u) > u
        finite = function('discrete', *args, 20, zeta_n, smearing, n_quad=64)
        assert finite < continuous if function is z_inf else finite > continuous


@pytest.mark.parametrize('model, n', [('discrete', 90), ('continuous', None)])
@pytest.mark.parametrize('smearing', [None, {'type': 'gaussian', 'a_int': .05},
                                      {'type': 'sigmoidal', 'k_int': 20., 'dk_int': 5.}])
def test_incompressible_limit_and_smearing(model, n, smearing):
    args = (model, [16] * 3, [4.] * 3, 1e4, n)
    for function in (z_inf, bare_segment_ratio):
        base = function(*args, None, smearing, n_quad=64)
        assert function(*args, 1e12, smearing, n_quad=64) == pytest.approx(base, abs=1e-8)
        assert function(*args, math.inf, smearing, n_quad=64) == base
        # Compressibility weakens both corrections
        assert function(*args, 10., smearing, n_quad=64) > base
    # Smearing weakens the renormalization of chi
    if smearing:
        assert z_inf(*args, None, smearing, n_quad=64) > z_inf(*args, None, None, n_quad=64)


def test_smearing_function():
    k2 = np.array([0., 1., 100.])
    gaussian = TOOL['smearing_function'](k2, {'type': 'gaussian', 'a_int': .1})
    assert np.allclose(gaussian, np.exp(-.01 * k2 / 2))
    sigmoidal = TOOL['smearing_function'](k2, {'type': 'sigmoidal', 'k_int': 20.})
    assert sigmoidal[0] == pytest.approx(1.) and np.all(np.diff(sigmoidal) < 0)
    with pytest.raises(ValueError, match='one number'):
        TOOL['smearing_function'](k2, {'type': 'gaussian', 'a_int': {'A': .1, 'B': .1}})
    with pytest.raises(ValueError, match='must be a number'):
        TOOL['smearing_function'](k2, {'type': 'gaussian'})


def write_fields(path, chain_model='discrete', segment_length=.99):
    params = {'nx': [16, 16, 16], 'lx': [4., 4., 4.], 'chain_model': chain_model, 'ds': 1 / 50,
              'segment_lengths': {'A': segment_length, 'B': .99}, 'chi_n': {'A,B': 12.}, 'zeta_n': 50.,
              'polymers': [{'volume_fraction': 1., 'blocks': [{'type': 'A', 'length': .4},
                                                             {'type': 'B', 'length': .6}]}],
              'smearing': {'type': 'gaussian', 'a_int': .05}, 'langevin': {'nbar': 1e4, 'dt': 1.}}
    savemat(path, {'initial_params': params, 'nx': params['nx'], 'lx': params['lx'], 'chain_model': chain_model,
                   'ds': params['ds'], 'nbar': 1e4, 'chi_n': params['chi_n']})
    return params


def run_cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


@pytest.mark.parametrize('chain_model', ['discrete', 'continuous'])
def test_cli_reads_fields_file(tmp_path, chain_model):
    p = write_fields(tmp_path / 'fields.mat', chain_model)
    out = run_cli(tmp_path / 'fields.mat', '--target', 10, '--quadrature', 64)
    assert out.returncode == 0, out.stderr
    settings = (chain_model, p['nx'], p['lx'], 1e4, 50, 50., p['smearing'])
    z = z_inf(*settings, n_quad=64)
    assert f'z_inf = {z:.7f}' in out.stdout
    assert f'a_b/a = {bare_segment_ratio(*settings, n_quad=64):.7f}' in out.stdout
    assert f'z_inf chi_b N = {12 * z:.7f}' in out.stdout
    assert f'z_inf chi_b N = 10: {10 / z:.7f}' in out.stdout
    # Options override the file: no smearing, incompressible
    plain = run_cli(tmp_path / 'fields.mat', '--a-int', 0, '--zeta-n', 'inf', '--quadrature', 64)
    assert f"z_inf = {z_inf(chain_model, p['nx'], p['lx'], 1e4, 50, n_quad=64):.7f}" in plain.stdout


def test_cli_errors(tmp_path):
    write_fields(tmp_path / 'fields.mat', segment_length=1.2)
    out = run_cli(tmp_path / 'fields.mat')
    assert out.returncode == 2 and 'conformationally symmetric' in out.stderr
    out = run_cli('--chain-model', 'discrete', '--nx', 8, 8, 8, '--lx', 4, 4, 4, '--nbar', 1e4)
    assert out.returncode == 2 and 'number of segments' in out.stderr
    out = run_cli('--chain-model', 'continuous', '--nx', 8, 8, 8, '--lx', 4, 4, 4)
    assert out.returncode == 2 and '--nbar' in out.stderr
    # Per-type smearing lengths of a CL-FTS fields file
    p = write_fields(tmp_path / 'fields.mat', segment_length=.99)
    p['smearing']['a_int'] = {'A': .05, 'B': .08}
    savemat(tmp_path / 'per_type.mat', {'initial_params': p, 'nx': p['nx'], 'lx': p['lx'],
                                        'chain_model': 'discrete', 'ds': p['ds'], 'nbar': 1e4})
    out = run_cli(tmp_path / 'per_type.mat', '--quadrature', 16)
    assert out.returncode == 2 and 'one number' in out.stderr
