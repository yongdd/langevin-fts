"""Execute the notebook's aggregation cell against independently known full spectra."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat


NOTEBOOK = Path(__file__).resolve().parents[2] / 'scripts/analysis/plot_structure_function.ipynb'
CELL = ''.join(json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells'][0]['source'])


def aggregate(tmp_path, monkeypatch, spectra, counts=None):
    """Write exactly the four windows selected by the notebook, then run its actual code."""
    directory = tmp_path / 'data_simulation'
    directory.mkdir()
    for step, full, count in zip(range(100000, 500000, 100000), spectra, counts or [None] * 4):
        nx = full.shape
        data = {'nx': nx, 'lx': [1., 1., 1.],
                'structure_function_A_A': full[:, :, :nx[2] // 2 + 1]}
        if count is not None:
            data['sample_count'] = count
        savemat(directory / f'structure_function_{step:06d}.mat', data)
    monkeypatch.chdir(tmp_path)
    result = {}
    exec(compile(CELL, str(NOTEBOOK) + ':cell0', 'exec'), result)
    return result['k2_unique'], result['y']


@pytest.mark.parametrize('nx', ([4, 4, 4], [5, 5, 5], [4, 5, 6]))
def test_half_spectrum_matches_full_fourier_shells(tmp_path, monkeypatch, nx):
    # A real field gives a valid Hermitian spectrum, including all Nyquist planes.
    field = np.random.default_rng(193).normal(size=nx)
    full = np.abs(np.fft.fftn(field)) ** 2
    k2, observed = aggregate(tmp_path, monkeypatch, [full] * 4, [3] * 4)
    modes = np.meshgrid(*(np.fft.fftfreq(n) * n for n in nx), indexing='ij')
    full_k2 = np.round(sum(k * k for k in modes), 7)
    expected = np.array([np.mean(full[full_k2 == shell]) for shell in k2])
    expected[0] = 0.
    np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize('axis', [0, 2])
def test_axis_rotation_does_not_change_shell_average(tmp_path, monkeypatch, axis):
    full = np.zeros((8, 8, 8))
    for dimension in range(3):
        for sign in (-1, 1):
            index = [0, 0, 0]
            index[dimension] = sign
            full[tuple(index)] = 10. if dimension == axis else 1.
    k2, mean = aggregate(tmp_path, monkeypatch, [full] * 4)
    assert mean[k2 == 1.].item() == pytest.approx(4.)


def test_measurement_windows_are_weighted_by_sample_count(tmp_path, monkeypatch):
    # sample_interval=30000 and statistics_save_interval=100000 give these counts.
    spectra = [np.full((4, 4, 4), value) for value in (1., 1., 10., 1.)]
    k2, mean = aggregate(tmp_path, monkeypatch, spectra, [3, 3, 4, 3])
    np.testing.assert_allclose(mean[k2 > 0], 49 / 13)


def test_legacy_windows_keep_equal_file_weights(tmp_path, monkeypatch):
    spectra = [np.full((4, 4, 4), value) for value in (1., 1., 10., 1.)]
    k2, mean = aggregate(tmp_path, monkeypatch, spectra)
    np.testing.assert_allclose(mean[k2 > 0], 3.25)


def test_legacy_and_counted_windows_cannot_be_mixed(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='with and without sample_count'):
        aggregate(tmp_path, monkeypatch, [np.ones((4, 4, 4))] * 4, [3, None, 4, 3])


def test_empty_windows_are_skipped(tmp_path, monkeypatch):
    spectra = [np.full((4, 4, 4), value) for value in (np.nan, 1., np.nan, 4.)]
    k2, mean = aggregate(tmp_path, monkeypatch, spectra, [0, 2, 0, 4])
    np.testing.assert_allclose(mean[k2 > 0], 3.)


def test_all_empty_windows_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='contain no samples'):
        aggregate(tmp_path, monkeypatch, [np.zeros((4, 4, 4))] * 4, [0] * 4)


@pytest.mark.parametrize('count', [-1, .5, np.nan, np.inf, [1, 2], 1 + 1j])
def test_invalid_sample_counts_are_rejected(tmp_path, monkeypatch, count):
    with pytest.raises(ValueError, match='finite nonnegative integer'):
        aggregate(tmp_path, monkeypatch, [np.ones((4, 4, 4))] * 4, [count, 1, 1, 1])
