"""scripts/analysis/drift_tail.py on synthetic records and on the drift of a complex Langevin toy
model with a known answer; no compiled backend or field simulation."""

import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/analysis/drift_tail.py'
TOOL = runpy.run_path(str(SCRIPT))


def write_records(directory, values, per_file, psi=None, ratios=None):
    """values: (steps, fields) as clfts.py records them; ratios: (steps, species)."""
    for start in range(0, len(values), per_file):
        step = start + per_file
        record = {'drift_max_step_history': values[start:step]}
        if psi is not None:
            record['psi_drift_max_step_history'] = psi[start:step]
        if ratios is not None:
            record['partition_ratio_step_history'] = ratios[start:step]
            record['partition_ratio_species_json'] = json.dumps([f'polymer {i} (S)' for i in range(ratios.shape[1])])
        savemat(directory / f'dH_{step:06d}.mat', record)


def test_records_are_read_in_step_order_and_skipped(tmp_path):
    values = np.column_stack([np.arange(1.0, 13.0), 10 * np.arange(1.0, 13.0)])
    write_records(tmp_path, values, 4, psi=-np.arange(1.0, 13.0))
    records = TOOL['read'](tmp_path, 5)
    assert list(records) == ['auxiliary field 0', 'auxiliary field 1', 'psi']
    assert np.array_equal(records['auxiliary field 0'], np.arange(6.0, 13.0))
    assert np.array_equal(records['auxiliary field 1'], 10 * np.arange(6.0, 13.0))
    assert np.array_equal(records['psi'], -np.arange(6.0, 13.0))


def test_power_law_and_exponential_tails(tmp_path):
    rng = np.random.default_rng(3)
    n = 400000
    # Pareto tail S(u) = u^-2 for u > 1, against an exponential tail
    _, _, _, power = TOOL['tail'](1 / np.sqrt(rng.uniform(size=n)))
    _, _, _, exponential = TOOL['tail'](rng.exponential(size=n))
    power = [s for s in power if s is not None]
    exponential = [s for s in exponential if s is not None]
    assert all(abs(s + 2) < 0.15 for s in power[1:]), power
    assert exponential[-1] < -4 and exponential[-1] < exponential[0], exponential


def one_pole(beta, walkers=1000, eps=1e-3, t_eq=2.0, t_meas=5.0, thin=10, seed=3):
    """Complex Langevin of rho(x) = (x - i)^2 exp(-beta x^2) (Aarts, Seiler, Sexty and Stamatescu,
    JHEP 05 (2017) 044, Sec. 3.1): Euler steps with real noise for independent walkers. Returns
    the means and standard errors of z^2 and of K and 2 + 2 z K, the Schwinger-Dyson combinations
    of z and z^2 (exact average 0), and the drift |K| of every thin-th step."""
    rng = np.random.default_rng(seed)
    z = 0.1 * rng.standard_normal(walkers) + 0j
    n_eq, n_meas = round(t_eq / eps), round(t_meas / eps)
    sums = np.zeros((3, walkers), dtype=complex)
    drift = []
    for step in range(n_eq + n_meas):
        K = 2 / (z - 1j) - 2 * beta * z
        if step >= n_eq:
            sums += np.array([z * z, K, 2 + 2 * z * K])
            if (step - n_eq) % thin == 0:
                drift.append(np.abs(K))
        z = z + eps * K + np.sqrt(2 * eps) * rng.standard_normal(walkers)
    means = sums / n_meas
    errors = (means.real.std(axis=1, ddof=1) + 1j * means.imag.std(axis=1, ddof=1)) / np.sqrt(walkers)
    return means.mean(axis=1), errors, np.concatenate(drift)


def test_one_pole_model():
    # beta = 1.6: the walkers reach the zero z = i of rho, the drift has a power-law tail and
    # Re <z^2> is wrong; beta = 4.8: they stay away from it and CL is correct. In both cases
    # Im <z^2> and the Schwinger-Dyson combinations vanish within their errors
    for beta, correct in ((1.6, False), (4.8, True)):
        mean, error, drift = one_pole(beta)
        exact = (3 / (4 * beta ** 2) - 1 / (2 * beta)) / (1 / (2 * beta) - 1)
        _, median, survival, slopes = TOOL['tail'](drift)
        assert abs(mean[0].imag) < 4 * error[0].imag, (beta, mean[0], error[0])
        for k in (1, 2):
            assert abs(mean[k].real) < 4 * error[k].real and abs(mean[k].imag) < 4 * error[k].imag, (beta, k, mean[k])
        if correct:
            assert abs(mean[0].real - exact) < 4 * error[0].real, (beta, mean[0], exact)
            assert survival(10 * median) == 0, survival(10 * median)
        else:
            assert abs(mean[0].real - exact) > 20 * error[0].real, (beta, mean[0], exact)
            assert survival(10 * median) > 1e-3, survival(10 * median)
            assert all(-3.5 < s < -1.5 for s in slopes[2:4]), slopes   # u from 3.16 to 31.6 times the median


def test_command_line(tmp_path):
    values = np.random.default_rng(4).exponential(size=(1000, 2)) + 0.1
    values[:, 1] = 0.0   # a field with zero mobility
    ratios = np.full((1000, 1), 0.5)
    ratios[:100] = np.nan          # steps of a checkpoint without the record
    ratios[600:610] = 0.05
    ratios[700:702] = 0.005
    write_records(tmp_path, values, 500, ratios=ratios)
    plot = tmp_path / 'tail.png'
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), '--plot', str(plot)],
                            capture_output=True, encoding='utf-8', errors='replace', check=True)
    assert 'auxiliary field 0: 1000 steps' in result.stdout, result.stdout
    assert 'auxiliary field 1: no drift' in result.stdout and plot.is_file()
    assert ('partition ratio of polymer 0 (S): 900 steps, minimum 0.005, steps below 0.1: 0.0133, '
            'below 0.01: 0.00222') in result.stdout, result.stdout
