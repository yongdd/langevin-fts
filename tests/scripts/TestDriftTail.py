"""scripts/analysis/drift_tail.py on synthetic records; no compiled backend or simulation."""

from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/analysis/drift_tail.py'
TOOL = runpy.run_path(str(SCRIPT))


def write_records(directory, values, per_file, psi=None):
    """values: (steps, fields) as clfts.py records them."""
    for start in range(0, len(values), per_file):
        step = start + per_file
        record = {'drift_max_step_history': values[start:step]}
        if psi is not None:
            record['psi_drift_max_step_history'] = psi[start:step]
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


def test_command_line(tmp_path):
    values = np.random.default_rng(4).exponential(size=(1000, 2)) + 0.1
    values[:, 1] = 0.0   # a field with zero mobility
    write_records(tmp_path, values, 500)
    plot = tmp_path / 'tail.png'
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), '--plot', str(plot)],
                            capture_output=True, encoding='utf-8', errors='replace', check=True)
    assert 'auxiliary field 0: 1000 steps' in result.stdout, result.stdout
    assert 'auxiliary field 1: no drift' in result.stdout and plot.is_file()
