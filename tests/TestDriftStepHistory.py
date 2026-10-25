#!/usr/bin/env python3
"""The largest CL-FTS drifts of every Langevin step (drift_max_step_history).

The distribution of the drift magnitude decides whether complex Langevin dynamics
converges correctly (Nagata, Nishimura & Shimasaki, PRD 94, 114515 (2016)); its rare
tail needs every step, not only the sampled ones:

- every step of a statistics interval is recorded, for each auxiliary field and for psi,
  and the sampled values of drift_max_history are the largest of their steps;
- a run continued from a checkpoint with pending steps reproduces the uninterrupted
  record bit for bit;
- the checkpoint check rejects a pending record of the wrong length or with invalid
  values, and accepts a checkpoint without it (earlier versions);
- the end of a run prints the summary of the distribution.
"""
import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from polymerfts import CLFTS, ValidationError
from polymerfts.utils import validate_checkpoint
from TestClftsElectrostatics import parameters as charged_parameters, initial_fields

M = 3   # monomer types A, B, C: three auxiliary fields


def parameters(directory, steps=6, electrostatics=True):
    params = charged_parameters(directory, steps, electrostatics)
    params["recording"].update(sample_interval=2, checkpoint_interval=3, statistics_save_interval=6)
    return params


def run(params, continue_from=None):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        sim = CLFTS(params, random_seed=5)
        if continue_from is None:
            sim.run(initial_fields=initial_fields())
        else:
            sim.continue_run(continue_from)
    return sim, output.getvalue()


def mat(path):
    return loadmat(path)   # without squeeze_me: (steps, fields) and (1, steps)


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _, output = run(parameters(root / "full"))
        full = mat(root / "full" / "dH_000006.mat")
        drift, psi = full["drift_max_step_history"], full["psi_drift_max_step_history"].ravel()
        assert drift.shape == (6, M) and psi.shape == (6,), (drift.shape, psi.shape)
        for values in (drift, psi):
            assert np.all(np.isfinite(values)) and np.all(values > 0), values
        # Samples at steps 2, 4 and 6 are the largest drifts of those steps
        assert np.array_equal(full["drift_max_history"].ravel(), drift.max(axis=1)[1::2])
        assert np.array_equal(full["psi_drift_max_history"].ravel(), psi[1::2])
        assert "Largest drift per step (6 steps of this run)" in output, output
        for label in ("auxiliary field 0", "auxiliary field 2", "psi:"):
            assert label in output, (label, output)

        # Without electrostatics only the auxiliary fields are recorded
        run(parameters(root / "neutral", electrostatics=False))
        neutral = mat(root / "neutral" / "dH_000006.mat")
        assert neutral["drift_max_step_history"].shape == (6, M) and "psi_drift_max_step_history" not in neutral

        # A continued run keeps the pending steps 1-3 of the checkpoint
        run(parameters(root / "part", steps=3))
        checkpoint = root / "part" / "fields_000003.mat"
        _, output = run(parameters(root / "part"), continue_from=checkpoint)
        assert "Largest drift per step (3 steps of this run)" in output, output
        part = mat(root / "part" / "dH_000006.mat")
        for name in ("drift_max_step_history", "psi_drift_max_step_history"):
            assert np.array_equal(full[name], part[name]), name

        data = loadmat(checkpoint, squeeze_me=True)
        with contextlib.redirect_stdout(io.StringIO()):
            sim = CLFTS(parameters(root / "check"), random_seed=5)
        validate_checkpoint(sim, data)
        assert np.size(data["record_drift_step_history"]) == 3 * M
        assert np.size(data["record_psi_drift_step_history"]) == 3
        for key in ("record_drift_step_history", "record_psi_drift_step_history"):
            for value in (data[key][:2], np.full(np.shape(data[key]), np.nan), -data[key]):
                modified = dict(data)
                modified[key] = value
                try:
                    validate_checkpoint(sim, modified)
                except ValidationError as error:
                    assert key in str(error), str(error)
                else:
                    raise AssertionError(f"A corrupt '{key}' was accepted")
            # Checkpoints of earlier versions lack the record; their continuation starts it anew
            modified = dict(data)
            modified.pop(key)
            validate_checkpoint(sim, modified)
    print("Drift step history: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
