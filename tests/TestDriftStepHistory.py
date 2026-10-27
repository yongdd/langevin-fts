#!/usr/bin/env python3
"""The largest CL-FTS drifts of every Langevin step (drift_max_step_history).

The distribution of the drift magnitude decides whether complex Langevin dynamics
converges correctly (Nagata, Nishimura & Shimasaki, PRD 94, 114515 (2016)); its rare
tail needs every step, not only the sampled ones:

- every step of a statistics interval is recorded, for each auxiliary field and for psi,
  and the sampled values of drift_max_history are the largest of their steps;
- a run continued from a checkpoint with pending steps reproduces the uninterrupted
  record bit for bit, also when the run was itself continued by hand or from a
  checkpoint without the record (the record then starts at its first step);
- the checkpoint check rejects a pending record of the wrong length or with invalid
  values, and accepts a checkpoint without it (earlier versions);
- the end of a run prints the summary of the distribution.
"""
import contextlib
import io
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from polymerfts import CLFTS, ValidationError
from polymerfts.utils import atomic_savemat, validate_checkpoint
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
        assert data["record_drift_step_start"] == 1
        for value in (0, 2, 5):
            modified = dict(data)
            modified["record_drift_step_start"] = value
            try:
                validate_checkpoint(sim, modified)
            except ValidationError as error:
                assert "record_drift_step_start" in str(error) or "drift_step_history" in str(error), str(error)
            else:
                raise AssertionError(f"A record start of {value} was accepted")

        # A run continued by hand records from its first step; its checkpoints before the next
        # statistics file continue it as the uninterrupted run does
        def hand_run(directory, steps, continue_from=None):
            params = parameters(directory, steps=steps)
            params["recording"]["checkpoint_interval"] = 2
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # psi restarts at zero
                sim = CLFTS(params, random_seed=5)
                if continue_from is None:
                    sim.run(initial_fields=initial_fields(), start_langevin_step=3)
                else:
                    sim.continue_run(continue_from)
        hand_run(root / "hand", 6)
        hand_run(root / "hand_part", 4)
        assert loadmat(root / "hand_part" / "fields_000004.mat")["record_drift_step_start"] == 3
        hand_run(root / "hand_part", 6, root / "hand_part" / "fields_000004.mat")
        hand, hand_part = mat(root / "hand" / "dH_000006.mat"), mat(root / "hand_part" / "dH_000006.mat")
        assert hand["drift_max_step_history"].shape == (4, M)
        for name in ("drift_max_step_history", "psi_drift_max_step_history"):
            assert np.array_equal(hand[name], hand_part[name]), name

        # So do the checkpoints of a run continued from a checkpoint of an earlier version
        params = parameters(root / "old", steps=2)
        params["recording"]["checkpoint_interval"] = 2
        run(params)
        old = {k: v for k, v in loadmat(root / "old" / "fields_000002.mat", squeeze_me=True).items()
               if not k.startswith("__") and k not in ("record_drift_step_history", "record_psi_drift_step_history",
                                                       "record_drift_step_start")}
        atomic_savemat(root / "old" / "old_000002.mat", old, long_field_names=True, do_compression=True)
        params["langevin"]["max_step"] = 4
        run(params, continue_from=root / "old" / "old_000002.mat")
        params["langevin"]["max_step"] = 6
        run(params, continue_from=root / "old" / "fields_000004.mat")
        assert mat(root / "old" / "dH_000006.mat")["drift_max_step_history"].shape == (4, M)
    print("Drift step history: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
