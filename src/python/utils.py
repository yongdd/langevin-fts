"""Shared utilities for polymer field theory simulations.

This module provides common utility functions and constants used across
the SCFT and LFTS simulation classes to reduce code duplication.
"""

import ast
import copy
import json
import operator
import itertools
import warnings
import hashlib
import os
import platform
import tempfile
import subprocess
from functools import lru_cache
from types import MappingProxyType
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import yaml

from .validation import normalize_params, ValidationError, parse_chi_n  # noqa: F401 (parse_chi_n: former utils API)


def json_ready(value):
    """Convert parameter dictionaries to portable JSON values."""
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_float": str(value)}
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def mat_ready(value):
    """Convert a parameter dictionary for scipy.io.savemat, which has no None: an option
    given as None (its default) is left out, and paths become strings."""
    if isinstance(value, dict):
        return {str(k): mat_ready(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [np.empty(0) if v is None else mat_ready(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def settings_json(settings):
    return json.dumps(json_ready(settings), sort_keys=True, separators=(",", ":"), allow_nan=False)


def input_hash(value):
    return hashlib.sha256(settings_json(value).encode()).hexdigest()


def fields_hash(fields):
    digest = hashlib.sha256()
    for name in sorted(fields):
        array = np.ascontiguousarray(fields[name])
        digest.update(settings_json([name, array.shape, array.dtype.str]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _package_hashes():
    from . import _core
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    native = hashlib.sha256()
    with open(_core.__file__, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            native.update(chunk)
    return {"python_source_sha256": source.hexdigest(), "native_binary_sha256": native.hexdigest()}


def atomic_savemat(path, data, **kwargs):
    """Replace a MAT file only after a complete write in the same directory."""
    from scipy.io import savemat
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".mat", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            savemat(stream, data, **kwargs)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_run_status(sim):
    """Persist completion/failure separately from periodic field checkpoints."""
    path = Path(sim.recording["dir"])
    path.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".run_status.", dir=path)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(json_ready({**sim.run_status, "metadata": sim.metadata}), stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path / "run_status.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def report_partition_consistency(prop_solver):
    """Print the forward/backward partition error of each polymer; True if all are within tolerance."""
    from . import _core
    errors = prop_solver.partition_errors()
    print("Relative contour partition error of each polymer: " +
          ", ".join(f"{p}: {error:.3e}" for p, error in enumerate(errors)))
    return all(np.isfinite(error) and error <= _core.PARTITION_TOLERANCE for error in errors)


def check_partition_functions(prop_solver, state):
    """Reject an accepted state whose partition functions are not positive and finite.

    With real fields Q is a sum of positive Boltzmann weights. Intermediate iterates can
    still give Q <= 0 (the Richardson extrapolation of rqm4 or Crank-Nicolson steps on a
    coarse grid in strong fields), so only accepted states are checked: the final SCFT
    fields and an L-FTS saddle point whose Hamiltonian, which contains log Q, is not
    finite. Complex fields (CL-FTS) have no sign to check.
    """
    if prop_solver.field_type != "real":
        return
    for p in range(prop_solver.get_n_polymer_types()):
        Q = prop_solver.get_partition_function(p)
        if not (np.isfinite(Q) and Q > 0):
            raise ValidationError(
                f"The partition function of polymer {p} is {Q:.6e} at {state}, but with real fields it must "
                "be positive and finite. Either the discretization is too coarse for the fields (refine the "
                "grid or the contour step), or the propagators fell below the round-off level of the spectral "
                "transforms, which is relative to their largest value on the grid (for example strong fields "
                "next to absorbing walls or a grafted source).")


def simulation_metadata(sim, random_seed=None):
    """Build provenance comes from the loaded extension, never the current cwd."""
    import scipy
    from . import _core
    solver = sim.prop_solver
    metadata = {
        "schema_version": 1, "build": dict(getattr(_core, "build_info", {})),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "system": platform.platform(), "machine": platform.machine(),
        "backend": solver.platform, "numerical_method": solver.numerical_method,
        "chain_model": solver.chain_model, "precision": "complex128" if solver.field_type == "complex" else "float64",
        "reduce_memory": solver.reduce_memory,
        "threads": {key: os.environ.get(key, "") for key in
                    ("OMP_NUM_THREADS", "OMP_MAX_ACTIVE_LEVELS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS")},
        "random_seed": random_seed, "input_hash": input_hash(sim.params),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        **_package_hashes(),
    }
    try:
        metadata["cpu"] = next(line.split(":", 1)[1].strip() for line in
                               Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except (OSError, StopIteration):
        metadata["cpu"] = platform.processor()
    if solver.platform == "cuda":
        command = ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            command.append("--id=" + visible)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
            metadata["gpu"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            metadata["gpu"] = "unavailable"
    if hasattr(_core, "runtime_info"):
        metadata["libraries"] = _core.runtime_info(solver.platform == "cuda")
    return json_ready(metadata)


def checkpoint_settings(sim):
    """Effective settings that define a continuation; output paths/max_step may change."""
    solver = sim.prop_solver
    result = {
        "simulation": type(sim).__name__ if type(sim).__name__ in ("LFTS", "CLFTS") else
                      ("CLFTS" if solver.field_type == "complex" else "LFTS"),
        "nx": solver.nx, "lx": solver.get_lx(), "angles": solver.get_angles_degrees(),
        "bc": solver.bc, "ds": sim.ds, "chain_model": sim.chain_model,
        "numerical_method": solver.numerical_method, "segment_lengths": sim.segment_lengths,
        "polymers": sim.polymers, "chi_n": sim.chi_n, "zeta_n": sim.zeta_n,
        "dt": sim.langevin["dt"], "nbar": sim.langevin["nbar"], "dt_scaling": sim.dt_scaling,
        "smearing": {"type": sim.smearing.type, "params": sim.smearing.params},
        # None without electrostatics, which earlier checkpoints also give
        "electrostatics": None if getattr(sim, "electrostatics", None) is None else sim.electrostatics.settings(),
        "sample_interval": sim.recording["sample_interval"],
        "statistics_save_interval": sim.recording["statistics_save_interval"],
        "extensions": extension_settings(sim),
    }
    if hasattr(sim, "saddle"):
        result["saddle"] = {**sim.saddle, "failure_policy": sim.saddle.get("failure_policy", "retry"),
                            "max_retries": sim.saddle.get("max_retries", 5)}
        result["compressor"] = sim.params["compressor"]
        wtmd = sim.wtmd
        result["wtmd"] = None if wtmd is None else {
            key: getattr(wtmd, key) for key in
            ("l", "kc", "dT", "sigma_psi", "psi_min", "psi_max", "dpsi", "bins", "update_freq")}
    else:
        result["dynamic_stabilization"] = sim.alpha_ds
    return json_ready(result)


def checkpoint_data(sim):
    """Version, compatibility settings and complete RNG/measurement state."""
    settings = checkpoint_settings(sim)
    data = {"checkpoint_format_version": 2, "checkpoint_settings_json": settings_json(settings),
            "checkpoint_settings_hash": input_hash(settings),
            "rng_state_json": json.dumps(sim.random_bg.state),
            "simulation_metadata_json": json.dumps(json_ready(sim.metadata)),
            "run_status_json": json.dumps(json_ready(sim.run_status))}
    data["extension_state_json"] = json.dumps(extension_state(sim), allow_nan=False)
    state = getattr(sim, "_recording_state", None)
    if state is None:
        raise ValidationError("Checkpointing requires the recording state of an accepted run step")
    if state is not None:
        data["record_H_history"] = np.asarray(state["H_history"])
        data["record_drift_history"] = np.asarray(state.get("drift_history", []))
        if "psi_drift_history" in state:
            data["record_psi_drift_history"] = np.asarray(state["psi_drift_history"])
        # CL-FTS: the drifts of every step since the last statistics file (per auxiliary field)
        if "drift_step_history" in state:
            data["record_drift_step_history"] = np.reshape(np.asarray(state["drift_step_history"], dtype=np.float64),
                                                           (-1, len(sim.monomer_types)))
            data["record_drift_step_start"] = int(state["drift_step_start"])
        if "psi_drift_step_history" in state:
            data["record_psi_drift_step_history"] = np.asarray(state["psi_drift_step_history"], dtype=np.float64)
        # CL-FTS: the chemical potentials (one row of all species per sample) and the pressure
        if "chemical_potential_history" in state:
            data["record_chemical_potential_history"] = np.reshape(
                np.asarray(state["chemical_potential_history"], dtype=np.complex128), (-1, len(sim.chain_densities)))
        if "pressure_history" in state:
            data["record_pressure_history"] = np.asarray(state["pressure_history"], dtype=np.complex128)
        data["record_sample_count"] = state["sample_count"]
        for key, values in state["dH_history"].items():
            data[_measurement_key("dH", key)] = np.asarray(values)
        for key, values in state["sf_average"].items():
            data[_measurement_key("sf", key)] = values
    return data


def _measurement_key(kind, pair):
    # Monomer names may contain underscores; joining names with '_' is ambiguous.
    return "record_" + kind + "_" + hashlib.sha256(pair.encode()).hexdigest()[:32]


def validate_checkpoint(sim, data, *, allow_legacy=False):
    version = data.get("checkpoint_format_version", 0)
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)):
        raise ValidationError("Checkpoint format version must be an integer")
    if version == 0:
        if not allow_legacy:
            raise ValidationError("Unversioned checkpoint: use allow_legacy=True to accept an unverified continuation")
        if sim.integrator.name != "lm" or sim.callbacks or getattr(sim.integrator, "settings", {}):
            raise ValidationError("Legacy checkpoints cannot restore custom integrators or callbacks")
        warnings.warn("Legacy checkpoint: settings are unverified and pending measurements cannot be recovered", RuntimeWarning)
        return
    if version not in (1, 2):
        raise ValidationError(f"Unsupported checkpoint format version {version}")
    try:
        stored = json.loads(str(data["checkpoint_settings_json"]))
        if input_hash(stored) != str(data["checkpoint_settings_hash"]):
            raise ValidationError("Checkpoint settings hash does not match its contents")
        # Verify the original settings hash before translating names. Earlier
        # checkpoints have legacy block keys and flat recording-period keys.
        stored = normalize_params(stored)
        for old, new in (("sf_computing_period", "sample_interval"),
                         ("sf_recording_period", "statistics_save_interval")):
            if old in stored:
                if new in stored:
                    raise ValidationError(f"Checkpoint specifies both '{old}' and '{new}'")
                stored[new] = stored.pop(old)
        current = checkpoint_settings(sim)
        # Version 1 predates extensions and always used LM without callbacks.
        if version == 1:
            stored.setdefault("extensions", {"integrator": {"name": "lm", "settings": {}}, "callbacks": {}})
        # JSON distinguishes 15 from 15.0; effective numerical settings do not.
        different = [key for key in current if current[key] != stored.get(key)]
        if different:
            raise ValidationError("Incompatible checkpoint settings: " + ", ".join(different))
        if version == 2:
            state = json.loads(str(data["extension_state_json"]))
            if not isinstance(state, dict) or set(state) != {"integrator", "callbacks"} or not isinstance(state["callbacks"], dict):
                raise ValidationError("Invalid extension checkpoint state")
            if set(state["callbacks"]) != set(sim.callbacks):
                raise ValidationError("Checkpoint callback state does not match registered callbacks")
            if not isinstance(state["integrator"], dict) or not all(isinstance(value, dict) for value in state["callbacks"].values()):
                raise ValidationError("Each extension checkpoint state must be a dictionary")
            json.dumps(state, allow_nan=False)
        stored_basis = np.asarray(data["matrix_a"])
        if stored_basis.size != sim.mpt.matrix_a.size or not np.allclose(
                stored_basis.reshape(sim.mpt.matrix_a.shape), sim.mpt.matrix_a, rtol=1e-12, atol=1e-12):
            raise ValidationError("Incompatible checkpoint auxiliary-field basis")
        step = data["langevin_step"]
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)) or step < 0:
            raise ValidationError("Checkpoint step must be a nonnegative integer")
        if step > sim.langevin["max_step"]:
            raise ValidationError("Checkpoint step exceeds langevin.max_step or is negative")
        for key in ("simulation_metadata_json", "run_status_json"):
            if key in data:
                value = json.loads(str(data[key]))
                if not isinstance(value, dict):
                    raise ValidationError(f"Checkpoint '{key}' must describe a dictionary")
                if key == "run_status_json":
                    if value.get("last_completed_step") != step:
                        raise ValidationError("Checkpoint run status does not match its step")
                    for counter in ("saddle_failures", "noise_resamples"):
                        count = value.get(counter, 0)
                        if type(count) is not int or count < 0:
                            raise ValidationError(f"Invalid checkpoint run counter '{counter}'")
        n_fields = len(sim.monomer_types)
        for key in ("w_aux", "normal_noise_prev"):
            if key not in data or not np.all(np.isfinite(data[key])):
                raise ValidationError(f"Checkpoint has missing or non-finite '{key}'")
            if np.iscomplexobj(data[key]) and (key == "normal_noise_prev" or sim.prop_solver.field_type == "real"):
                raise ValidationError(f"Checkpoint '{key}' must be real")
        if np.size(data["w_aux"]) != n_fields * sim.prop_solver.n_grid:
            raise ValidationError("Checkpoint auxiliary field shape is incompatible")
        if "record_sample_count" not in data or "rng_state_json" not in data:
            raise ValidationError("Checkpoint is missing measurement or RNG state")
        n_noise = n_fields if sim.prop_solver.field_type == "complex" else len(sim.mpt.aux_fields_real_idx)
        if np.size(data["normal_noise_prev"]) != n_noise * sim.prop_solver.n_grid:
            raise ValidationError("Checkpoint noise shape is incompatible")
        rng = np.random.PCG64(0)
        rng.state = json.loads(str(data["rng_state_json"]))
        count = data["record_sample_count"]
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValidationError("Invalid checkpoint sample count")
        for key in ["record_H_history", "record_drift_history"] + [_measurement_key("dH", key) for key in sim.chi_n]:
            if np.size(data[key]) != count or not np.all(np.isfinite(data[key])):
                raise ValidationError(f"Invalid checkpoint measurement '{key}'")
        if np.iscomplexobj(data["record_drift_history"]) or np.any(data["record_drift_history"] < 0):
            raise ValidationError("Checkpoint drift magnitudes must be real and nonnegative")
        # The drifts of every step since the last statistics file, or since the first step of a run
        # continued by hand or from a checkpoint without the record (record_drift_step_start).
        # Checkpoints of earlier versions lack the record, whose continued runs record from the restart on
        pending_steps = step % sim.recording["statistics_save_interval"]
        if "record_drift_step_start" in data:
            first = data["record_drift_step_start"]
            if (isinstance(first, (bool, np.bool_)) or not isinstance(first, (int, np.integer))
                    or not step - pending_steps < first <= step + 1):
                raise ValidationError("Invalid checkpoint measurement 'record_drift_step_start'")
            pending_steps = step + 1 - int(first)
        for key, per_step in (("record_drift_step_history", n_fields), ("record_psi_drift_step_history", 1)):
            if key in data:
                values = np.atleast_1d(np.asarray(data[key]))
                if (np.size(values) != pending_steps * per_step or np.iscomplexobj(values)
                        or not np.all(np.isfinite(values)) or np.any(values < 0)):
                    raise ValidationError(f"Invalid checkpoint measurement '{key}'")
        # Pending chemical potentials and pressures (absent in checkpoints of earlier versions or
        # of runs without them; continue_run then records NaN for those samples, which later
        # checkpoints keep). A species without chains has the chemical potential -inf
        for key, per_sample in (("record_chemical_potential_history", len(getattr(sim, "chain_densities", []))),
                                ("record_pressure_history", 1)):
            if key in data:
                values = np.atleast_1d(np.asarray(data[key]))
                if np.size(values) != count * per_sample or np.any(np.isposinf(np.real(values))):
                    raise ValidationError(f"Invalid checkpoint measurement '{key}'")
        for monomer in sim.monomer_types:
            names = [kind + monomer + suffix for kind in ("w_", "phi_") for suffix in
                     (("",) if sim.prop_solver.field_type == "real" else ("_real", "_imag"))]
            for name in names:
                if np.size(data[name]) != sim.prop_solver.n_grid or not np.all(np.isfinite(data[name])) or np.iscomplexobj(data[name]):
                    raise ValidationError(f"Invalid checkpoint field '{name}'")
        sf_size = int(np.prod(sim.prop_solver.nx)) if sim.prop_solver.field_type == "complex" else int(
            np.prod(sim.prop_solver.nx[:-1])) * (sim.prop_solver.nx[-1] // 2 + 1)
        for pair in itertools.combinations_with_replacement(sim.monomer_types, 2):
            name = _measurement_key("sf", ",".join(pair))
            if np.size(data[name]) != sf_size or not np.all(np.isfinite(data[name])):
                raise ValidationError(f"Invalid checkpoint structure function '{name}'")
        electrostatics = getattr(sim, "electrostatics", None)
        if electrostatics is not None:
            # ψ, ψ(k), the previous noise and the stream of ψ (Electrostatics.checkpoint_data), the
            # pending drifts of ψ and, with Coulomb coupling, the pending charge structure sums
            missing = [key for key in electrostatics.CHECKPOINT_KEYS + ("record_psi_drift_history",) if key not in data]
            if missing:
                raise ValidationError(f"The checkpoint lacks electrostatic data: {', '.join(missing)}")
            for key in ("psi_real", "psi_imag", "psi_k_real", "psi_k_imag", "psi_noise_prev"):
                if (np.size(data[key]) != sim.prop_solver.n_grid or not np.all(np.isfinite(data[key]))
                        or np.iscomplexobj(data[key])):
                    raise ValidationError(f"Invalid checkpoint electrostatic array '{key}'")
            try:
                np.random.PCG64(0).state = json.loads(str(data["psi_rng_state_json"]))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValidationError("Invalid checkpoint random-number state of psi") from exc
            psi_drift = data["record_psi_drift_history"]
            if (np.size(psi_drift) != count or np.iscomplexobj(psi_drift) or not np.all(np.isfinite(psi_drift))
                    or np.any(psi_drift < 0)):
                raise ValidationError("Invalid checkpoint measurement 'record_psi_drift_history'")
            if electrostatics.coupling > 0:
                for key in electrostatics.CHARGE_SUMS:
                    name = _measurement_key("sf", key)
                    if (name not in data or np.size(data[name]) != int(np.prod(sim.prop_solver.nx))
                            or not np.all(np.isfinite(data[name]))):
                        raise ValidationError(f"Invalid checkpoint charge structure sum '{key}'")
        wtmd = getattr(sim, "wtmd", None)
        if wtmd is not None:
            def check_wtmd_array(key):
                if np.size(data[key]) != wtmd.bins or not np.all(np.isfinite(data[key])) or np.iscomplexobj(data[key]):
                    raise ValidationError(f"Invalid checkpoint WTMD array '{key}'")
            def check_wtmd_copies(name):
                # Both copies are written from one array; the restore uses the unscaled
                # one, so a checkpoint whose scaled copy was edited is refused
                check_wtmd_array("wtmd_unscaled_" + name)
                if not np.allclose(np.ravel(data["wtmd_unscaled_" + name]) * wtmd.CV,
                                   np.ravel(data["wtmd_" + name]), rtol=1e-12, atol=0):
                    raise ValidationError(f"Checkpoint WTMD arrays 'wtmd_{name}' and 'wtmd_unscaled_{name}' disagree")
            # Newer checkpoints also hold the bias arrays without the factor CV ("wtmd_unscaled_*")
            unscaled = "wtmd_unscaled_u" in data
            for key in ("wtmd_u", "wtmd_up", "wtmd_I0"):
                check_wtmd_array(key)
            if unscaled:
                check_wtmd_copies("u")
                check_wtmd_copies("up")
            pending = np.size(data["wtmd_order_parameter_history"])
            if (pending >= wtmd.update_freq or not np.all(np.isfinite(data["wtmd_order_parameter_history"])) or
                    np.iscomplexobj(data["wtmd_order_parameter_history"])):
                raise ValidationError("Invalid checkpoint pending WTMD samples")
            for pair in sim.chi_n:
                suffix = "_".join(sorted(pair.split(",")))
                key = "wtmd_I1_" + suffix
                if np.any(data["wtmd_I0"]) or key in data:
                    check_wtmd_array(key)
                    if unscaled:
                        check_wtmd_copies("I1_" + suffix)
                key = "wtmd_dH_history_" + suffix
                if pending and (np.size(data[key]) != pending or not np.all(np.isfinite(data[key])) or np.iscomplexobj(data[key])):
                    raise ValidationError(f"Invalid checkpoint WTMD history '{key}'")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid checkpoint: {exc}") from exc


def restore_recording_state(data, H_history, dH_history, sf_average, drift_history=None, *, psi_drift_history=None,
                            drift_step_history=None, n_fields=1, psi_drift_step_history=None,
                            chemical_potential_history=None, pressure_history=None, n_polymers=1):
    """Restore pending samples without MATLAB's squeeze changing their shape."""
    if data is None or "record_sample_count" not in data:
        return 0
    # Chemical potentials and pressures that the checkpoint does not hold are NaN
    count = int(data["record_sample_count"])
    if chemical_potential_history is not None:
        if "record_chemical_potential_history" in data:
            values = np.reshape(np.asarray(data["record_chemical_potential_history"], dtype=np.complex128), (-1, n_polymers))
        else:
            values = np.full((count, n_polymers), np.nan + 0j)
        chemical_potential_history.extend(values)
    if pressure_history is not None:
        if "record_pressure_history" in data:
            values = np.atleast_1d(np.asarray(data["record_pressure_history"], dtype=np.complex128))
        else:
            values = np.full(count, np.nan + 0j)
        pressure_history.extend(values.tolist())
    H_history.extend(np.atleast_1d(data["record_H_history"]).tolist())
    if drift_history is not None and "record_drift_history" in data:
        drift_history.extend(np.atleast_1d(data["record_drift_history"]).tolist())
    if psi_drift_history is not None:
        psi_drift_history.extend(np.atleast_1d(data["record_psi_drift_history"]).tolist())
    if drift_step_history is not None and "record_drift_step_history" in data:
        drift_step_history.extend(np.reshape(data["record_drift_step_history"], (-1, n_fields)))
    if psi_drift_step_history is not None and "record_psi_drift_step_history" in data:
        psi_drift_step_history.extend(np.atleast_1d(data["record_psi_drift_step_history"]).tolist())
    for key in dH_history:
        dH_history[key].extend(np.atleast_1d(data[_measurement_key("dH", key)]).tolist())
    for key, array in sf_average.items():
        array[:] = np.asarray(data[_measurement_key("sf", key)]).reshape(array.shape)
    return int(data["record_sample_count"])


class DriftTailSummary:
    """Distribution of the largest drift of each step, summarized at the end of a CL-FTS run.

    Complex Langevin averages need the distribution of the drift magnitude to fall off
    exponentially or faster (Nagata, Nishimura & Shimasaki, Phys. Rev. D 94, 114515 (2016)).
    The counts stay in memory, twenty bins per decade (the printed median and maximum are
    bin edges); the step values themselves are recorded as drift_max_step_history and
    psi_drift_max_step_history. The summary gives
    numbers, not a verdict: a finite run cannot show an exponential tail, and the weight of
    a power-law tail sets the size of the bias.
    """
    EDGES = 10.0 ** (np.arange(-120, 241) / 20)

    def __init__(self, labels):
        self.labels = list(labels)
        self.counts = np.zeros((len(self.labels), self.EDGES.size + 1), dtype=np.int64)

    def add(self, values):
        self.counts[np.arange(len(self.labels)), np.searchsorted(self.EDGES, values)] += 1

    def report(self):
        steps = int(self.counts[0].sum()) if self.labels else 0
        if steps == 0:
            return
        print(f"Largest drift per step ({steps} steps of this run); S(u) = P(drift > u) at 10 and 100 "
              "times the median, and the log-log slope of S between them (scripts/analysis/drift_tail.py):")
        for label, counts in zip(self.labels, self.counts):
            beyond = steps - np.cumsum(counts)          # steps above EDGES[k]
            k_median = int(np.searchsorted(-beyond, -steps / 2))
            if k_median >= self.EDGES.size:
                continue
            median = self.EDGES[k_median]
            if median <= self.EDGES[0]:
                print(f"  {label}: no drift")
                continue
            k10 = min(np.searchsorted(self.EDGES, 10 * median), self.EDGES.size - 1)
            k100 = min(np.searchsorted(self.EDGES, 100 * median), self.EDGES.size - 1)
            s10, s100 = beyond[k10] / steps, beyond[k100] / steps
            slope = f"{np.log(s100 / s10) / np.log(self.EDGES[k100] / self.EDGES[k10]):.2f}" \
                if beyond[k100] >= 20 else "-"
            largest = self.EDGES[min(np.flatnonzero(counts).max(), self.EDGES.size - 1)]
            print(f"  {label}: median ~{median:.3g}, maximum < {largest:.3g}, "
                  f"S(10x) {s10:.3g}, S(100x) {s100:.3g}, slope {slope}")


# Small extension mechanics; the physical order of calls stays in each run loop.
def initialize_extensions(sim, integrator, callbacks):
    from .integrators import make_integrator
    sim.integrator = make_integrator(integrator)
    if callbacks is not None and not isinstance(callbacks, dict):
        raise ValidationError("callbacks must be a dictionary of stable names to callables")
    sim.callbacks = dict(callbacks or {})
    for name, callback in sim.callbacks.items():
        if not isinstance(name, str) or not name or not callable(callback):
            raise ValidationError("Each callback needs a nonempty string name and must be callable")
        interval = getattr(callback, "interval", 1)
        if type(interval) is not int or interval < 1:
            raise ValidationError("Callback interval must be a positive integer")
    for obj in [sim.integrator, *sim.callbacks.values()]:
        for method in ("state_dict", "load_state_dict"):
            if hasattr(obj, method) and not callable(getattr(obj, method)):
                raise ValidationError(f"Extension {method} must be callable")
        save = callable(getattr(obj, "state_dict", None))
        load = callable(getattr(obj, "load_state_dict", None))
        if save != load:
            raise ValidationError("Extensions must implement both state_dict and load_state_dict, or neither")
    # Fail at construction, rather than at the first scheduled checkpoint.
    sim._initial_extension_state = copy.deepcopy(extension_state(sim))
    sim.metadata["extensions"] = extension_settings(sim)


def extension_settings(sim):
    def settings(obj):
        value = getattr(obj, "settings", {})
        if not isinstance(value, dict):
            raise ValidationError("Extension settings must be a JSON dictionary")
        return value
    result = {"integrator": {"name": sim.integrator.name, "settings": settings(sim.integrator)},
              "callbacks": {name: {"callable": f"{getattr(cb, '__module__', type(cb).__module__)}."
                                              f"{getattr(cb, '__qualname__', type(cb).__qualname__)}",
                                   "interval": getattr(cb, "interval", 1),
                                   "settings": settings(cb)} for name, cb in sim.callbacks.items()}}
    try:
        return json.loads(json.dumps(result, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValidationError("Extension settings must contain finite JSON values") from exc


def extension_state(sim):
    return {"integrator": extension_object_state(sim.integrator),
            "callbacks": {name: extension_object_state(cb) for name, cb in sim.callbacks.items()}}


def extension_object_state(obj):
    value = obj.state_dict() if callable(getattr(obj, "state_dict", None)) else {}
    if not isinstance(value, dict):
        raise ValidationError("Extension state_dict must return a JSON dictionary")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValidationError("Extension state must contain finite JSON values") from exc


def restore_extensions(sim, data=None):
    """Load before the RNG is changed; roll back extension objects on a bad state."""
    state = (json.loads(str(data["extension_state_json"]))
             if data is not None and "extension_state_json" in data else sim._initial_extension_state)
    previous = extension_state(sim)

    def load(values):
        objects = [(sim.integrator, values["integrator"])]
        objects.extend((obj, values["callbacks"][name]) for name, obj in sim.callbacks.items())
        for obj, value in objects:
            if callable(getattr(obj, "load_state_dict", None)):
                obj.load_state_dict(copy.deepcopy(value))
            elif value != {}:
                raise ValidationError("A stateless extension cannot load nonempty state")
    try:
        load(state)
    except Exception as exc:
        load(previous)
        raise ValidationError(f"Cannot restore extension state: {exc}") from exc


def notify_callbacks(sim, step, w_aux, phi, hamiltonian, drift_max, error_level, step_seconds):
    """Observers receive detached, read-only snapshots of accepted steps only."""
    callbacks = [cb for cb in sim.callbacks.values() if step % getattr(cb, "interval", 1) == 0]
    if not callbacks:
        return

    def frozen(array):
        # A bytes-backed array cannot be made writeable, even with setflags.
        array = np.asarray(array)
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)

    sample = MappingProxyType({"step": step, "dt": sim.langevin["dt"],
                               "time": step * sim.langevin["dt"],
                               "w_aux": frozen(w_aux), "w": frozen(sim.mpt.to_monomer_fields(w_aux)),
                               "phi": MappingProxyType({m: frozen(v) for m, v in phi.items()}),
                               "hamiltonian": hamiltonian, "drift_max": drift_max,
                               "error_level": frozen(error_level), "step_seconds": step_seconds,
                               "status": MappingProxyType(dict(sim.run_status))})
    for callback in callbacks:
        callback(sample)


# =============================================================================
# Polymer architecture drawing (SCFT into polymer_figure_dir, LFTS/CLFTS into the recording directory)
# =============================================================================

DEFAULT_MONOMER_COLORS: List[str] = [
    "red", "blue", "green", "cyan", "magenta", "yellow"
]


def create_monomer_color_dict(
    segment_lengths: Dict[str, float],
    colors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create a color dictionary for monomer types.

    Parameters
    ----------
    segment_lengths : dict
        Dictionary mapping monomer types to their statistical segment lengths.
    colors : list of str, optional
        List of color names to use. Defaults to DEFAULT_MONOMER_COLORS.

    Returns
    -------
    dict
        Dictionary mapping monomer types to colors.
    """
    if colors is None:
        colors = DEFAULT_MONOMER_COLORS

    # Colors beyond the list come from a fixed local stream: the same in every run, and the
    # global NumPy random state stays untouched
    extra_colors = np.random.default_rng(0)
    dict_color = {}
    for count, monomer_type in enumerate(segment_lengths.keys()):
        if count < len(colors):
            dict_color[monomer_type] = colors[count]
        else:
            dict_color[monomer_type] = extra_colors.random(3)

    return dict_color


def draw_polymer_architectures(polymers, ds, segment_lengths, prefix="polymer", directory="."):
    """Draw every polymer of a simulation to ``<directory>/<prefix>_<id>.png``.

    SCFT draws them into ``polymer_figure_dir`` when the input gives one; LFTS and CLFTS
    draw them into the recording directory when a run starts. To draw them elsewhere:
    ``draw_polymer_architectures(calc.polymers, calc.ds, calc.segment_lengths, directory=...)``.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    dict_color = create_monomer_color_dict(segment_lengths)
    for idx, polymer in enumerate(polymers):
        draw_polymer_architecture(polymer, idx, ds, dict_color,
                                  output_file=os.path.join(directory, f"{prefix}_{idx}.png"))


def draw_polymer_architecture(
    polymer: Dict[str, Any],
    polymer_id: int,
    ds: float,
    dict_color: Dict[str, Any],
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 100
) -> None:
    """Draw polymer chain architecture as a graph.

    Parameters
    ----------
    polymer : dict
        Polymer dictionary containing 'blocks_input' with block definitions.
    polymer_id : int
        Identifier for the polymer (used in title and filename).
    ds : float
        Contour step size for calculating segment numbers.
    dict_color : dict
        Dictionary mapping monomer types to colors.
    output_file : str, optional
        Output filename. Defaults to "polymer_{id}.png".
    figsize : tuple of int
        Figure size in inches. Defaults to (8, 8).
    dpi : int
        Resolution in dots per inch. Defaults to 100.
    """
    # Make a graph
    import networkx as nx               # imported here: only the drawing needs them
    import matplotlib.pyplot as plt

    G = nx.Graph()
    for block in polymer["blocks_input"]:
        monomer_type = block[0]
        length = round(block[1] / ds)
        v = block[2]
        u = block[3]
        G.add_edge(v, u, weight=length, monomer_type=monomer_type)

    # Set node colors
    color_map = []
    for node in G:
        if len(G.edges(node)) == 1:
            color_map.append('yellow')
        else:
            color_map.append('gray')

    labels = nx.get_edge_attributes(G, 'weight')
    try:
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='twopi')
    except ImportError:
        # pygraphviz not available, use spring_layout as fallback
        pos = nx.spring_layout(G)
    edge_colors = [dict_color[G[u][v]['monomer_type']] for u, v in G.edges()]

    plt.figure(figsize=figsize)
    title = f"Polymer ID: {polymer_id:2d},"
    title += f"\nColors of monomers: {dict_color},"
    title += "\nColor of chain ends: 'yellow',"
    title += "\nColor of junctions: 'gray',"
    title += "\nPlease note that the length of each edge is not proportional to the number of monomers in this image."
    plt.title(title)
    nx.draw(G, pos, node_color=color_map, edge_color=edge_colors, width=4, with_labels=True)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=labels, rotate=False,
        bbox=dict(boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0), alpha=0.5)
    )

    if output_file is None:
        output_file = f"polymer_{polymer_id:01d}.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()


def process_polymer_blocks(
    polymer: Dict[str, Any]
) -> List[List[Any]]:
    """Process polymer block definitions into blocks_input format.

    Parameters
    ----------
    polymer : dict
        Polymer dictionary containing 'blocks' list.

    Returns
    -------
    list
        Processed blocks_input list with [type, length, v, u] format.

    Raises
    ------
    ValidationError
        If block vertex indices are inconsistent.
    """
    blocks_input = []
    has_node_number = "v" not in polymer["blocks"][0]

    for block in polymer["blocks"]:
        if has_node_number:
            if "v" in block:
                raise ValidationError(
                    "Index v should exist in all blocks, or it should not exist in all blocks for each polymer."
                )
            if "u" in block:
                raise ValidationError(
                    "Index u should exist in all blocks, or it should not exist in all blocks for each polymer."
                )
            blocks_input.append([block["type"], block["length"], len(blocks_input), len(blocks_input) + 1])
        else:
            if "v" not in block:
                raise ValidationError(
                    "Index v should exist in all blocks, or it should not exist in all blocks for each polymer."
                )
            if "u" not in block:
                raise ValidationError(
                    "Index u should exist in all blocks, or it should not exist in all blocks for each polymer."
                )
            blocks_input.append([block["type"], block["length"], block["v"], block["u"]])

    return blocks_input


def process_random_copolymer(
    polymer: Dict[str, Any],
    segment_lengths: Dict[str, float]
) -> Optional[Tuple[str, float, Dict[str, float]]]:
    """Process random copolymer definitions.

    Parameters
    ----------
    polymer : dict
        Polymer dictionary potentially containing random copolymer definition.
    segment_lengths : dict
        Current segment lengths dictionary.

    Returns
    -------
    tuple or None
        If random copolymer: (type_string, statistical_segment_length, fraction_dict)
        Otherwise: None

    Raises
    ------
    ValidationError
        If random copolymer definition is invalid.
    """
    # Check if this is a random copolymer
    is_random = any("monomer_fractions" in block for block in polymer["blocks"])
    if not is_random:
        return None

    if len(polymer["blocks"]) != 1:
        raise ValidationError("Only single block random copolymer is allowed.")

    statistical_segment_length = 0
    total_random_fraction = 0
    for monomer_type in polymer["blocks"][0]["monomer_fractions"]:
        statistical_segment_length += (
            segment_lengths[monomer_type] ** 2 * polymer["blocks"][0]["monomer_fractions"][monomer_type]
        )
        total_random_fraction += polymer["blocks"][0]["monomer_fractions"][monomer_type]
    statistical_segment_length = np.sqrt(statistical_segment_length)

    if not np.isclose(total_random_fraction, 1.0):
        raise ValidationError("The sum of volume fractions of random copolymer must be equal to 1.")

    random_type_string = polymer["blocks"][0]["type"]
    if random_type_string in segment_lengths:
        raise ValidationError(
            f"The name of random copolymer '{random_type_string}' is already used as a type "
            "in 'segment_lengths' or other random copolymer"
        )

    return (random_type_string, statistical_segment_length, polymer["blocks"][0]["monomer_fractions"])


# =============================================================================
# Langevin noise
# =============================================================================
def calculate_sigma(langevin_nbar, langevin_dt, n_grids, volume):
    """Calculate Langevin noise strength for L-FTS simulations.

    Computes the standard deviation of Gaussian noise to be added at each
    Langevin step, ensuring correct statistical weighting of field configurations
    in the field-theoretic ensemble.

    Parameters
    ----------
    langevin_nbar : float
        Average polymerization index N̄ for the Langevin scheme. Related to
        the prefactor of the Hamiltonian. Typical values: 1-100.
    langevin_dt : float
        Langevin time step size Δt. Controls the magnitude of field updates.
        Typical values: 0.001-0.1.
    n_grids : int
        Total number of grid points M (product of nx dimensions).
    volume : float
        System volume V in units of (a_Ref * N_Ref^0.5)³.

    Returns
    -------
    sigma : float
        Standard deviation of Gaussian noise for each field component.
    """
    return np.sqrt(2*langevin_dt*n_grids/(volume*np.sqrt(langevin_nbar)))

# =============================================================================
# Field compressors for the saddle-point search of the imaginary fields
# =============================================================================


# =============================================================================
# Parameter files (YAML / JSON)
# =============================================================================

class ConfigError(Exception):
    """Exception raised for configuration file errors."""
    pass


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load simulation parameters from a configuration file.

    Supports YAML (.yaml, .yml) and JSON (.json) formats.
    Automatically detects format from file extension.

    String values that use numpy, such as ``"4.0*np.sqrt(3)/2"``, are
    evaluated: numbers, ``+ - * / **``, parentheses, ``np.pi``, ``np.e`` and
    the elementwise functions in ``CONFIG_NP_FUNCTIONS`` are allowed, and
    nothing else is executed. Any other string containing ``np.`` is kept as
    it is, with a warning.

    Parameters
    ----------
    path : str or Path
        Path to configuration file.

    Returns
    -------
    dict
        Simulation parameters dictionary.

    Raises
    ------
    ConfigError
        If file format is not supported or file cannot be read.
    FileNotFoundError
        If configuration file does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()

    try:
        with open(path, 'r') as f:
            if suffix in ('.yaml', '.yml'):
                params = yaml.safe_load(f)
            elif suffix == '.json':
                params = json.load(f)
            else:
                raise ConfigError(
                    f"Unsupported configuration file format: {suffix}. "
                    "Use .yaml, .yml, or .json"
                )
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML syntax in {path}: {e}")
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON syntax in {path}: {e}")

    if params is None:
        raise ConfigError(f"Configuration file is empty: {path}")

    if not isinstance(params, dict):
        raise ConfigError(f"Configuration must be a dictionary, got {type(params).__name__}")

    # Process special values (e.g., numpy expressions)
    params = _process_config_values(params)

    return params


# Numpy names a configuration expression may use
CONFIG_NP_CONSTANTS = ("pi", "e")
CONFIG_NP_FUNCTIONS = ("sqrt", "cbrt", "exp", "log", "log10", "sin", "cos", "tan",
                       "arcsin", "arccos", "arctan", "deg2rad", "rad2deg", "abs")
_CONFIG_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                     ast.Div: operator.truediv, ast.Pow: operator.pow}


def _eval_config_expression(text: str) -> float:
    """Value of an arithmetic expression such as "4.0*np.sqrt(3)/2".

    Evaluates the parsed expression node by node instead of calling eval, so
    only numbers, + - * / **, unary signs, np.pi, np.e and the functions in
    CONFIG_NP_FUNCTIONS (one argument each) can appear, and every intermediate
    value must be a finite real number. Raises ValueError otherwise.
    """
    def is_np(node):
        return isinstance(node, ast.Name) and node.id in ("np", "numpy")

    def finite_real(x, node):
        # (-8)**0.5 is complex in Python, np.sqrt(-1) is nan, np.log(0) is -inf
        if isinstance(x, complex) or not np.isfinite(x):
            raise ValueError(f"'{ast.unparse(node)}' is not a finite real number")
        return float(x)

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _CONFIG_OPERATORS:
            return finite_real(_CONFIG_OPERATORS[type(node.op)](value(node.left), value(node.right)), node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return value(node.operand) if isinstance(node.op, ast.UAdd) else -value(node.operand)
        if isinstance(node, ast.Attribute) and is_np(node.value) and node.attr in CONFIG_NP_CONSTANTS:
            return float(getattr(np, node.attr))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and is_np(node.func.value)
                and node.func.attr in CONFIG_NP_FUNCTIONS and len(node.args) == 1 and not node.keywords):
            return finite_real(getattr(np, node.func.attr)(value(node.args[0])), node)
        raise ValueError(f"'{ast.unparse(node)}' is not allowed in a configuration expression")

    with np.errstate(all="ignore"):
        return value(ast.parse(text.strip(), mode="eval").body)


def _process_config_values(obj: Any) -> Any:
    """Process configuration values, evaluating numpy expressions.

    Handles:
    - String expressions like "4.0*np.sqrt(3)/2" for lx (see _eval_config_expression)
    - Nested dictionaries and lists

    Parameters
    ----------
    obj : Any
        Configuration value to process.

    Returns
    -------
    Any
        Processed value.
    """
    if isinstance(obj, dict):
        return {k: _process_config_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_process_config_values(item) for item in obj]
    elif isinstance(obj, str):
        if 'np.' in obj or 'numpy.' in obj:
            try:
                return _eval_config_expression(obj)
            except (SyntaxError, ValueError, ArithmeticError, RecursionError, MemoryError) as e:
                # A string that LOOKS like a numpy expression but cannot be
                # evaluated is almost certainly a typo in the config; passing
                # it through silently would surface later as a confusing
                # type error. Warn so the user can fix the expression.
                warnings.warn(
                    f"Config value {obj!r} looks like a numpy expression but "
                    f"was not evaluated ({e}); using the raw string instead.")
                return obj
        return obj
    else:
        return obj


# =============================================================================
# VTK output for ParaView
# =============================================================================

def save_vtk(
    path: Union[str, Path],
    fields: Union[np.ndarray, Dict[str, np.ndarray]],
    lx: List[float],
    field_name: str = "phi",
    nx: Optional[List[int]] = None,
    angles: Optional[List[float]] = None,
) -> None:
    """Save fields as a legacy VTK file for ParaView.

    The grid points sit at the cell centres, (i + 1/2) * lx / nx along each
    lattice vector, as in the solvers, so the outline ParaView draws through
    the points lies half a cell inside the box (the walls of a non-periodic
    box are at 0 and lx).

    Parameters
    ----------
    path : str or Path
        Output file path (should end with .vtk).
    fields : np.ndarray or dict
        One real field, or {name: field} to store several fields in one file
        (for example SCFTResult.phi). A field has the grid shape (1D, 2D or 3D),
        or is flat when nx is given.
    lx : list of float
        Box lengths, one per dimension.
    field_name : str
        Name of the field when a single array is given (default: "phi").
    nx : list of int, optional
        Grid size, needed for flat arrays.
    angles : list of float, optional
        Lattice angles [alpha, beta, gamma] in degrees, or [gamma] for a 2D
        cell. A non-orthogonal cell is written as a structured grid with
        explicit point coordinates.
    """
    if not isinstance(fields, dict):
        fields = {field_name: fields}
    if not fields:
        raise ValueError("No field to save")

    if nx is None:
        nx = list(np.shape(next(iter(fields.values()))))
    nx = [int(n) for n in nx]
    dim = len(nx)
    if not 1 <= dim <= 3 or len(lx) != dim:
        raise ValueError(f"nx and lx must have the same length 1, 2 or 3, got nx={nx}, lx={list(lx)}")
    n_points = int(np.prod(nx))

    # Fields in VTK point order (x fastest); names without spaces
    data = {}
    for name, field in fields.items():
        field = np.asarray(field)
        if np.iscomplexobj(field):
            raise ValueError(f"Field '{name}' is complex; VTK stores real values "
                             "(save the real and imaginary parts separately)")
        if field.size != n_points:
            raise ValueError(f"Field '{name}' has {field.size} values, the grid {nx} has {n_points}")
        data[str(name).replace(" ", "_")] = np.reshape(field, nx).ravel(order="F")

    # Pad to three dimensions
    nx3 = nx + [1] * (3 - dim)
    lx3 = [float(x) for x in lx] + [1.0] * (3 - dim)
    if angles is not None and len(angles) == 1:
        angles = [90.0, 90.0, angles[0]]          # 2D: [gamma]
    if angles is not None and len(angles) != 3:
        raise ValueError(f"angles must be [alpha, beta, gamma] or [gamma], got {list(angles)}")
    alpha, beta, gamma = np.deg2rad(angles) if angles is not None else (np.pi/2,)*3
    orthogonal = np.allclose([alpha, beta, gamma], np.pi/2, atol=1e-10)

    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Polymer field data\n")
        f.write("ASCII\n")
        if orthogonal:
            spacing = [lx3[d]/nx3[d] for d in range(3)]
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write("DIMENSIONS %d %d %d\n" % tuple(nx3))
            f.write("ORIGIN %.10g %.10g %.10g\n" % tuple(0.5*h for h in spacing))
            f.write("SPACING %.10g %.10g %.10g\n" % tuple(spacing))
        else:
            # Lattice vectors as in ComputationBox: a along x, b in the xy plane
            a = lx3[0]*np.array([1.0, 0.0, 0.0])
            b = lx3[1]*np.array([np.cos(gamma), np.sin(gamma), 0.0])
            cx = np.cos(beta)
            cy = (np.cos(alpha) - np.cos(beta)*np.cos(gamma))/np.sin(gamma)
            c = lx3[2]*np.array([cx, cy, np.sqrt(max(1.0 - cx*cx - cy*cy, 0.0))])
            frac = [(np.arange(nx3[d]) + 0.5)/nx3[d] for d in range(3)]
            fz, fy, fx = np.meshgrid(frac[2], frac[1], frac[0], indexing="ij")   # x fastest after ravel
            points = np.outer(fx.ravel(), a) + np.outer(fy.ravel(), b) + np.outer(fz.ravel(), c)
            f.write("DATASET STRUCTURED_GRID\n")
            f.write("DIMENSIONS %d %d %d\n" % tuple(nx3))
            f.write("POINTS %d double\n" % n_points)
            np.savetxt(f, points, fmt="%.10g")
        f.write("POINT_DATA %d\n" % n_points)
        for name, values in data.items():
            f.write("SCALARS %s float 1\n" % name)
            f.write("LOOKUP_TABLE default\n")
            np.savetxt(f, values, fmt="%.7g")
