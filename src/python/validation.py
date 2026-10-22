"""Parameter checks for SCFT, L-FTS and CL-FTS.

The checks raise ValidationError (never assert, which disappears under
``python -O``) before any expensive C++ object is created.
"""

from typing import Any, Dict
from os import PathLike

import numpy as np


class ValidationError(Exception):
    """Raised when a simulation parameter is missing or invalid."""
    pass


class SaddlePointError(RuntimeError):
    """A Langevin run stopped before accepting a step; see ``sim.run_status``."""


def _finite_number(value, name, *, positive=False, nonnegative=False):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating)) or
            not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0)):
        qualifier = "positive " if positive else "non-negative " if nonnegative else ""
        raise ValidationError(f"'{name}' must be a finite {qualifier}number, got {value!r}")


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValidationError(f"'{name}' must be an integer >= {minimum}, got {value!r}")


def validate_system_params(params: Dict[str, Any]) -> None:
    """Check the parameters shared by SCFT, L-FTS and CL-FTS.

    Required: nx, lx, ds, chain_model, segment_lengths, chi_n, distinct_polymers.
    The first problem found raises ValidationError with a message naming the key.
    """
    if not isinstance(params, dict):
        raise ValidationError("Simulation parameters must be a dict")
    for key in ("nx", "lx", "ds", "chain_model", "segment_lengths", "chi_n", "distinct_polymers"):
        if key not in params:
            raise ValidationError(f"Missing required parameter: '{key}'")

    nx, lx = params["nx"], params["lx"]
    if not isinstance(nx, (list, tuple, np.ndarray)) or not isinstance(lx, (list, tuple, np.ndarray)):
        raise ValidationError("'nx' and 'lx' must be lists")
    if len(nx) not in (1, 2, 3):
        raise ValidationError("'nx' must have 1, 2, or 3 dimensions")
    for n in nx:
        _integer(n, "nx")
    if len(lx) != len(nx):
        raise ValidationError(f"'lx' must have {len(nx)} elements, got {len(lx)}")
    for length in lx:
        _finite_number(length, "lx", positive=True)

    ds = params["ds"]
    _finite_number(ds, "ds", positive=True)
    if ds > 1:
        raise ValidationError(f"'ds' must be a number in (0, 1], got {ds}")
    if params["chain_model"] not in ("continuous", "discrete"):
        raise ValidationError(f"'chain_model' must be 'continuous' or 'discrete', got '{params['chain_model']}'")

    segment_lengths = params["segment_lengths"]
    if not isinstance(segment_lengths, dict) or len(segment_lengths) == 0:
        raise ValidationError("'segment_lengths' must be a non-empty dict {monomer_type: length}")
    for monomer, length in segment_lengths.items():
        if not isinstance(monomer, str) or not monomer:
            raise ValidationError("Monomer names must be non-empty strings")
        _finite_number(length, f"segment_lengths.{monomer}", positive=True)

    if not isinstance(params["chi_n"], dict):
        raise ValidationError("'chi_n' must be a dict, e.g. {'A,B': 15.0}")
    for pair, value in params["chi_n"].items():
        _finite_number(value, f"chi_n.{pair}")

    polymers = params["distinct_polymers"]
    if not isinstance(polymers, list) or len(polymers) == 0:
        raise ValidationError("'distinct_polymers' must be a non-empty list")
    total_volume_fraction = 0.0
    for i, polymer in enumerate(polymers):
        if not isinstance(polymer, dict) or "volume_fraction" not in polymer or "blocks" not in polymer:
            raise ValidationError(f"distinct_polymers[{i}] must be a dict with 'volume_fraction' and 'blocks'")
        vf = polymer["volume_fraction"]
        _finite_number(vf, f"distinct_polymers[{i}].volume_fraction", nonnegative=True)
        if vf > 1:
            raise ValidationError(f"distinct_polymers[{i}]['volume_fraction'] must be in [0, 1], got {vf}")
        total_volume_fraction += vf
        if not isinstance(polymer["blocks"], list) or len(polymer["blocks"]) == 0:
            raise ValidationError(f"distinct_polymers[{i}]['blocks'] must be a non-empty list")
        for j, block in enumerate(polymer["blocks"]):
            name = f"distinct_polymers[{i}].blocks[{j}]"
            if not isinstance(block, dict) or not isinstance(block.get("type"), str) or "length" not in block:
                raise ValidationError(f"'{name}' needs 'type' and 'length'")
            _finite_number(block["length"], name + ".length", positive=True)
            for vertex in ("v", "u"):
                if vertex in block:
                    _integer(block[vertex], name + "." + vertex, minimum=0)
            if "fraction" in block:
                fractions = block["fraction"]
                if not isinstance(fractions, dict) or not fractions:
                    raise ValidationError(f"'{name}.fraction' must be a non-empty dict")
                for monomer, fraction in fractions.items():
                    if monomer not in segment_lengths:
                        raise ValidationError(f"Unknown monomer '{monomer}' in '{name}.fraction'")
                    _finite_number(fraction, name + ".fraction." + monomer, nonnegative=True)
    if abs(total_volume_fraction - 1.0) > 1e-10:
        raise ValidationError(f"Sum of volume fractions must be 1.0, got {total_volume_fraction}")

    if "platform" in params and params["platform"] not in ("auto", "cuda", "cpu-fftw", "cpu-mkl"):
        raise ValidationError(f"'platform' must be 'auto', 'cuda', 'cpu-fftw' or 'cpu-mkl', got '{params['platform']}'")
    if params.get("zeta_n") is not None:
        _finite_number(params["zeta_n"], "zeta_n", nonnegative=True)


def validate_lfts_params(params: Dict[str, Any], *, complex_fields=False) -> None:
    """Validate the shared Langevin options before constructing numerical objects."""
    validate_system_params(params)
    langevin = params.get("langevin", None)
    if not isinstance(langevin, dict):
        raise ValidationError("Missing required parameter: 'langevin' (dict with 'nbar', 'dt', 'max_step')")
    for key in ("nbar", "dt"):
        _finite_number(langevin.get(key), "langevin." + key, positive=True)
    _integer(langevin.get("max_step"), "langevin.max_step", minimum=0)
    recording = params.get("recording")
    if not isinstance(recording, dict) or not isinstance(recording.get("dir"), (str, PathLike)):
        raise ValidationError("'recording' needs a 'dir' path")
    for key in ("recording_period", "sf_computing_period", "sf_recording_period"):
        _integer(recording.get(key), "recording." + key)
    _integer(params.get("verbose_level"), "verbose_level", minimum=0)
    if params["verbose_level"] > 2:
        raise ValidationError("'verbose_level' must be 0, 1 or 2")
    if not complex_fields:
        saddle = params.get("saddle")
        if not isinstance(saddle, dict):
            raise ValidationError("Missing required dict: 'saddle'")
        _integer(saddle.get("max_iter"), "saddle.max_iter")
        _finite_number(saddle.get("tolerance"), "saddle.tolerance", positive=True)
        _integer(saddle.get("max_retries", 5), "saddle.max_retries", minimum=0)
        if saddle.get("failure_policy", "retry") not in ("retry", "abort", "resample"):
            raise ValidationError("saddle.failure_policy must be 'retry', 'abort' or 'resample'")
        compressor = params.get("compressor")
        if not isinstance(compressor, dict) or compressor.get("name") not in ("am", "lr", "lram"):
            raise ValidationError("compressor.name must be 'am', 'lr' or 'lram'")
        if compressor["name"] in ("am", "lram"):
            _integer(compressor.get("max_hist"), "compressor.max_hist", minimum=0)
            for key in ("start_error", "mix_min", "mix_init"):
                _finite_number(compressor.get(key), "compressor." + key, nonnegative=True)
    for key in ("dynamic_stabilization", "alpha_ds"):
        if key in params:
            _finite_number(params[key], key, nonnegative=True)
    smearing = params.get("smearing")
    if smearing is not None:
        if not isinstance(smearing, dict):
            raise ValidationError("'smearing' must be a dict or None")
        kind = smearing.get("type", "gaussian")
        if kind not in ("gaussian", "sigmoidal"):
            raise ValidationError("Unknown smearing type")
        keys = ("a_int",) if kind == "gaussian" else ("k_int", "dk_int")
        for key in keys:
            _finite_number(smearing.get(key, 5.0 if key == "dk_int" else None), "smearing." + key,
                           positive=key != "a_int", nonnegative=key == "a_int")
    wtmd = params.get("wtmd")
    if wtmd is not None:
        if not isinstance(wtmd, dict):
            raise ValidationError("'wtmd' must be a dict or None")
        for key in ("update_freq", "recording_period"):
            _integer(wtmd.get(key, 1000 if key == "update_freq" else 100000), "wtmd." + key)
        for key, default in (("ell", 4), ("kc", 6.02), ("delta_t", 5.0), ("sigma_psi", 0.16), ("dpsi", 1e-3)):
            _finite_number(wtmd.get(key, default), "wtmd." + key, positive=True)
        lo, hi = wtmd.get("psi_min", 0.0), wtmd.get("psi_max", 10.0)
        _finite_number(lo, "wtmd.psi_min")
        _finite_number(hi, "wtmd.psi_max")
        if hi <= lo or (hi - lo) / wtmd.get("dpsi", 1e-3) < 2:
            raise ValidationError("WTMD range must contain at least two bins")
