"""Parameter checks for SCFT, L-FTS and CL-FTS.

The checks raise ValidationError (never assert, which disappears under
``python -O``) before any expensive C++ object is created.
"""

from typing import Any, Dict

import numpy as np


class ValidationError(Exception):
    """Raised when a simulation parameter is missing or invalid."""
    pass


def validate_system_params(params: Dict[str, Any]) -> None:
    """Check the parameters shared by SCFT, L-FTS and CL-FTS.

    Required: nx, lx, ds, chain_model, segment_lengths, chi_n, distinct_polymers.
    The first problem found raises ValidationError with a message naming the key.
    """
    for key in ("nx", "lx", "ds", "chain_model", "segment_lengths", "chi_n", "distinct_polymers"):
        if key not in params:
            raise ValidationError(f"Missing required parameter: '{key}'")

    nx, lx = params["nx"], params["lx"]
    if not isinstance(nx, (list, tuple, np.ndarray)) or not isinstance(lx, (list, tuple, np.ndarray)):
        raise ValidationError("'nx' and 'lx' must be lists")
    if len(nx) not in (1, 2, 3):
        raise ValidationError("'nx' must have 1, 2, or 3 dimensions")
    if any(not isinstance(n, (int, np.integer)) or n <= 0 for n in nx):
        raise ValidationError(f"'nx' must contain positive integers, got {list(nx)}")
    if len(lx) != len(nx):
        raise ValidationError(f"'lx' must have {len(nx)} elements, got {len(lx)}")
    if any(not isinstance(l, (int, float, np.integer, np.floating)) or l <= 0 for l in lx):
        raise ValidationError(f"'lx' must contain positive numbers, got {list(lx)}")

    ds = params["ds"]
    if not isinstance(ds, (int, float)) or ds <= 0 or ds > 1:
        raise ValidationError(f"'ds' must be a number in (0, 1], got {ds}")
    if params["chain_model"] not in ("continuous", "discrete"):
        raise ValidationError(f"'chain_model' must be 'continuous' or 'discrete', got '{params['chain_model']}'")

    segment_lengths = params["segment_lengths"]
    if not isinstance(segment_lengths, dict) or len(segment_lengths) == 0:
        raise ValidationError("'segment_lengths' must be a non-empty dict {monomer_type: length}")
    for monomer, length in segment_lengths.items():
        if not isinstance(monomer, str) or not isinstance(length, (int, float)) or length <= 0:
            raise ValidationError(f"segment_lengths['{monomer}'] must be a positive number, got {length}")

    if not isinstance(params["chi_n"], dict):
        raise ValidationError("'chi_n' must be a dict, e.g. {'A,B': 15.0}")
    for pair, value in params["chi_n"].items():
        if not isinstance(value, (int, float)):
            raise ValidationError(f"chi_n['{pair}'] must be a number, got {type(value).__name__}")

    polymers = params["distinct_polymers"]
    if not isinstance(polymers, list) or len(polymers) == 0:
        raise ValidationError("'distinct_polymers' must be a non-empty list")
    total_volume_fraction = 0.0
    for i, polymer in enumerate(polymers):
        if not isinstance(polymer, dict) or "volume_fraction" not in polymer or "blocks" not in polymer:
            raise ValidationError(f"distinct_polymers[{i}] must be a dict with 'volume_fraction' and 'blocks'")
        vf = polymer["volume_fraction"]
        if not isinstance(vf, (int, float)) or vf < 0 or vf > 1:
            raise ValidationError(f"distinct_polymers[{i}]['volume_fraction'] must be in [0, 1], got {vf}")
        total_volume_fraction += vf
        if not isinstance(polymer["blocks"], list) or len(polymer["blocks"]) == 0:
            raise ValidationError(f"distinct_polymers[{i}]['blocks'] must be a non-empty list")
    if abs(total_volume_fraction - 1.0) > 1e-10:
        raise ValidationError(f"Sum of volume fractions must be 1.0, got {total_volume_fraction}")

    if "platform" in params and params["platform"] not in ("cuda", "cpu-fftw", "cpu-mkl"):
        raise ValidationError(f"'platform' must be 'cuda', 'cpu-fftw', or 'cpu-mkl', got '{params['platform']}'")


def validate_lfts_params(params: Dict[str, Any]) -> None:
    """Check the L-FTS parameters: the shared ones plus a 'langevin' dict with positive 'nbar' and 'dt'."""
    validate_system_params(params)
    langevin = params.get("langevin", None)
    if not isinstance(langevin, dict):
        raise ValidationError("Missing required parameter: 'langevin' (dict with 'nbar', 'dt', 'max_step')")
    for key in ("nbar", "dt"):
        if key not in langevin or not isinstance(langevin[key], (int, float)) or langevin[key] <= 0:
            raise ValidationError(f"langevin['{key}'] must be a positive number")
