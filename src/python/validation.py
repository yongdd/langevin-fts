"""
Parameter validation utilities for polymer field theory simulations.

This module provides validation functions for SCFT and L-FTS simulation
parameters, ensuring proper types, values, and consistency before
expensive computations begin.

These validations raise descriptive exceptions rather than assertions,
ensuring they work even when Python optimization is enabled (-O flag).
"""

from typing import Dict, List, Any, Optional, Union
import itertools
import numpy as np


class ValidationError(Exception):
    """Exception raised for simulation parameter validation errors."""
    pass


def validate_type(value: Any, expected_type: type, name: str) -> None:
    """Validate that a value has the expected type.

    Parameters
    ----------
    value : Any
        The value to check
    expected_type : type
        Expected type (or tuple of types)
    name : str
        Parameter name for error messages

    Raises
    ------
    ValidationError
        If value is not of expected type
    """
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_names = " or ".join(t.__name__ for t in expected_type)
        else:
            type_names = expected_type.__name__
        raise ValidationError(
            f"'{name}' must be {type_names}, got {type(value).__name__}"
        )


def validate_required_keys(params: Dict, required: List[str]) -> None:
    """Validate that all required keys are present in params dict.

    Parameters
    ----------
    params : dict
        Parameter dictionary to check
    required : list of str
        List of required key names

    Raises
    ------
    ValidationError
        If any required key is missing
    """
    missing = [key for key in required if key not in params]
    if missing:
        raise ValidationError(
            f"Missing required parameters: {', '.join(missing)}"
        )


def validate_positive(value: Union[int, float], name: str) -> None:
    """Validate that a numeric value is positive.

    Parameters
    ----------
    value : int or float
        Value to check
    name : str
        Parameter name for error messages

    Raises
    ------
    ValidationError
        If value is not positive
    """
    if value <= 0:
        raise ValidationError(
            f"'{name}' must be positive, got {value}"
        )


def validate_list_length(value: List, expected_len: int, name: str) -> None:
    """Validate that a list has expected length.

    Parameters
    ----------
    value : list
        List to check
    expected_len : int
        Expected length
    name : str
        Parameter name for error messages

    Raises
    ------
    ValidationError
        If length doesn't match
    """
    if len(value) != expected_len:
        raise ValidationError(
            f"'{name}' must have {expected_len} elements, got {len(value)}"
        )


def validate_scft_params(params: Dict[str, Any]) -> None:
    """Validate SCFT simulation parameters.

    Performs comprehensive validation of parameter dictionary for SCFT
    simulations, checking types, required fields, and value constraints.

    Parameters
    ----------
    params : dict
        SCFT parameter dictionary containing:
        - nx : list of int - Grid dimensions
        - lx : list of float - Box dimensions
        - ds : float - Contour step size
        - segment_lengths : dict - Monomer segment lengths
        - chi_n : dict - Flory-Huggins parameters
        - distinct_polymers : list - Polymer definitions

    Raises
    ------
    ValidationError
        If any parameter is invalid

    Examples
    --------
    >>> params = {"nx": [32, 32], "lx": [4.0, 4.0], ...}
    >>> validate_scft_params(params)  # raises ValidationError if invalid
    """
    # Check required keys
    required = ["nx", "lx", "ds", "segment_lengths", "chi_n", "distinct_polymers"]
    validate_required_keys(params, required)

    # Validate nx (grid dimensions)
    nx = params["nx"]
    validate_type(nx, (list, tuple, np.ndarray), "nx")
    if len(nx) not in (1, 2, 3):
        raise ValidationError("'nx' must have 1, 2, or 3 dimensions")
    for i, n in enumerate(nx):
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValidationError(f"nx[{i}] must be a positive integer, got {n}")

    # Validate lx (box dimensions)
    lx = params["lx"]
    validate_type(lx, (list, tuple, np.ndarray), "lx")
    validate_list_length(lx, len(nx), "lx")
    for i, l in enumerate(lx):
        if not isinstance(l, (int, float, np.integer, np.floating)) or l <= 0:
            raise ValidationError(f"lx[{i}] must be a positive number, got {l}")

    # Validate ds (contour step)
    ds = params["ds"]
    validate_type(ds, (int, float), "ds")
    validate_positive(ds, "ds")
    if ds > 1:
        raise ValidationError(f"'ds' should be <= 1.0, got {ds}")

    # Validate segment_lengths
    segment_lengths = params["segment_lengths"]
    validate_type(segment_lengths, dict, "segment_lengths")
    if len(segment_lengths) == 0:
        raise ValidationError("'segment_lengths' cannot be empty")
    for monomer, length in segment_lengths.items():
        validate_type(monomer, str, f"segment_lengths key '{monomer}'")
        if not isinstance(length, (int, float)) or length <= 0:
            raise ValidationError(
                f"segment_lengths['{monomer}'] must be positive, got {length}"
            )

    # Validate chi_n
    chi_n = params["chi_n"]
    validate_type(chi_n, dict, "chi_n")
    monomer_types = list(segment_lengths.keys())
    for pair_str, value in chi_n.items():
        if not isinstance(value, (int, float)):
            raise ValidationError(
                f"chi_n['{pair_str}'] must be a number, got {type(value).__name__}"
            )

    # Validate distinct_polymers
    distinct_polymers = params["distinct_polymers"]
    validate_type(distinct_polymers, list, "distinct_polymers")
    if len(distinct_polymers) == 0:
        raise ValidationError("'distinct_polymers' cannot be empty")

    total_volume_fraction = 0.0
    for i, polymer in enumerate(distinct_polymers):
        validate_type(polymer, dict, f"distinct_polymers[{i}]")

        if "volume_fraction" not in polymer:
            raise ValidationError(
                f"distinct_polymers[{i}] missing 'volume_fraction'"
            )
        vf = polymer["volume_fraction"]
        if not isinstance(vf, (int, float)) or vf < 0 or vf > 1:
            raise ValidationError(
                f"distinct_polymers[{i}]['volume_fraction'] must be in [0, 1], got {vf}"
            )
        total_volume_fraction += vf

        if "blocks" not in polymer:
            raise ValidationError(
                f"distinct_polymers[{i}] missing 'blocks'"
            )
        blocks = polymer["blocks"]
        validate_type(blocks, list, f"distinct_polymers[{i}]['blocks']")
        if len(blocks) == 0:
            raise ValidationError(
                f"distinct_polymers[{i}]['blocks'] cannot be empty"
            )

    if abs(total_volume_fraction - 1.0) > 1e-10:
        raise ValidationError(
            f"Sum of volume fractions must be 1.0, got {total_volume_fraction}"
        )

    # Validate optional parameters
    if "chain_model" in params:
        chain_model = params["chain_model"]
        if chain_model not in ("continuous", "discrete"):
            raise ValidationError(
                f"'chain_model' must be 'continuous' or 'discrete', got '{chain_model}'"
            )

    if "platform" in params:
        platform = params["platform"]
        if platform not in ("cuda", "cpu-fftw"):
            raise ValidationError(
                f"'platform' must be 'cuda' or 'cpu-fftw', got '{platform}'"
            )


def validate_lfts_params(params: Dict[str, Any]) -> None:
    """Validate L-FTS simulation parameters.

    Performs comprehensive validation of parameter dictionary for Langevin
    field-theoretic simulations, checking types, required fields, and
    value constraints.

    Parameters
    ----------
    params : dict
        L-FTS parameter dictionary containing all SCFT parameters plus:
        - langevin: dict with 'nbar' and 'dt' for Langevin dynamics

    Raises
    ------
    ValidationError
        If any parameter is invalid
    """
    # First validate common SCFT parameters
    validate_scft_params(params)

    # Validate Langevin-specific parameters
    if "langevin" not in params:
        raise ValidationError("Missing required parameter: 'langevin'")

    langevin = params["langevin"]
    validate_type(langevin, dict, "langevin")

    if "nbar" not in langevin:
        raise ValidationError("'langevin' dict missing 'nbar'")
    nbar = langevin["nbar"]
    if not isinstance(nbar, (int, float)) or nbar <= 0:
        raise ValidationError(
            f"langevin['nbar'] must be positive, got {nbar}"
        )

    if "dt" not in langevin:
        raise ValidationError("'langevin' dict missing 'dt'")
    dt = langevin["dt"]
    if not isinstance(dt, (int, float)) or dt <= 0:
        raise ValidationError(
            f"langevin['dt'] must be positive, got {dt}"
        )


# -------------------- Runtime Guardrails --------------------

# Methods that are known to have approximate (not exact) material conservation.
NON_EXACT_CONSERVATION_METHODS = set()

# Conservative defaults that should not trip existing workflows while still
# catching large regressions or numerical blow-ups.
DEFAULT_RUNTIME_VALIDATION_EXACT = {
    "enabled": True,
    "mass_tol": 5e-8,
    "partition_check": True,
    "partition_enforce": True,
    "report": True,
}

DEFAULT_RUNTIME_VALIDATION_APPROX = {
    "enabled": True,
    "mass_tol": 1e-6,
    "partition_check": True,
    # For approximate methods, report partition issues but do not raise by default.
    "partition_enforce": False,
    "report": True,
}


def _coerce_runtime_validation_config(
    user_config: Optional[Dict[str, Any]],
    numerical_method: Optional[str],
) -> Dict[str, Any]:
    """Merge user runtime validation config with method-aware defaults."""
    method = (numerical_method or "").lower()
    if method in NON_EXACT_CONSERVATION_METHODS:
        config = dict(DEFAULT_RUNTIME_VALIDATION_APPROX)
    else:
        config = dict(DEFAULT_RUNTIME_VALIDATION_EXACT)

    if user_config is None:
        return config
    if not isinstance(user_config, dict):
        raise ValidationError(
            f"'validation_runtime' must be a dict when provided, got {type(user_config).__name__}"
        )

    config.update(user_config)
    return config


def validate_runtime_state(
    prop_solver: Any,
    phi: Dict[str, np.ndarray],
    monomer_types: List[str],
    numerical_method: Optional[str],
    user_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Validate material conservation and partition consistency at runtime.

    This function is intentionally lightweight so it can be called inside
    iteration loops without significant overhead.
    """
    config = _coerce_runtime_validation_config(user_config, numerical_method)
    if not config.get("enabled", True):
        return {
            "mass_error_mean": 0.0,
            "mass_error_max": 0.0,
            "partition_ok": 1.0,
        }

    # Material conservation: mean(sum(phi_i)) should be close to 1.
    total_phi = np.zeros_like(phi[monomer_types[0]])
    for monomer_type in monomer_types:
        total_phi = total_phi + phi[monomer_type]

    mass_error_field = total_phi - 1.0
    mass_error_mean = float(prop_solver.mean(mass_error_field))
    mass_error_max = float(np.max(np.abs(mass_error_field)))

    mass_tol = float(config.get("mass_tol", DEFAULT_RUNTIME_VALIDATION_EXACT["mass_tol"]))
    if abs(mass_error_mean) > mass_tol:
        raise ValidationError(
            "Material conservation failed: "
            f"mean(sum(phi)-1)={mass_error_mean:.3e} exceeds tolerance {mass_tol:.3e}"
        )

    # Partition consistency: use the C++ check when available.
    partition_ok = True
    if config.get("partition_check", True):
        try:
            partition_ok = bool(prop_solver.check_total_partition())
        except Exception:
            # Some solvers/methods may not expose this cleanly; treat as not ok.
            partition_ok = False

        if not partition_ok and config.get("partition_enforce", True):
            raise ValidationError(
                "Partition consistency check failed (forward/backward mismatch)."
            )

    return {
        "mass_error_mean": mass_error_mean,
        "mass_error_max": mass_error_max,
        "partition_ok": 1.0 if partition_ok else 0.0,
    }
