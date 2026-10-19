"""Configuration file support for polymer field theory simulations.

This module provides utilities for loading simulation parameters from
YAML and JSON configuration files, enabling users to run simulations
without writing Python scripts.
"""

import json
import warnings
from typing import Dict, Any, Union
from pathlib import Path

import yaml
import numpy as np


class ConfigError(Exception):
    """Exception raised for configuration file errors."""
    pass


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load simulation parameters from a configuration file.

    Supports YAML (.yaml, .yml) and JSON (.json) formats.
    Automatically detects format from file extension.

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

    Examples
    --------
    >>> params = load_config("simulation.yaml")
    >>> calc = SCFT(params=params)

    >>> # Or use the class method
    >>> calc = SCFT.from_file("simulation.yaml")
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


def _process_config_values(obj: Any) -> Any:
    """Process configuration values, evaluating special expressions.

    Handles:
    - String expressions like "4.0*np.sqrt(3)/2" for lx
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
        # Try to evaluate numpy expressions
        if 'np.' in obj or 'numpy.' in obj:
            try:
                # Safe evaluation with only numpy functions
                result = eval(obj, {"np": np, "numpy": np, "__builtins__": {}})
                return result
            except Exception as e:
                # A string that LOOKS like a numpy expression but fails to
                # evaluate is almost certainly a typo in the config; passing
                # it through silently would surface later as a confusing
                # type error. Warn so the user can fix the expression.
                warnings.warn(
                    f"Config value {obj!r} looks like a numpy expression but "
                    f"failed to evaluate ({e}); using the raw string instead.")
                return obj  # Return original string if evaluation fails
        return obj
    else:
        return obj
