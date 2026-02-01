"""Configuration file support for polymer field theory simulations.

This module provides utilities for loading simulation parameters from
YAML and JSON configuration files, enabling users to run simulations
without writing Python scripts.
"""

import json
import os
from typing import Dict, Any, Optional, Union
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


def save_config(params: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save simulation parameters to a configuration file.

    Parameters
    ----------
    params : dict
        Simulation parameters dictionary.
    path : str or Path
        Output file path. Format determined by extension.

    Raises
    ------
    ConfigError
        If file format is not supported.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Convert numpy arrays to lists for serialization
    params_serializable = _make_serializable(params)

    with open(path, 'w') as f:
        if suffix in ('.yaml', '.yml'):
            yaml.dump(params_serializable, f, default_flow_style=None,
                     width=90, sort_keys=False)
        elif suffix == '.json':
            json.dump(params_serializable, f, indent=2)
        else:
            raise ConfigError(
                f"Unsupported configuration file format: {suffix}. "
                "Use .yaml, .yml, or .json"
            )


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
            except Exception:
                return obj  # Return original string if evaluation fails
        return obj
    else:
        return obj


def _make_serializable(obj: Any) -> Any:
    """Convert objects to JSON/YAML serializable format.

    Parameters
    ----------
    obj : Any
        Object to convert.

    Returns
    -------
    Any
        Serializable version of object.
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def create_template_config(path: Union[str, Path], simulation_type: str = "scft") -> None:
    """Create a template configuration file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    simulation_type : str
        Type of simulation: "scft" or "lfts".

    Examples
    --------
    >>> create_template_config("template.yaml", simulation_type="scft")
    """
    if simulation_type == "scft":
        template = {
            "nx": [32, 32, 32],
            "lx": [4.0, 4.0, 4.0],

            "chain_model": "continuous",
            "ds": 0.01,

            "segment_lengths": {"A": 1.0, "B": 1.0},
            "chi_n": {"A,B": 15.0},

            "distinct_polymers": [{
                "volume_fraction": 1.0,
                "blocks": [
                    {"type": "A", "length": 0.5},
                    {"type": "B", "length": 0.5}
                ]
            }],

            "optimizer": {
                "name": "am",
                "max_hist": 20,
                "start_error": 0.1,
                "mix_min": 0.1,
                "mix_init": 0.1
            },

            "box_is_altering": False,
            "max_iter": 2000,
            "tolerance": 1e-8,
            "verbose_level": 1
        }
    elif simulation_type == "lfts":
        template = {
            "nx": [32, 32, 32],
            "lx": [4.0, 4.0, 4.0],

            "chain_model": "discrete",
            "ds": 0.01,

            "segment_lengths": {"A": 1.0, "B": 1.0},
            "chi_n": {"A,B": 15.0},

            "distinct_polymers": [{
                "volume_fraction": 1.0,
                "blocks": [
                    {"type": "A", "length": 0.5},
                    {"type": "B", "length": 0.5}
                ]
            }],

            "langevin": {
                "nbar": 10000,
                "dt": 0.8
            },

            "recording": {
                "dir": "data",
                "recording_period": 1000,
                "sf_computing_period": 10,
                "sf_recording_period": 100
            },

            "box_is_altering": False,
            "verbose_level": 1
        }
    else:
        raise ConfigError(f"Unknown simulation type: {simulation_type}")

    save_config(template, path)
