"""Shared utilities for polymer field theory simulations.

This module provides common utility functions and constants used across
the SCFT and LFTS simulation classes to reduce code duplication.
"""

import functools
import logging
import os
import re
import sys
import warnings
from typing import Dict, List, Any, Optional, Tuple, Callable

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .validation import ValidationError

# Module-level logger
logger = logging.getLogger(__name__)


# =============================================================================
# Logging Configuration
# =============================================================================

def configure_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    stream: Any = None
) -> None:
    """Configure logging for the polymerfts package.

    Sets up logging based on parameters, environment variables, or defaults.
    Priority: parameters > environment variables > defaults.

    Parameters
    ----------
    level : str, optional
        Logging level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        If not specified, reads from POLYMERFTS_LOG_LEVEL env var,
        defaults to "WARNING".
    format_string : str, optional
        Log message format string.
        If not specified, reads from POLYMERFTS_LOG_FORMAT env var.
    stream : file-like, optional
        Output stream. Defaults to sys.stderr.

    Examples
    --------
    >>> from polymerfts.utils import configure_logging
    >>> configure_logging(level="DEBUG")  # Enable debug output
    >>> configure_logging(level="ERROR")  # Only show errors

    >>> # Or via environment variable before running
    >>> # export POLYMERFTS_LOG_LEVEL=DEBUG
    """
    # Determine level
    if level is None:
        level = os.environ.get("POLYMERFTS_LOG_LEVEL", "WARNING")

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if level.upper() not in level_map:
        raise ValueError(
            f"Invalid log level: {level}. "
            f"Choose from: {', '.join(level_map.keys())}"
        )

    log_level = level_map[level.upper()]

    # Determine format
    if format_string is None:
        format_string = os.environ.get(
            "POLYMERFTS_LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Configure root logger for polymerfts
    pkg_logger = logging.getLogger("polymerfts")
    pkg_logger.setLevel(log_level)

    # Remove existing handlers
    pkg_logger.handlers.clear()

    # Add new handler
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(format_string))
    pkg_logger.addHandler(handler)


def get_verbose_level(params: Dict[str, Any]) -> int:
    """Get verbose level from params dict.

    Parameters
    ----------
    params : dict
        Simulation parameters.

    Returns
    -------
    int
        Verbose level (0=quiet, 1=normal, 2=verbose).
    """
    return params.get("verbose_level", 1)


# =============================================================================
# Deprecation Warnings
# =============================================================================

def deprecated(
    reason: str = "",
    version: Optional[str] = None,
    replacement: Optional[str] = None
) -> Callable:
    """Decorator to mark functions/methods as deprecated.

    Parameters
    ----------
    reason : str
        Explanation for why it's deprecated.
    version : str, optional
        Version when it will be removed.
    replacement : str, optional
        Name of replacement function/method.

    Returns
    -------
    Callable
        Decorated function that emits deprecation warning.

    Examples
    --------
    >>> @deprecated(reason="Use new_function instead", version="2.0")
    ... def old_function():
    ...     pass

    >>> @deprecated(replacement="SCFT.run")
    ... def legacy_run():
    ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"{func.__name__} is deprecated"
            if reason:
                msg += f": {reason}"
            if replacement:
                msg += f". Use {replacement} instead"
            if version:
                msg += f". Will be removed in version {version}"

            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        # Update docstring
        doc = func.__doc__ or ""
        deprecation_note = f"\n\n.. deprecated::\n   {reason or 'This function is deprecated.'}"
        if replacement:
            deprecation_note += f"\n   Use :func:`{replacement}` instead."
        wrapper.__doc__ = doc + deprecation_note

        return wrapper
    return decorator


def warn_deprecated_param(
    param_name: str,
    replacement: Optional[str] = None,
    version: Optional[str] = None
) -> None:
    """Emit a deprecation warning for a parameter.

    Parameters
    ----------
    param_name : str
        Name of the deprecated parameter.
    replacement : str, optional
        Name of replacement parameter.
    version : str, optional
        Version when it will be removed.

    Examples
    --------
    >>> if "old_param" in params:
    ...     warn_deprecated_param("old_param", replacement="new_param")
    ...     params["new_param"] = params["old_param"]
    """
    msg = f"Parameter '{param_name}' is deprecated"
    if replacement:
        msg += f". Use '{replacement}' instead"
    if version:
        msg += f". Will be removed in version {version}"

    warnings.warn(msg, DeprecationWarning, stacklevel=2)

# Default colors for monomer visualization
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

    dict_color = {}
    for count, monomer_type in enumerate(segment_lengths.keys()):
        if count < len(colors):
            dict_color[monomer_type] = colors[count]
        else:
            dict_color[monomer_type] = np.random.rand(3,)

    logger.debug(f"Monomer color mapping: {dict_color}")
    return dict_color


def draw_polymer_architecture(
    polymer: Dict[str, Any],
    polymer_id: int,
    ds: float,
    dict_color: Dict[str, Any],
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 20)
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
        Figure size in inches. Defaults to (20, 20).
    """
    # Make a graph
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
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='twopi')
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
    plt.savefig(output_file)
    plt.close()


def parse_chi_n(
    chi_n_params: Dict[str, float],
    segment_lengths: Dict[str, float]
) -> Dict[str, float]:
    """Parse and validate Flory-Huggins interaction parameters.

    Parameters
    ----------
    chi_n_params : dict
        Dictionary of chi_n values with monomer pair keys (e.g., "A,B": 15.0).
    segment_lengths : dict
        Dictionary of segment lengths for validation.

    Returns
    -------
    dict
        Validated and normalized chi_n dictionary with sorted monomer pair keys.

    Raises
    ------
    ValidationError
        If monomer types are invalid, self-interactions are specified,
        or duplicate parameters are found.
    """
    chi_n = {}
    for monomer_pair_str, chin_value in chi_n_params.items():
        monomer_pair = re.split(',| |_|/', monomer_pair_str)
        if monomer_pair[0] not in segment_lengths:
            raise ValidationError(f"Monomer type '{monomer_pair[0]}' is not in 'segment_lengths'.")
        if monomer_pair[1] not in segment_lengths:
            raise ValidationError(f"Monomer type '{monomer_pair[1]}' is not in 'segment_lengths'.")
        if monomer_pair[0] == monomer_pair[1]:
            raise ValidationError(f"Do not add self interaction parameter, {monomer_pair_str}.")
        monomer_pair.sort()
        sorted_monomer_pair = monomer_pair[0] + "," + monomer_pair[1]
        if sorted_monomer_pair in chi_n:
            raise ValidationError(f"There are duplicated χN ({sorted_monomer_pair}) parameters.")
        chi_n[sorted_monomer_pair] = chin_value
    return chi_n


def validate_polymers(
    distinct_polymers: List[Dict[str, Any]]
) -> None:
    """Validate polymer chain definitions.

    Parameters
    ----------
    distinct_polymers : list of dict
        List of polymer chain specifications.

    Raises
    ------
    ValidationError
        If no polymers are defined or volume fractions don't sum to 1.
    """
    if len(distinct_polymers) < 1:
        raise ValidationError("There is no polymer chain.")

    total_volume_fraction = sum(p["volume_fraction"] for p in distinct_polymers)
    if not np.isclose(total_volume_fraction, 1.0):
        raise ValidationError("The sum of volume fractions must be equal to 1.")


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
    is_random = any("fraction" in block for block in polymer["blocks"])
    if not is_random:
        return None

    if len(polymer["blocks"]) != 1:
        raise ValidationError("Only single block random copolymer is allowed.")

    statistical_segment_length = 0
    total_random_fraction = 0
    for monomer_type in polymer["blocks"][0]["fraction"]:
        statistical_segment_length += (
            segment_lengths[monomer_type] ** 2 * polymer["blocks"][0]["fraction"][monomer_type]
        )
        total_random_fraction += polymer["blocks"][0]["fraction"][monomer_type]
    statistical_segment_length = np.sqrt(statistical_segment_length)

    if not np.isclose(total_random_fraction, 1.0):
        raise ValidationError("The sum of volume fractions of random copolymer must be equal to 1.")

    random_type_string = polymer["blocks"][0]["type"]
    if random_type_string in segment_lengths:
        raise ValidationError(
            f"The name of random copolymer '{random_type_string}' is already used as a type "
            "in 'segment_lengths' or other random copolymer"
        )

    return (random_type_string, statistical_segment_length, polymer["blocks"][0]["fraction"])
