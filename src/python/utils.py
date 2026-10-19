"""Shared utilities for polymer field theory simulations.

This module provides common utility functions and constants used across
the SCFT and LFTS simulation classes to reduce code duplication.
"""

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import yaml

from .validation import ValidationError


# =============================================================================
# Polymer architecture drawing (SCFT/LFTS draw at construction; draw_polymer_architectures redraws)
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

    dict_color = {}
    for count, monomer_type in enumerate(segment_lengths.keys()):
        if count < len(colors):
            dict_color[monomer_type] = colors[count]
        else:
            dict_color[monomer_type] = np.random.rand(3,)

    return dict_color


def draw_polymer_architectures(polymers, ds, segment_lengths, prefix="polymer"):
    """Draw every polymer of a simulation to ``<prefix>_<id>.png`` in the current directory.

    Example: ``draw_polymer_architectures(calc.distinct_polymers, calc.ds, calc.segment_lengths)``
    after ``calc = SCFT(params)`` (or LFTS / CLFTS).
    """
    dict_color = create_monomer_color_dict(segment_lengths)
    for idx, polymer in enumerate(polymers):
        draw_polymer_architecture(polymer, idx, ds, dict_color, output_file=f"{prefix}_{idx}.png")


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


# =============================================================================
# VTK output for ParaView
# =============================================================================

def save_vtk(
    path: Union[str, Path],
    field: np.ndarray,
    lx: List[float],
    field_name: str = "phi"
) -> None:
    """Save 3D field as VTK file for ParaView visualization.

    Parameters
    ----------
    path : str or Path
        Output file path (should end with .vtk).
    field : np.ndarray
        3D field data, shape (nx, ny, nz).
    lx : list of float
        Box dimensions [lx, ly, lz].
    field_name : str
        Name for the scalar field (default: "phi").
    """
    if field.ndim != 3:
        raise ValueError(f"Field must be 3D, got {field.ndim}D")

    nx, ny, nz = field.shape

    with open(path, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Polymer field data\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write(f"SPACING {lx[0]/nx} {lx[1]/ny} {lx[2]/nz}\n")
        f.write(f"POINT_DATA {nx*ny*nz}\n")
        f.write(f"SCALARS {field_name} float 1\n")
        f.write("LOOKUP_TABLE default\n")

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{field[i, j, k]:.6f}\n")
