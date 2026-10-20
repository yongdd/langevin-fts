"""Shared utilities for polymer field theory simulations.

This module provides common utility functions and constants used across
the SCFT and LFTS simulation classes to reduce code duplication.
"""

import ast
import json
import operator
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
            except (SyntaxError, ValueError, ArithmeticError) as e:
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
        Lattice angles [alpha, beta, gamma] in degrees. A non-orthogonal cell
        is written as a structured grid with explicit point coordinates.
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
