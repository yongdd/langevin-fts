"""I/O utilities for polymer field theory simulations.

This module provides functions for saving and loading simulation data
in various formats including MATLAB (.mat), JSON, YAML, and VTK.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Union, Optional, List

import numpy as np
import yaml


def save_fields(
    path: Union[str, Path],
    phi: Dict[str, np.ndarray],
    w: np.ndarray,
    nx: List[int],
    lx: List[float],
    monomer_types: List[str],
    **kwargs: Any
) -> None:
    """Save field data to file.

    Parameters
    ----------
    path : str or Path
        Output file path. Format determined by extension:
        - .mat : MATLAB format
        - .json : JSON format
        - .yaml : YAML format
        - .npz : NumPy compressed format
    phi : dict
        Concentration fields {monomer_type: array}.
    w : np.ndarray
        Potential fields, shape (M, total_grid).
    nx : list of int
        Grid dimensions.
    lx : list of float
        Box dimensions.
    monomer_types : list of str
        Monomer type labels.
    **kwargs : Any
        Additional data to save (e.g., free_energy, chi_n).

    Examples
    --------
    >>> save_fields("output.mat", calc.phi, calc.w,
    ...             params["nx"], params["lx"], calc.monomer_types,
    ...             free_energy=calc.free_energy)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Build data dictionary
    data = {
        "nx": np.array(nx),
        "lx": np.array(lx),
        "monomer_types": monomer_types,
    }

    # Add concentration fields
    for mt in monomer_types:
        data[f"phi_{mt}"] = phi[mt]

    # Add potential fields
    for i, mt in enumerate(monomer_types):
        data[f"w_{mt}"] = w[i]

    # Add extra data
    data.update(kwargs)

    if suffix == ".mat":
        _save_mat(path, data)
    elif suffix == ".json":
        _save_json(path, data)
    elif suffix in (".yaml", ".yml"):
        _save_yaml(path, data)
    elif suffix == ".npz":
        _save_npz(path, data)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Use .mat, .json, .yaml, or .npz"
        )


def load_fields(path: Union[str, Path]) -> Dict[str, Any]:
    """Load field data from file.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    dict
        Dictionary containing loaded data.

    Examples
    --------
    >>> data = load_fields("output.mat")
    >>> phi_A = data["phi_A"]
    >>> w_A = data["w_A"]
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".mat":
        return _load_mat(path)
    elif suffix == ".json":
        return _load_json(path)
    elif suffix in (".yaml", ".yml"):
        return _load_yaml(path)
    elif suffix == ".npz":
        return _load_npz(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Use .mat, .json, .yaml, or .npz"
        )


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

    Examples
    --------
    >>> phi_A = calc.phi["A"].reshape(params["nx"])
    >>> save_vtk("concentration.vtk", phi_A, params["lx"])
    >>> # Open with: paraview concentration.vtk
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


def save_vtk_binary(
    path: Union[str, Path],
    field: np.ndarray,
    lx: List[float],
    field_name: str = "phi"
) -> None:
    """Save 3D field as binary VTK file (faster, smaller).

    Parameters
    ----------
    path : str or Path
        Output file path (should end with .vtk).
    field : np.ndarray
        3D field data, shape (nx, ny, nz).
    lx : list of float
        Box dimensions [lx, ly, lz].
    field_name : str
        Name for the scalar field.
    """
    if field.ndim != 3:
        raise ValueError(f"Field must be 3D, got {field.ndim}D")

    nx, ny, nz = field.shape

    with open(path, 'wb') as f:
        # Write header as ASCII
        header = (
            "# vtk DataFile Version 3.0\n"
            "Polymer field data\n"
            "BINARY\n"
            "DATASET STRUCTURED_POINTS\n"
            f"DIMENSIONS {nx} {ny} {nz}\n"
            "ORIGIN 0 0 0\n"
            f"SPACING {lx[0]/nx} {lx[1]/ny} {lx[2]/nz}\n"
            f"POINT_DATA {nx*ny*nz}\n"
            f"SCALARS {field_name} float 1\n"
            "LOOKUP_TABLE default\n"
        )
        f.write(header.encode('ascii'))

        # Write data as big-endian binary (VTK requirement)
        data = field.astype('>f4')  # Big-endian float32
        f.write(data.tobytes(order='F'))  # Fortran order for VTK


# Private helper functions

def _save_mat(path: Path, data: Dict[str, Any]) -> None:
    """Save data in MATLAB format."""
    try:
        import scipy.io
    except ImportError:
        raise ImportError("scipy required for .mat format. Install with: pip install scipy")

    # Convert to serializable format
    mat_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            mat_data[key] = value
        elif isinstance(value, (list, tuple)):
            mat_data[key] = np.array(value)
        else:
            mat_data[key] = value

    scipy.io.savemat(str(path), mat_data, long_field_names=True, do_compression=True)


def _load_mat(path: Path) -> Dict[str, Any]:
    """Load data from MATLAB format."""
    try:
        import scipy.io
    except ImportError:
        raise ImportError("scipy required for .mat format. Install with: pip install scipy")

    data = scipy.io.loadmat(str(path), squeeze_me=True)

    # Remove MATLAB metadata keys
    return {k: v for k, v in data.items() if not k.startswith('__')}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    """Save data in JSON format."""
    serializable = _make_serializable(data)
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


def _load_json(path: Path) -> Dict[str, Any]:
    """Load data from JSON format."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Convert lists back to numpy arrays for field data
    for key in list(data.keys()):
        if key.startswith(('phi_', 'w_')) or key in ('nx', 'lx'):
            data[key] = np.array(data[key])

    return data


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Save data in YAML format."""
    serializable = _make_serializable(data)
    with open(path, 'w') as f:
        yaml.dump(serializable, f, default_flow_style=None, width=90, sort_keys=False)


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load data from YAML format."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    # Convert lists back to numpy arrays for field data
    for key in list(data.keys()):
        if key.startswith(('phi_', 'w_')) or key in ('nx', 'lx'):
            data[key] = np.array(data[key])

    return data


def _save_npz(path: Path, data: Dict[str, Any]) -> None:
    """Save data in NumPy compressed format."""
    np.savez_compressed(str(path), **data)


def _load_npz(path: Path) -> Dict[str, Any]:
    """Load data from NumPy compressed format."""
    with np.load(str(path), allow_pickle=True) as npz:
        return dict(npz)


def _make_serializable(obj: Any) -> Any:
    """Convert objects to JSON/YAML serializable format."""
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
