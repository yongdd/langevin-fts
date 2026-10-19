"""I/O utilities for polymer field theory simulations.

This module provides a VTK writer for visualizing 3D concentration fields
(e.g., in ParaView). Simulation results themselves are written by
``SCFT.save_results`` / ``LFTS.save_simulation_data``.
"""

from pathlib import Path
from typing import Union, List

import numpy as np


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


