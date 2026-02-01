#!/usr/bin/env python3
"""Convert PSCF output files to langevin-fts .mat format.

This script reads PSCF omega (field) and rho (density) files and
converts them to a MATLAB/Python compatible .mat file.

Usage:
    1. Place omega and rho files from PSCF in this directory
    2. Run: python to_matlab.py

Input:
    omega  - PSCF field file
    rho    - PSCF density file

Output:
    fields.mat - MATLAB/Python compatible file with fields and densities

Supported crystal systems:
    orthorhombic, cubic, tetragonal
"""

import string
import numpy as np
from scipy.io import savemat


def main():
    # Read PSCF output files
    with open("omega", 'r') as fp:
        lines_omega = fp.readlines()
    with open("rho", 'r') as fp:
        lines_rho = fp.readlines()

    # Parse header parameters
    params = parse_header(lines_omega)
    print(f"Parameters: {params}")

    # Handle crystal system specific cell parameters
    if params["crystal_system"] == "cubic":
        a = params["cell_param"][0]
        params["cell_param"] = [a, a, a]
    elif params["crystal_system"] == "tetragonal":
        a, c = params["cell_param"][0], params["cell_param"][1]
        params["cell_param"] = [a, a, c]

    # Reverse order (PSCF uses z,y,x; we use x,y,z)
    params["mesh"].reverse()
    params["cell_param"].reverse()

    # Find data start line
    data_start = find_data_start(lines_omega)
    print(f"Data lines: {len(lines_omega[data_start:])}")
    print(f"Expected grid points: {np.prod(params['mesh'])}")

    # Read field and density data
    w, phi = read_field_data(lines_omega, lines_rho, data_start, params)

    # Create output dictionary
    monomer_types = string.ascii_uppercase[:params["N_monomer"]]
    mdic = {
        "dim": params["dim"],
        "crystal_system": params["crystal_system"],
        "N_cell_param": params["N_cell_param"],
        "group_name": params["group_name"],
        "nx": params["mesh"],
        "lx": params["cell_param"],
        "chain_model": "continuous",
        "monomer_types": monomer_types,
    }

    # Add field and density arrays
    for i, name in enumerate(monomer_types):
        mdic[f"w_{name}"] = w[i]
        mdic[f"phi_{name}"] = phi[i]

    # Save to .mat file
    savemat("fields.mat", mdic, long_field_names=True, do_compression=True)
    print("Written: fields.mat")


def parse_header(lines):
    """Parse PSCF file header to extract parameters."""
    params = {}
    for i, line in enumerate(lines):
        if "format" in line:
            params["format"] = lines[i].split()[1:]
        elif "dim" in line:
            params["dim"] = int(lines[i+1].split()[0])
        elif "crystal_system" in line:
            params["crystal_system"] = lines[i+1].split()[0]
        elif "N_cell_param" in line:
            params["N_cell_param"] = int(lines[i+1].split()[0])
        elif "cell_param" in line:
            params["cell_param"] = [float(f) for f in lines[i+1].split()]
        elif "group_name" in line:
            params["group_name"] = lines[i+1].split()[0]
        elif "N_monomer" in line:
            params["N_monomer"] = int(lines[i+1].split()[0])
        elif "mesh" in line:
            params["mesh"] = [int(x) for x in lines[i+1].split()]
            break
    return params


def find_data_start(lines):
    """Find the line number where field data begins."""
    for i, line in enumerate(lines):
        if "mesh" in line:
            return i + 2
    raise ValueError("Could not find 'mesh' keyword in file")


def read_field_data(lines_omega, lines_rho, data_start, params):
    """Read field and density data from PSCF files."""
    n_monomer = params["N_monomer"]
    n_grid = np.prod(params["mesh"])

    w = np.zeros([n_monomer, n_grid])
    phi = np.zeros([n_monomer, n_grid])

    for i, line in enumerate(lines_omega[data_start:]):
        w[:, i] = np.array([float(f) for f in line.split()])

    for i, line in enumerate(lines_rho[data_start:]):
        phi[:, i] = np.array([float(f) for f in line.split()])

    return w, phi


if __name__ == "__main__":
    main()
