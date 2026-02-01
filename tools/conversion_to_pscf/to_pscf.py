#!/usr/bin/env python3
"""Convert langevin-fts Python script to PSCF input files.

This script parses a langevin-fts simulation script and generates
PSCF-compatible parameter and field files.

Usage:
    python to_pscf.py --file_name /path/to/your_scft_script.py

Output:
    param     - PSCF parameter file
    in/omega  - PSCF field file

Supported features:
    - Linear and branched polymer architectures
    - Multi-monomer systems
    - Automatic vertex re-numbering for branched polymers
"""

import sys
import ast
import numpy as np
import argparse


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Convert langevin-fts script to PSCF format')
    parser.add_argument('--file_name', type=str, required=True,
                        help='Path to langevin-fts simulation script')
    args, unknown = parser.parse_known_args()

    file_name = args.file_name
    print(f"Reading: {file_name}")

    # Read the simulation script
    with open(file_name, 'r') as fp:
        code = fp.read()

    # Remove simulation-specific imports and calls
    lines = code.splitlines()
    filtered_id = ""
    new_lines = []
    for line in lines:
        if "import scft" in line or "import lfts" in line:
            continue
        elif "scft." in line or "lfts." in line:
            filtered_id = line.split()[0]
            continue
        elif filtered_id and (filtered_id + ".") in line:
            continue
        new_lines.append(line)

    filtered_code = '\n'.join(new_lines)

    # Execute to extract params and fields
    exec(filtered_code, globals())

    # Print info
    print(f"Segment lengths: {params['segment_lengths']}")
    print(f"Grid: {params['nx']}")
    for monomer_type in params["segment_lengths"]:
        field = eval(f"w_{monomer_type}")
        print(f"w_{monomer_type}.shape: {field.shape}")

    # Generate PSCF param file
    nx = params["nx"]
    pscf_param = generate_param_file(params)

    # Generate PSCF omega file
    pscf_omega = generate_omega_file(params, nx)

    # Write output files
    with open("param", 'w') as f:
        f.write(pscf_param)
    print("Written: param")

    with open("in/omega", 'w') as f:
        f.write(pscf_omega)
    print("Written: in/omega")


def generate_param_file(params):
    """Generate PSCF parameter file content."""
    nx = params["nx"]

    # Build monomer index mapping
    monomer_to_idx = {}
    for count, monomer_type in enumerate(params["segment_lengths"]):
        monomer_to_idx[monomer_type] = count

    # Header
    pscf = f"""System{{
  Mixture{{
    nMonomer  {len(params['segment_lengths'])}
    monomers[
"""
    # Monomer segment lengths
    for monomer_type in params["segment_lengths"]:
        pscf += f"      {params['segment_lengths'][monomer_type]}\n"
    pscf += "    ]\n"

    # Polymers
    pscf += f"    nPolymer  {len(params['distinct_polymers'])}\n"
    for polymer in params["distinct_polymers"]:
        pscf += "    Polymer{\n"

        # Determine polymer type
        poly_type = "branched" if "v" in polymer["blocks"][0] else "linear"
        pscf += f"      type    {poly_type}\n"
        pscf += f"      nBlock  {len(polymer['blocks'])}\n"
        pscf += "      blocks\n"

        # Re-number vertices for branched polymers
        if poly_type == "branched":
            indices = set()
            for block in polymer["blocks"]:
                indices.update([block["v"], block["u"]])
            vertex_to_index = {v: i for i, v in enumerate(sorted(indices))}
            print(f"Vertex mapping: {vertex_to_index}")

        # Write blocks
        for block in polymer["blocks"]:
            if poly_type == "linear":
                pscf += f"        {monomer_to_idx[block['type']]}  {block['length']}\n"
            else:
                pscf += (f"        {monomer_to_idx[block['type']]}  {block['length']} "
                        f"{vertex_to_index[block['v']]} {vertex_to_index[block['u']]}\n")

        pscf += f"      phi  {polymer['volume_fraction']}\n"
        pscf += "    }\n"

    pscf += f"    ds  {params['ds']}\n"
    pscf += "  }\n"

    # Interactions
    pscf += "  Interaction{\n    chi(\n"
    for chi_n_pair in params["chi_n"]:
        m1, m2 = chi_n_pair.split(",")
        pscf += f"      {monomer_to_idx[m1]}  {monomer_to_idx[m2]}  {params['chi_n'][chi_n_pair]}\n"
    pscf += "    )\n  }\n"

    # Domain
    mesh_str = "   ".join(map(str, nx))
    is_flexible = 1 if params.get("box_is_altering", False) else 0
    pscf += f"""  Domain{{
    mesh        {mesh_str}
    lattice     orthorhombic
    groupName   P_1
  }}
  AmIteratorGrid{{
    epsilon  1.0e-5
    maxItr   1000
    maxHist  20
    isFlexible   {is_flexible}
    scaleStress  1.0
  }}
}}
"""
    return pscf


def generate_omega_file(params, nx):
    """Generate PSCF omega (field) file content."""
    S = len(params["segment_lengths"])
    total_grid = np.prod(nx)

    # Build monomer index mapping
    monomer_to_idx = {}
    for count, monomer_type in enumerate(params["segment_lengths"]):
        monomer_to_idx[monomer_type] = count

    # Header
    omega = f"""format   1   0
dim
                  {len(nx)}
crystal_system
          orthorhombic
N_cell_param
                  {len(nx)}
cell_param
                  {"   ".join(map(str, params['lx']))}
group_name
    P_1
N_monomer
                  {S}
mesh
                  {"   ".join(map(str, nx))}
"""

    # Collect and reorder fields
    w = np.zeros([S, total_grid], dtype=np.float64)
    for count, monomer_type in enumerate(params["segment_lengths"]):
        field = eval(f"w_{monomer_type}")
        # Transpose from (x,y,z) to (z,y,x) order for PSCF
        w[count, :] = np.reshape(np.reshape(field, nx).transpose(2, 1, 0), total_grid)

    # Write field data
    for i in range(w.shape[1]):
        for n in monomer_to_idx.values():
            omega += f"  {w[n, i]:15.10e}"
        omega += "\n"

    return omega


if __name__ == "__main__":
    main()
