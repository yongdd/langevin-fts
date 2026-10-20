#!/usr/bin/env python3
"""save_vtk: legacy VTK output for ParaView.

Writes fields and reads the files back: values in VTK point order (x fastest),
cell-centred grid points, several fields in one file, 1D/2D grids, flat arrays
with nx, a hexagonal cell as a structured grid, and errors for complex or
wrongly sized fields.
"""

import os
import sys
import tempfile

import numpy as np

from polymerfts import save_vtk


def read_vtk(path):
    """Header lines, point coordinates (structured grid only) and {name: values}."""
    lines = open(path).read().split("\n")
    header = {l.split()[0]: l.split()[1:] for l in lines[3:] if l and l.split()[0] in
              ("DATASET", "DIMENSIONS", "ORIGIN", "SPACING", "POINTS", "POINT_DATA")}
    n = int(header["POINT_DATA"][0])
    points, fields = None, {}
    for i, l in enumerate(lines):
        if l.startswith("POINTS"):
            points = np.loadtxt(lines[i+1:i+1+n])
        if l.startswith("SCALARS"):
            fields[l.split()[1]] = np.array(lines[i+2:i+2+n], dtype=float)
    return header, points, fields


def main():
    failed = []
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "f.vtk")

        # 3D field: x fastest, cell-centred origin
        field = rng.normal(size=(4, 3, 2))
        save_vtk(path, field, [2.0, 3.0, 4.0])
        header, _, fields = read_vtk(path)
        if header["DIMENSIONS"] != ["4", "3", "2"] or not np.allclose([float(v) for v in header["ORIGIN"]], [0.25, 0.5, 1.0]):
            failed.append(f"3D header {header}")
        if not np.allclose(fields["phi"], field.ravel(order="F"), rtol=1e-6, atol=1e-6):
            failed.append("3D values are not in x-fastest order")

        # Several fields from a dict, flat arrays with nx, a 2D grid
        a, b = rng.normal(size=12), rng.normal(size=12)
        save_vtk(path, {"phi A": a, "phi_B": b}, [1.0, 1.5], nx=[4, 3])
        header, _, fields = read_vtk(path)
        if header["DIMENSIONS"] != ["4", "3", "1"] or sorted(fields) != ["phi_A", "phi_B"]:
            failed.append(f"2D dict header {header}, fields {sorted(fields)}")
        elif not np.allclose(fields["phi_B"], b.reshape(4, 3).ravel(order="F"), rtol=1e-6, atol=1e-6):
            failed.append("2D flat values")

        # 1D grid
        line = rng.normal(size=5)
        save_vtk(path, line, [2.5], field_name="w")
        header, _, fields = read_vtk(path)
        if header["DIMENSIONS"] != ["5", "1", "1"] or abs(float(header["ORIGIN"][0]) - 0.25) > 1e-12 \
                or not np.allclose(fields["w"], line, rtol=1e-6, atol=1e-6):
            failed.append(f"1D grid {header}")

        # Hexagonal cell (gamma = 120 degrees): structured grid with lattice coordinates
        save_vtk(path, rng.normal(size=(3, 3, 2)), [1.0, 1.0, 2.0], angles=[90.0, 90.0, 120.0])
        header, points, _ = read_vtk(path)
        b_vec = np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3), 0.0])
        expected_last = (2.5/3)*np.array([1.0, 0.0, 0.0]) + (2.5/3)*b_vec + (1.5/2)*np.array([0.0, 0.0, 2.0])
        if header["DATASET"] != ["STRUCTURED_GRID"] or points.shape != (18, 3) \
                or not np.allclose(points[-1], expected_last, atol=1e-9):
            failed.append(f"hexagonal grid: {header['DATASET']}, last point {points[-1] if points is not None else None}")

        # Errors
        for bad, kwargs in ((np.ones((2, 2, 2)) + 1j, {}), (np.ones(7), {"nx": [2, 2, 2]})):
            try:
                save_vtk(path, bad, [1.0, 1.0, 1.0], **kwargs)
                failed.append("invalid field accepted")
            except ValueError:
                pass

    for f in failed:
        print("FAILED:", f)
    if failed:
        sys.exit(1)
    print("save_vtk: OK")


if __name__ == "__main__":
    main()
