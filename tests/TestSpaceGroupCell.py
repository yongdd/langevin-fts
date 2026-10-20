#!/usr/bin/env python3
"""The cell must belong to the crystal system of the space group.

SpaceGroup.check_cell runs when a space group is attached to a computation box
and whenever the box changes (set_lx, set_lattice_parameters). Cells that break
the symmetry must be rejected, for every crystal system: cubic a = b = c,
tetragonal a = b, orthorhombic right angles, hexagonal (and trigonal in
hexagonal axes) a = b with gamma = 120, rhombohedral a = b = c with
alpha = beta = gamma, and monoclinic with the unique axis perpendicular to the
other two. Matching cells must be accepted.
"""

import sys

from polymerfts import PropagatorSolver, SpaceGroup, PlatformSelector


def cpu_platform():
    avail = PlatformSelector.avail_platforms()
    cpu = [p for p in avail if p.startswith("cpu")]
    return "cpu-mkl" if "cpu-mkl" in avail else (cpu[0] if cpu else None)


def solver(lx, symbol, hall=None, angles=None):
    nx = [8, 8, 8]
    sg = SpaceGroup(nx, symbol, hall) if hall is not None else SpaceGroup(nx, symbol)
    s = PropagatorSolver(nx=nx, lx=lx, ds=0.1, bond_lengths={"A": 1.0}, bc=["periodic"] * 6,
                         chain_model="continuous", numerical_method="rqm4", platform=cpu_platform(),
                         angles=angles, space_group=sg)
    s.add_polymer(1.0, [["A", 1.0, 0, 1]])
    s.propagator_computation                     # creates the box and attaches the space group
    return s


def main():
    if cpu_platform() is None:
        print("SKIP: no CPU platform in this build")
        return
    failed = []

    def expect_accept(what, make):
        try:
            return make()
        except Exception as e:
            failed.append(f"{what} rejected: {e}")

    def expect_reject(what, make):
        try:
            make()
            failed.append(f"{what} accepted")
        except Exception as e:
            if "crystal system" not in str(e):
                failed.append(f"{what}: unexpected error {e}")

    hex_angles = [90.0, 90.0, 120.0]

    # Cubic, tetragonal, orthorhombic
    expect_accept("Im-3m on a cube", lambda: solver([2.0, 2.0, 2.0], "Im-3m", 529))
    expect_accept("P4/mmm with a = b", lambda: solver([2.0, 2.0, 3.0], "P4/mmm"))
    expect_accept("Pmmm on a box", lambda: solver([2.0, 2.5, 3.0], "Pmmm"))
    expect_reject("Pm-3m with lx = [2, 2, 2.2]", lambda: solver([2.0, 2.0, 2.2], "Pm-3m"))
    expect_reject("P4/mmm with lx = [2, 2.5, 3]", lambda: solver([2.0, 2.5, 3.0], "P4/mmm"))
    expect_reject("Pmmm on a 120-degree cell", lambda: solver([2.0, 2.5, 3.0], "Pmmm", angles=hex_angles))

    # Hexagonal, trigonal in hexagonal axes, rhombohedral axes
    expect_accept("P6/mmm on a hexagonal cell", lambda: solver([2.0, 2.0, 2.5], "P6/mmm", angles=hex_angles))
    expect_reject("P6/mmm on an orthogonal cell", lambda: solver([2.0, 2.0, 2.5], "P6/mmm"))
    expect_reject("P-3m1 on an orthogonal cell", lambda: solver([2.0, 2.0, 2.5], "P-3m1"))
    expect_accept("R-3m (rhombohedral axes) with alpha = beta = gamma",
                  lambda: solver([2.0, 2.0, 2.0], "R-3m", 459, angles=[80.0, 80.0, 80.0]))
    expect_reject("R-3m (rhombohedral axes) on a hexagonal cell",
                  lambda: solver([2.0, 2.0, 2.0], "R-3m", 459, angles=hex_angles))

    # Monoclinic P2/m with unique axis b: alpha = gamma = 90, beta free
    expect_accept("P2/m (unique b) with beta = 70", lambda: solver([2.0, 2.0, 2.0], "P2/m", 57, angles=[90.0, 70.0, 90.0]))
    expect_reject("P2/m (unique b) with alpha = 70", lambda: solver([2.0, 2.0, 2.0], "P2/m", 57, angles=[70.0, 90.0, 90.0]))

    # Box changes: set_lx and set_lattice_parameters
    s = expect_accept("Im-3m on a cube", lambda: solver([2.0, 2.0, 2.0], "Im-3m", 529))
    if s is not None:
        expect_accept("Im-3m box change to [2.1, 2.1, 2.1]", lambda: s.set_lx([2.1, 2.1, 2.1]))
        expect_reject("Im-3m box change to [2.1, 2.1, 2.3]", lambda: s.set_lx([2.1, 2.1, 2.3]))
    s = expect_accept("P6/mmm on a hexagonal cell", lambda: solver([2.0, 2.0, 2.5], "P6/mmm", angles=hex_angles))
    if s is not None:
        expect_accept("P6/mmm lattice change keeping a = b",
                      lambda: s.set_lattice_parameters([2.1, 2.1, 2.6], hex_angles))
        expect_reject("P6/mmm lattice change to a != b",
                      lambda: s.set_lattice_parameters([2.1, 2.2, 2.6], hex_angles))
    s = expect_accept("Pmmm on a box", lambda: solver([2.0, 2.5, 3.0], "Pmmm"))
    if s is not None:
        expect_reject("Pmmm lattice change to gamma = 120",
                      lambda: s.set_lattice_parameters([2.0, 2.5, 3.0], hex_angles))

    for f in failed:
        print("FAILED:", f)
    if failed:
        sys.exit(1)
    print("Space-group cells: OK")


if __name__ == "__main__":
    main()
