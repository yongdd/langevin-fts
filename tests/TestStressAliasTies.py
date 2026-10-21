#!/usr/bin/env python3
"""
Stress at alias ties.

On a periodic grid each Fourier mode is defined modulo N, and the propagator
uses the alias with the smallest |k|^2. Representatives can tie: on the
Brillouin-zone boundary of a hexagonal cell (e.g. 2 m1 + m2 = N, three ways at
the K points) and on the Nyquist planes m = +-N/2 of any cell. |k|^2 is the
same for them, but the stress table v (x) v is not; the stress averages it over
the tied representatives. Until 2026-10 it took the first one, so for a field
with hexagonal symmetry dH/da and dH/db differed (1.7e-6 relative for the field
below), and the angle derivatives of an orthogonal cell
missed the central difference (0.15 relative for the white-noise field below,
whose Nyquist modes are strong; 1e-7 for a smooth field).

Checks (discrete chains, whose stress is the exact derivative of their ln Q):
1. P6_3/mmc field on 18x18x16 (three-way ties): dH/da = dH/db to 1e-12, their
   sum equals the central difference of the a = b deformation, and the stress
   with the space group on equals the stress with it off.
2. Hexagonal field on 16x16x16: dH/da equals the central difference with respect
   to a alone (a consistency check; this grid has no ties that change it).
3. Orthogonal cell with forced angle derivatives: dH/dgamma, dH/dbeta and
   dH/dalpha equal their central differences (relative to the largest, 1e-7).
"""

import io
import sys
import contextlib

import numpy as np
from polymerfts import _core
from polymerfts.propagator_solver import PropagatorSolver

n_pass = 0
n_fail = 0


def check(cond, msg):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"[PASS] {msg}")
    else:
        n_fail += 1
        print(f"[FAIL] {msg}")


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def cpu_platform():
    cpu = [p for p in _core.PlatformSelector.avail_platforms() if p.startswith("cpu")]
    return cpu[0] if cpu else "cuda"


BLOCKS = [["A", 0.4, 0, 1], ["B", 0.6, 1, 2]]


def solver(nx, lx, angles, sg=None):
    s = quiet(PropagatorSolver, nx=nx, lx=lx, ds=0.05, bond_lengths={"A": 1.0, "B": 1.3}, bc=["periodic"] * 6,
              chain_model="discrete", platform=cpu_platform(), angles=angles, space_group=sg)
    s.add_polymer(1.0, BLOCKS)
    return s


def stress(s, w):
    quiet(s.compute_propagators, w)
    s.compute_stress()
    return np.array(s.get_stress())


def minus_ln_q(s, w, lx, angles):
    s.set_lattice_parameters(lx, angles)
    s.update_laplacian_operator()
    quiet(s.compute_propagators, w)
    return -np.log(s.get_partition_function(0))


def central_difference(s, w, lx, angles, direction):
    """d(-ln Q)/dt along lx(t) = lx + t*direction[:3], angles(t) = angles + t*direction[3:] (degrees per radian).

    Where a deformation changes which alias is minimal (ties), -ln Q has terms like t|t|,
    and the central difference D(h) has an O(h) error besides the O(h^2) one;
    (8 D(h/4) - 6 D(h/2) + D(h))/3 removes both.
    """
    values = {}
    for h in (2e-3, 1e-3, 5e-4):
        f = []
        for sign in (1, -1):
            L = [lx[d] + sign * h * direction[d] for d in range(3)]
            A = [angles[d] + sign * np.rad2deg(h) * direction[3 + d] for d in range(3)]
            f.append(minus_ln_q(s, w, L, A))
        values[h] = (f[0] - f[1]) / (2 * h)
    s.set_lattice_parameters(lx, angles)
    s.update_laplacian_operator()
    return (8 * values[5e-4] - 6 * values[1e-3] + values[2e-3]) / 3


def hexagonal_field(nx):
    with contextlib.redirect_stdout(io.StringIO()):
        sg = _core.SpaceGroup(nx, "P6_3/mmc", 488)
    rng = np.random.default_rng(3)
    raw = rng.normal(0.0, 1.0, size=(2, int(np.prod(nx))))
    sym = sg.from_reduced_basis(sg.to_reduced_basis(raw))
    sym *= 3.0 / np.max(np.abs(sym))
    return sym, sg


def main():
    angles = [90.0, 90.0, 120.0]

    # 1. Three-way ties (18 is divisible by 3: the K points lie on the grid)
    nx, lx = [18, 18, 16], [1.8, 1.8, 2.3]
    sym, sg = hexagonal_field(nx)
    w = {"A": sym[0], "B": sym[1]}
    s = solver(nx, lx, angles)
    st = stress(s, w)
    rel = abs(st[0] - st[1]) / abs(st[0])
    check(rel < 1e-12, f"P6_3/mmc 18x18x16: dH/da {st[0]:.12e} = dH/db {st[1]:.12e} (rel. {rel:.1e})")
    fd_ab = central_difference(s, w, lx, angles, [lx[0], lx[1], 0, 0, 0, 0])
    total = st[0] * lx[0] + st[1] * lx[1]
    check(abs(total - fd_ab) < 1e-8 * abs(fd_ab),
          f"P6_3/mmc 18x18x16: a dH/da + b dH/db = d/dt at a = b ({abs(total - fd_ab) / abs(fd_ab):.1e})")
    with contextlib.redirect_stdout(io.StringIO()):
        sg_on = _core.SpaceGroup(nx, "P6_3/mmc", 488)
        s_on = solver(nx, lx, angles, sg_on)
        st_on = stress(s_on, {"A": sg_on.to_reduced_basis(sym[:1])[0], "B": sg_on.to_reduced_basis(sym[1:])[0]})
    diff = np.max(np.abs(st_on[:3] - st[:3])) / np.max(np.abs(st[:3]))
    check(diff < 1e-12, f"P6_3/mmc 18x18x16: stress with the space group = without ({diff:.1e})")

    # 2. Two-way ties only: the single-axis derivative is the central difference
    nx, lx = [16, 16, 16], [1.8, 1.8, 2.3]
    sym, _ = hexagonal_field(nx)
    w = {"A": sym[0], "B": sym[1]}
    s = solver(nx, lx, angles)
    st = stress(s, w)
    fd_a = central_difference(s, w, lx, angles, [1, 0, 0, 0, 0, 0])
    rel = abs(st[0] - fd_a) / abs(fd_a)
    check(rel < 1e-8 and abs(st[0] - st[1]) < 1e-12 * abs(st[0]),
          f"P6_3/mmc 16x16x16: dH/da = central difference in a ({rel:.1e}), dH/da = dH/db")

    # 3. Orthogonal cell, forced angle derivatives (Nyquist planes)
    nx, lx, angles = [10, 12, 14], [1.6, 1.9, 2.2], [90.0, 90.0, 90.0]
    rng = np.random.default_rng(5)
    w = {"A": rng.normal(0.0, 1.0, int(np.prod(nx))), "B": rng.normal(0.0, 1.0, int(np.prod(nx)))}
    s = solver(nx, lx, angles)
    s.set_force_off_diagonal_stress(True)
    st = stress(s, w)
    fd = np.array([central_difference(s, w, lx, angles, direction)
                   for direction in ([0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0])])
    err = np.max(np.abs(st[3:6] - fd)) / np.max(np.abs(fd))
    check(err < 1e-7, f"orthogonal 10x12x14: forced angle derivatives = central differences ({err:.1e})")

    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
