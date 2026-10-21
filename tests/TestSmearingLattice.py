#!/usr/bin/env python3
"""
Smearing in non-orthogonal cells and with box relaxation, and the cell checks
that go with them.

1. Wavenumbers: Smearing.k_sq must equal the minimum of |k|^2 over the aliases
   of each grid mode, computed independently from Cartesian reciprocal vectors
   (oblique, hexagonal and triclinic cells). The hexagonal first shell
   (1,0), (0,1), (1,-1) must get one value of Gamma (six-fold symmetry). The
   orthogonal formula is unchanged bit for bit. Until 2026-10 the angles were
   ignored: in a hexagonal cell (1,-1) got twice the k^2 of (1,0).

2. Stress with smearing: H = -sum_p (phi_p/alpha_p) ln Q_p[Gamma(theta) * w]
   depends on the lattice parameters theta also through Gamma(k(theta)). The
   solver's stress plus Smearing.stress() must equal the central difference of
   H at fixed w. Discrete chains, whose stress is the exact derivative of their
   Q: all cells agree to ~1e-11 (the orthogonal cell with angle derivatives
   needs the averaged stress of its tied Nyquist modes; before 2026-10 it was
   1e-7 off). The check also requires the smearing term to matter (without it
   the error is > 1e-3).

3. SCFT box relaxation with smearing: the relaxed period must be the minimum
   of F(L) of fixed-box runs whose kernel is built at L. Until 2026-10 the
   kernel stayed at the initial box (smearing length a_int L/L0) and the
   stress had no Gamma term: the relaxed L was 5 % off.

4. L-FTS and CL-FTS noise: sigma uses the cell volume (with the angle factor),
   not the product of the box lengths.

5. Walls need an orthogonal cell, and a space group needs periodic boundaries:
   both combinations are rejected (they used to give silently wrong results).

6. Smearing with reflecting or absorbing walls uses the image of the field (the
   DCT-II/DST-II basis of the propagators): it must equal the periodic smearing of
   the even or odd extension. Until 2026-10 it wrapped around the box (a ramp in a
   reflecting box gave 0.89 instead of 0.08 in the first cell). Item 2 includes
   boxes with walls.
"""

import os
import io
import sys
import contextlib
import tempfile
import shutil

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.ndimage import gaussian_filter
from polymerfts import _core, scft, LFTS, CLFTS
from polymerfts.smearing import Smearing
from polymerfts.utils import calculate_sigma
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


def lattice_vectors(lx, angles):
    """Cartesian lattice vectors (rows): a along x, b in the xy plane."""
    dim = len(lx)
    alpha, beta, gamma = np.deg2rad(angles)
    if dim == 1:
        return np.array([[lx[0]]])
    if dim == 2:
        return np.array([[lx[0], 0.0], [lx[1] * np.cos(gamma), lx[1] * np.sin(gamma)]])
    cx = np.cos(beta)
    cy = (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    return np.array([[lx[0], 0.0, 0.0],
                     [lx[1] * np.cos(gamma), lx[1] * np.sin(gamma), 0.0],
                     [lx[2] * cx, lx[2] * cy, lx[2] * np.sqrt(1.0 - cx ** 2 - cy ** 2)]])


def brute_force_k_sq(nx, lx, angles):
    """min over aliases m + N t, t in {-2..2}^dim, of |sum_d m_d b_d|^2."""
    dim = len(nx)
    recip = 2 * np.pi * np.linalg.inv(lattice_vectors(lx, angles)).T   # rows b_d, a_i . b_j = 2 pi delta_ij
    m = np.array(np.meshgrid(*[np.fft.fftfreq(n) * n for n in nx], indexing="ij")).reshape(dim, -1)
    best = None
    for t in np.array(np.meshgrid(*[range(-2, 3)] * dim, indexing="ij")).reshape(dim, -1).T:
        k = (m + (t * np.array(nx))[:, None]).T @ recip
        k_sq = np.sum(k ** 2, axis=1)
        best = k_sq if best is None else np.minimum(best, k_sq)
    return best.reshape(nx)


def smooth_fields(nx, seed, n=2):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, size=(n, *nx))
    for c in range(n):
        w[c] = gaussian_filter(w[c], sigma=1.5, mode="wrap")
        w[c] *= 5.0 / max(w[c].std(), 1e-12)
    return w.reshape(n, -1)


def test_wavenumbers():
    cases = [("2D oblique 70", [12, 14], [1.6, 1.9], [90.0, 90.0, 70.0]),
             ("2D hexagonal", [16, 16], [1.8, 1.8], [90.0, 90.0, 120.0]),
             ("3D triclinic", [8, 10, 12], [1.6, 1.9, 2.2], [75.0, 100.0, 70.0]),
             ("3D hexagonal", [10, 10, 8], [1.7, 1.7, 2.3], [90.0, 90.0, 120.0])]
    for name, nx, lx, angles in cases:
        sm = quiet(Smearing, nx, lx, {"type": "gaussian", "a_int": 0.2}, angles=angles)
        ref = brute_force_k_sq(nx, lx, angles)
        err = np.max(np.abs(sm.k_sq - ref) / np.maximum(ref, 1e-30))
        check(err < 1e-12, f"{name}: k^2 = minimal alias |k|^2 (max rel. diff {err:.1e})")

    sm = quiet(Smearing, [16, 16], [1.8, 1.8], {"type": "sigmoidal", "k_int": 10.0}, angles=[90.0, 90.0, 120.0])
    shell = [sm.fourier[i % 16, j % 16] for i, j in ((1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1))]
    check(np.ptp(shell) < 1e-14 * shell[0], f"2D hexagonal: Gamma equal on the six first-shell modes (spread {np.ptp(shell):.1e})")

    # Orthogonal cells keep the separable formula bit for bit
    nx, lx = [12, 10, 8], [1.6, 1.9, 2.2]
    sm = quiet(Smearing, nx, lx, {"type": "gaussian", "a_int": 0.2})
    k = [2 * np.pi * np.fft.fftfreq(nx[i], d=lx[i] / nx[i]) for i in range(3)]
    kx, ky, kz = np.meshgrid(*k, indexing="ij")
    check(np.array_equal(sm.fourier, np.exp(-0.2 ** 2 * (kx ** 2 + ky ** 2 + kz ** 2) / 2)),
          "3D orthogonal: Gamma unchanged bit for bit")


def test_stress():
    blocks = [["A", 0.4, 0, 1], ["B", 0.6, 1, 2]]
    smearings = [{"type": "gaussian", "a_int": 0.15}, {"type": "sigmoidal", "k_int": 12.0, "dk_int": 4.0}]
    cells = [("1D", [32], [1.7], None, None),
             ("2D oblique 110", [16, 18], [1.6, 1.9], [90.0, 90.0, 110.0], None),
             ("3D orthogonal (with angle derivatives)", [10, 12, 14], [1.6, 1.9, 2.2], [90.0, 90.0, 90.0], None),
             ("3D triclinic", [10, 12, 14], [1.6, 1.9, 2.2], [75.0, 100.0, 70.0], None),
             ("2D reflecting walls", [16, 18], [1.6, 1.9], None, ["reflecting"] * 4),
             ("1D absorbing walls", [32], [1.7], None, ["absorbing"] * 2)]
    for smearing in smearings:
        for name, nx, lx, angles, bc in cells:
            dim = len(nx)
            bc = bc or ["periodic"] * (2 * dim)
            w = smooth_fields(nx, seed=dim)
            s = quiet(PropagatorSolver, nx=nx, lx=lx, ds=0.05, bond_lengths={"A": 1.0, "B": 1.3},
                      bc=bc, chain_model="discrete", platform=cpu_platform(), angles=angles)
            s.add_polymer(1.0, blocks)
            if angles is not None:
                s.set_force_off_diagonal_stress(True)
            sm = quiet(Smearing, nx, lx, smearing, angles=angles, bc=bc)

            def hamiltonian():
                quiet(s.compute_propagators, sm.apply_to_dict({"A": w[0], "B": w[1]}))
                return -np.log(s.get_partition_function(0))

            hamiltonian()
            s.compute_concentrations()
            s.compute_stress()
            phi = [np.array(s.get_concentration(m)) for m in ("A", "B")]
            solver_stress = np.array(s.get_stress())
            smearing_stress = sm.stress(w, phi)

            # Parameters (name, stress slot): lengths, then the angles (per radian)
            params = [(f"L{d + 1}", d) for d in range(dim)]
            if dim == 3:
                params += [("gamma", 3), ("beta", 4), ("alpha", 5)]
            fd = []
            for pname, slot in params:
                diffs = {}
                for h in (1e-3, 5e-4):
                    values = []
                    for sign in (1, -1):
                        L = list(lx)
                        A = list(angles) if angles is not None else None
                        if pname.startswith("L"):
                            L[slot] += sign * h * lx[slot]
                            step = h * lx[slot]
                        else:
                            A[{"gamma": 2, "beta": 1, "alpha": 0}[pname]] += sign * np.rad2deg(h)
                            step = h
                        if A is None:
                            s.set_lx(L)
                        else:
                            s.set_lattice_parameters(L, A)
                        s.update_laplacian_operator()
                        sm.set_box(L, A)
                        values.append(hamiltonian())
                    diffs[h] = (values[0] - values[1]) / (2 * step)
                fd.append((4 * diffs[5e-4] - diffs[1e-3]) / 3)
            fd = np.array(fd)
            slots = [slot for _, slot in params]
            total = solver_stress[slots] + smearing_stress[slots]
            scale = np.max(np.abs(fd))
            err = np.max(np.abs(total - fd)) / scale
            err_without = np.max(np.abs(solver_stress[slots] - fd)) / scale
            check(err < 1e-6 and err_without > 1e-3,
                  f"{smearing['type']}, {name}: solver + smearing stress = FD of H "
                  f"(rel. err {err:.1e}; without the smearing term {err_without:.1e})")


def test_walls():
    """Smearing across a wall is the periodic smearing of the image of the field:
    even across a reflecting wall, odd across an absorbing one."""
    rng = np.random.default_rng(4)
    nx, lx = [12, 10], [1.5, 1.3]
    field = rng.normal(0.0, 1.0, size=nx)
    for smearing in ({"type": "gaussian", "a_int": 0.12}, {"type": "sigmoidal", "k_int": 10.0, "dk_int": 3.0}):
        for walls in (("periodic", "reflecting"), ("absorbing", "periodic"), ("reflecting", "absorbing")):
            bc = [walls[0]] * 2 + [walls[1]] * 2
            sm = quiet(Smearing, nx, lx, smearing, bc=bc)
            # Image: double every wall axis (even or odd extension), smear periodically, cut back
            image, image_lx = field.copy(), list(lx)
            for axis, kind in enumerate(walls):
                if kind != "periodic":
                    sign = 1.0 if kind == "reflecting" else -1.0
                    image = np.concatenate([image, sign * np.flip(image, axis=axis)], axis=axis)
                    image_lx[axis] = 2 * lx[axis]
            periodic = quiet(Smearing, list(image.shape), image_lx, smearing)
            reference = np.reshape(periodic.apply(image.ravel()), image.shape)[:nx[0], :nx[1]]
            smeared = np.reshape(sm.apply(field.ravel()), nx)
            err = np.max(np.abs(smeared - reference))
            complex_err = np.max(np.abs(sm.apply((field + 2j * field).ravel()) - (1 + 2j) * smeared.ravel()))
            check(err < 1e-12 and complex_err < 1e-12,
                  f"{smearing['type']} smearing, {walls[0]} x / {walls[1]} y: equals the smeared image "
                  f"(max diff {err:.1e}; complex fields {complex_err:.1e})")
    try:
        quiet(Smearing, [8], [1.0], {"type": "gaussian", "a_int": 0.1}, bc=["reflecting", "absorbing"])
        check(False, "different walls on the two sides of an axis are rejected")
    except NotImplementedError:
        check(True, "different walls on the two sides of an axis are rejected")


def test_scft_box_relaxation():
    def params(lx, box):
        return {"nx": [32], "lx": [lx], "box_is_altering": box, "stress_interval": 1,
                "chain_model": "discrete", "ds": 1 / 50, "segment_lengths": {"A": 1.0, "B": 1.0},
                "chi_n": {"A,B": 20.0}, "platform": cpu_platform(),
                "distinct_polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5},
                                                                          {"type": "B", "length": 0.5}]}],
                "smearing": {"type": "gaussian", "a_int": 0.1},
                "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
                "max_iter": 4000, "tolerance": 1e-10}

    x = np.arange(32)
    w0 = {"A": 5 * np.cos(2 * np.pi * x / 32), "B": -5 * np.cos(2 * np.pi * x / 32)}
    calc = quiet(scft.SCFT, params=params(1.4, True))
    quiet(calc.run, initial_fields=w0)
    L_relaxed = calc.prop_solver.get_lx()[0]
    w = {"A": np.array(calc.w[0]), "B": np.array(calc.w[1])}
    delta = 0.002 * L_relaxed
    F = []
    for L in (L_relaxed - delta, L_relaxed, L_relaxed + delta):
        fixed = quiet(scft.SCFT, params=params(L, False))
        quiet(fixed.run, initial_fields=w)
        F.append(fixed.free_energy)
    # Minimum of the parabola through the three points
    L_min = L_relaxed - delta * (F[2] - F[0]) / (2 * (F[2] - 2 * F[1] + F[0]))
    rel = abs(L_min - L_relaxed) / L_relaxed
    check(F[1] < min(F[0], F[2]) and rel < 2e-5,
          f"SCFT smearing box relaxation: relaxed L {L_relaxed:.6f}, minimum of F(L) {L_min:.6f} (rel. {rel:.1e})")


def test_noise_volume():
    root = tempfile.mkdtemp()
    try:
        base = {"nx": [16, 16], "lx": [3.0, 3.0], "chain_model": "continuous", "ds": 1 / 20,
                "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 15.0},
                "distinct_polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5},
                                                                          {"type": "B", "length": 0.5}]}],
                "platform": cpu_platform(), "langevin": {"nbar": 10000, "dt": 8.0, "max_step": 2},
                "recording": {"dir": root, "recording_period": 2, "sf_computing_period": 1, "sf_recording_period": 2},
                "saddle": {"max_iter": 200, "tolerance": 1e-4},
                "compressor": {"name": "am", "max_hist": 20, "start_error": 5e-1, "mix_min": 0.01, "mix_init": 0.01},
                "smearing": {"type": "gaussian", "a_int": 0.2}, "verbose_level": 1}
        for name, cls in (("L-FTS", LFTS), ("CL-FTS", CLFTS)):
            p = dict(base, angles=[90.0, 90.0, 120.0])
            if cls is CLFTS:
                del p["saddle"], p["compressor"]
            sim = quiet(cls, p, random_seed=1)
            volume = sim.prop_solver.get_volume()
            expected = calculate_sigma(10000, 8.0, 256, volume)
            check(abs(volume - 9.0 * np.sin(np.deg2rad(120.0))) < 1e-12
                  and abs(sim.langevin["sigma"] - expected) < 1e-14 * expected,
                  f"{name} hexagonal: sigma from the cell volume ({sim.langevin['sigma']:.6e})")
            check(not sim.smearing.orthogonal, f"{name} hexagonal: smearing uses the cell angles")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cell_checks():
    def solver(bc, angles=None, space_group=None):
        s = quiet(PropagatorSolver, nx=[12, 12, 12], lx=[1.6, 1.6, 1.6], ds=0.05, bond_lengths={"A": 1.0, "B": 1.0},
                  bc=bc, chain_model="discrete", platform=cpu_platform(), angles=angles, space_group=space_group)
        s.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
        return s

    def rejected(fn, text):
        try:
            quiet(fn)
        except Exception as e:
            return text in str(e)
        return False

    walls = ["reflecting"] * 6
    check(rejected(lambda: solver(walls, angles=[90.0, 90.0, 110.0]).computation_box, "orthogonal box"),
          "reflecting walls in a non-orthogonal cell are rejected")
    check(rejected(lambda: solver(["absorbing"] * 6, angles=[80.0, 90.0, 90.0]).computation_box, "orthogonal box"),
          "absorbing walls in a non-orthogonal cell are rejected")
    s = solver(walls)
    quiet(lambda: s.computation_box)
    check(rejected(lambda: s.set_lattice_parameters([1.6, 1.6, 1.6], [90.0, 90.0, 110.0]), "orthogonal box")
          and list(s.get_angles_degrees()) == [90.0, 90.0, 90.0],
          "set_lattice_parameters: walls keep the cell orthogonal, the box is unchanged")
    sg = quiet(_core.SpaceGroup, [12, 12, 12], "Ia-3d", 530)
    check(rejected(lambda: solver(walls, space_group=sg).computation_box, "periodic boundary conditions"),
          "a space group with walls is rejected")


def main():
    test_wavenumbers()
    test_stress()
    test_walls()
    test_scft_box_relaxation()
    test_noise_volume()
    test_cell_checks()
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
