#!/usr/bin/env python3
"""PropagatorSolver.fourier: transforms and Fourier multipliers against NumPy/SciPy.

On every available platform (CPU-FFTW, CPU-MKL, CUDA), with random fields of standard
deviation 5:
1. Periodic real fields (3D, odd and even sizes): forward() is numpy.fft.rfftn and
   backward() its inverse; k_squared() is (2π m/L)²; a diagonal Gaussian kernel applied
   to three fields and a sum of two terms with complex 2x3 and 2x2 matrix multipliers
   (a real operator: a derivative along x without its Nyquist mode) equal ifftn(Σ M fftn).
   Multipliers and coefficients that are not Hermitian on the planes of the half spectrum
   that hold both k and -k (i·k with its Nyquist mode, random coefficients) act as
   (M(k) + conj M(-k))/2 there, the same on every backend.
2. Periodic complex fields (2D): forward() is numpy.fft.fftn; a complex multiplier without
   symmetry and real input fields (converted to complex) give ifftn(Σ M fftn).
3. Oblique cells (2D at 120 degrees with 4x32 points, 3D triclinic): k_squared() is the
   minimum over the aliases m + N t of 4π² mᵀ g⁻¹ m; after set_lx() it follows the new box.
4. Walls (1D, 2D, 3D): cosine (reflecting) and sine (absorbing) series of real and complex
   fields are scipy.fft.dctn/dstn of type 2 divided by 2 per axis, with k = πm/L and
   π(m+1)/L, also with a cosine axis next to a sine axis; a multiplier applied through them
   equals the SciPy round trip; PropagatorSolver.fourier of a box with absorbing walls
   takes the cosine series.
5. 1D periodic grids (odd and even) and several real fields with an odd number of grid
   points (periodic 5x5x5, 1D reflecting 33), whose fields are not 16-byte aligned when packed.
6. CPU and CUDA give the same transforms and products (when both are available).
7. Input checks: multiplier shapes, a handle of another transform, field counts,
   non-finite values, complex values on real cosine coefficients, complex fields and
   coefficients for real transforms, a missing box, mixed periodic and wall axes; a cosine
   transform keeps its box orthogonal while it exists.
"""

import contextlib
import io

import numpy as np
import scipy.fft
from polymerfts import PlatformSelector, PropagatorSolver

RNG = np.random.default_rng(11)
failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def rel_err(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def solver(platform, nx, lx, bc=None, field_type="real", angles=None):
    bc = bc or ["periodic"] * (2 * len(nx))
    method = "cn-adi2" if any(b != "periodic" for b in bc) and any(b == "periodic" for b in bc) else None
    with contextlib.redirect_stdout(io.StringIO()):
        s = PropagatorSolver(nx=nx, lx=lx, ds=0.1, bond_lengths={"A": 1.0}, bc=bc,
                             chain_model="continuous", numerical_method=method, platform=platform,
                             field_type=field_type, angles=angles)
        s.add_polymer(1.0, [["A", 1.0, 0, 1]])
    return s


def wavevectors(nx, lx):
    """Cartesian k of the full FFT grid of an orthogonal box, shape (dim, *nx)."""
    axes = [2 * np.pi * np.fft.fftfreq(n, d=l / n) for n, l in zip(nx, lx)]
    return np.array(np.meshgrid(*axes, indexing="ij"))


def periodic_real(platform):
    nx, lx = [6, 5, 8], [2.0, 1.7, 2.5]
    s = solver(platform, nx, lx)
    ft = s.fourier
    n = int(np.prod(nx))
    half = tuple(nx[:-1]) + (nx[-1] // 2 + 1,)
    fields = RNG.normal(0.0, 5.0, (3, n))
    grids = fields.reshape((3,) + tuple(nx))

    coef = ft.forward(fields)
    ref = np.fft.rfftn(grids, axes=(1, 2, 3))
    check(f"{platform} periodic real: forward is rfftn", ft.coefficient_shape == half and ft.complex_coefficients
          and coef.shape == (3,) + half and rel_err(coef, ref) < 1e-13, f"err {rel_err(coef, ref):.1e}")
    back = ft.backward(coef)
    check(f"{platform} periodic real: backward(forward(f)) = f", back.shape == (3, n) and rel_err(back, fields) < 1e-13,
          f"err {rel_err(back, fields):.1e}")
    single = ft.forward(fields[0])
    check(f"{platform} periodic real: one flat field", single.shape == half and rel_err(single, ref[0]) < 1e-13)

    k = wavevectors(nx, lx)
    k_sq_full = np.sum(k ** 2, axis=0)
    k_sq = ft.k_squared()
    check(f"{platform} periodic real: k_squared", k_sq.shape == half and rel_err(k_sq, k_sq_full[..., :half[-1]]) < 1e-14,
          f"err {rel_err(k_sq, k_sq_full[..., :half[-1]]):.1e}")

    # Diagonal Gaussian kernel on three fields
    gamma_full = np.exp(-0.3 ** 2 * k_sq_full / 2)
    h = ft.set_multiplier(gamma_full[..., :half[-1]])
    out = ft.apply([(h, fields)])
    ref = np.real(np.fft.ifftn(gamma_full * np.fft.fftn(grids, axes=(1, 2, 3)), axes=(1, 2, 3))).reshape(3, n)
    check(f"{platform} periodic real: diagonal multiplier", h.diagonal and out.shape == (3, n) and rel_err(out, ref) < 1e-13,
          f"err {rel_err(out, ref):.1e}")
    out1 = ft.apply([(h, fields[1])])
    check(f"{platform} periodic real: flat field gives a flat result", out1.shape == (n,) and rel_err(out1, ref[1]) < 1e-13)

    # Two terms with complex matrix multipliers of a real operator: c_ij exp(-b_ij k²) + i d_ij k_x,
    # k_x set to zero on its Nyquist plane so that M(-k) = conj M(k)
    kx = k[0].copy()
    kx[np.isclose(np.abs(np.fft.fftfreq(nx[0]) * nx[0]), nx[0] / 2)] = 0.0
    def operator(n_out, n_in):
        c, b, d = (RNG.normal(size=(n_out, n_in, 1, 1, 1)) for _ in range(3))
        return c * np.exp(-np.abs(b) * k_sq_full) + 1j * d * kx
    m1, m2 = operator(2, 3), operator(2, 2)
    h1, h2 = ft.set_multiplier(m1[..., :half[-1]]), ft.set_multiplier(m2[..., :half[-1]])
    inputs2 = RNG.normal(0.0, 5.0, (2, n))
    out = ft.apply([(h1, fields), (h2, inputs2)])
    f1 = np.fft.fftn(grids, axes=(1, 2, 3))
    f2 = np.fft.fftn(inputs2.reshape((2,) + tuple(nx)), axes=(1, 2, 3))
    ref_c = np.fft.ifftn(np.einsum("ij...,j...->i...", m1, f1) + np.einsum("ij...,j...->i...", m2, f2), axes=(1, 2, 3))
    ref = np.real(ref_c).reshape(2, n)
    check(f"{platform} periodic real: two matrix terms", (h1.n_out, h1.n_in) == (2, 3) and out.shape == (2, n)
          and rel_err(out, ref) < 1e-12 and np.max(np.abs(np.imag(ref_c))) < 1e-10, f"err {rel_err(out, ref):.1e}")
    return {"forward": coef, "apply": out}


def periodic_complex(platform):
    nx, lx = [9, 8], [2.3, 1.9]
    s = solver(platform, nx, lx, field_type="complex")
    ft = s.fourier
    n = int(np.prod(nx))
    fields = RNG.normal(0.0, 5.0, (2, n)) + 1j * RNG.normal(0.0, 5.0, (2, n))
    grids = fields.reshape((2,) + tuple(nx))
    coef = ft.forward(fields)
    ref = np.fft.fftn(grids, axes=(1, 2))
    check(f"{platform} periodic complex: forward is fftn", ft.coefficient_shape == tuple(nx)
          and rel_err(coef, ref) < 1e-13, f"err {rel_err(coef, ref):.1e}")
    check(f"{platform} periodic complex: backward(forward(f)) = f", rel_err(ft.backward(coef), fields) < 1e-13)

    m = RNG.normal(size=(3, 2) + tuple(nx)) + 1j * RNG.normal(size=(3, 2) + tuple(nx))
    d = RNG.normal(size=tuple(nx)) + 1j * RNG.normal(size=tuple(nx))
    real_noise = RNG.normal(0.0, 5.0, (3, n))
    hm, hd = ft.set_multiplier(m), ft.set_multiplier(d)
    out = ft.apply([(hm, fields), (hd, real_noise)])
    ref = np.fft.ifftn(np.einsum("ij...,j...->i...", m, ref)
                       + d * np.fft.fftn(real_noise.reshape((3,) + tuple(nx)), axes=(1, 2)), axes=(1, 2)).reshape(3, n)
    check(f"{platform} periodic complex: matrix and diagonal terms, real inputs", out.dtype == np.complex128
          and rel_err(out, ref) < 1e-12, f"err {rel_err(out, ref):.1e}")
    return {"forward": coef, "apply": out}


def hermitian_planes(half, n_last):
    """(M(k) + conj M(-k))/2 on the planes of a half spectrum that hold both k and -k."""
    out = np.array(half, dtype=complex)
    lead = half.shape[:-1]
    for l in [0] + ([n_last // 2] if n_last % 2 == 0 else []):
        plane = np.array(half[..., l], dtype=complex)
        partner = plane[np.ix_(*[(-np.arange(n)) % n for n in lead])] if lead else plane
        out[..., l] = 0.5 * (plane + np.conj(partner))
    return out


def hermitian(platform):
    nx, lx = [8, 6, 8], [2.0, 1.7, 2.5]
    ft = solver(platform, nx, lx).fourier
    n = int(np.prod(nx))
    h = nx[-1] // 2 + 1
    fields = RNG.normal(0.0, 5.0, (2, n))
    rfft = np.fft.rfftn(fields.reshape((2,) + tuple(nx)), axes=(1, 2, 3))
    k = wavevectors(nx, lx)[..., :h]
    results = {}
    for axis in range(3):
        m = 1j * k[axis]   # an odd derivative with its Nyquist mode: not Hermitian there
        out = ft.apply([(ft.set_multiplier(m), fields)])
        ref = np.fft.irfftn(hermitian_planes(m, nx[-1]) * rfft, s=nx, axes=(1, 2, 3)).reshape(2, n)
        check(f"{platform} periodic real: i k_{'xyz'[axis]} with its Nyquist mode acts as (M + conj M(-k))/2",
              rel_err(out, ref) < 1e-12, f"err {rel_err(out, ref):.1e}")
        results[f"apply_{axis}"] = out
    coef = RNG.normal(size=(2,) + tuple(nx[:-1]) + (h,)) + 1j * RNG.normal(size=(2,) + tuple(nx[:-1]) + (h,))
    back = ft.backward(coef)
    ref = np.fft.irfftn(np.array([hermitian_planes(c, nx[-1]) for c in coef]), s=nx, axes=(1, 2, 3)).reshape(2, n)
    check(f"{platform} periodic real: backward of non-Hermitian coefficients", rel_err(back, ref) < 1e-12,
          f"err {rel_err(back, ref):.1e}")
    results["backward"] = back
    return results


def odd_and_1d(platform):
    """1D grids and several real fields with an odd number of grid points."""
    results = {}
    for nx, lx, kind in [([10], [2.0], "periodic"), ([9], [1.7], "periodic"), ([5, 5, 5], [1.5, 1.6, 1.7], "periodic"),
                         ([33], [3.0], "reflecting")]:
        s = solver(platform, nx, lx, bc=[kind] * (2 * len(nx)))
        ft = s.fourier
        n = int(np.prod(nx))
        axes = tuple(range(1, len(nx) + 1))
        fields = RNG.normal(0.0, 5.0, (2, n))
        grids = fields.reshape((2,) + tuple(nx))
        k2 = ft.k_squared()
        gamma = np.exp(-0.2 * k2)
        label = f"{platform} {kind} {nx}"
        coef = ft.forward(fields)
        out = ft.apply([(ft.set_multiplier(gamma), fields)])
        if kind == "periodic":
            h = nx[-1] // 2 + 1
            ref_coef = np.fft.rfftn(grids, axes=axes)
            k = wavevectors(nx, lx)[..., :h]
            ref_k2 = np.sum(k ** 2, axis=0)
            ref_out = np.fft.irfftn(gamma * ref_coef, s=nx, axes=axes).reshape(2, n)
        else:
            ref_coef = scipy.fft.dctn(grids, type=2, axes=axes) / 2 ** len(nx)
            ref_k2 = (np.pi * np.arange(nx[0]) / lx[0]) ** 2
            ref_out = scipy.fft.idctn(gamma * scipy.fft.dctn(grids, type=2, axes=axes), type=2, axes=axes).reshape(2, n)
        check(f"{label}: forward", rel_err(coef, ref_coef) < 1e-13, f"err {rel_err(coef, ref_coef):.1e}")
        check(f"{label}: backward(forward(f)) = f", rel_err(ft.backward(coef), fields) < 1e-13)
        check(f"{label}: k_squared", rel_err(k2, ref_k2) < 1e-13)
        check(f"{label}: diagonal multiplier on two fields", rel_err(out, ref_out) < 1e-12, f"err {rel_err(out, ref_out):.1e}")
        results[(kind, tuple(nx))] = {"forward": coef, "apply": out}
    return results


def minimal_alias_k_sq(nx, recip_metric):
    """min over t in {-2..2}^d of 4π² (m + N t)ᵀ g⁻¹ (m + N t) on the full grid."""
    dim = len(nx)
    g11, g12, g13, g22, g23, g33 = recip_metric   # upper triangle of the 3x3 metric
    g = np.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]])[:dim, :dim]
    m = np.array(np.meshgrid(*[np.fft.fftfreq(n) * n for n in nx], indexing="ij")).reshape(dim, -1)
    best = None
    for t in np.array(np.meshgrid(*[range(-2, 3)] * dim, indexing="ij")).reshape(dim, -1).T:
        u = m + (t * np.array(nx))[:, None]
        f = 4 * np.pi ** 2 * np.einsum("an,ab,bn->n", u, g, u)
        best = f if best is None else np.minimum(best, f)
    return best.reshape(nx)


def oblique(platform):
    results = {}
    for nx, lx, angles, field_type in [([4, 32], [1.5, 1.5], [120.0], "complex"),
                                       ([5, 6, 7], [1.6, 1.9, 2.1], [80.0, 100.0, 65.0], "real")]:
        s = solver(platform, nx, lx, field_type=field_type, angles=angles)
        ft = s.fourier
        ref = minimal_alias_k_sq(nx, s._computation_box.get_recip_metric())
        if field_type == "real":
            ref = ref[..., :nx[-1] // 2 + 1]
        k_sq = ft.k_squared()
        check(f"{platform} oblique {len(nx)}D: k_squared is the minimal alias", rel_err(k_sq, ref) < 1e-13,
              f"err {rel_err(k_sq, ref):.1e}")
        s.set_lx([1.3 * l for l in lx])
        k_sq2 = ft.k_squared()
        check(f"{platform} oblique {len(nx)}D: k_squared after set_lx", rel_err(k_sq2, k_sq / 1.3 ** 2) < 1e-13)
        results[len(nx)] = k_sq
    return results


def series(grids, kinds, inverse=False):
    """Type-2 cosine/sine series (or their inverses) of SciPy along the axes after the first."""
    for axis, kind in enumerate(kinds, start=1):
        if kind == "reflecting":
            grids = (scipy.fft.idct if inverse else scipy.fft.dct)(grids, type=2, axis=axis)
        else:
            grids = (scipy.fft.idst if inverse else scipy.fft.dst)(grids, type=2, axis=axis)
    return grids


def walls(platform):
    results = {}
    # Odd cosine axes; the CUDA sine series needs even lengths
    for kinds, field_type, nx, lx in [(["reflecting"] * 2, "real", [7, 6], [1.8, 2.2]),
                                      (["absorbing"] * 2, "real", [8, 6], [1.8, 2.2]),
                                      (["reflecting"] * 2, "complex", [7, 6], [1.8, 2.2]),
                                      (["absorbing"] * 2, "complex", [8, 6], [1.8, 2.2]),
                                      (["reflecting", "absorbing"], "real", [7, 6], [1.8, 2.2]),
                                      (["reflecting"] * 3, "real", [6, 4, 5], [1.2, 1.4, 1.1]),
                                      (["absorbing"] * 3, "complex", [6, 4, 6], [1.2, 1.4, 1.1]),
                                      (["absorbing"], "real", [10], [2.1])]:
        n = int(np.prod(nx))
        s = solver(platform, nx, lx, bc=[b for b in kinds for _ in range(2)], field_type=field_type)
        if all(b == "reflecting" for b in kinds):
            ft = s.fourier   # walls use the cosine series
        else:
            ft = s._factory.create_fourier_transform(s._computation_box, kinds)
        fields = RNG.normal(0.0, 5.0, (2, n))
        if field_type == "complex":
            fields = fields + 1j * RNG.normal(0.0, 5.0, (2, n))
        grids = fields.reshape((2,) + tuple(nx))
        coef = ft.forward(fields)
        ref = series(grids, kinds) / 2 ** len(nx)   # Σ_j f_j cos(πm(j+½)/N): half of SciPy's, per axis
        label = f"{platform} {'/'.join(kinds)} {field_type} {nx}"
        check(f"{label}: forward is the type-2 series", ft.complex_coefficients == (field_type == "complex")
              and coef.shape == (2,) + tuple(nx) and rel_err(coef, ref) < 1e-13, f"err {rel_err(coef, ref):.1e}")
        check(f"{label}: backward(forward(f)) = f", rel_err(ft.backward(coef), fields) < 1e-13)
        k_axes = [np.pi * (np.arange(nn) + (b == "absorbing")) / l for nn, l, b in zip(nx, lx, kinds)]
        k_sq_ref = sum(k ** 2 for k in np.meshgrid(*k_axes, indexing="ij"))
        check(f"{label}: k_squared", rel_err(ft.k_squared(), k_sq_ref) < 1e-14)
        gamma = np.exp(-0.25 * k_sq_ref)
        if field_type == "complex":
            gamma = gamma * (1.0 + 0.5j)
        out = ft.apply([(ft.set_multiplier(gamma), fields)])
        ref = series(gamma * series(grids, kinds), kinds, inverse=True).reshape(2, n)
        check(f"{label}: diagonal multiplier", rel_err(out, ref) < 1e-12, f"err {rel_err(out, ref):.1e}")
        results[(tuple(kinds), field_type, tuple(nx))] = {"forward": coef, "apply": out}

        if any(b == "absorbing" for b in kinds):
            # PropagatorSolver.fourier takes the cosine series also with absorbing walls
            cosine = s.fourier
            k_axes = [np.pi * np.arange(nn) / l for nn, l in zip(nx, lx)]
            k_sq_ref = sum(k ** 2 for k in np.meshgrid(*k_axes, indexing="ij"))
            ref = series(grids, ["reflecting"] * len(nx)) / 2 ** len(nx)
            check(f"{label}: PropagatorSolver.fourier takes the cosine series",
                  rel_err(cosine.k_squared(), k_sq_ref) < 1e-14 and rel_err(cosine.forward(fields), ref) < 1e-13)
    return results


def input_checks(platform):
    s = solver(platform, [6, 4], [1.0, 1.0])
    ft = s.fourier
    other = solver(platform, [6, 4], [1.0, 1.0]).fourier
    shape = ft.coefficient_shape
    field = RNG.normal(size=24)
    h = ft.set_multiplier(np.ones(shape))
    h23 = ft.set_multiplier(np.ones((2, 3) + shape))

    def raises(call):
        try:
            call()
        except (RuntimeError, ValueError, TypeError):
            return True
        return False

    walls_real = solver(platform, [6, 4], [1.0, 1.0], bc=["reflecting"] * 4).fourier
    complex_field = field + 1j * field
    cases = [
        ("multiplier of a wrong shape", lambda: ft.set_multiplier(np.ones((6, 4)))),
        ("multiplier with one leading axis", lambda: ft.set_multiplier(np.ones((2,) + shape))),
        ("non-finite multiplier", lambda: ft.set_multiplier(np.full(shape, np.nan))),
        ("complex multiplier on real cosine coefficients",
         lambda: walls_real.set_multiplier(np.full(walls_real.coefficient_shape, 1j))),
        ("handle of another transform", lambda: other.apply([(h, field)])),
        ("matrix multiplier with too few fields", lambda: ft.apply([(h23, np.ones((2, 24)))])),
        ("terms with different output counts", lambda: ft.apply([(h23, np.ones((3, 24))), (h, field)])),
        ("fields of a wrong size", lambda: ft.apply([(h, np.ones(25))])),
        ("no terms", lambda: ft.apply([])),
        ("missing multiplier", lambda: ft.apply([(None, field)])),
        ("mixed periodic and wall axes",
         lambda: solver(platform, [6, 4], [1.0, 1.0], bc=["periodic"] * 2 + ["reflecting"] * 2).fourier),
        ("a missing box", lambda: s._factory.create_fourier_transform(None, ["periodic"] * 2)),
        ("complex fields for a real transform (forward)", lambda: ft.forward(complex_field)),
        ("complex fields for a real transform (apply)", lambda: ft.apply([(h, complex_field)])),
        ("complex coefficients for real cosine coefficients",
         lambda: walls_real.backward(np.full(walls_real.coefficient_shape, 1j))),
    ]
    for name, call in cases:
        check(f"{platform} rejects {name}", raises(call))

    # A cosine transform keeps its (periodic) box orthogonal while it exists
    box = solver(platform, [6, 4], [1.0, 1.0])._computation_box
    cosine = s._factory.create_fourier_transform(box, ["reflecting"] * 2)
    held = raises(lambda: box.set_lattice_parameters([1.0, 1.0], [120.0]))
    del cosine
    released = not raises(lambda: box.set_lattice_parameters([1.0, 1.0], [120.0]))
    check(f"{platform} a cosine transform keeps its box orthogonal while it exists", held and released)


def main():
    platforms = PlatformSelector.avail_platforms()
    results = {}
    for platform in platforms:
        results[platform] = {"real": periodic_real(platform), "complex": periodic_complex(platform),
                             "oblique": oblique(platform), "walls": walls(platform)}
        hermitian(platform)
        odd_and_1d(platform)
        input_checks(platform)

    cpu = [p for p in platforms if p.startswith("cpu")]
    if "cuda" in platforms and cpu:
        # Same random draws on both platforms: rerun the CPU after resetting the generator
        for p in [cpu[0], "cuda"]:
            global RNG
            RNG = np.random.default_rng(23)
            results[p] = {"real": periodic_real(p), "complex": periodic_complex(p), "hermitian": hermitian(p),
                          **{f"walls {key}": value for key, value in walls(p).items()},
                          **{f"grid {key}": value for key, value in odd_and_1d(p).items()}}
        a, b = results[cpu[0]], results["cuda"]
        err = max(rel_err(b[key][q], a[key][q]) for key in a for q in a[key])
        check(f"CUDA agrees with {cpu[0]} (forward, apply, backward of {len(a)} cases)", err < 1e-12,
              f"max rel diff {err:.1e}")

    print(f"\n{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
