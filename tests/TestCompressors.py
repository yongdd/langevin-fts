#!/usr/bin/env python3
"""LR and LRAM field compressors used outside an LFTS run.

Model problem: the residual of the imaginary field is linear and diagonal in
Fourier space, -dH/dw = J(k) (w* - w). With the exact J(k) one LR step lands
on w*; LRAM must equal Anderson mixing applied to the LR step; both must keep
the shape of their input, (M,) or (1, M); and LRAM must converge when J(k) is
only approximately known.
"""

import sys

import numpy as np

from polymerfts import LR, LRAM, PlatformSelector


def cpu_platform():
    avail = PlatformSelector.avail_platforms()
    cpu = [p for p in avail if p.startswith("cpu")]
    return "cpu-mkl" if "cpu-mkl" in avail else (cpu[0] if cpu else None)


def main():
    if cpu_platform() is None:
        print("SKIP: no CPU platform in this build")
        sys.exit(77)  # ctest SKIP_RETURN_CODE
    failed = []
    nx, lx = [8, 6, 4], [2.0, 1.5, 1.0]
    m = int(np.prod(nx))
    k2 = sum(np.meshgrid(*[(2*np.pi*np.fft.fftfreq(n, l/n))**2 for n, l in zip(nx[:-1], lx[:-1])]
                         + [(2*np.pi*np.fft.rfftfreq(nx[-1], lx[-1]/nx[-1]))**2], indexing="ij"))
    jk = 1.0 + k2                                       # J(k), positive
    rng = np.random.default_rng(5)
    w_star = rng.normal(size=m)

    def residual(w):
        """-dH/dw = J(k) (w* - w) for a flat or (1, M) field."""
        r = np.fft.irfftn(jk*np.fft.rfftn(np.reshape(w_star - np.ravel(w), nx)), nx, axes=tuple(range(len(nx))))
        return np.reshape(r, np.shape(w))

    factory = PlatformSelector.create_factory(cpu_platform(), False, "real")
    def anderson():
        return factory.create_anderson_mixing(m, 20, 1e-1, 0.1, 0.1)

    # LR with the exact J(k) is exact in one step, for both shapes
    lr = LR(nx, lx, jk)
    for w0 in (np.zeros(m), np.zeros((1, m))):
        w1 = lr.calculate_new_fields(w0, residual(w0), 1.0, 1.0)
        if np.shape(w1) != np.shape(w0) or not np.allclose(np.ravel(w1), w_star, atol=1e-12):
            failed.append(f"LR step for input shape {np.shape(w0)}")

    # LRAM = Anderson mixing of the LR step
    lr_approx = LR(nx, lx, 0.8*jk)
    w0 = rng.normal(size=(1, m))
    lram = LRAM(lr_approx, anderson())
    w_lram = lram.calculate_new_fields(w0, residual(w0), 1.0, 1.0)
    w_lr = lr_approx.calculate_new_fields(w0, residual(w0), 1.0, 1.0)
    w_ref = anderson().calculate_new_fields(w_lr, w_lr - w0, 1.0, 1.0)
    if np.shape(w_lram) != (1, m) or not np.allclose(np.ravel(w_lram), w_ref, rtol=0, atol=1e-14):
        failed.append("LRAM differs from Anderson mixing of the LR step")

    # LRAM converges with an approximate J(k), flat fields
    lram = LRAM(lr_approx, anderson())
    w, old_error = np.zeros(m), 1.0
    for it in range(50):
        r = residual(w)
        error = np.std(r)
        if error < 1e-10:
            break
        w = lram.calculate_new_fields(w, r, old_error, error)
        old_error = error
    if np.shape(w) != (m,) or not np.max(np.abs(w - w_star)) <= 1e-8:
        failed.append(f"LRAM did not converge: max |w - w*| = {np.max(np.abs(np.ravel(w) - w_star)):.2e} after {it} steps")

    for f in failed:
        print("FAILED:", f)
    if failed:
        sys.exit(1)
    print(f"LR/LRAM: OK (LRAM converged in {it} steps)")


if __name__ == "__main__":
    main()
