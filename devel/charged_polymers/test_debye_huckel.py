"""Deterministic Debye-Hueckel validation of the charged-polymer L-FTS.

Symmetric +-1 salt of single-segment ions (discrete, ds=1, alpha=1),
phibar = 0.5 each, smearing radius a for both. A small exchange-field
perturbation delta w_- = eps cos(k0 x) is applied and the saddle
(pressure field + psi) converged; linear response predicts the smeared
charge amplitude

    |c(k0)| = eps * hhat^2(k0) / (1 + g(k0)),    g = E hhat^2 / k0^2,
    hhat^2  = exp(-a^2 k0^2)   (pair kernel of two equal Gaussians)

i.e. Debye screening with kappa^2 R0^2 = E sum_i z_i^2 phibar_i. The test
checks |c|/prediction == 1 and the Poisson relation psi_hat = E c_hat/k0^2
for several k0 and E (including E=0, the unscreened bare response).

This is sampling-free: it validates the Coulomb coupling normalization E,
the per-species smearing kernels, the W_i = h_i*(W_SPT + z_i psi)
assembly, and the psi partial saddle, independent of Langevin statistics.

NOTE (measurement): fields live on the CELL-CENTERED grid x_i=(i+0.5)dx,
so a cos(k0 x) perturbation has FFT phase e^{i k0 dx/2}; use |c_hat|,
not Re(c_hat), or the ratio picks up a spurious cos(k0 dx/2).

Expected output: every ratio equal to 1 within ~1e-6 (saddle tolerance).
"""
import os
import numpy as np

os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from lfts_charged import ChargedLFTS

nx = [16, 16, 16]
lx = [4.0, 4.0, 4.0]
n_grid = int(np.prod(nx))
a = 0.2
eps = 1e-4

x = (np.arange(nx[0]) + 0.5) * lx[0] / nx[0]
X = np.broadcast_to(x[:, None, None], nx)

all_ok = True
for E in (0.0, 6.25, 25.0):
    params = {
        "nx": nx, "lx": lx, "chain_model": "discrete", "ds": 1.0,
        "segment_lengths": {"P": 1.0, "M": 1.0},
        "chi_n": {"P,M": 0.5}, "zeta_n": 100.0,
        "distinct_polymers": [
            {"volume_fraction": 0.5, "blocks": [{"type": "P", "length": 1.0}]},
            {"volume_fraction": 0.5, "blocks": [{"type": "M", "length": 1.0}]}],
        "charges": {"P": 1.0, "M": -1.0},
        "radiuses": {"P": a, "M": a},
        "bjerrum_e": E,
        "langevin": {"max_step": 1, "dt": 0.5, "nbar": 1000.0},
        "recording": {"dir": "/tmp/dh_dump", "recording_period": 10**9,
                      "sf_computing_period": 10**9, "sf_recording_period": 10**9},
        "saddle": {"max_iter": 500, "tolerance": 1e-9},
        "compressor": {"name": "am", "max_hist": 20, "start_error": 1e-1,
                       "mix_min": 0.1, "mix_init": 0.1},
        "platform": "cuda", "verbose_level": 0,
    }
    sim = ChargedLFTS(params=params, random_seed=1)
    ridx = sim.mpt.aux_fields_real_idx[0]
    print(f"E = {E}")
    for n0 in (1, 2, 3, 4, 5, 6):
        k0 = 2 * np.pi * n0 / lx[0]
        w_aux = np.zeros((2, n_grid))
        w_aux[ridx] = eps * np.cos(k0 * X).reshape(-1)
        phi, ham, it, err = sim.find_saddle_point(w_aux=w_aux)

        c = sim.electro.charge_density(phi).reshape(nx)
        c_amp = 2 * np.abs(np.fft.fftn(c)[n0, 0, 0]) / n_grid
        h2 = np.exp(-a * a * k0 * k0)
        g = E * h2 / k0**2
        pred = h2 * eps / (1 + g)
        r1 = c_amp / pred

        if E > 0:
            psi_amp = 2 * np.abs(np.fft.fftn(
                np.asarray(sim.electro.psi).reshape(nx))[n0, 0, 0]) / n_grid
            r2 = psi_amp / (E * c_amp / k0**2)
        else:
            r2 = 1.0
        ok = abs(r1 - 1) < 1e-4 and abs(r2 - 1) < 1e-4
        all_ok = all_ok and ok
        print(f"  k0={k0:6.3f}  |c|/pred={r1:.6f}  psi/Poisson={r2:.6f}  "
              f"g={g:7.3f}  {'OK' if ok else 'FAIL'}")

print("DEBYE-HUCKEL TEST", "PASSED" if all_ok else "FAILED")
