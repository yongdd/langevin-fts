"""Verify smearing implementation: SCFT vs L-FTS vs CL-FTS.

Tests that all three methods agree (within fluctuation corrections)
for both incompressible and compressible models in 1D and 3D.

Usage:
    python verify_smearing.py [--dim 1|3] [--nbar 1e9] [--max_step 10000]
"""
import os, sys, argparse
import numpy as np
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"

from polymerfts import scft, LFTS, CLFTS
from scipy.io import loadmat


def run_scft(common, w_A, w_B):
    params = {
        **common,
        "box_is_altering": False,
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2,
                      "mix_min": 0.1, "mix_init": 0.1},
        "max_iter": 5000, "tolerance": 1e-8, "verbose_level": 0,
    }
    calc = scft.SCFT(params=params)
    result = calc.run(initial_fields={"A": w_A.copy(), "B": w_B.copy()},
                      return_result=True)
    H = result.free_energy + calc.mpt.h_const
    return H, result.converged, result.error_level, result.iterations


def run_lfts(common, w_A, w_B, nbar, max_step, data_dir, dt=0.5):
    os.makedirs(data_dir, exist_ok=True)
    params = {
        **common,
        "langevin": {"max_step": max_step, "dt": dt, "nbar": nbar},
        "saddle": {"max_iter": 100, "tolerance": 1e-4},
        "compressor": {"name": "lram", "max_hist": 20, "start_error": 1e-2,
                       "mix_min": 0.1, "mix_init": 0.1},
        "recording": {"dir": data_dir, "recording_period": max_step,
                      "sf_computing_period": 1, "sf_recording_period": max_step},
        "verbose_level": 1,
    }
    sim = LFTS(params=params, random_seed=42)
    sim.run(initial_fields={"A": w_A.copy(), "B": w_B.copy()})

    data = loadmat(f"{data_dir}/dH_{max_step:06d}.mat")
    H = data["H_history"].flatten()
    H_eq = H[len(H)//2:]
    return np.mean(H_eq), np.std(H_eq) / np.sqrt(len(H_eq))


def run_clfts(common, w_A, w_B, nbar, max_step, data_dir, dt=0.5):
    os.makedirs(data_dir, exist_ok=True)
    params = {
        **common,
        "langevin": {"max_step": max_step, "dt": dt, "nbar": nbar},
        "recording": {"dir": data_dir, "recording_period": max_step,
                      "sf_computing_period": 1, "sf_recording_period": max_step},
        "dynamic_stabilization": 0.001,
        "verbose_level": 0,
    }
    sim = CLFTS(params=params, random_seed=42)
    sim.run(initial_fields={"A": w_A.copy(), "B": w_B.copy()})

    data = loadmat(f"{data_dir}/dH_{max_step:06d}.mat")
    H = data["H_history_real"].flatten()
    H_eq = H[len(H)//2:]
    return np.mean(H_eq), np.std(H_eq) / np.sqrt(len(H_eq))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1, choices=[1, 3])
    parser.add_argument("--nbar", type=float, default=1e9)
    parser.add_argument("--max_step", type=int, default=10000)
    parser.add_argument("--dt", type=float, default=8.0)
    args = parser.parse_args()

    f = 0.4
    chi_n = 17.0
    a_int = 0.1
    zeta_n = 500.0

    if args.dim == 1:
        nx, lx = [64], [4.36]
    else:
        nx, lx = [32, 32, 32], [4.36, 4.36, 4.36]

    common_base = {
        "nx": nx, "lx": lx,
        "chain_model": "discrete", "ds": 1/90,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": chi_n},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1-f},
            ],
        }],
        "smearing": {"type": "gaussian", "a_int": a_int},
    }
    if args.dim == 3:
        common_base["platform"] = "cuda"

    # Initial fields (lamellar)
    x = np.linspace(0, lx[0], nx[0], endpoint=False)
    w_1d = 2.0 * np.cos(2 * np.pi * x / (lx[0] / 2))
    if args.dim == 1:
        w_A, w_B = w_1d.copy(), -w_1d.copy()
    else:
        w_A = np.tile(w_1d[:, None, None], (1, nx[1], nx[2])).flatten()
        w_B = -w_A.copy()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_base = os.path.join(base_dir, f"data_{args.dim}d")

    print(f"{'='*60}")
    print(f"Smearing verification: {args.dim}D, f={f}, chiN={chi_n}, a_int={a_int}")
    print(f"nbar={args.nbar:.0e}, max_step={args.max_step}, dt={args.dt}")
    print(f"{'='*60}\n")

    results = []

    for model_name, zeta in [("Incompressible", None), ("Compressible", zeta_n)]:
        common = {**common_base}
        if zeta is not None:
            common["zeta_n"] = zeta

        tag = f"{model_name} (zeta_n={zeta})" if zeta else model_name
        print(f"--- {tag} ---")

        # SCFT
        H_scft, conv, err, iters = run_scft(common, w_A, w_B)
        print(f"  SCFT:   H={H_scft:.10f}  (converged={conv}, err={err:.1e}, iters={iters})")

        # L-FTS
        H_lfts, se_lfts = run_lfts(
            common, w_A, w_B, args.nbar, args.max_step,
            os.path.join(data_base, f"lfts_{model_name.lower()}"), dt=args.dt)
        diff_lfts = H_lfts - H_scft
        print(f"  L-FTS:  <H>={H_lfts:.10f} +/- {se_lfts:.1e}  (diff={diff_lfts:+.4e})")

        # CL-FTS (compressible only)
        if zeta is not None:
            H_clfts, se_clfts = run_clfts(
                common, w_A, w_B, args.nbar, args.max_step,
                os.path.join(data_base, f"clfts_{model_name.lower()}"), dt=args.dt)
            diff_clfts = H_clfts - H_scft
            print(f"  CL-FTS: <H>={H_clfts:.10f} +/- {se_clfts:.1e}  (diff={diff_clfts:+.4e})")

        print()


if __name__ == "__main__":
    main()
