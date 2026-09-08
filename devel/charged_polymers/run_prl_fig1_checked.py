"""Recompute Fig. 1 parameter cases with explicit approximation labels.

This runner uses the existing PRLBrushSCFT equations without parameter fitting.
LDA and the existing nonlocal backend are approximations, not the full SI theory.
Each valence, closure and grid must be submitted as a separate SLURM job.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from prl_scft import PRLBrushSCFT


def coupled_newton(s):
    """Solve the same equal-radius equations for V and one base self-energy.

    With equal Born radii the three self-energies are exactly z_K^2 U.
    PB is solved inside every residual evaluation; no closure is changed.
    """
    from scipy.optimize import newton_krylov
    assert s.aP == s.aplus == s.aion
    def residual(values):
        V, U = values[:s.Nz], values[s.Nz:]
        s.W = s.N * V.copy()
        s.uP, s.up, s.um = s.zP**2 * U, s.zp**2 * U, U.copy()
        rho = s._chain_density(V)
        c, p, m = s._solve_pb(rho)
        phi = np.clip(s.v * rho, 0, .999)
        target = s.chi * (1 - 2 * phi) - np.log1p(-phi) + s.zP * s.psi + s.uP
        energies = s._self_energies(rho, c, p, m)
        return np.r_[target - V, energies[2] - U]
    def callback(values, error):
        print(json.dumps(dict(newton_max_residual=float(abs(error).max()))), flush=True)
    values = newton_krylov(residual, np.r_[s.W / s.N, s.um], f_tol=3e-8,
        rdiff=1e-5, maxiter=80, inner_maxiter=50, callback=callback, line_search="armijo")
    residual(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zplus", type=int, choices=[0, 1, 2, 3], required=True)
    ap.add_argument("--closure", choices=["mf", "lda", "nonlocal"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--warm", type=Path, required=True)
    ap.add_argument("--nz", type=int, default=1600)
    ap.add_argument("--L", type=float, default=200.)
    ap.add_argument("--gc-only", action="store_true")
    ap.add_argument("--coupled-newton", action="store_true")
    ap.add_argument("--nonlocal-continuation", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    s = PRLBrushSCFT(zplus=max(1, args.zplus), zP=-1. if args.zplus else 0.,
        correlations=args.closure != "mf" and args.zplus != 0,
        counterions=not args.gc_only, L=args.L, Nz=args.nz, Ib_molar=.3)
    if args.closure == "nonlocal":
        s.selfenergy = "nonlocal"
        s.u_every = 10
    warm = np.load(args.warm)
    suffix = "mf" if args.closure == "mf" else "corr"
    for attr, key in [("W", "W_" + suffix), ("psi", "psi_" + suffix),
                      ("uP", "uP"), ("up", "up"), ("um", "um")]:
        if key not in warm and attr in warm:
            key = attr
        if key in warm and (s.corr or attr in ["W", "psi"]):
            setattr(s, attr, np.interp(s.z, warm["z"], warm[key]))
    result = dict(params=vars(args).copy(), status="running", stages=[])
    result["params"] = {k: str(v) if isinstance(v, Path) else v for k, v in result["params"].items()}
    try:
        if args.nonlocal_continuation:
            if args.closure != "nonlocal" or not args.coupled_newton:
                raise ValueError("Nonlocal continuation requires --closure nonlocal --coupled-newton")
            original = s._self_energies
            for fraction in [.1, .2, .4, .6, .8, 1.]:
                def blended(*densities, fraction=fraction):
                    s.selfenergy = "lda"
                    local = original(*densities)
                    s.selfenergy = "nonlocal"
                    nonlocal_values = original(*densities)
                    return tuple((1-fraction)*a + fraction*b for a, b in zip(local, nonlocal_values))
                s._self_energies = blended
                print(json.dumps(dict(nonlocal_fraction=fraction)), flush=True)
                coupled_newton(s)
                np.savez_compressed(args.out / f"continuation_{fraction:g}.npz", z=s.z,
                                    W=s.W, psi=s.psi, uP=s.uP, up=s.up, um=s.um)
            s._self_energies = original
        elif args.coupled_newton:
            coupled_newton(s)
        attempts = [(False, .001)] if args.coupled_newton else [(True, .01), (False, .01), (False, .003)]
        for attempt, (am, mixing) in enumerate(attempts):
            s.solve(max_iter=12 if args.coupled_newton else 10000, tol=1e-7, lam_w=mixing, lam_u=.05,
                    psi_exact=True, use_am=am, dW_cap=.05)
            stage = dict(attempt=attempt, am=am, mixing=mixing, iterations=s.n_iter,
                         residuals=s.residuals, height_2_mean_z=float(s.height()))
            result["stages"].append(stage)
            print(json.dumps(stage), flush=True)
            np.savez_compressed(args.out / "checkpoint.npz", z=s.z, W=s.W, psi=s.psi,
                                uP=s.uP, up=s.up, um=s.um)
            if s.converged:
                break
        if not s.converged:
            raise RuntimeError(f"Full residual did not reach 1e-7: {s.residuals}")
        rho = s.rhoP.copy()
        s.chain_backend = "direct"
        direct = s._chain_density(s.W / s.N)
        error = float(np.max(abs(direct - rho)))
        if error > 1e-10:
            raise RuntimeError(f"Independent chain comparison failed: {error}")
        if max(abs(s.W / s.N).max(), abs(s.psi + s.um).max(),
               abs(s.zp * s.psi + s.up).max(), abs(-s.psi + s.um).max()) >= 290:
            raise RuntimeError("Returned state approaches exponential clipping in existing solver")
        mass = float(rho.sum() * s.dz)
        c_mass = float(s.rhoC.sum() * s.dz)
        assert abs(mass - s.sigma * s.N) < 1e-10
        if s.with_counterions:
            assert abs(c_mass - abs(s.zP) * s.sigma * s.N) < 1e-10
        result.update(status="completed", residuals=s.residuals,
            height_2_mean_z_nm=float(s.height(rho)), max_phi=float(s.v * rho.max()),
            polymer_mass=mass, canonical_counterion_mass=c_mass,
            independent_chain_error=error, contour_partition_error=s._direct_chain.partition_error,
            log_partition=float(s.log_partition),
            far_field_potential=float(s.psi[-1]),
            note="Finite Gaussian graft width 0.5b; selected existing self-energy closure; not full PRL reproduction.")
        np.savez_compressed(args.out / "profiles.npz", z=s.z, phiP=s.v * rho,
            rhoP=rho, rhoC=s.rhoC, rhop=s.rhop, rhom=s.rhom, W=s.W, psi=s.psi,
            uP=s.uP, up=s.up, um=s.um, rhoP_direct=direct)
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - start
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
