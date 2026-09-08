"""Recompute a PRL-parameter mean-field reference with explicit model choices.

This is not the correlation-augmented PRL result: it uses a Gaussian graft
sheet of width 0.5 b, truncated discrete Gaussian bonds and a finite box.
The reference and CL direct backend are compared at identical imposed fields.
Submit a separate SLURM job for each valence/ensemble/grid configuration.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from prl_scft import PRLBrushSCFT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zplus", type=int, choices=[1,2,3], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gc-only", action="store_true")
    ap.add_argument("--L", type=float, default=200.)
    ap.add_argument("--nz", type=int, default=1600)
    ap.add_argument("--warm", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    s = PRLBrushSCFT(zplus=args.zplus, Ib_molar=.3, correlations=False,
                     counterions=not args.gc_only, L=args.L, Nz=args.nz)
    if args.warm:
        archive = np.load(args.warm)
        s.W = np.interp(s.z, archive["z"], archive["W_mf"])
        s.psi = np.interp(s.z, archive["z"], archive["psi_mf"])
        ladder = [1.]
    else:
        ladder = [.05,.2,.4,.6,.8,1.]
    start = time.time()
    for charge in ladder:
        s.zP = -charge
        s.solve(max_iter=10000, tol=1e-7, lam_w=.01, lam_u=.1,
                dW_cap=.05, psi_exact=True, use_am=True)
        print(f"alpha={charge:g}, iterations={s.n_iter}, W residual={s.err:.3e}",flush=True)
        if s.err >= 1e-7:
            raise RuntimeError("Mean-field continuation failed; do not report a converged reference.")
    rho = s._chain_density(s.W/s.N)
    c, p, m = s._solve_pb(rho)
    phi = s.v*rho
    if np.max(phi) >= 1:
        raise RuntimeError("Mean-field polymer exceeds incompressible volume bound.")
    target = s.N*(s.chi*(1-2*phi)-np.log1p(-phi)+s.zP*s.psi)
    w_error = float(np.max(abs(target-s.W))/s.N)
    lap = np.diff(np.pad(s.psi,1,mode="edge"),2)/s.dz**2
    pb_error = float(np.max(abs(lap+4*np.pi*s.lB*(s.zP*rho+c+s.zp*p-m))))
    # Same potential, source and boundary; independent chain implementations.
    s.chain_backend = "direct"
    direct = s._chain_density(s.W/s.N)
    profile_error = float(np.max(abs(direct-rho)))
    bond_error = float(np.max(abs(s._direct_chain.bond(s.q_init["G"])-s._bond_conv(s.q_init["G"]))))
    if max(w_error, pb_error) > 1e-7 or profile_error > 1e-10:
        raise RuntimeError(f"Final residual/chain comparison failed: {w_error}, {pb_error}, {profile_error}")
    result = dict(zplus=args.zplus, correlations=False,
        canonical_monovalent_counterions=not args.gc_only,
        N=s.N, b_nm=s.b, v_nm3=s.v, sigma_nm_minus2=s.sigma,
        Ib_molar=.3, L_nm=s.L, nz=s.Nz, graft_width_nm=.5*s.b,
        chain_wall="truncated bottom / reflecting top", height_2_mean_z_nm=float(s.height(rho)),
        max_phi=float(phi.max()), polymer_mass_per_area=float(rho.sum()*s.dz),
        counterion_mass_per_area=float(c.sum()*s.dz), max_W_residual=w_error,
        max_PB_residual=pb_error, direct_reference_profile_error=profile_error,
        direct_reference_bond_error=bond_error, log_partition=s.log_partition,
        wall_seconds=time.time()-start,
        status="Numerically checked mean-field reference; not a quantitative PRL reproduction.")
    np.savez_compressed(args.out/"profiles.npz",z=s.z,rhoP=rho,rhoC=c,rhop=p,rhom=m,
                        W=s.W,psi=s.psi,rhoP_direct=direct)
    (args.out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
