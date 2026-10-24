#!/usr/bin/env python3
r"""UV renormalization of the Flory-Huggins parameter and the bare segment length in FTS.

    python scripts/analysis/renormalization.py fields_001000.mat
    python scripts/analysis/renormalization.py --chain-model discrete --n 90 --nbar 1e5 \
        --nx 32 32 32 --lx 4.5 4.5 4.5 --zeta-n 90 --a-int 0.02 --target 9

On a grid the effective interaction is chi = z_inf chi_b, where chi_b N is the `chi_n` of the
simulation, and CL-FTS needs the bare segment length a_b. Both follow Eqs. (29) and (26) of
Matsen, Ramirez, Willis and Pina, J. Chem. Phys. 164, 014905 (2026), with compressibility and
smearing: as written for discrete chains, and in their N -> inf limit for continuous chains.
Options override the values read from the file. See docs/references/Tools.md.
"""

import argparse
import math

import numpy as np
from scipy.io import loadmat


def smearing_function(k2, smearing):
    """Gamma(k) of polymerfts smearing (src/python/smearing.py); 1 without smearing."""
    if not smearing:
        return np.ones_like(k2)
    kind = str(smearing.get("type", "gaussian")).lower()
    if kind == "gaussian":
        return np.exp(-smearing["a_int"] ** 2 * k2 / 2)
    if kind == "sigmoidal":
        k_int, dk_int = smearing["k_int"], smearing.get("dk_int", 5.0)
        c0 = 1.0 / (1 - np.tanh(-k_int / dk_int))
        return c0 * (1 - np.tanh((np.sqrt(k2) - k_int) / dk_int))
    raise ValueError(f"Unknown smearing type '{kind}'")


def brillouin_integral(integrand, nx, lx, n_quad):
    """Integral of an even function of k^2 over -pi/dx <= k_i <= pi/dx (Gauss-Legendre)."""
    nodes, weights = [], []
    x, w = np.polynomial.legendre.leggauss(n_quad)
    for n, length in zip(nx, lx):
        k_max = math.pi * n / length
        nodes.append(0.5 * k_max * (x + 1))
        weights.append(0.5 * k_max * w)
    total = 0.0
    yz = nodes[1][:, None] ** 2 + nodes[2][None, :] ** 2
    w_yz = weights[1][:, None] * weights[2][None, :]
    for kx, wx in zip(nodes[0], weights[0]):
        total += wx * np.sum(w_yz * integrand(kx ** 2 + yz))
    return 8.0 * total


def _chain(chain_model, n_segments, nx, lx):
    """N tanh(k^2/12N) and N sinh(k^2/12N) (k in 1/R0, a^2 = 1/N); both k^2/12 for N -> inf."""
    if len(nx) != 3 or len(lx) != 3:
        raise ValueError("The renormalization is defined for 3D boxes")
    if chain_model == "discrete":
        if not n_segments:
            raise ValueError("Discrete chains need the number of segments N")
        n = float(n_segments)
        return lambda k2: n * np.tanh(k2 / (12 * n)), lambda k2: n * np.sinh(k2 / (12 * n))
    if chain_model == "continuous":
        return (lambda k2: k2 / 12,) * 2
    raise ValueError(f"Unknown chain model '{chain_model}'")


def _compressibility(k2, h, zeta_n, smearing):
    """1/(1 + t/zeta) with t/zeta = h/(zeta N Gamma^2), written without dividing by Gamma."""
    gamma2 = smearing_function(k2, smearing) ** 2
    if zeta_n is None or math.isinf(zeta_n):
        return gamma2, 1.0
    return gamma2, gamma2 / (gamma2 + h / zeta_n)


def z_inf(chain_model, nx, lx, nbar, n_segments=None, zeta_n=None, smearing=None, n_quad=128):
    """Factor z_inf of chi = z_inf chi_b, Eq. (29) of Matsen et al. (2026).

    In units of R0 (a^2 = 1/N, rho0 = Nbar^(1/2) N, zeta_n = zeta N):
        z_inf = 1 - 1/((2 pi)^3 Nbar^(1/2)) int Gamma^2 / (h (1 + h/(zeta N Gamma^2))) dk,
    with h = N tanh(k^2/12N) for discrete chains and h = k^2/12 for continuous chains. For
    zeta -> inf and Gamma = 1 the discrete z_inf is the bond sum of Beardsley and Matsen,
    J. Chem. Phys. 150, 174902 (2019).
    """
    h_of, _ = _chain(chain_model, n_segments, nx, lx)

    def integrand(k2):
        h = h_of(k2)
        gamma2, factor = _compressibility(k2, h, zeta_n, smearing)
        return gamma2 * factor / h

    return 1.0 - brillouin_integral(integrand, nx, lx, n_quad) / ((2 * math.pi) ** 3 * math.sqrt(nbar))


def bare_segment_ratio(chain_model, nx, lx, nbar, n_segments=None, zeta_n=None, smearing=None, n_quad=128):
    """a_b/a of CL-FTS, Eq. (26) of Matsen et al. (2026); L-FTS and SCFT use a_b = a.

        a_b/a = [1 + 1/((2 pi)^3 Nbar^(1/2)) int k^2 / (72 q^2 (1 + h/(zeta N Gamma^2))) dk]^(-1),
    with q = N sinh(k^2/12N) for discrete chains and q = k^2/12 for continuous chains.
    """
    h_of, q_of = _chain(chain_model, n_segments, nx, lx)

    def integrand(k2):
        _, factor = _compressibility(k2, h_of(k2), zeta_n, smearing)
        return k2 * factor / (72 * q_of(k2) ** 2)

    return 1.0 / (1.0 + brillouin_integral(integrand, nx, lx, n_quad) / ((2 * math.pi) ** 3 * math.sqrt(nbar)))


def read_simulation(path):
    """Settings of a polymerfts fields_*.mat file."""
    data = loadmat(path, simplify_cells=True)
    params = data.get("initial_params", {})
    angles = params.get("angles", [90.0, 90.0, 90.0])
    if any(abs(float(angle) - 90.0) > 1e-12 for angle in np.atleast_1d(angles)):
        raise ValueError("The renormalization is implemented for orthogonal boxes only")
    zeta_n = data.get("zeta_n", params.get("zeta_n"))
    nbar = data.get("nbar", params.get("langevin", {}).get("nbar"))
    ds = data.get("ds", params.get("ds"))
    chain_model = data.get("chain_model", params.get("chain_model"))
    return {
        "chain_model": None if chain_model is None else str(chain_model).lower(),
        "nx": [int(n) for n in np.atleast_1d(data.get("nx", params.get("nx")))],
        "lx": [float(length) for length in np.atleast_1d(data.get("lx", params.get("lx")))],
        "nbar": None if nbar is None else float(nbar),
        "n_segments": None if ds is None else round(1.0 / float(ds)),
        "zeta_n": None if zeta_n is None else float(zeta_n),
        "smearing": params.get("smearing") or None,
        "chi_n": data.get("chi_n", params.get("chi_n", {})),
        "segment_lengths": params.get("segment_lengths", {}),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fields", nargs="?", help="polymerfts fields_*.mat file (optional)")
    parser.add_argument("--chain-model", choices=("discrete", "continuous"))
    parser.add_argument("--nx", type=int, nargs=3, help="grid points")
    parser.add_argument("--lx", type=float, nargs=3, help="box lengths in R0")
    parser.add_argument("--n", type=int, help="segments per reference chain, 1/ds (discrete chains)")
    parser.add_argument("--nbar", type=float, help="invariant polymerization index")
    parser.add_argument("--zeta-n", type=float, help="compressibility zeta N; inf: incompressible")
    parser.add_argument("--a-int", type=float, help="Gaussian smearing length in R0 (0: none)")
    parser.add_argument("--k-int", type=float, help="sigmoidal smearing cutoff in 1/R0")
    parser.add_argument("--dk-int", type=float, default=5.0, help="sigmoidal transition width (default 5)")
    parser.add_argument("--target", type=float, help="print chi_b N for this z_inf chi_b N")
    parser.add_argument("--quadrature", type=int, default=128, help="Gauss-Legendre points per axis")
    args = parser.parse_args(argv)
    try:
        s = read_simulation(args.fields) if args.fields else {
            "chain_model": None, "nx": None, "lx": None, "nbar": None, "n_segments": None,
            "zeta_n": None, "smearing": None, "chi_n": {}, "segment_lengths": {}}
        for key, value in (("chain_model", args.chain_model), ("nx", args.nx), ("lx", args.lx),
                           ("n_segments", args.n), ("nbar", args.nbar), ("zeta_n", args.zeta_n)):
            if value is not None:
                s[key] = value
        if args.a_int is not None:
            s["smearing"] = {"type": "gaussian", "a_int": args.a_int} if args.a_int > 0 else None
        if args.k_int is not None:
            s["smearing"] = {"type": "sigmoidal", "k_int": args.k_int, "dk_int": args.dk_int}
        missing = [name for name in ("chain_model", "nx", "lx", "nbar") if s[name] is None]
        if missing:
            raise ValueError("Give a fields file or --" + ", --".join(m.replace("_", "-") for m in missing))
        lengths = {float(v) for v in s["segment_lengths"].values()}
        if len(lengths) > 1:
            raise ValueError("Only conformationally symmetric chains (equal segment_lengths) are supported")

        settings = (s["chain_model"], s["nx"], s["lx"], s["nbar"], s["n_segments"], s["zeta_n"],
                    s["smearing"], args.quadrature)
        z, ratio = z_inf(*settings), bare_segment_ratio(*settings)
        zeta = "incompressible" if s["zeta_n"] is None or math.isinf(s["zeta_n"]) else f"zeta N = {s['zeta_n']:g}"
        chain = f"N = {s['n_segments']}" if s["chain_model"] == "discrete" else "N -> inf"
        print(f"{s['chain_model']} chains, {chain}, Nbar = {s['nbar']:g}, "
              f"grid {s['nx']}, box {s['lx']}, {zeta}, smearing: {s['smearing'] or 'none'}")
        print(f"z_inf = {z:.7f}")
        print(f"a_b/a = {ratio:.7f}  (CL-FTS segment_lengths; L-FTS and SCFT use a_b = a)")
        if lengths and abs(next(iter(lengths)) - 1.0) > 1e-12:
            print(f"segment_lengths in the file: {next(iter(lengths)):.7f}")
        for pair, chi_b_n in dict(s["chi_n"]).items():
            print(f"chi_b N ({pair}) = {float(chi_b_n):.7f}  ->  z_inf chi_b N = {z * float(chi_b_n):.7f}")
        if args.target is not None:
            print(f"chi_b N for z_inf chi_b N = {args.target:g}: {args.target / z:.7f}")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    main()
