"""SCFT of counterions between two charged plates: the Poisson-Boltzmann (Gouy-Chapman) limit of
Netz, Eur. Phys. J. E 5, 557 (2001), Section 3.3.

Two plates of surface charge -sigma at distance d enclose point counterions of valence q that
neutralize them; there is no salt. In the weak-coupling limit (coupling parameter Xi -> 0) the
counterion density is the Poisson-Boltzmann solution (Netz Eqs. 63-65)

    n(z) = 2 pi l_B sigma^2 Lambda0 / cos^2(sqrt(Lambda0) (z/mu - d~/2)),
    sqrt(Lambda0) tan(d~ sqrt(Lambda0)/2) = 1,     d~ = d/mu,   mu = 1/(2 pi q l_B sigma),

and the pressure between the plates is P/kT = n(midplane) = 2 pi l_B sigma^2 Lambda0 (Eq. 69), which
the contact-value theorem (Eq. 67) also gives as n(contact) - 2 pi l_B sigma^2.

The plates are fixed charges (electrostatics.fixed_charge) in the first and the last cell of a
closed box (reflecting walls: psi is a cosine series with no field through the walls, which in
this planar mean-field geometry is the field of two isolated plates). The counterions are point
ions (electrostatics.ions) with a small Gaussian smearing length, and a neutral solvent S of one
segment, which the ions do not feel, fills the box. Code units: lengths in R0, densities per
segment of rho0; an ion of valence q feels q psi/N, and the Poisson equation -psi'' = E c makes
E/N = 4 pi l_B rho0 R0^2. Hence mu = 2/(q (E/N) sigma) and 2 pi l_B sigma^2 -> (E/N) sigma^2/2.

For d~ = 0.5, 2, 8 and 32 the script prints the PB pressure Lambda0, the SCFT midplane density and
the contact value extrapolated from the first cells (both divided by 2 pi l_B sigma^2), and the
largest relative deviation of the profile. With 1024 cells and the smearing a = dx/2 the midplane
pressure agrees to 3e-5 or better, the profile to 1e-3 or better (both converge as dx^2), and
the contact theorem holds to 1.4e-4 for d~ <= 8; at d~ = 32 the pressure is 1/117 of the contact
density, so the 2e-4 error of the extrapolated contact density becomes 2 % of P (it vanishes
as dx -> 0). Writes counterions_between_plates.json and counterions_between_plates.png.
"""

import contextlib
import io
import json
import os

import numpy as np
from scipy.optimize import brentq

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from polymerfts import SCFT

n_segments = 1.0   # The reference chain is the solvent molecule (one segment); an ion feels q psi/N
coupling = 100.0   # E = 4 pi l_B rho0 N R0^2
q = 1.0            # Counterion valence
d = 1.0            # Plate separation (box width) in R0
nx = 1024          # Grid cells across the box
smearing = 0.5 * d / nx   # Gaussian smearing length of the counterions (small: point-charge limit)


def pb_lambda(d_tilde):
    """Lambda0 of Netz Eq. (64): sqrt(L) tan(d~ sqrt(L)/2) = 1, 0 < sqrt(L) < pi/d~."""
    s = brentq(lambda s: s * np.tan(d_tilde * s / 2) - 1.0, 1e-12, np.pi / d_tilde - 1e-12, xtol=1e-15)
    return s * s


def plates(d_tilde):
    mu = d / d_tilde                          # Gouy-Chapman length
    sigma = 2.0 / (q * (coupling / n_segments) * mu)
    dx = d / nx
    fixed_charge = np.zeros(nx)
    fixed_charge[0] = fixed_charge[-1] = -sigma / dx   # charges per segment of rho0 in the end cells
    params = {
        "nx": [nx], "lx": [d],
        "bc": ["reflecting", "reflecting"],       # closed box: cosine series, no field through the walls
        "chain_model": "discrete", "ds": 1.0 / n_segments,
        "segment_lengths": {"S": 1.0}, "chi_n": {}, "zeta_n": 10.0,
        "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "S", "length": 1.0}]}],
        "smearing": {"type": "gaussian", "a_int": smearing},
        # The counterions Ci (no salt: concentration 0) neutralize the fixed charge automatically
        "electrostatics": {"charges": {"S": 0.0}, "coupling": coupling, "n_segments": n_segments,
                           "fixed_charge": fixed_charge,
                           "ions": {"Ci": {"valence": q, "concentration": 0.0, "smearing_length": smearing}}},
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
        "relax_cell": False, "max_iter": 20000, "tolerance": 1e-10, "verbose_level": 0,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        calculation = SCFT(params=params)
        result = calculation.run(initial_fields={"S": np.zeros(nx)}, return_result=True)
    (c,), _ = calculation.electrostatics.ion_densities(result.psi)
    x = (np.arange(nx) + 0.5) * dx
    scale = (coupling / n_segments) * sigma ** 2 / 2       # 2 pi l_B sigma^2 in code units
    lam = pb_lambda(d_tilde)
    c_pb = scale * lam / np.cos(np.sqrt(lam) * (x / mu - d_tilde / 2)) ** 2
    midplane = 0.5 * (c[nx // 2 - 1] + c[nx // 2]) / scale
    contact = np.polyval(np.polyfit(x[1:4], c[1:4], 2), 0.0) / scale   # cell 0 holds the plate
    inner = slice(2, nx - 2)
    return {"d_tilde": d_tilde, "converged": bool(result.converged), "P_pb": lam, "P_midplane": float(midplane),
            "P_contact": float(contact - 1.0),
            "max_rel_dev": float(np.max(np.abs(c[inner] / c_pb[inner] - 1))),
            "z_over_mu": (x / mu).tolist(), "n_tilde": (c / scale).tolist(), "n_tilde_pb": (c_pb / scale).tolist()}


rows = [plates(d_tilde) for d_tilde in (0.5, 2.0, 8.0, 32.0)]
print("   d/mu   P_PB (Netz Eq. 69)   P = n(mid)    rel. dev.   P = n(contact) - 1   rel. dev.   max |n/n_PB - 1|")
for r in rows:
    print(f"  {r['d_tilde']:5.1f}   {r['P_pb']:16.8f}   {r['P_midplane']:11.8f}   {r['P_midplane'] / r['P_pb'] - 1:+9.1e}"
          f"   {r['P_contact']:18.8f}   {r['P_contact'] / r['P_pb'] - 1:+9.1e}   {r['max_rel_dev']:12.1e}   {r['converged']}")
json.dump(rows, open("counterions_between_plates.json", "w"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for r in rows:
        line, = ax.plot(r["z_over_mu"][::8], r["n_tilde"][::8], "o", ms=3, label=f"SCFT, d/mu = {r['d_tilde']:g}")
        ax.plot(r["z_over_mu"], r["n_tilde_pb"], "-", color=line.get_color(), lw=1)
    z = np.linspace(0, 16, 200)
    ax.plot(z, 1 / (1 + z) ** 2, "k:", lw=1, label="one plate, 1/(1 + z/mu)^2 (Eq. 52)")
    ax.set_xlim(0, 16)
    ax.set_yscale("log")
    ax.set_xlabel("z / mu")
    ax.set_ylabel("n / (2 pi l_B sigma^2)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("counterions_between_plates.png", dpi=150)
except ImportError:
    pass
