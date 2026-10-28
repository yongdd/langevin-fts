"""Free energy of AB diblock cylinders against the angle of a fixed 2D cell.

SCFT runs in fixed cells (relax_cell false) at a series of angles gamma. The angle entry of the
stress (SCFT.compute_stress, the derivative of the free energy with respect to gamma in radians)
changes sign at the optimal angle, where the script finds its root. With the lengths of the
relaxed hexagonal cell (a = b = 1.7019 R0, from ObliqueCylinder2D.py) the optimum is
gamma = 120 degrees. In fixed cells with other lengths the angle also sets the area of the cell,
and the optimum moves away from 120 degrees, also with a = b: for a = b = 1.8 R0, longer than the
spacing the cylinders prefer, opening the angle shortens the distance |a + b| = 2a cos(gamma/2)
between neighbors, and the optimum is about 123.5 degrees. No crystal system relaxes the angle
alone: Oblique2D varies a, b and gamma together and reaches the hexagonal cell
(ObliqueCylinder2D.py), so the optimum at fixed lengths comes from the stress, as here.
"""

import os
import contextlib
import io
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import SCFT

f = 1/3                 # A-fraction: the A blocks form the cylinders
chi_n = 18.0


def free_energy(a, b, gamma):
    """Free energy and the angle entry of the stress of cylinders in the cell (a, b, gamma)."""
    params = {
        "nx":[32, 32], "lx":[a, b], "angles":[gamma],
        "crystal_system":"Oblique2D",   # the stress holds the derivative with respect to gamma
        "relax_cell":False,             # Fixed cell
        "chain_model":"continuous", "ds":1/90,
        "segment_lengths":{"A":1.0, "B":1.0}, "chi_n":{"A,B":chi_n},
        "polymers":[{"volume_fraction":1.0,
                     "blocks":[{"type":"A", "length":f}, {"type":"B", "length":1-f}]}],
        "optimizer":{"name":"am", "max_hist":20, "start_error":1e-2, "mix_min":0.1, "mix_init":0.1},
        "max_iter":2000, "tolerance":1e-8,
    }
    nx, ny = params["nx"]
    u = np.arange(nx)[:, None] / nx
    v = np.arange(ny)[None, :] / ny
    u, v = u - np.round(u), v - np.round(v)
    r = np.sqrt(np.maximum(u**2 + v**2 + 2 * np.cos(np.radians(gamma)) * u * v, 0.0))
    w_A = -gaussian_filter(0.5 * (1.0 - np.tanh((r - 0.2) / 0.05)), sigma=1.0, mode="wrap")
    w_A *= 0.5 * chi_n / np.abs(w_A).max()
    with contextlib.redirect_stdout(io.StringIO()):
        calculation = SCFT(params=params)
        result = calculation.run(initial_fields={"A": w_A, "B": -w_A}, return_result=True)
        w = np.array([result.w[m] for m in calculation.monomer_types])
        calculation.compute_concentrations(w)
        stress = calculation.compute_stress(w, result.phi)
    assert result.converged, (a, b, gamma)
    return result.free_energy, stress[2]


for a, b in ((1.7019, 1.7019), (1.8, 1.8), (2.0, 1.5)):
    print(f"\nCell a = {a}, b = {b}:")
    print("  gamma   free energy   dF/dgamma (stress)")
    rows = [(gamma,) + free_energy(a, b, gamma) for gamma in range(100, 145, 5)]
    for gamma, energy, stress in rows:
        print(f"  {gamma:5.0f}   {energy:11.7f}   {stress:+.3e}")
    # The root of the stress between the two angles where it changes sign
    for (g1, _, s1), (g2, _, s2) in zip(rows, rows[1:]):
        if s1 < 0 <= s2:
            optimum = brentq(lambda gamma: free_energy(a, b, gamma)[1], g1, g2, xtol=1e-3)
            print(f"  optimal angle (root of the stress): {optimum:.2f} degrees")
