"""SCFT of an AB diblock lamella in a monoclinic cell that relaxes its lengths and the angle beta.

The lamellae are stacked along the third grid axis, so the fields vary along c only and the layer
spacing is the distance between the (001) planes, d = c sin(beta) (alpha = gamma = 90 degrees).
With crystal_system "Monoclinic" box relaxation varies a, b, c and beta; the free energy depends
on c and beta only through d, so the cell can end at any beta with the optimal d. The script
compares d with the period of the same lamella relaxed in a 1D box.
"""

import os
import contextlib
import io
import numpy as np

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import SCFT

f = 0.5                 # A-fraction of the diblock
chi_n = 15.0
beta = 80.0             # Starting angle between a and c

params = {
    "nx":[16, 16, 32],          # Simulation grid numbers
    "lx":[2.0, 2.0, 4.0],       # Lengths of the cell vectors a, b and c
    "angles":[90.0, beta, 90.0],    # [alpha, beta, gamma]; monoclinic: alpha = gamma = 90 degrees

    "crystal_system":"Monoclinic",  # a, b, c and beta vary
    "relax_cell":True,          # Relax the lengths and beta with the fields
    "stress_scale":0.1,         # Scaling of the stress entries in the optimizer vector

    "chain_model":"continuous",
    "ds":1/50,

    "segment_lengths":{"A":1.0, "B":1.0},
    "chi_n":{"A,B":chi_n},

    "polymers":[{
        "volume_fraction":1.0,
        "blocks":[{"type":"A", "length":f}, {"type":"B", "length":1-f}], }],

    "optimizer":{
        "name":"am",
        "max_hist":20,
        "start_error":1e-2,
        "mix_min":0.02,
        "mix_init":0.02,
    },

    "max_iter":3000,
    "tolerance":1e-8,
}

# Lamellae along the third axis
z = np.arange(params["nx"][2]) / params["nx"][2]
w_A = np.broadcast_to(np.cos(2 * np.pi * z), params["nx"]).copy()

calculation = SCFT(params=params)
result = calculation.run(initial_fields={"A": w_A, "B": -w_A}, return_result=True)
a, b, c = result.lx
alpha, beta_final, gamma = result.angles
spacing = c * np.sin(np.radians(beta_final))

# The same lamella in a 1D box
params_1d = dict(params, nx=[32], lx=[4.0], angles=[90.0, 90.0, 90.0], crystal_system="Orthorhombic")
with contextlib.redirect_stdout(io.StringIO()):
    period = SCFT(params=params_1d).run(initial_fields={"A": np.cos(2 * np.pi * z), "B": -np.cos(2 * np.pi * z)},
                                        return_result=True).lx[0]

print(f"\nCell: a = {a:.4f}, b = {b:.4f}, c = {c:.4f}, alpha = {alpha:.2f}, beta = {beta:.1f} -> {beta_final:.2f}, "
      f"gamma = {gamma:.2f} degrees, converged = {result.converged}")
print(f"Layer spacing c sin(beta) = {spacing:.5f}; period of the lamella in a 1D box = {period:.5f}")
