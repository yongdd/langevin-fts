"""SCFT of AB diblock cylinders in an oblique 2D cell that relaxes its lengths and its angle.

The cell starts at gamma = 115 degrees with a = b. With crystal_system "Oblique2D", box relaxation
varies a, b and gamma independently, and the cell reaches the hexagonal packing of the
cylinders: gamma = 120 degrees and a = b. A start far from 120 degrees (below about 105 degrees)
can end in the rectangular packing near gamma = 90 degrees instead, a local minimum.
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import SCFT

f = 1/3                 # A-fraction: the A blocks form the cylinders
chi_n = 18.0
gamma = 115.0           # Starting angle between a and b
L = 1.8                 # Starting lengths a = b, in units of R0

params = {
    "nx":[32, 32],              # Simulation grid numbers
    "lx":[L, L],                # Lengths of the cell vectors a and b
    "angles":[gamma],           # 2D: the angle gamma between a and b

    "crystal_system":"Oblique2D",   # a, b and gamma vary independently
    "relax_cell":True,          # Relax the lengths and the angle with the fields
    "stress_scale":0.3,         # Scaling of the stress entries in the optimizer vector

    "chain_model":"continuous",
    "ds":1/90,

    "segment_lengths":{"A":1.0, "B":1.0},
    "chi_n":{"A,B":chi_n},

    "polymers":[{
        "volume_fraction":1.0,
        "blocks":[{"type":"A", "length":f}, {"type":"B", "length":1-f}], }],

    "optimizer":{
        "name":"am",
        "max_hist":20,
        "start_error":1e-2,
        "mix_min":0.05,
        "mix_init":0.05,
    },

    "max_iter":2000,
    "tolerance":1e-8,
}

# A cylinder at the origin, with distances measured in the metric of the oblique cell
nx, ny = params["nx"]
u = np.arange(nx)[:, None] / nx
v = np.arange(ny)[None, :] / ny
u, v = u - np.round(u), v - np.round(v)
r = np.sqrt(np.maximum(u**2 + v**2 + 2 * np.cos(np.radians(gamma)) * u * v, 0.0))
w_A = -gaussian_filter(0.5 * (1.0 - np.tanh((r - 0.2) / 0.05)), sigma=1.0, mode="wrap")
w_A *= 0.5 * chi_n / np.abs(w_A).max()

calculation = SCFT(params=params)
result = calculation.run(initial_fields={"A": w_A, "B": -w_A}, return_result=True)

print(f"\nStart:  a = b = {L:.4f}, gamma = {gamma:.2f} degrees")
print(f"Result: a = {result.lx[0]:.4f}, b = {result.lx[1]:.4f}, gamma = {result.angles[-1]:.2f} degrees "
      f"(hexagonal packing: a = b, gamma = 120), converged = {result.converged}")
