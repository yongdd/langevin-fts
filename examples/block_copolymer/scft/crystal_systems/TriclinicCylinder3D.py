"""SCFT of AB diblock cylinders in a triclinic cell that relaxes all lengths and angles.

The cylinders run along c, and the cell starts with gamma = 115 degrees between a and b. With
crystal_system "Triclinic" box relaxation varies a, b, c, alpha, beta and gamma; the cross-section
reaches the hexagonal packing (a = b, gamma = 120 degrees), while c, alpha and beta, along which
the fields do not vary, keep their values (no stress acts on them).
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
    "nx":[24, 24, 8],           # Few points along the cylinder axis c
    "lx":[L, L, L/2],           # Lengths of the cell vectors a, b and c
    "angles":[90.0, 90.0, gamma],   # [alpha, beta, gamma]

    "crystal_system":"Triclinic",   # All lengths and angles vary
    "relax_cell":True,          # Relax the lengths and angles with the fields
    "stress_scale":0.2,         # Scaling of the stress entries in the optimizer vector

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

# A cylinder along c through the origin, with distances measured in the metric of the a-b plane
nx, ny, nz = params["nx"]
u = np.arange(nx)[:, None] / nx
v = np.arange(ny)[None, :] / ny
u, v = u - np.round(u), v - np.round(v)
r = np.sqrt(np.maximum(u**2 + v**2 + 2 * np.cos(np.radians(gamma)) * u * v, 0.0))
w_A = -gaussian_filter(0.5 * (1.0 - np.tanh((r - 0.2) / 0.05)), sigma=1.0, mode="wrap")
w_A = np.repeat((0.5 * chi_n / np.abs(w_A).max() * w_A)[:, :, None], nz, axis=2)

calculation = SCFT(params=params)
result = calculation.run(initial_fields={"A": w_A, "B": -w_A}, return_result=True)

print(f"\nStart:  a = b = {L:.4f}, c = {L/2:.4f}, angles = 90, 90, {gamma:.1f} degrees")
print(f"Result: a = {result.lx[0]:.4f}, b = {result.lx[1]:.4f}, c = {result.lx[2]:.4f}, angles = "
      + ", ".join(f"{angle:.2f}" for angle in result.angles) + f" degrees, converged = {result.converged}")
