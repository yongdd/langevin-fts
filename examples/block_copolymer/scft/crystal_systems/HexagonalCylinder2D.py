"""SCFT of AB diblock cylinders in a hexagonal 2D cell that relaxes its lattice constant.

With crystal_system "Hexagonal2D" the cell keeps gamma = 120 degrees and a = b; box relaxation
varies the single lattice constant a, the distance between neighboring cylinders.
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
L = 1.9                 # Starting lattice constant a = b, in units of R0

params = {
    "nx":[32, 32],              # Simulation grid numbers
    "lx":[L, L],                # Hexagonal cell: a = b
    "angles":[120.0],           # 2D: gamma = 120 degrees

    "crystal_system":"Hexagonal2D", # a = b, gamma fixed at 120 degrees
    "relax_cell":True,          # Relax the lattice constant with the fields

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
        "mix_min":0.1,
        "mix_init":0.1,
    },

    "max_iter":2000,
    "tolerance":1e-8,
}

# A cylinder at the origin, with distances measured in the metric of the hexagonal cell
nx, ny = params["nx"]
u = np.arange(nx)[:, None] / nx
v = np.arange(ny)[None, :] / ny
u, v = u - np.round(u), v - np.round(v)
r = np.sqrt(np.maximum(u**2 + v**2 + 2 * np.cos(np.radians(120.0)) * u * v, 0.0))
w_A = -gaussian_filter(0.5 * (1.0 - np.tanh((r - 0.2) / 0.05)), sigma=1.0, mode="wrap")
w_A *= 0.5 * chi_n / np.abs(w_A).max()

calculation = SCFT(params=params)
result = calculation.run(initial_fields={"A": w_A, "B": -w_A}, return_result=True)

print(f"\nLattice constant: {L:.4f} -> {result.lx[0]:.4f} R0 "
      f"(gamma = {calculation.prop_solver.get_angles_degrees()[-1]:.1f} degrees), "
      f"free energy {result.free_energy:.7f}, converged = {result.converged}")
