"""Charged-polymer CL-FTS demo: polyelectrolyte solution with counter-ions.

Model (see THEORY.md): compressible (zeta_n) SPT + smeared charges, run
with the devel-only ChargedCLFTS — the electrostatic potential psi is a
FULLY FLUCTUATING imaginary-type field with complex Langevin dynamics
(semi-implicit high-k treatment), not a partial saddle. Mainline
polymerfts is untouched.

Species: P = polyelectrolyte segment (charge fraction z_P per segment),
S = solvent, C = counter-ion. Each solvent/ion "chain" is one segment
(length = ds). All charged interactions use per-species Gaussian smearing
of radius a_i ("radiuses").

Electroneutrality: sum_i z_i * phibar_i = 0 (per-segment fractions) —
the counter-ion volume fraction is COMPUTED from the polymer charge,
never chosen independently.

Stability notes:
- Exchange-field stiffness ~ 1/chi_n sets an explicit-Euler limit
  dt < chi_n (in these units); small-chi systems need small dt.
- alpha_ds (Willis & Matsen) recommendations: ~0.01-0.02 for nbar >= 1e5,
  ~0.1 for nbar ~ 1e4.
"""
import os
import time
import numpy as np

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"        # 1 ~ 4

from clfts_charged import ChargedCLFTS

# ---------------- Composition (electroneutral by construction) ----------
z_polymer = 0.2          # charge per P segment (charge fraction)
z_counter = -1.0         # counter-ion valence
polymer_fraction = 0.1

# sum_i z_i phibar_i = 0  =>  phi_C = z_P * phi_P / |z_C|
counter_fraction = z_polymer * polymer_fraction / abs(z_counter)
solvent_fraction = 1.0 - polymer_fraction - counter_fraction

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[40, 40, 40],          # Simulation grid numbers
    "lx":[4.36, 4.36, 4.36],    # Box size in a_Ref * N_Ref^(1/2) units
                                # dx = 0.109; keep min(radiuses) >~ dx/2

    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/100,                 # Contour step interval (= 1/N_Ref)

    "segment_lengths":{         # Relative statistical segment lengths
        "P":1.0,
        "S":1.0,
        "C":1.0,
        },

    "chi_n": {"P,S":50},        # Flory-Huggins params * N_Ref (others 0)
    "zeta_n": 100.0,            # Helfand compressibility * N_Ref (required
                                # in charged mode - compressible model)

    "charges":{                 # Charge per segment (None or 0 = neutral)
        "P": z_polymer,
        "S": None,
        "C": z_counter,
        },

    "radiuses":{                # Smearing length / Born radius a_i (R0 units)
        "P": 0.1,
        "S": 0.1,
        "C": 0.1,
        },

    # Coulomb coupling: E = 4 pi (l_B/R0) N_Ref^2 sqrt(nbar).
    # Give l_B/R0 here ("bjerrum_length"), or the precomputed E directly
    # as "bjerrum_e".
    "bjerrum_length": 0.01,

    # Mobility scaling of the psi update (default 1.0). The lap/E part is
    # integrated semi-implicitly, so this mainly sets how fast psi tracks
    # the charge density.
    "psi_dt_scaling": 1.0,

    "distinct_polymers":[
        {   # Polyelectrolyte
            "volume_fraction":polymer_fraction,
            "blocks":[
                {"type":"P", "length":1.0},
            ],
        },
        {   # Solvent (single segment)
            "volume_fraction":solvent_fraction,
            "blocks":[
                {"type":"S", "length":0.01},
            ],
        },
        {   # Counter-ion (single segment)
            "volume_fraction":counter_fraction,
            "blocks":[
                {"type":"C", "length":0.01},
            ],
        },
        ],

    "langevin":{                # Complex Langevin dynamics
        "max_step":10000,       # Langevin steps for simulation
        "dt":0.5,               # Langevin step interval, delta tau*N_Ref
        "nbar":10000,           # Invariant polymerization index
    },

    "recording":{                       # Recording simulation data
        "dir":"data_simulation",        # Directory name
        "recording_period":1000,        # Period for fields/concentrations
        "sf_computing_period":10,       # Period for structure function
        "sf_recording_period":10000,    # Period for recording str. func.
    },

    # Dynamical stabilization (Willis & Matsen 2024): damps Im[W-] hot
    # spots. 0.1 is appropriate for nbar ~ 1e4 (may slightly bias).
    "alpha_ds": 0.1,

    "platform":"cuda",
    "verbose_level":1,      # 1: print each Langevin step
}

# Set random seed
# If you want different results for each execution, set random_seed=None
random_seed = 12345
np.random.seed(random_seed)

# Initial fields: seed a lamellar modulation in P vs S
w_P = np.zeros(params["nx"], dtype=np.float64)
w_S = np.zeros(params["nx"], dtype=np.float64)
w_C = np.zeros(params["nx"], dtype=np.float64)
for i in range(params["nx"][2]):
    w_P[:, :, i] =  np.cos(3*2*np.pi*i/params["nx"][2])
    w_S[:, :, i] = -np.cos(3*2*np.pi*i/params["nx"][2])

# Initialize calculation (validates electroneutrality at construction)
simulation = ChargedCLFTS(params=params, random_seed=random_seed)

time_start = time.time()

# # Continue simulation with recorded fields and random state
# simulation.continue_run(file_name="data_simulation/fields_010000.mat")

# Run
simulation.run(initial_fields={"P": w_P, "S": w_S, "C": w_C})

print(f"total time: {time.time()-time_start:.2f} s")
