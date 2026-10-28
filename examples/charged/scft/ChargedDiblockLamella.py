"""SCFT of a charged AB diblock lamella in a salt of point ions.

The A blocks carry the valence z_A per segment. Cl- counterions neutralize them, and Na+ Cl- salt
is added at the concentration c_salt (ions per segment of the reference density). The ions are
point charges without volume and without chi (electrostatics.ions); SCFT solves the Poisson
equation for the electrostatic potential psi together with the fields
(docs/theory/SelfConsistentFieldTheory.md, Section 2.5). For each salt concentration the period of
the lamella is relaxed, starting from the converged fields, psi and period of the previous one.
The table compares the periods and free energies with those of the neutral diblock: the charges
shorten the period and lower the free-energy gain of the ordered state; the salt screens them at
the Debye length 1/kappa_D, (kappa_D R0)^2 = E sum_s z_s^2 c_s / N, and the charged lamella
approaches the neutral one.
"""

import os
import numpy as np

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import SCFT

f = 0.5             # A-fraction of the diblock
z_a = 0.2           # Valence of an A segment
n_segments = 50     # Segments of the diblock, 1/ds
coupling = 2000.0   # E = 4 pi l_B rho_0 N R0^2 = 4 pi (l_B/R0) N^2 Nbar^(1/2)
salts = [0.0, 0.01, 0.03, 0.1, 0.3]   # Added salt, ions per segment

params = {
    # "platform":"cpu-fftw",       # choose platform among [cuda, cpu-fftw, cpu-mkl]

    "nx":[64],                  # Simulation grid numbers
    "lx":[4.3],                 # Simulation box size as a_Ref * N_Ref^(1/2) unit (initial period)

    "relax_cell":True,          # Relax the period (periodic box)
    "chain_model":"discrete",   # Discrete chains: N = 1/ds is the number of segments
    "ds":1/n_segments,          # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment lengths
        "A":1.0,
        "B":1.0, },

    "chi_n": {"A,B": 18.0},     # Interaction parameter, Flory-Huggins params * N_Ref

    "polymers":[{               # AB diblock copolymer
        "volume_fraction":1.0,
        "blocks":[
            {"type":"A", "length":f, },
            {"type":"B", "length":1-f}, ],}, ],

    # Smearing of the fields and charges; SCFT takes one length for all monomer types
    "smearing": {"type": "gaussian", "a_int": 0.05},

    "optimizer":{
        "name":"am",
        "max_hist":20,          # Maximum number of history
        "start_error":1e-2,     # When switch to AM from simple mixing
        "mix_min":0.1,          # Minimum mixing rate of simple mixing
        "mix_init":0.1,         # Initial mixing rate of simple mixing
    },

    "max_iter":3000,            # The maximum relaxation iterations
    "tolerance":1e-8,           # Terminate iteration if the self-consistency error is less than tolerance
}


def electrostatics(c_salt):
    """Charges of A, point ions Na+ and Cl- with the salt c_salt; Cl- also receives the counterions."""
    ions = {"Cl": {"valence": -1.0, "concentration": c_salt, "smearing_length": 0.05}}
    if c_salt > 0.0:
        ions["Na"] = {"valence": 1.0, "concentration": c_salt, "smearing_length": 0.05}
    return {"charges": {"A": z_a}, "coupling": coupling, "ions": ions}


# Initial fields of a lamella with one period in the box
x = np.arange(params["nx"][0]) / params["nx"][0]
fields = {"A": 3.0 * np.cos(2 * np.pi * x), "B": -3.0 * np.cos(2 * np.pi * x)}

# Charged lamellae, from the salt-free melt to high salt
rows = []
psi, lx = None, params["lx"]
for c_salt in salts:
    charged = SCFT(params=dict(params, lx=lx, electrostatics=electrostatics(c_salt)))
    result = charged.run(initial_fields=fields, initial_psi=psi, return_result=True)
    # Every point ion screens: Cl- holds the salt and the counterions z_A f
    kappa_sq = coupling * (2.0 * c_salt + z_a * f) / n_segments
    rows.append((c_salt, 1.0 / np.sqrt(kappa_sq), result.lx[0], result.free_energy, np.ptp(result.psi), result.converged))
    fields, psi, lx = dict(result.w), result.psi, result.lx

# The neutral diblock for comparison
neutral = SCFT(params=params).run(initial_fields=fields, return_result=True)

# beta dF/n: free energy per chain relative to the homogeneous state with the same ions
print()
print("   c_salt   1/kappa_D      period   beta dF/n     max-min psi  converged")
for c_salt, debye, period, energy, psi_range, converged in rows:
    print(f"  {c_salt:7.3f}  {debye:10.4f}  {period:10.5f}  {energy:10.6f}  {psi_range:12.5f}  {converged}")
print(f"  neutral  {'':10s}  {neutral.lx[0]:10.5f}  {neutral.free_energy:10.6f}  {'':12s}  {neutral.converged}")

# Save the last charged state (.mat, .json or .yaml format), with psi and the ion chemical potentials
charged.save_results("fields_charged_lamella.mat")
