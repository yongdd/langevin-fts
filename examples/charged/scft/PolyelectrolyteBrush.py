"""SCFT of a polyelectrolyte brush in a salt solution between two walls.

Charged A chains are grafted to the wall at x = 0 of a closed box (reflecting walls on both sides)
and swollen by an athermal solvent S of molecules of the size of one segment (contour length 1/N).
The A segments carry the valence z_A, and the substrate a fixed surface charge of the same sign
(electrostatics.fixed_charge, in the first grid cell). Na+ counterions neutralize both, and
Na+ Cl- salt is added. The ions are point charges without volume (electrostatics.ions). With walls
the electrostatic potential psi is a cosine series, which has no field through the walls
(docs/theory/PolymerFieldTheory.md, Section 5.3).

The salt decreases from run to run, and each run starts from the fields and psi of the previous
one. Salt screens the charges at the Debye length, so the brush swells as the salt is removed,
until the counterions trapped inside it set its height (the osmotic regime). For each salt
concentration the script prints the brush height <x> (the first moment of phi_A), the
concentrations of both ions at the far wall and the potential difference between the two walls
per unit charge. At c_salt >= 0.03 the far wall lies in a neutral salt solution (equal
concentrations); at lower salt the box is too narrow for that, and the counterions exceed the
co-ions there (by 5 % at c_salt = 0.01 and a factor of 3 at 0.001).

The propagators use the real-space method cn-adi2. Those of the grafted chains fall by many
orders of magnitude between the grafting wall and the far wall (phi_A drops from 0.37 to 4e-17 at
c_salt = 0.1), and the round-off of the pseudo-spectral methods (rqm4, rk2 and the discrete chain
model), about 1e-16 of the largest value of a propagator, then grows into spurious and negative
concentrations far from the brush. In this box rqm4 fails at every salt concentration
(docs/internals/ExampleValidation.md).
"""

import os
import numpy as np

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import SCFT

z_a = -0.2          # Valence of an A segment
phi_brush = 0.1     # Volume fraction of the brush in the box (sets the grafting density)
n_segments = 50     # Segments N of a brush chain (the contour step ds is a numerical choice)
coupling = 2000.0   # E = 4 pi l_B rho_0 N R0^2 = 4 pi (l_B/R0) N^2 Nbar^(1/2)
substrate = 0.25    # Fixed charge of the substrate relative to the charge of the brush (same sign)
salts = [0.1, 0.03, 0.01, 0.003, 0.001]   # Added salt in the box, ions per segment

nx, lx = 128, 6.0
params = {
    # "platform":"cpu-fftw",       # choose platform among [cuda, cpu-fftw, cpu-mkl]

    "nx":[nx],                  # Simulation grid numbers
    "lx":[lx],                  # Simulation box size as a_Ref * N_Ref^(1/2) unit
    "bc":["reflecting", "reflecting"],   # Walls on both sides: a closed box
    "relax_cell":False,         # Fixed box: box relaxation of charged systems needs a periodic box

    "chain_model":"continuous", # cn-adi2 takes continuous chains
    "ds":0.01,                  # Contour step
    "numerical_method":"cn-adi2",   # Real-space propagators (finite differences)

    "segment_lengths":{         # Relative statistical segment lengths
        "A":1.0,
        "S":1.0, },

    "chi_n": {"A,S": 0.0},      # Athermal (good) solvent; SCFT notes that this chi N matrix is singular

    "polymers":[
        {   # Brush: node 0 starts at the grafting points, given by q_init["G"]
            "volume_fraction":phi_brush,
            "blocks":[{"type":"A", "length":1.0, "v":0, "u":1}],
            "initial_conditions":{0:"G"}, },
        {   # Solvent of the size of one segment
            "volume_fraction":1.0-phi_brush,
            "blocks":[{"type":"S", "length":1/n_segments}], }, ],

    # Smearing of the fields and charges; SCFT takes one length for all monomer types
    "smearing": {"type": "gaussian", "a_int": 0.05},

    "optimizer":{
        "name":"am",
        "max_hist":20,          # Maximum number of history
        "start_error":1e-2,     # When switch to AM from simple mixing
        "mix_min":0.1,          # Minimum mixing rate of simple mixing
        "mix_init":0.1,         # Initial mixing rate of simple mixing
    },

    "max_iter":5000,            # The maximum relaxation iterations
    "tolerance":1e-8,           # Terminate iteration if the self-consistency error is less than tolerance
}

# Grafting points: all chains start in the first grid cell (the scale of q_init does not matter)
dx = lx / nx
x = (np.arange(nx) + 0.5) * dx
q_graft = np.zeros(nx)
q_graft[0] = 1.0 / dx

# Substrate charge: charges per segment of the reference density in the first grid cell. Per unit
# area the brush holds z_A phi_brush L charges (in these units), the cell fixed_charge[0] dx.
fixed_charge = np.zeros(nx)
fixed_charge[0] = substrate * z_a * phi_brush * lx / dx


def electrostatics(c_salt):
    """Charges of A, the substrate and point ions Na+ (also the counterions) and Cl- with the salt c_salt."""
    # Continuous chains give N through n_segments: a segment and an ion feel z psi/N
    return {"charges": {"A": z_a}, "coupling": coupling, "n_segments": n_segments, "fixed_charge": fixed_charge,
            "ions": {"Na": {"valence": 1.0, "concentration": c_salt, "smearing_length": 0.05},
                     "Cl": {"valence": -1.0, "concentration": c_salt, "smearing_length": 0.05}}}


rows = []
fields = {"A": np.zeros(nx), "S": np.zeros(nx)}
psi = None
for c_salt in salts:
    calculation = SCFT(params=dict(params, electrostatics=electrostatics(c_salt)))
    result = calculation.run(initial_fields=fields, initial_psi=psi, q_init={"G": q_graft}, return_result=True)
    fields, psi = dict(result.w), result.psi

    height = np.sum(x * result.phi["A"]) / np.sum(result.phi["A"])
    # Ion profiles c_s(x), in ions per segment of the reference density
    (na, cl), _ = calculation.electrostatics.ion_densities(psi)
    # Potential difference between the grafting wall and the far wall, per unit charge in units of kT
    potential = (psi[0] - psi[-1]) / n_segments
    rows.append((c_salt, na[-1], cl[-1], height, potential, result.converged))

print()
print("   c_salt   Na+ (far wall)   Cl- (far wall)   height <x>   (psi(0) - psi(L))/N   converged")
for c_salt, na_wall, cl_wall, height, potential, converged in rows:
    print(f"  {c_salt:7.3f}  {na_wall:15.5f}  {cl_wall:15.5f}  {height:11.4f}  {potential:20.4f}   {converged}")

# Save the last state; YAML keeps the integer keys of initial_conditions, JSON and .mat do not
calculation.save_results("fields_brush.yaml")
