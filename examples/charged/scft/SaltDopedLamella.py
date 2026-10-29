"""SCFT of a salt-doped AB diblock lamella whose B block solvates the ions.

The cation P and the anion M of the salt are molecules of one segment in `polymers`, not point
ions: they have volume and interact with the monomers through chi, like a salt in the
poly(ethylene oxide) block of PS-b-PEO. Both prefer B, the cation strongly (chi_AP N) and the
anion weakly (chi_AM N); they carry the valences +1 and -1 (electrostatics.charges). For each
salt fraction phi_s (volume fraction of each ion species) the period of the lamella is relaxed,
starting from the converged fields, psi and period of the previous one. The ratios of the ion
concentrations at the centers of the B and A domains show the selective solvation: on its own
the anion prefers B only weakly, but the Coulomb interaction drags it after the cation, and the
salt swells the B domains and lengthens the period.

Two references follow at the largest salt. With the charges switched off, the anion is
distributed by its own chi only. When the Debye length is much shorter than the domains, the
salt is neutral everywhere and acts as one neutral molecule of one segment with the volume
fraction 2 phi_s and the mean chi N of the two ions. The ion partitions and the free energy of
the charged lamella lie between these two references, close to the neutral salt at this coupling.
"""

import os
import numpy as np

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import SCFT

n_segments = 50     # Segments of the diblock, 1/ds
chi_ab = 15.0       # chi N between the blocks
chi_ap = 100.0      # chi N between A and the cation (B-selective)
chi_am = 25.0       # chi N between A and the anion (weakly B-selective)
coupling = 1e5      # E = 4 pi l_B rho_0 N R0^2 = 4 pi (l_B/R0) N^2 Nbar^(1/2)
salts = [0.005, 0.01, 0.02, 0.04]   # Volume fraction of each ion species


def ion(name, phi_s):
    """A molecule of one segment of the monomer type `name`."""
    return {"volume_fraction": phi_s, "blocks": [{"type": name, "length": 1 / n_segments}]}


def parameters(phi_s, lx, charged=True):
    params = {
        # "platform":"cpu-fftw",       # choose platform among [cuda, cpu-fftw, cpu-mkl]

        "nx":[128],                 # Simulation grid numbers
        "lx":[lx],                  # Simulation box size as a_Ref * N_Ref^(1/2) unit

        "relax_cell":True,          # Relax the period (periodic box)
        "chain_model":"discrete",   # Discrete chains: N = 1/ds is the number of segments
        "ds":1/n_segments,          # Contour step interval, which is equal to 1/N_Ref.

        "segment_lengths":{         # Relative statistical segment lengths
            "A":1.0,
            "B":1.0,
            "P":1.0,
            "M":1.0, },

        # Interaction parameters, Flory-Huggins params * N_Ref; pairs not given do not interact
        "chi_n": {"A,B": chi_ab, "A,P": chi_ap, "A,M": chi_am},

        "polymers":[{               # AB diblock copolymer
            "volume_fraction":1-2*phi_s,
            "blocks":[
                {"type":"A", "length":0.5, },
                {"type":"B", "length":0.5}, ],},
            ion("P", phi_s),        # Cation
            ion("M", phi_s), ],     # Anion

        # Smearing of the fields and charges; SCFT takes one length for all monomer types
        "smearing": {"type": "gaussian", "a_int": 0.05},

        "optimizer":{
            "name":"am",
            "max_hist":20,          # Maximum number of history
            "start_error":1e-2,     # When switch to AM from simple mixing
            "mix_min":0.02,         # Minimum mixing rate of simple mixing
            "mix_init":0.02,        # Initial mixing rate of simple mixing
        },

        "max_iter":3000,            # The maximum relaxation iterations
        "tolerance":1e-8,           # Terminate iteration if the self-consistency error is less than tolerance
    }
    if charged:
        params["electrostatics"] = {"charges": {"P": 1.0, "M": -1.0}, "coupling": coupling}
    return params


def partition(result, name):
    """Concentration of `name` at the center of the B domain over that at the center of the A domain."""
    phi_a = result.phi["A"]
    return result.phi[name][np.argmin(phi_a)] / result.phi[name][np.argmax(phi_a)]


# Initial fields of a lamella with one period in the box: the mean-field fields
# w_i = sum_j chi_ij N phi_j of a cosine profile of A
x = np.arange(128) / 128
phi_a = 0.5 + 0.4 * np.cos(2 * np.pi * x)
fields = {"A": chi_ab * (1 - phi_a), "B": chi_ab * phi_a, "P": chi_ap * phi_a, "M": chi_am * phi_a}

# Salt-doped lamellae
rows = []
psi, lx = None, 4.7
for phi_s in salts:
    doped = SCFT(params=parameters(phi_s, lx))
    result = doped.run(initial_fields=fields, initial_psi=psi, return_result=True)
    # Debye length of the salt at its mean concentration: (kappa_D R0)^2 = E sum_s z_s^2 c_s / N
    debye = 1.0 / np.sqrt(coupling * 2.0 * phi_s / n_segments)
    rows.append((f"{phi_s:7.3f}", f"{debye:10.4f}", result))
    fields, psi, lx = dict(result.w), result.psi, result.lx[0]

# At the largest salt: the ions without charges, and the locally neutral salt as one molecule of
# one segment S with the volume fraction 2 phi_s and the mean chi N of the two ions
uncharged = SCFT(params=parameters(salts[-1], lx, charged=False)).run(initial_fields=fields, return_result=True)
rows.append(("no charge", "", uncharged))
neutral_salt = parameters(salts[-1], lx, charged=False)
neutral_salt["segment_lengths"] = {"A": 1.0, "B": 1.0, "S": 1.0}
neutral_salt["chi_n"] = {"A,B": chi_ab, "A,S": (chi_ap + chi_am) / 2}
neutral_salt["polymers"] = [neutral_salt["polymers"][0], ion("S", 2 * salts[-1])]
neutral = SCFT(params=neutral_salt).run(
    initial_fields={"A": fields["A"], "B": fields["B"], "S": 0.5 * (fields["P"] + fields["M"])}, return_result=True)

# The neat diblock for comparison
neat_params = parameters(0.0, lx, charged=False)
neat_params.update(segment_lengths={"A": 1.0, "B": 1.0}, chi_n={"A,B": chi_ab}, polymers=neat_params["polymers"][:1])
neat = SCFT(params=neat_params).run(initial_fields={"A": fields["A"], "B": fields["B"]}, return_result=True)

# beta dF: free energy per reference chain (in units of rho_0 V/N) relative to the homogeneous state
# of the same composition; P and M: concentration at the center of the B domain over that at the
# center of the A domain
print()
print("      phi_s   1/kappa_D      period     beta dF        P        M  max-min psi  converged")
for label, debye, result in rows:
    psi_range = f"{np.ptp(result.psi):11.4f}" if result.psi is not None else ""
    print(f"  {label:>9s}  {debye:10s}  {result.lx[0]:10.5f}  {result.free_energy:10.6f}  "
          f"{partition(result, 'P'):7.3f}  {partition(result, 'M'):7.3f}  {psi_range:11s}  {result.converged}")
print(f"  {'neutral':>9s}  {'':10s}  {neutral.lx[0]:10.5f}  {neutral.free_energy:10.6f}  "
      f"{partition(neutral, 'S'):7.3f}  {partition(neutral, 'S'):7.3f}  {'':11s}  {neutral.converged}")
print(f"  {'neat':>9s}  {'':10s}  {neat.lx[0]:10.5f}  {neat.free_energy:10.6f}  {'':7s}  {'':7s}  {'':11s}  {neat.converged}")

# Save the last salt-doped state (.mat, .json or .yaml format), with psi
doped.save_results("fields_salt_doped_lamella.mat")
