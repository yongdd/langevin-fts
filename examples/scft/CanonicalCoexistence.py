"""A binary binodal from CE SCFT and analytic FH states, without a GCE solver.

Run from a scratch directory: SCFT constructors draw polymer architectures.
For modulated phases, give each callback its own phase-specific initial fields
and reuse that branch's converged fields and cell for nearby compositions.
"""

import contextlib
import io
import numpy as np
from polymerfts import SCFT, find_binary_coexistence, homogeneous_thermodynamics

params = {
    "nx": [16], "lx": [1.0], "chain_model": "continuous", "ds": 0.02,
    "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 2.0},
    "box_is_altering": False, "max_iter": 100, "tolerance": 1e-10,
    "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
}
previous = None


def canonical_phase(x):
    global previous
    polymers = [
        {"volume_fraction": 1-x, "blocks": [{"type": "A", "length": 1.0}]},
        {"volume_fraction": x, "blocks": [{"type": "B", "length": 2.0}]},
    ]
    fields = previous.w if previous is not None else {"A": np.zeros(16), "B": np.zeros(16)}
    with contextlib.redirect_stdout(io.StringIO()):
        calculation = SCFT(dict(params, distinct_polymers=polymers))
        previous = calculation.run(fields, return_result=True)
    return previous


def disordered_phase(x):
    return homogeneous_thermodynamics([1-x, x], [1.0, 2.0], np.eye(2), [[0, 2], [2, 0]])


coexistence = find_binary_coexistence(
    canonical_phase, disordered_phase, initial_compositions=[0.06, 0.85],
    composition_bounds=((0.01, 0.25), (0.65, 0.98)), tolerance=1e-10,
)
print("Coexisting polymer-1 volume fractions:", coexistence["compositions"])
print("Exchange chemical potential:", coexistence["exchange_chemical_potential"])
print("Reduced grand potential:", coexistence["reduced_grand_potential"])
print("Coexistence residuals:", coexistence["residuals"])
