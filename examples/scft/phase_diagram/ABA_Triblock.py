"""Phase diagram of an ABA triblock copolymer melt in the (f, chi N) plane.

f is the total A fraction; each A block has length f/2. Only the system
definition differs from AB_Diblock.py. The seeds are the diblock phases, placed
where each phase exists for the triblock, which orders at higher chi N than the
diblock (chi N = 18.0 at f = 0.5).

Run on a GPU: about 1500 SCFT solves, 45 minutes on an A10. The results are saved
in ABA_Triblock.json and ABA_Triblock.png.
"""

import os

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

import numpy as np
import phase_diagram

# Parameters shared by all phases
base_params = {
    "relax_cell": True,
    "stress_interval": 1,
    "chain_model": "continuous",
    "ds": 1/100,
    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 20.0},
    "polymers": [{"volume_fraction": 1.0, "blocks": [
        {"type": "A", "length": 0.25},
        {"type": "B", "length": 0.5},
        {"type": "A", "length": 0.25}]}],
    "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2,
                  "mix_min": 0.1, "mix_init": 0.1},
    "max_iter": 2000,
    "tolerance": 1e-6,
}


def set_point(params, f, chi_n):
    params["polymers"][0]["blocks"][0]["length"] = f/2
    params["polymers"][0]["blocks"][1]["length"] = 1 - f
    params["polymers"][0]["blocks"][2]["length"] = f/2
    params["chi_n"]["A,B"] = chi_n


phases = {
    "L": {"seed": "L1D.json", "point": (0.5, 24.0), "params": {"nx": [32]}},
    "C": {"seed": "C2D.json", "point": (0.34, 24.0), "params": {"nx": [48, 32]}},
    "G": {"seed": "DG.json", "point": (0.38, 24.0),
          "params": {"nx": [32, 32, 32], "space_group": {"symbol": "Ia-3d", "hall_number": 530}}},
    "S": {"seed": "BCC.json", "point": (0.30, 32.0),
          "params": {"nx": [32, 32, 32], "space_group": {"symbol": "Im-3m", "hall_number": 529}}},
}

phase_diagram.run(
    base_params, set_point, phases,
    x_grid=np.round(np.arange(0.26, 0.5001, 0.04), 6),
    y_grid=np.round(np.arange(14.0, 34.0001, 2.0), 6),
    x_tol=1e-3, y_tol=1e-2,
    derivatives={"y": phase_diagram.chi_derivative("A", "B")},
    seed_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"),
    labels=("f (total A fraction)", r"$\chi N$"), mirror_x=0.5, output="ABA_Triblock")
