"""Phase diagram of an AB diblock copolymer melt in the (f, chi N) plane.

Candidate phases: lamellae (L), cylinders (C), double gyroid (G), BCC spheres (S)
and the disordered melt (DIS). A and B have equal segment lengths, so the diagram
is symmetric about f = 0.5; it is computed for f <= 0.5 and mirrored in the plot.
The order-disorder transition at f = 0.5 is the mean-field critical point
chi N = 10.495.

Run on a GPU: about 2000 SCFT solves, about an hour on an A10. The results are
saved in AB_Diblock.json and AB_Diblock.png.
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
        {"type": "A", "length": 0.5},
        {"type": "B", "length": 0.5}]}],
    "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2,
                  "mix_min": 0.1, "mix_init": 0.1},
    "max_iter": 2000,       # a stalled run is stopped much earlier
    # The free energy converges faster than the fields: at this tolerance it is
    # within about 1e-8 of its converged value, far below the differences that
    # locate the boundaries
    "tolerance": 1e-6,
}


def set_point(params, f, chi_n):
    params["polymers"][0]["blocks"][0]["length"] = f
    params["polymers"][0]["blocks"][1]["length"] = 1 - f
    params["chi_n"]["A,B"] = chi_n


# Candidate phases: seed, the point (f, chi N) of the seed and the parameters of
# the phase. Each phase uses its cheapest cell: lamellae in 1D, cylinders in a 2D
# rectangular cell holding two of them, gyroid and BCC spheres in 3D with their
# space groups.
phases = {
    "L": {"seed": "L1D.json", "point": (0.5, 13.27), "params": {"nx": [32]}},
    "C": {"seed": "C2D.json", "point": (1/3, 15.0), "params": {"nx": [48, 32]}},
    "G": {"seed": "DG.json", "point": (0.4, 15.0),
          "params": {"nx": [32, 32, 32], "space_group": {"symbol": "Ia-3d", "hall_number": 530}}},
    "S": {"seed": "BCC.json", "point": (24/90, 18.1),
          "params": {"nx": [32, 32, 32], "space_group": {"symbol": "Im-3m", "hall_number": 529}}},
}

phase_diagram.run(
    base_params, set_point, phases,
    # Grid values of f are multiples of ds so that both blocks fit the contour
    # grid; boundary points between them use local contour steps
    x_grid=np.round(np.arange(0.18, 0.5001, 0.04), 6),
    y_grid=np.round(np.arange(10.0, 30.0001, 2.0), 6),
    x_tol=1e-3, y_tol=1e-2,
    derivatives={"y": phase_diagram.chi_derivative("A", "B")},
    seed_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"),
    labels=("f", r"$\chi N$"), mirror_x=0.5, output="AB_Diblock")
