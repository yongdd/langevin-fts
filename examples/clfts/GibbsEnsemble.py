"""Coexisting phases of a binary blend from a Gibbs-ensemble CL-FTS.

Two boxes of a symmetric blend of A and B homopolymers (chi N = 3, above the critical value 2)
exchange chains and volume at fixed totals until their exchange potentials
mu_p - alpha_p Pi agree (polymerfts.GibbsEnsemble, docs/theory/ComplexLangevinSimulation.md,
Section 8). Each cycle runs 200 Langevin steps in each box. The compositions of the second half
of the cycles are compared with the Flory-Huggins binodal ln(phi/(1 - phi)) = chi N (2 phi - 1);
the fluctuations at this Nbar shift it. The CL-FTS output of the boxes goes to
data_gibbs/box0 and data_gibbs/box1, the history of the cycles to data_gibbs/gibbs_history.json.
"""

import os

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import brentq

from polymerfts import GibbsEnsemble

chi_n = 3.0

params = {
    "nx": [16, 16],                 # Grid of each box
    "lx": [2.0, 2.0],               # Not used: each box has its own lengths (gibbs_ensemble.boxes)
    "chain_model": "discrete",
    "ds": 1 / 32,
    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": chi_n},
    "zeta_n": 100.0,

    # Two homopolymers; their volume fractions in each box come from gibbs_ensemble.boxes
    "polymers": [{"volume_fraction": 0.5, "blocks": [{"type": "A", "length": 1.0}]},
                 {"volume_fraction": 0.5, "blocks": [{"type": "B", "length": 1.0}]}],

    "langevin": {"dt": 0.5, "nbar": 10000, "max_step": 1},   # max_step: set by the cycles

    "recording": {"dir": "data_gibbs", "checkpoint_interval": 1, "sample_interval": 2,
                  "statistics_save_interval": 1},                # intervals: set by the cycles

    "gibbs_ensemble": {
        # Starting compositions and box lengths; the totals stay fixed
        "boxes": [{"lx": [2.0, 2.0], "volume_fractions": [0.8, 0.2]},
                  {"lx": [2.0, 2.0], "volume_fractions": [0.2, 0.8]}],
        "cycles": 40,               # Exchanges
        "cycle_steps": 200,         # Langevin steps of each box per exchange
        "skip_steps": 40,           # Steps at the start of each cycle left out of the averages
        "exchange_rate": 0.5,       # Rate of the exchange of chains and volume
    },
    "verbose_level": 0,
}

gibbs = GibbsEnsemble(params, random_seed=1)
summary = gibbs.run()

binodal = brentq(lambda phi: np.log(phi / (1 - phi)) - chi_n * (2 * phi - 1), 0.51, 0.999)
print("Volume fraction of A over the last %d cycles:" % summary["averaged_cycles"])
for b, box in enumerate(summary["boxes"]):
    print("  box %d: %.4f (volume %.3f, pressure %.4f)" % (b, box["volume_fractions"][0], box["volume"], box["pressure"]))
print("Flory-Huggins binodal: %.4f and %.4f" % (binodal, 1 - binodal))
