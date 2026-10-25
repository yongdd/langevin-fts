#!/usr/bin/env python3
"""Grid-size limits of the 32-bit grid indices (GridLimits.h).

A computation box larger than MAX_GRID_POINTS, Anderson mixing of more than
MAX_GRID_ARRAY variables, and a CUDA launch configuration of more than 2^20
threads must be rejected with a clear message before anything is allocated.
"""

import os
import subprocess
import sys

from polymerfts import _core, PlatformSelector


def main():
    failed = []
    max_points, max_array = _core.MAX_GRID_POINTS, _core.MAX_GRID_ARRAY
    if max_array != 2**31 - 1 - 2**20 or max_points != max_array // 2:
        failed.append(f"limits {max_points}, {max_array}")

    # The space group is built before the box and must reject the same grids
    try:
        _core.SpaceGroup([1024, 1024, 1024], "Im-3m", 529)
        failed.append("space group on a 1024^3 grid accepted")
    except Exception as e:
        if "32-bit" not in str(e):
            failed.append(f"space group on a 1024^3 grid: unexpected error {e}")

    avail = PlatformSelector.avail_platforms()
    for platform in avail:
        factory = PlatformSelector.create_factory(platform, False, "real")

        # 1024^3 is just above the limit; 2048^3 overflowed a 32-bit product
        for nx in ([1024, 1024, 1024], [2048, 2048, 2048]):
            try:
                factory.create_computation_box(nx=nx, lx=[1.0] * 3, bc=["periodic"] * 6)
                failed.append(f"{platform}: box {nx} accepted")
            except Exception as e:
                if "32-bit" not in str(e):
                    failed.append(f"{platform}: box {nx}: unexpected error {e}")

        # Anderson mixing just above the limit (rejected before allocating)
        try:
            factory.create_anderson_mixing(max_array + 1, 20, 0.1, 0.1, 0.1)
            failed.append(f"{platform}: Anderson mixing of {max_array + 1} variables accepted")
        except Exception as e:
            if "32-bit" not in str(e):
                failed.append(f"{platform}: Anderson mixing: unexpected error {e}")

    # CUDA launch configuration of 2^21 threads
    if "cuda" in avail:
        # The launch configuration is read when the first kernel runs
        code = ("import numpy as np\n"
                "from polymerfts import PropagatorSolver\n"
                "s = PropagatorSolver(nx=[8, 8, 8], lx=[1.0] * 3, ds=0.1, bond_lengths={'A': 1.0},\n"
                "                     bc=['periodic'] * 6, chain_model='continuous', numerical_method='rqm4',\n"
                "                     platform='cuda')\n"
                "s.add_polymer(1.0, [['A', 1.0, 0, 1]])\n"
                "s.compute_propagators({'A': np.zeros(512)})\n")
        env = dict(os.environ, LFTS_GPU_NUM_BLOCKS="8192", LFTS_GPU_NUM_THREADS="256")
        run = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        if run.returncode == 0 or "threads in total" not in run.stdout + run.stderr:
            failed.append("CUDA launch of 2^21 threads accepted or rejected without the message")

    for f in failed:
        print("FAILED:", f)
    if failed:
        sys.exit(1)
    print("Grid limits: OK (platforms: " + ", ".join(avail) + ")")


if __name__ == "__main__":
    main()
