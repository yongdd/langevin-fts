#!/usr/bin/env python3
"""SCFT.run progress callback (on_iteration) and IterationInfo.

Runs a small 1D lamellar SCFT with box relaxation on the CPU. A callback that
returns False must stop the run at that iteration; a callback that returns
None must let the run converge. Every IterationInfo must carry the iteration
number, the errors, the partition functions and the box size.
"""

import contextlib
import io
import sys

import numpy as np

from polymerfts import SCFT, IterationInfo, PlatformSelector


def cpu_platform():
    avail = PlatformSelector.avail_platforms()
    cpu = [p for p in avail if p.startswith("cpu")]
    return "cpu-mkl" if "cpu-mkl" in avail else (cpu[0] if cpu else None)


def lamella_params():
    return {
        "nx": [64], "lx": [4.0],
        "chain_model": "continuous", "ds": 1/50,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 20.0},
        "polymers": [{"volume_fraction": 1.0,
                               "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
        "platform": cpu_platform(),
        "relax_cell": True,
        "max_iter": 300, "tolerance": 1e-8,
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
    }


def initial_fields():
    x = (np.arange(64) + 0.5)/64
    return {"A": np.cos(2*np.pi*x), "B": -np.cos(2*np.pi*x)}


def main():
    if cpu_platform() is None:
        print("SKIP: no CPU platform in this build")
        sys.exit(77)  # ctest SKIP_RETURN_CODE
    failed = []

    # A callback returning False stops the run
    seen = []
    def stop_at_three(info):
        seen.append(info)
        return False if info.iteration == 3 else None

    with contextlib.redirect_stdout(io.StringIO()):
        calc = SCFT(lamella_params())
        calc.run(initial_fields=initial_fields(), on_iteration=stop_at_three)
    if [info.iteration for info in seen] != [1, 2, 3] or calc.iter != 3:
        failed.append(f"stop at 3: iterations {[info.iteration for info in seen]}, calc.iter {calc.iter}")
    for info in seen:
        if not (isinstance(info, IterationInfo) and np.isfinite(info.error_level) and np.isfinite(info.energy)
                and np.isfinite(info.mass_error) and len(info.partition_functions) == 1
                and info.lx is not None and len(info.lx) == 1 and info.angles is None):
            failed.append(f"incomplete IterationInfo: {info}")
            break

    # A callback returning None lets the run converge
    count = []
    with contextlib.redirect_stdout(io.StringIO()):
        calc = SCFT(lamella_params())
        calc.run(initial_fields=initial_fields(), on_iteration=lambda info: count.append(info.iteration))
    if calc.error_level >= 1e-8 or count != list(range(1, calc.iter + 1)):
        failed.append(f"callback without stop: error {calc.error_level}, {len(count)} calls for {calc.iter} iterations")

    for f in failed:
        print("FAILED:", f)
    if failed:
        sys.exit(1)
    print(f"SCFT callback: OK (converged in {calc.iter} iterations)")


if __name__ == "__main__":
    main()
