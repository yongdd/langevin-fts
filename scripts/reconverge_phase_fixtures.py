#!/usr/bin/env python3
"""Re-converge the space-group phase fixtures used by tests/TestSpaceGroupPhases.py.

The test starts every phase from tests/data/<PHASE>.mat and expects it to be
converged already (1-2 iterations). Whenever the discretization or the
symmetry convention changes, the fixtures must be re-converged under the new
convention and the expected free energies in the test updated. This script
does the first part and prints the converged values for the second.

For each phase it takes the parameters from the test's own PHASES table
(grid, box from the fixture, chi N, ds, space group, ...), runs SCFT with the
box fixed until the self-consistency error is below --tolerance, and
overwrites the fixture with the converged fields.

Usage:
    python scripts/reconverge_phase_fixtures.py [--platform cuda] [--out DIR] [PHASE ...]
    python scripts/reconverge_phase_fixtures.py --verify [PHASE ...]

--verify does not write anything: it reruns each fixture with the test's own
settings (max_iter 5, tolerance 2e-8) and reports how many iterations were
needed and the resulting free energy.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")

# Import the library before the test module: the test module forces
# OMP_NUM_THREADS=1 at import time, which must not override the setting above
# (OpenMP reads the environment when the library is loaded).
from polymerfts import scft  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import TestSpaceGroupPhases as T  # noqa: E402  (PHASES, build_params, load_mat_fields, DATA_DIR)


def run_phase(name, platform, tolerance, max_iter, write):
    phase = T.PHASES[name]
    mat_file = os.path.join(T.DATA_DIR, phase["mat_file"])
    lx, w_A, w_B = T.load_mat_fields(mat_file, phase["nx"])
    if lx is None:
        raise RuntimeError(f"{mat_file} not found or invalid")

    params = T.build_params(phase, lx, platform, False)
    params["max_iter"] = max_iter
    params["tolerance"] = tolerance

    t0 = time.perf_counter()
    calc = scft.SCFT(params=params)
    result = calc.run(initial_fields={"A": w_A.flatten(), "B": w_B.flatten()}, return_result=True)
    elapsed = time.perf_counter() - t0

    info = {
        "free_energy": float(calc.free_energy),
        "error_level": float(calc.error_level),
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
        "nx": [int(n) for n in phase["nx"]],
        "lx": [float(v) for v in lx],
        "platform": platform,
        "time_s": elapsed,
    }
    if write:
        if not result.converged:
            raise RuntimeError(f"{name}: did not converge (error {calc.error_level:.3e} after {result.iterations} iterations)")
        calc.save_results(mat_file)
        info["written"] = mat_file
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phases", nargs="*", help="phase names (default: all phases of the test)")
    ap.add_argument("--platform", default="cuda")
    ap.add_argument("--tolerance", type=float, default=1e-9, help="convergence tolerance for re-converging (default 1e-9)")
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--out", default=None, help="directory for one JSON result file per phase")
    ap.add_argument("--verify", action="store_true", help="rerun fixtures with the test settings instead of writing them")
    args = ap.parse_args()

    names = args.phases or list(T.PHASES)
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    results = {}
    for name in names:
        if args.verify:
            info = run_phase(name, args.platform, tolerance=2e-8, max_iter=5, write=False)
            diff = abs(info["free_energy"] - T.PHASES[name]["expected_fe"])
            print(f"{name:6s} iterations={info['iterations']} FE={info['free_energy']:.15f} "
                  f"diff_vs_expected={diff:.2e} error={info['error_level']:.2e} time={info['time_s']:.1f}s")
        else:
            info = run_phase(name, args.platform, tolerance=args.tolerance, max_iter=args.max_iter, write=True)
            print(f"{name:6s} converged in {info['iterations']} iterations: FE={info['free_energy']:.15f} "
                  f"error={info['error_level']:.2e} -> {info['written']} ({info['time_s']:.1f}s)")
        results[name] = info
        if args.out:
            with open(os.path.join(args.out, f"{name}.json"), "w") as f:
                json.dump({name: info}, f, indent=1)

    if not args.verify:
        print("\nexpected_fe values for tests/TestSpaceGroupPhases.py:")
        for name, info in results.items():
            print(f'    "{name}": {info["free_energy"]:.15f},')
    return 0


if __name__ == "__main__":
    sys.exit(main())
