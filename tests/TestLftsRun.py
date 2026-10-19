#!/usr/bin/env python3
"""Short L-FTS and CL-FTS runs on the CPU (small 2D grid).

Checks that the run loops work for every compressor and verbose level, that the
recorded files are written, that a run continued from a checkpoint reproduces
the uninterrupted run exactly, and that invalid options are rejected.
"""

import os
import sys
import copy
import io
import contextlib
import atexit
import shutil
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import LFTS, CLFTS, ValidationError, PlatformSelector


def cpu_platform():
    avail = PlatformSelector.avail_platforms()
    cpu = [p for p in avail if p.startswith("cpu")]
    return cpu[0] if cpu else None


def params_for(work_dir, max_step, compressor="am", verbose_level=1):
    return {
        "nx": [16, 16],
        "lx": [3.0, 3.0],
        "chain_model": "continuous",
        "ds": 1/20,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15.0},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}],
        }],
        "platform": cpu_platform(),
        "langevin": {"nbar": 10000, "dt": 8.0, "max_step": max_step},
        "recording": {"dir": work_dir, "recording_period": 2, "sf_computing_period": 1, "sf_recording_period": 2},
        "saddle": {"max_iter": 200, "tolerance": 1e-4},
        "compressor": {"name": compressor, "max_hist": 20, "start_error": 5e-1, "mix_min": 0.01, "mix_init": 0.01},
        "verbose_level": verbose_level,
    }


def initial_fields(seed=3):
    rng = np.random.default_rng(seed)
    return {"A": rng.normal(0.0, 0.5, 256), "B": rng.normal(0.0, 0.5, 256)}


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def main():
    if cpu_platform() is None:
        print("SKIP: no CPU platform in this build")
        return 0
    ok = True
    root = tempfile.mkdtemp(prefix="polymerfts_lfts_")
    atexit.register(shutil.rmtree, root, ignore_errors=True)

    # ---- every compressor and verbose level completes and records its files
    for compressor in ("am", "lr", "lram"):
        for verbose_level in (0, 1, 2):
            work_dir = os.path.join(root, f"{compressor}_{verbose_level}")
            sim = quiet(LFTS, params_for(work_dir, 4, compressor, verbose_level), random_seed=5)
            quiet(sim.run, initial_fields=initial_fields())
            for name in ("fields_000002.mat", "fields_000004.mat", "dH_000002.mat",
                         "dH_000004.mat", "structure_function_000002.mat"):
                if not os.path.exists(os.path.join(work_dir, name)):
                    print(f"FAILED: {compressor} verbose {verbose_level}: {name} not written")
                    ok = False
            data = loadmat(os.path.join(work_dir, "dH_000004.mat"), squeeze_me=True)
            if not np.all(np.isfinite(data["H_history"])):
                print(f"FAILED: {compressor} verbose {verbose_level}: Hamiltonian not finite")
                ok = False

    # ---- continuing from a checkpoint reproduces the uninterrupted run
    dir_full = os.path.join(root, "full")
    sim = quiet(LFTS, params_for(dir_full, 4), random_seed=11)
    quiet(sim.run, initial_fields=initial_fields())
    dir_split = os.path.join(root, "split")
    sim = quiet(LFTS, params_for(dir_split, 2), random_seed=11)
    quiet(sim.run, initial_fields=initial_fields())
    sim = quiet(LFTS, params_for(dir_split, 4), random_seed=99)   # the seed is restored from the file
    quiet(sim.continue_run, os.path.join(dir_split, "fields_000002.mat"))
    w_full = loadmat(os.path.join(dir_full, "fields_000004.mat"), squeeze_me=True)["w_A"]
    w_split = loadmat(os.path.join(dir_split, "fields_000004.mat"), squeeze_me=True)["w_A"]
    # Equal to round-off: the continued run recomputes the concentrations of the loaded
    # (pressure-gauge-shifted) fields, which differs from the uninterrupted run at ~1e-11
    # (the saddle-point tolerance itself is 1e-4). A wrong noise or RNG state gives O(1).
    if not np.allclose(w_full, w_split, rtol=0.0, atol=1e-8):
        print(f"FAILED: continued run differs from the full run, max |dw| = {np.max(np.abs(w_full - w_split)):.3e}")
        ok = False

    # ---- CL-FTS run (complex fields, compressible)
    work_dir = os.path.join(root, "clfts")
    params = params_for(work_dir, 4)
    del params["compressor"], params["saddle"]
    params["dynamic_stabilization"] = 0.05
    sim = quiet(CLFTS, params, random_seed=5)
    quiet(sim.run, initial_fields=initial_fields())
    data = loadmat(os.path.join(work_dir, "dH_000004.mat"), squeeze_me=True)
    if not (os.path.exists(os.path.join(work_dir, "fields_000004.mat"))
            and np.all(np.isfinite(data["H_history_real"]))):
        print("FAILED: CL-FTS run")
        ok = False

    # ---- invalid options
    for bad, what in ((("compressor", "name", "foo"), "unknown compressor"),
                      (("verbose_level", None, 3), "verbose_level 3")):
        params = params_for(os.path.join(root, "bad"), 4)
        key, sub, value = bad
        if sub is None:
            params[key] = value
        else:
            params[key][sub] = value
        try:
            quiet(LFTS, params, random_seed=1)
            print(f"FAILED: {what} accepted")
            ok = False
        except ValidationError:
            pass
    params = params_for(os.path.join(root, "bad"), 4)
    params["wtmd"] = {"ell": 4}
    try:
        quiet(LFTS, params, random_seed=1)
        print("FAILED: WTMD accepted on a 2D grid")
        ok = False
    except ValidationError:
        pass

    if ok:
        print("Test passed, L-FTS and CL-FTS runs")
        return 0
    print("Test failed, L-FTS and CL-FTS runs")
    return 1


if __name__ == "__main__":
    sys.exit(main())
