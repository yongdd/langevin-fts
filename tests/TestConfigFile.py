#!/usr/bin/env python3
"""Parameter files: load_config, ConfigError and SCFT/LFTS/CLFTS.from_file.

Writes small YAML and JSON files and checks that numpy expressions are
evaluated, that nothing outside the allowed expressions is executed, that bad
files raise ConfigError, and that the three simulation classes build from a
file with keyword overrides applied.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import warnings

import numpy as np

from polymerfts import SCFT, LFTS, CLFTS, PlatformSelector, load_config, ConfigError


def cpu_platform():
    avail = PlatformSelector.avail_platforms()
    cpu = [p for p in avail if p.startswith("cpu")]
    return "cpu-mkl" if "cpu-mkl" in avail else (cpu[0] if cpu else None)


YAML = """
nx: [8, 8, 8]
lx: ["4.0*np.sqrt(3)/2", "np.pi", 4.0]
chain_model: continuous
ds: 0.1
segment_lengths: {A: 1.0, B: 1.0}
chi_n: {"A,B": 15.0}
distinct_polymers:
  - volume_fraction: 1.0
    blocks: [{type: A, length: 0.5}, {type: B, length: 0.5}]
platform: PLATFORM
box_is_altering: false
optimizer: {name: am, max_hist: 20, start_error: 0.01, mix_min: 0.1, mix_init: 0.1}
langevin: {nbar: 10000, dt: 8.0, max_step: 10}
compressor: {name: am, max_hist: 20, start_error: 0.1, mix_min: 0.1, mix_init: 0.1}
saddle: {max_iter: 100, tolerance: 1.0e-4}
recording: {dir: data, recording_period: 10, sf_computing_period: 1, sf_recording_period: 10}
verbose_level: 1
note: "np.sqrt(4) + np.save('INJECTED', 0)"
complex: "(-8)**0.5*np.pi"
"""


def main():
    if cpu_platform() is None:
        print("SKIP: no CPU platform in this build")
        sys.exit(77)  # ctest SKIP_RETURN_CODE
    failed = []
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            with open("p.yaml", "w") as f:
                f.write(YAML.replace("PLATFORM", cpu_platform()))

            # Numpy expressions are evaluated; anything else is kept as text and not executed
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                p = load_config("p.yaml")
            if not np.allclose(p["lx"], [2.0*np.sqrt(3.0), np.pi, 4.0], rtol=0, atol=1e-15):
                failed.append(f"lx expressions evaluated to {p['lx']}")
            # (the former eval with numpy in scope ran np.save and wrote INJECTED.npy)
            if os.path.exists("INJECTED.npy") or not isinstance(p["note"], str) or len(caught) < 2:
                failed.append("an expression outside the allowed set was executed or not reported")
            if not isinstance(p["complex"], str):
                failed.append(f"a complex value was accepted: {p['complex']!r}")

            # JSON gives the same dictionary
            with open("p.json", "w") as f:
                json.dump({k: v for k, v in p.items() if k not in ("note", "complex")}, f)
            q = load_config("p.json")
            if q["chi_n"] != p["chi_n"] or q["lx"] != list(p["lx"]):
                failed.append("JSON and YAML differ")

            # Bad files
            for name, text in (("p.txt", "nx: [8]"), ("empty.yaml", ""), ("list.yaml", "- 1\n- 2\n")):
                with open(name, "w") as f:
                    f.write(text)
                try:
                    load_config(name)
                    failed.append(f"{name} accepted")
                except ConfigError:
                    pass
            try:
                load_config("missing.yaml")
                failed.append("missing file accepted")
            except FileNotFoundError:
                pass

            # The simulation classes build from the file, with overrides applied
            # (the warnings were checked above)
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scft = SCFT.from_file("p.yaml", max_iter=7)
                lfts = LFTS.from_file("p.yaml", random_seed=1, ds=0.05)
                clfts = CLFTS.from_file("p.yaml", random_seed=1)
            if scft.max_iter != 7:
                failed.append("SCFT.from_file override")
            if abs(lfts.ds - 0.05) > 1e-15:
                failed.append("LFTS.from_file override")
            if not np.allclose(clfts.prop_solver.get_lx(), p["lx"]):
                failed.append("CLFTS.from_file box")
        finally:
            os.chdir(cwd)

    for f in failed:
        print("FAILED:", f)
    if failed:
        sys.exit(1)
    print("Parameter files: OK")


if __name__ == "__main__":
    main()
