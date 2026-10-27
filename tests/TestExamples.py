#!/usr/bin/env python3
"""Every example builds its simulation with the current code and input checks.

Each script under examples/ runs in its own process, in its directory of a
temporary copy of the examples tree (it finds its input files there, also in a
neighbouring directory, and writes nothing into the source tree). The SCFT, LFTS
and CLFTS classes are wrapped: a platform the build lacks
is replaced by a CPU platform, and the first run() or continue_run() stops the
script, so the example is checked up to the start of its simulation (inputs,
parameter validation, construction), not run. The examples of tools that build no
simulation (WITHOUT_SIMULATION) run to the end instead and check their own results.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cuda_part import cuda_part_missing

ROOT = Path(__file__).resolve().parents[1]

# Examples that build no SCFT/LFTS/CLFTS simulation, and modules of examples; they
# must finish without an error
WITHOUT_SIMULATION = {"examples/fourier/FourierTransform.py",
                      "examples/scft/phase_diagram/phase_diagram.py",
                      "examples/scft/phase_diagram/seeds.py"}

DRIVER = r"""
import os, runpy, sys
import polymerfts
from polymerfts import PlatformSelector

# The process output, kept for the message below: an example may redirect file
# descriptor 1 around run(), e.g. to silence the native library
output = os.dup(1)

platforms = PlatformSelector.avail_platforms()
fallback = next((p for p in platforms if p.startswith("cpu")), platforms[0])


def wrap(cls):
    init = cls.__init__

    def __init__(self, params, *args, **kwargs):
        if params.get("platform") is not None and params["platform"] not in platforms:
            params = dict(params, platform=fallback)
        init(self, params, *args, **kwargs)
    cls.__init__ = __init__
    for name in ("run", "continue_run"):
        def stop(self, *args, _name=cls.__name__ + "." + name, **kwargs):
            # Straight to the process output: an example may redirect sys.stdout around run()
            os.write(output, ("EXAMPLE CONSTRUCTED " + _name + "\n").encode())
            os._exit(0)
        setattr(cls, name, stop)


for cls in (polymerfts.SCFT, polymerfts.LFTS, polymerfts.CLFTS):
    wrap(cls)
sys.argv = [sys.argv[1]]
runpy.run_path(sys.argv[0], run_name="__main__")
print("EXAMPLE FINISHED WITHOUT A SIMULATION", flush=True)
"""


def main():
    scripts = sorted(p for p in (ROOT / "examples").rglob("*.py") if "__pycache__" not in p.parts)
    if not scripts:
        print("No examples found")
        return 1
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "examples"
        shutil.copytree(ROOT / "examples", tree, ignore=shutil.ignore_patterns("__pycache__", "*.png"))
        for script in scripts:
            copy = tree / script.relative_to(ROOT / "examples")
            child = subprocess.run([sys.executable, "-c", DRIVER, str(copy)], cwd=copy.parent,
                                   capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   timeout=600, env=dict(os.environ))
            name = script.relative_to(ROOT).as_posix()
            finished = "EXAMPLE FINISHED WITHOUT A SIMULATION" in child.stdout and name in WITHOUT_SIMULATION
            ok = child.returncode == 0 and ("EXAMPLE CONSTRUCTED" in child.stdout or finished)
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
            if not ok:
                tail = (child.stdout + child.stderr).strip().splitlines()[-6:]
                print("    " + "\n    ".join(tail))
                failures.append(name)
    print(f"\n{len(scripts) - len(failures)} passed, {len(failures)} failed")
    return 1 if cuda_part_missing() or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
