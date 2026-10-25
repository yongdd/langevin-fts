#!/usr/bin/env python3
"""Every example builds its simulation with the current code and input checks.

Each script under examples/ runs in its own process, in a temporary copy of its
directory (it finds its input files there and writes nothing into the source
tree). The SCFT, LFTS and CLFTS classes are wrapped: a platform the build lacks
is replaced by a CPU platform, and the first run() or continue_run() stops the
script, so the example is checked up to the start of its simulation (inputs,
parameter validation, construction), not run.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DRIVER = r"""
import os, runpy, sys
import polymerfts
from polymerfts import PlatformSelector

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
            os.write(1, ("EXAMPLE CONSTRUCTED " + _name + "\n").encode())
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
        copies = {}
        for script in scripts:
            directory = script.parent
            if directory not in copies:
                copies[directory] = Path(tmp) / str(len(copies))
                shutil.copytree(directory, copies[directory], ignore=shutil.ignore_patterns("__pycache__", "*.png"))
            copy = copies[directory] / script.name
            child = subprocess.run([sys.executable, "-c", DRIVER, str(copy)], cwd=copies[directory],
                                   capture_output=True, text=True, timeout=600, env=dict(os.environ))
            name = script.relative_to(ROOT).as_posix()
            ok = child.returncode == 0 and "EXAMPLE CONSTRUCTED" in child.stdout
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
            if not ok:
                tail = (child.stdout + child.stderr).strip().splitlines()[-6:]
                print("    " + "\n    ".join(tail))
                failures.append(name)
    print(f"\n{len(scripts) - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
