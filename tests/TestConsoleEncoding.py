#!/usr/bin/env python3
"""
Console output survives a stream that cannot encode its Greek letters.

polymerfts prints quantities as χN, κ or βF/(ρ₀V/N). A stream redirected to a
file or pipe uses the locale encoding (cp949, cp1252, ... on Windows), which
cannot encode all of them, and print() raised UnicodeEncodeError: at the end of
every SCFT run under cp949, and at the start of every SCFT, L-FTS and CL-FTS run
under code pages without Greek letters. The package writes such characters as
backslash escapes. This test runs a short SCFT in a child process whose stdout
is ASCII, with the strict error handler and with surrogateescape (the default
of a redirected stdout with Python 3.14 on Windows), and checks that the run
completes with the escapes in its output.
"""

import os
import subprocess
import sys
import tempfile

CHILD = r'''
import numpy as np
from polymerfts import scft, _core
params = {
    "platform": _core.PlatformSelector.avail_platforms()[0],
    "nx": [8, 8, 8],
    "lx": [2.0, 2.0, 2.0],
    "relax_cell": False,
    "chain_model": "continuous",
    "ds": 1 / 20,
    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 15.0},
    "polymers": [{"volume_fraction": 1.0,
                  "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
    "optimizer": {"name": "am", "max_hist": 10, "start_error": 1e-2,
                  "mix_min": 0.1, "mix_init": 0.1},
    "max_iter": 3,
    "tolerance": 1e-8,
}
calc = scft.SCFT(params=params)
rng = np.random.default_rng(0)
calc.run(initial_fields={m: rng.normal(0.0, 1.0, 512) for m in "AB"})
'''


def main():
    failures = []
    for io_encoding in ("ascii", "ascii:surrogateescape"):
        env = dict(os.environ, PYTHONIOENCODING=io_encoding)
        env.pop("PYTHONUTF8", None)
        with tempfile.TemporaryDirectory() as tmp:   # SCFT writes polymer_<id>.png here
            result = subprocess.run([sys.executable, "-c", CHILD], cwd=tmp, env=env,
                                    capture_output=True)
        out = result.stdout.decode("ascii", errors="replace")
        if result.returncode != 0:
            failures.append(f"{io_encoding}: the child exited with {result.returncode}:\n"
                            + result.stderr.decode("ascii", errors="replace"))
        for escape, where in (("\\u03c7N", "chi N at the start of the run"),
                              ("\\u2080", "the free energy at its end")):
            if escape not in out:
                failures.append(f"{io_encoding}: {escape} ({where}) is missing from the output")
    for f in failures:
        print("FAIL:", f)
    if failures:
        return 1
    print("SCFT output on an ASCII stdout (strict and surrogateescape): PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
