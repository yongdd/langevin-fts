#!/usr/bin/env python3
"""Refresh the web editor's offline symmetry catalog (development requires spglib).

The editor itself uses only the standard library. This catalog records the Hall
settings accepted by SCFT, with the cell/grid rules used by SpaceGroup.cpp.
"""

from fractions import Fraction
import json
import math
from pathlib import Path

import spglib


def build_catalog():
    settings = []
    for hall in range(1, 531):
        group = spglib.get_spacegroup_type(hall)
        number = group.number
        # SCFT currently accepts these five systems when a space group is set.
        if number < 16:
            continue
        system = ("Orthorhombic" if number <= 74 else "Tetragonal" if number <= 142
                  else "Trigonal" if number <= 167 else "Hexagonal" if number <= 194 else "Cubic")
        operations = spglib.get_symmetry_from_database(hall)
        rotations, translations = operations["rotations"], operations["translations"]
        hex_axes = system in ("Hexagonal", "Trigonal") and all(
            r[0, 2] == r[1, 2] == r[2, 0] == r[2, 1] == 0 and abs(r[2, 2]) == 1 for r in rotations)
        if not hex_axes:
            assert all(sum(row) % 2 for r in rotations for row in r), hall
        divisors = [1, 1, 1]
        for t in translations:
            for axis in range(3):
                fraction = Fraction(float(t[axis])).limit_denominator(24)
                assert abs(float(fraction) - t[axis]) < 1e-9
                divisors[axis] = math.lcm(divisors[axis], fraction.denominator)
        equal = {(min(i, j), max(i, j)) for r in rotations for i in range(3)
                 for j in range(3) if i != j and r[i, j] != 0}
        if system in ("Tetragonal", "Trigonal", "Hexagonal"):
            equal.add((0, 1))
        if system == "Cubic":
            equal.update(((0, 1), (0, 2), (1, 2)))
        settings.append(dict(hall=hall, symbol=group.international_short, setting=group.choice,
                             crystal_system=system, hexagonal_axes=hex_axes,
                             grid_divisors=divisors, equal_grid_axes=sorted(equal)))
    return {"spglib_version": spglib.__version__, "settings": settings}


if __name__ == "__main__":
    catalog = build_catalog()
    path = Path(__file__).with_name("simulation_gui") / "space-groups.json"
    lines = ["{", '  "spglib_version": ' + json.dumps(catalog["spglib_version"]) + ",", '  "settings": [']
    lines.extend("    " + json.dumps(s, separators=(",", ":")) + ("," if i + 1 < len(catalog["settings"]) else "")
                 for i, s in enumerate(catalog["settings"]))
    path.write_text("\n".join(lines + ["  ]", "}", ""]), encoding="utf-8")
    print(f"Wrote {len(catalog['settings'])} Hall settings to {path}")
