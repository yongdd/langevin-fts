#!/usr/bin/env python3
"""Write a small continuous-chain initial-field file for PSCF conversion.

No simulation or compiled polymerfts package is needed.
    python scripts/pscf/pscf_example.py -o initial.mat
    python scripts/pscf/pscf_converter.py export initial.mat -o pscf-input
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("initial.mat"))
    args = parser.parse_args()
    params = {
        "nx": [16, 12, 8], "lx": [4., 3., 2.], "angles": [90., 90., 90.],
        "chain_model": "continuous", "ds": 1/100, "relax_cell": False,
        "crystal_system": "Orthorhombic", "segment_lengths": {"A": 1., "B": 1.},
        "chi_n": {"A,B": 20.},
        "polymers": [{"volume_fraction": 1., "blocks": [
            {"type": "A", "length": .5}, {"type": "B", "length": .5}]}],
        "optimizer": {"name": "am", "max_hist": 20, "start_error": .01, "mix_min": .1, "mix_init": .1},
        "max_iter": 5000, "tolerance": 1e-8,
    }
    x = (np.arange(params["nx"][0]) + .5) / params["nx"][0]
    w = np.broadcast_to(np.cos(2 * np.pi * x)[:, None, None], params["nx"]).copy()
    data = {"initial_params": params, "nx": params["nx"], "lx": params["lx"], "angles": params["angles"],
            "w_A": w.ravel(), "w_B": -w.ravel(), "monomer_types": np.asarray(["A", "B"], dtype=object)}
    try:
        with args.output.open("xb") as handle:
            savemat(handle, data, long_field_names=True, do_compression=True)
    except OSError as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Written: {args.output} (lamellar initial fields; no simulation was run)")


if __name__ == "__main__":
    main()
