"""Compare saved finite-lattice CL densities with conditional enumeration.

Checks the final saved fields, not every state or the largest sampled drift.
Uses the independently constructed matrices of validate_clfts_exact.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from validate_clfts_exact import C, CHARGES, TYPES, enumerate_particles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    rows = []
    references = {}
    for path in sorted(args.root.glob("cl_*/summary.json")):
        result = json.loads(path.read_text())
        if result["status"] != "completed":
            continue
        p = result["params"]
        key = (p["nx"][0], p["bjerrum_e"])
        if key not in references:
            references[key] = enumerate_particles(*key)
        exact = references[key]
        checkpoint = path.parent / f"fields_{p['langevin']['max_step']:06d}.mat"
        data = loadmat(checkpoint, simplify_cells=True)
        w = np.array([data[f"w_{t}_real"]+1j*data[f"w_{t}_imag"] for t in TYPES])
        psi = data["psi_real"]+1j*data["psi_imag"]
        fields = (w+CHARGES[:, None]*psi) @ exact["smear"].T
        # Uniform species shifts cancel in canonical normalized densities.
        exponent = exact["log_bond"]-C*np.einsum("cts,ts->c", exact["phi"], fields)/key[0]
        weight = np.exp(exponent-exponent.real.max())
        conditional = np.einsum("c,cts->ts", weight, exact["phi"])/weight.sum()
        saved = np.array([data[f"phi_{t}_real"]+1j*data[f"phi_{t}_imag"] for t in TYPES])
        error = float(np.max(abs(saved-conditional)))
        rows.append(dict(run=path.parent.name, checkpoint=str(checkpoint),
                         max_conditional_density_error=error,
                         passed=bool(np.isfinite(error) and error < 1e-11)))
    report = dict(checkpoints=len(rows), results=rows,
                  all_passed=bool(rows and all(r["passed"] for r in rows)))
    (args.root/"checkpoint_checks.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
