"""Compare unchanged CL trajectories with the exact positive marginal sampler."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--cl-root", type=Path, required=True)
    args = ap.parse_args()
    positive = []
    for p in sorted(args.root.glob("positive_*/summary.json")):
        s = json.loads(p.read_text())
        assert s["status"] == "completed", p
        assert s["completed_steps"] == round(s["duration"] / s["dt"]), p
        assert s["n_samples"] == 180000, p
        positive.append(dict(run=p.parent.name, coupling=s["coupling"], dt=s["dt"],
            seed=s["seed"], labels=s["labels"], exact=s["exact"], real=s["statistics"],
            max_standardized_discrepancy=max(s["standardized_discrepancy"]),
            checks=s["checks"], max_species_mass_error=s["max_species_mass_error"]))
    identities = [json.loads(p.read_text()) for p in sorted(args.root.glob("identities_*/summary.json"))]
    assert len(positive) == 12 and len(identities) == 6
    cl = json.loads((args.cl_root / "analysis.json").read_text())
    cl_by_name = {s["run"]: s for s in cl}
    diagnostics = []
    for s in identities:
        zscores = []
        for key in ["uncut_real", "uncut_imag"]:
            mean, se = np.array(s[key]["mean"]), np.array(s[key]["se"])
            zscores.extend(np.divide(abs(mean), se, out=np.zeros_like(se), where=se > 0))
        cut = next(row for row in s["pole_cutoffs"] if row["epsilon"] == .1)
        np.testing.assert_allclose(np.array(cut["bias_inside"]["mean"]) + cut["bias_outside"]["mean"],
                                   s["total_bias"]["mean"], atol=1e-13, rtol=1e-12)
        diagnostics.append(dict(run=s["run"], coupling=s["coupling"],
            max_generator_standardized_discrepancy=float(max(zscores)),
            total_bias=s["total_bias"], pole_cutoff_point1=cut))
    result = dict(positive=positive, identities=diagnostics,
        max_positive_standardized_discrepancy=max(s["max_standardized_discrepancy"] for s in positive),
        max_generator_standardized_discrepancy=max(s["max_generator_standardized_discrepancy"] for s in diagnostics),
        note="Fixed finite-model diagnostic. Cutoff contributions use the full sample count; no corrected estimator.")
    (args.root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.root / "positive_comparison.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "E", "dt", "seed", "observable", "exact", "mean", "se"])
        for s in positive:
            for label, exact, mean, se in zip(s["labels"], s["exact"], s["real"]["mean"], s["real"]["se"]):
                writer.writerow([s["run"], s["coupling"], s["dt"], s["seed"], label, exact, mean, se])

    selected = [(cl_by_name[name], label, "#b95340") for name, label in [
        ("cl_E100_dt0.00015625_lm-etd_s501_mob1.0", "CL LM/ETD, seed 501"),
        ("cl_E100_dt0.00015625_heun_s501_mob1.0", "CL Heun, seed 501"),
        ("cl_E100_dt0.00015625_heun_s601_mob1.0", "CL Heun, seed 601"),
        ("cl_E100_dt0.00015625_adaptive-euler_s701_mob1.0", "CL adaptive Euler, seed 701")]]
    selected += [(s, f"Real marginal, seed {s['seed']}", "#21816c") for s in positive
                 if s["coupling"] == 100 and s["dt"] == .00125]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained", sharey=True)
    for ax, label, title in zip(axes, ["phi_C_cell0", "field_2"],
                               ["Counterion density in cell 0", "Auxiliary field amplitude (native mode 4)"]):
        for y, (s, _, color) in enumerate(selected):
            i = s["labels"].index(label)
            ax.errorbar(s["real"]["mean"][i], y, xerr=s["real"]["se"][i],
                        fmt="o", color=color, capsize=4)
        i = selected[0][0]["labels"].index(label)
        ax.axvline(selected[0][0]["exact"][i], color="#333333", linestyle="--", label="Exact particle sum")
        ax.set_title(title)
        ax.set_xlabel("Mean ± 1 SE")
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(np.arange(len(selected)), [s[1] for s in selected])
    axes[0].invert_yaxis()
    axes[1].legend(loc="center right", fontsize=9)
    fig.suptitle("Same two-cell action, E=100; different sampling representation\n"
                 "CL dt ≤ 0.00015625; real marginal dt=0.00125; T=20,000, discard 2,000", fontsize=11)
    for suffix in ["png", "pdf"]:
        fig.savefig(args.root / f"sampling_comparison.{suffix}", dpi=180)
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k not in ["positive", "identities"]}, indent=2))


if __name__ == "__main__":
    main()
