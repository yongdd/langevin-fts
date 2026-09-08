"""Summarize the 1D positive sampler and compare PRL Fig. 1 plateau values.

PRL values are digitized from the published raster, not authors' raw data.
Only five unambiguous interior positions are used for numerical comparison.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def digitize(root):
    path = root / "paper_fig1.jpeg"
    rgb = np.asarray(Image.open(path)).astype(float)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    masks = dict(red=(r > 170) & (r - g > 70) & (r - b > 70),
                 blue=(b > 150) & (b - r > 70) & (b - g > 70),
                 green=(g > 65) & (g - r > 35) & (g - b > 25),
                 black=np.max(rgb, axis=2) < 90)
    # Axis centers, inspected on the embedded 1612x958 image: z/b=0..60, phi=0..1.
    x0, x1, top, bottom = 213, 1579, 24, 789
    rows = []
    for color, zplus in [("black", 0), ("red", 1), ("blue", 2), ("green", 3)]:
        for z in [2, 5, 8, 10, 12]:
            x = round(x0 + (x1 - x0) * z / 60)
            ys = np.flatnonzero(masks[color][top + 4:bottom - 4, x - 4:x + 5].any(axis=1)) + top + 4
            clusters = np.split(ys, np.flatnonzero(np.diff(ys) > 2) + 1)
            values = [(bottom - group.mean()) / (bottom - top) for group in clusters if len(group)]
            assert len(values) == (1 if zplus == 0 else 2), (color, z, values)
            # In this inspected interval, solid curves are above the same-color dashed curves.
            for closure, value in zip(["mf"] if zplus == 0 else ["corr", "mf"], values):
                rows.append(dict(zplus=zplus, paper_curve=closure, z_over_b=z, phi=float(value),
                                 digitization_uncertainty_phi=.004))
    metadata = dict(source="Duan et al., PRL 134, 048101 (2025), Fig. 1, embedded image xref 53",
        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), image_size=[rgb.shape[1], rgb.shape[0]],
        axes=dict(x0=x0, x60=x1, phi1=top, phi0=bottom),
        method="Color masks, +/-4 pixel column window, separated row clusters; inspected solid/dashed ordering at z/b=2,5,8,10,12",
        limitation="Raster estimates of five interior points, not a full-profile digitization or original numerical data",
        points=rows)
    (root / "paper_fig1_digitized.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    root = args.root
    points = digitize(root)
    positive = [dict(run=p.parent.name, **json.loads(p.read_text())) for p in sorted(root.glob("positive_*/summary.json"))]
    comparisons = []
    for s in positive:
        if s["mode"] != "joint":
            continue
        ref = next(t for t in positive if t["mode"] == "particles" and
                   t["sites"] == s["sites"] and t["coupling"] == s["coupling"])
        delta = np.array(s["statistics"]["mean"]) - ref["statistics"]["mean"]
        se = np.hypot(s["statistics"]["se"], ref["statistics"]["se"])
        z = np.divide(abs(delta), se, out=np.zeros_like(delta), where=se > 0)
        comparisons.append(dict(run=s["run"], reference=ref["run"],
            max_standardized_difference=float(z.max()), height_difference=float(delta[0]),
            height_combined_se=float(se[0]), n_correlated_observables=len(delta)))
    prl = []
    for p in sorted(root.glob("prl_*/summary.json")):
        s = json.loads(p.read_text())
        row = dict(run=p.parent.name, **s)
        if s["status"] == "completed":
            d = np.load(p.parent / "profiles.npz")
            curve = "mf" if s["params"]["closure"] == "mf" else "corr"
            selected = [t for t in points if t["zplus"] == s["params"]["zplus"] and t["paper_curve"] == curve]
            samples = []
            for t in selected:
                value = float(np.interp(t["z_over_b"], d["z"], d["phiP"]))
                samples.append(dict(**t, computed_phi=value, difference=value - t["phi"]))
            row.update(paper_comparison=samples,
                max_interior_phi_difference=max(abs(t["difference"]) for t in samples),
                interior_phi_rmse=float(np.sqrt(np.mean([t["difference"]**2 for t in samples]))))
        prl.append(row)
    result = dict(positive_runs=positive, sampler_comparisons=comparisons, prl_runs=prl)
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "prl_comparison.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "z_over_b", "paper_phi", "computed_phi", "difference", "raster_uncertainty"])
        for s in prl:
            for p in s.get("paper_comparison", []):
                writer.writerow([s["run"], p["z_over_b"], p["phi"], p["computed_phi"], p["difference"], .004])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {0: "#333333", 1: "#de493b", 2: "#376bc2", 3: "#31865a"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained", sharey=True)
    for ax, closure, title in zip(axes, ["mf", "lda"], ["Mean field", "Existing smoothed LDA closure"]):
        for zplus in [0, 1, 2, 3]:
            if zplus == 0:
                name = "prl_z0_mf_M3200"
            else:
                name = f"prl_z{zplus}_{closure}_M3200"
            paths = root / name / "profiles.npz"
            if not paths.exists():
                name = name.replace("M3200", "M1600")
                if zplus in [2, 3] and closure == "lda":
                    name = name.replace("lda_M1600", "lda_newton_M1600")
                paths = root / name / "profiles.npz"
            d = np.load(paths)
            ax.plot(d["z"], d["phiP"], color=colors[zplus], label="Neutral" if zplus == 0 else f"z+={zplus}")
            paper = [p for p in points if p["zplus"] == zplus and p["paper_curve"] == ("mf" if zplus == 0 or closure == "mf" else "corr")]
            ax.errorbar([p["z_over_b"] for p in paper], [p["phi"] for p in paper], yerr=.004,
                        color=colors[zplus], fmt="o", ms=4, mfc="white", capsize=2)
        ax.set(xlim=(0, 60), ylim=(0, .85), xlabel="z / b", title=title)
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Polymer volume fraction")
    axes[0].legend(fontsize=9)
    fig.suptitle("PRL Fig. 1 parameters: recomputed lines and digitized paper markers\n"
                 "Canonical counterions, L=200 nm, Gaussian graft width=0.5 b; no parameter fitting", fontsize=11)
    for suffix in ["png", "pdf"]:
        fig.savefig(root / f"prl_fig1_checked.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1), layout="constrained", sharey=True)
    for ax, valence in zip(axes, [1, 2, 3]):
        for name, label, color in [
            (f"prl_z{valence}_lda_M3200", "LDA, canonical C", "#376bc2"),
            (f"prl_z{valence}_nonlocal" + ("_continuation" if valence == 3 else "") + "_M1600",
             "Existing nonlocal, canonical C", "#b55643"),
            (f"prl_z{valence}_lda_gc_M1600", "LDA, GC salt only", "#31865a")]:
            path = root / name / "profiles.npz"
            if path.exists():
                d = np.load(path)
                ax.plot(d["z"], d["phiP"], color=color, label=label)
        selected = [p for p in points if p["zplus"] == valence and p["paper_curve"] == "corr"]
        ax.errorbar([p["z_over_b"] for p in selected], [p["phi"] for p in selected],
                    yerr=.004, color="black", fmt="o", ms=4, mfc="white", label="Paper raster")
        ax.set(xlim=(0, 50), ylim=(0, .85), xlabel="z / b", title=f"z+={valence}")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Polymer volume fraction")
    axes[0].legend(fontsize=8)
    fig.suptitle("Existing correlation closures and ensemble controls\n"
                 "L=200 nm; nonlocal backend retains documented SI/boundary approximations", fontsize=11)
    for suffix in ["png", "pdf"]:
        fig.savefig(root / f"prl_closure_comparison.{suffix}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for s in positive:
        if s["coupling"] not in [0, 100]:
            continue
        offset = {1101: -.2, 1201: 0., 2101: .2}[s["seed"]]
        color = "#b55643" if s["coupling"] else "#3269a7"
        x = [4, 16, 32].index(s["sites"]) + offset
        ax.errorbar(x, s["statistics"]["mean"][0], yerr=s["statistics"]["se"][0],
                    fmt="o" if s["mode"] == "joint" else "s", color=color, capsize=3)
    for E in [0, 100]:
        exact = next(s["exact"][0] for s in positive if s["sites"] == 4 and s["coupling"] == E)
        ax.hlines(exact, -.35, .35, color="#333333", linestyle="--")
    ax.set(xticks=[0, 1, 2], xticklabels=[4, 16, 32], xlabel="Number of cells in 1D", ylabel="Mean polymer height / L")
    ax.set_title("Same finite 9-bead action; positive sampling without enumeration\n"
                 "Circles: real-field/particle Gibbs; squares: particle Gibbs\n"
                 "Blue E=0; red E=100; dashes: exact four-cell values", fontsize=10)
    ax.grid(alpha=.15)
    for suffix in ["png", "pdf"]:
        fig.savefig(root / f"positive_1d_comparison.{suffix}", dpi=180)
    plt.close(fig)
    print(json.dumps(dict(positive_runs=len(positive), prl_completed=sum(s["status"] == "completed" for s in prl),
        max_exact_z=max(s.get("max_standardized_discrepancy", 0) for s in positive),
        max_joint_particle_z=max(s["max_standardized_difference"] for s in comparisons)), indent=2))


if __name__ == "__main__":
    main()
