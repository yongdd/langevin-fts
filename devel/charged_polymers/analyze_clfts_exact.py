"""Autocorrelation/block diagnostics for the exact finite-lattice CL benchmark."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_validation import correlation_time


def statistics(values, interval):
    values = np.asarray(values)
    means = values.mean(axis=0)
    block_errors = {}
    for count in [10, 20, 40]:
        if len(values) < count*20:
            continue
        blocks = np.array([v.mean(axis=0) for v in np.array_split(values, count)])
        block_errors[str(count)] = (blocks.std(axis=0, ddof=1)/np.sqrt(count)).tolist()
    tau = np.array([correlation_time(v) for v in values.T])
    effective = len(values)/(2*tau)
    autocorr_se = values.std(axis=0, ddof=1)/np.sqrt(effective)
    se = np.maximum.reduce([autocorr_se]+[np.array(v) for v in block_errors.values()])
    return dict(mean=means.tolist(), se=se.tolist(), block_errors=block_errors,
                autocorrelation_se=autocorr_se.tolist(),
                tau_physical=(tau*interval).tolist(), effective_samples=effective.tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    rows = []
    for path in sorted(args.root.glob("cl_*/summary.json")):
        result = json.loads(path.read_text())
        p = result["params"]
        row = dict(run=path.parent.name, status=result["status"], sites=p["nx"][0],
                   coupling=p["bjerrum_e"], dt=p["langevin"]["dt"],
                   stabilization=p["alpha_ds"], error=result.get("error"),
                   failure_step=result.get("failure_step"))
        data = np.load(path.parent/"samples.npz")
        profiles = data["profiles"]
        if len(profiles) > 1:
            n = p["nx"][0]
            height = profiles[:, 0]@data["z"]/(n*4/9)
            obs = np.column_stack([height, profiles.reshape(len(profiles), -1)])
            labels = ["height"]+[f"phi_{t}_cell{i}" for t in result["species"] for i in range(n)]
            exact = np.r_[result["exact_height"], np.asarray(result["exact_profiles"]).ravel()]
            interval = float(np.median(np.diff(data["times"])))
            real = statistics(obs.real, interval)
            imag = statistics(obs.imag, interval)
            se = np.array(real["se"])
            ise = np.array(imag["se"])
            delta = np.array(real["mean"])-exact
            zscore = np.divide(abs(delta), se, out=np.zeros_like(se), where=se > 0)
            izscore = np.divide(np.abs(imag["mean"]), ise, out=np.zeros_like(ise), where=ise > 0)
            worst = int(np.argmax(zscore))
            drift = data["drift"][:, 0]
            cancellation = data["partition_cancellation"].min(axis=1)
            row.update(labels=labels, exact=exact.tolist(), real=real, imaginary=imag,
                standardized_real_discrepancy=zscore.tolist(),
                standardized_imaginary_discrepancy=izscore.tolist(),
                worst_real_observable=labels[worst], worst_real_sigma=float(zscore[worst]),
                max_imaginary_sigma=float(izscore.max()),
                max_species_mass_error=result.get("max_species_mass_error"),
                drift_quantiles=np.quantile(drift,[.5,.9,.99,.999,1.]).tolist(),
                min_partition_cancellation=float(cancellation.min()),
                partition_near_zero_fraction=float(np.mean(cancellation < .01)),
                n_samples=len(obs))
        rows.append(row)
    (args.root/"analysis.json").write_text(json.dumps(rows,indent=2)+"\n")
    print(json.dumps([dict(run=r["run"], status=r["status"],
        height=r.get("real",{}).get("mean",[None])[0],
        height_se=r.get("real",{}).get("se",[None])[0],
        exact_height=r.get("exact",[None])[0], worst_sigma=r.get("worst_real_sigma"),
        pole_ratio_min=r.get("min_partition_cancellation"), error=r.get("error")) for r in rows],indent=2))
    complete = [r for r in rows if r["status"] == "completed" and "real" in r]
    if not complete:
        return
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(2,2,figsize=(10,7),layout="constrained")
    axes = axes.ravel()
    colors = {0:"#46709d",25:"#267c66",100:"#bd603d"}
    for r in complete:
        if r["sites"] != 2 or r["stabilization"] != 0:
            continue
        color = colors[r["coupling"]]
        axes[0].errorbar(r["dt"], r["real"]["mean"][0]-r["exact"][0],
                         yerr=r["real"]["se"][0],fmt="o",color=color,alpha=.75,capsize=3)
        counterion = r["labels"].index("phi_C_cell0")
        axes[1].errorbar(r["dt"], r["real"]["mean"][counterion]-r["exact"][counterion],
                         yerr=r["real"]["se"][counterion],fmt="o",color=color,alpha=.75,capsize=3)
    for ax in axes[:2]:
        ax.axhline(0,color="black",lw=1)
    axes[0].set(xlabel="Langevin time step",ylabel="CL height − exact height",
                title="Two cells, stabilization off")
    axes[1].set(xlabel="Langevin time step",ylabel="CL counterion density − exact density",
                title="Monovalent C, cell 0")
    for e,color in colors.items():
        axes[0].plot([],[],"o",color=color,label=f"E={e}")
    axes[0].legend()
    # Empirical survival curves are diagnostics; no tail fit certifies CL.
    for e,color in colors.items():
        selected = [r for r in complete if r["sites"]==2 and r["stabilization"]==0
                    and r["coupling"]==e and r["dt"]==.005]
        if not selected:
            continue
        ds = [np.load(args.root/r["run"]/"samples.npz") for r in selected]
        for ax,key,transform in [(axes[2],"drift",lambda x:x[:,0]),
                                  (axes[3],"partition_cancellation",lambda x:-np.log10(np.maximum(x.min(axis=1),1e-300)))]:
            values = np.sort(np.concatenate([transform(d[key]) for d in ds]))
            survival = np.arange(len(values),0,-1)/len(values)
            ax.step(values,survival,where="pre",color=color,label=f"E={e}")
            ax.set_yscale("log")
    axes[2].set(xlabel="Maximum bare drift",ylabel="Empirical survival probability",title="Drift tail (dt=0.005)")
    axes[3].set(xlabel="−log10(min |Q(W)| / Q(Re W))",ylabel="Empirical survival probability",title="Partition cancellation (dt=0.005)")
    axes[2].legend(); axes[3].legend()
    fig.suptitle("Exact finite-lattice CL check — no claim of PRL reproduction")
    fig.savefig(args.root/"exact_cl_validation.png",dpi=180)
    fig.savefig(args.root/"exact_cl_validation.pdf")


if __name__ == "__main__":
    main()
