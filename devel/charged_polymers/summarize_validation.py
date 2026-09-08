"""Summarize validate_* outputs and export a scientific validation figure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def correlation_time(values):
    """Initial positive paired autocorrelation sum, in sampling intervals."""
    x = np.asarray(values, dtype=float)-np.mean(values)
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    c = np.fft.irfft(f*f.conj(), n=2*n)[:n]/np.arange(n,0,-1)
    if c[0] == 0:
        return .5
    r = c/c[0]
    pairs = r[1:2*((n-1)//2)+1].reshape(-1,2).sum(axis=1)
    first = np.flatnonzero(pairs <= 0)
    positive = pairs[:first[0]] if len(first) else pairs
    return .5+np.minimum.accumulate(positive).sum() if len(positive) else .5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    rows = []
    for file in sorted(args.root.glob("cl_*/summary.json")):
        d = json.loads(file.read_text())
        data = np.load(file.parent/"samples.npz")
        step_interval = np.median(np.diff(data["steps"]))*d["params"]["langevin"]["dt"]
        tau = [float(correlation_time(data["observables"][:,i].real)) for i in range(2)]
        effective = len(data["steps"])/(2*np.array(tau))
        rows.append(dict(run=file.parent.name, dt=d["params"]["langevin"]["dt"],
            nbar=d["params"]["langevin"]["nbar"], mean=d["cl_observables"],
            block_se=d["block_standard_error"], imaginary_mean=d["cl_imaginary_mean"],
            imaginary_block_se=d["imaginary_block_error"],
            tau_integrated_physical=(np.array(tau)*step_interval).tolist(),
            effective_sample_estimates=effective.tolist(),
            autocorrelation_se=(data["observables"].real.std(axis=0,ddof=1)/np.sqrt(effective)).tolist(),
            max_species_mass_error=d["max_species_mass_error"],
            max_partition_error=d["max_partition_error"], drift_quantiles=d["drift_quantiles"]))
    (args.root/"sampling_analysis.json").write_text(json.dumps(rows,indent=2)+"\n")
    plt.rcParams.update({"font.size":10, "axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1,3,figsize=(14,4.3), layout="constrained")
    det = json.loads((args.root/"deterministic.json").read_text())
    strong = json.loads((args.root/"strong_field.json").read_text())
    bad = [d for d in strong if d["dtype"] == "complex" and d["per_chain_field_contrast"] == 100.]
    height = det["cpu-mkl"]["strong_field"]["height"]
    axes[0].bar([0,1],[d["height_core"] for d in bad], color="#c86152",width=.55)
    axes[0].bar([2],[height],color="#25867e",width=.55)
    axes[0].axhline(height,color="#25867e",linestyle="--",linewidth=1)
    axes[0].set(xticks=[0,1,2],xticklabels=["Spectral\nCPU","Spectral\nCUDA","Direct\nCPU / CUDA"],
                ylabel=r"Mean height $\langle z\rangle/R_0$",title="Strong-field counterexample")
    axes[0].text(.05,.94,"Old spectral results now rejected",transform=axes[0].transAxes,va="top",fontsize=9)
    mf = json.loads((args.root/"saddle"/"summary.json").read_text())["mf_observables"][0]
    axes[1].axhline(mf,color="black",linestyle="--",label="Same-model MF")
    for row in rows:
        long_run = "seed52" in row["run"]
        high = row["nbar"] > 1e6
        color = "#25867e" if high else ("#354e91" if long_run else "#9eaccd")
        shift = .0015 if high else (-.0015 if long_run else 0)
        label = (r"$\bar N=10^8$, seed 51" if high else
                 (r"$\bar N=10^4$, seed 52 (long)" if long_run else r"$\bar N=10^4$, seed 51"))
        err = max(row["block_se"][0],row["autocorrelation_se"][0])
        axes[1].errorbar(row["dt"]+shift,row["mean"][0],yerr=err,fmt="s" if high else "o",
                         color=color,capsize=3,label=label if row["dt"] == .025 else None)
    axes[1].set(xlabel=r"Langevin $\Delta\tau$ (points offset)",ylabel=r"Mean height $\langle z\rangle/R_0$",
                xticks=[.025,.05],title="2D CL benchmark; no stabilization")
    axes[1].legend(fontsize=8,loc="best")
    colors = ["#354e91","#25867e","#bc7045"]
    for zplus,color in zip([1,2,3],colors):
        f = args.root/f"prl_mf_z{zplus}_semicanonical"/"profiles.npz"
        a = np.load(f)
        axes[2].plot(a["z"],a["rhoP"],color=color,label=f"z+ = {zplus}, fixed counterions")
    a = np.load(args.root/"prl_mf_z3_gc"/"profiles.npz")
    axes[2].plot(a["z"],a["rhoP"],"--",color=colors[-1],label="z+ = 3, GC only")
    axes[2].set(xlim=(0,65),xlabel="z (nm)",ylabel=r"Polymer volume fraction $\phi_P$",
                title="PRL-parameter MF reference")
    axes[2].legend(fontsize=8)
    fig.suptitle("Numerical validation — quantitative PRL reproduction remains unestablished",fontsize=13)
    fig.savefig(args.root/"validation_summary.png",dpi=180)
    fig.savefig(args.root/"validation_summary.pdf")
    print(json.dumps(rows,indent=2))


if __name__ == "__main__":
    main()
