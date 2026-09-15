"""Consolidated cl analysis workflows. Run through workflow.py."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from studies.statistics import correlation_time, statistics
import csv

# Original workflow: analyze_clfts_exact.py



def analyze_exact():
    ap = argparse.ArgumentParser(description='Autocorrelation/block diagnostics for the exact finite-lattice CL benchmark.')
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


# Original workflow: analyze_clfts_diagnosis.py



def analyze_reduced():
    ap = argparse.ArgumentParser(description='Compare reduced CL integrators, including entire field observables.')
    ap.add_argument("root",type=Path)
    args = ap.parse_args()
    rows = []
    for path in sorted(args.root.glob("cl_*/summary.json")):
        result = json.loads(path.read_text())
        data = np.load(path.parent/"samples.npz")
        ph, x = data["profiles"], data["states"]
        row = dict(run=path.parent.name,status=result["status"],method=result["method"],
            dt=result["params"]["langevin"]["dt"],coupling=result["params"]["bjerrum_e"],
            mobility=result["params"]["psi_dt_scaling"],seed=result["seed"],
            duration=result["duration"],equil=result["equil"],
            n_samples=len(ph),max_drift=result["max_drift_every_step"],
            min_cancellation=result["min_cancellation_every_step"])
        if len(ph) < 2:
            rows.append(row)
            continue
        h = ph[:,0]@data["z"]/(8/9)
        observables = np.column_stack([h,ph.reshape(len(ph),-1),x,x*x])
        labels = ["height"]+[f"phi_{t}_cell{i}" for t in result["species"] for i in range(2)]
        labels += [f"field_{i}" for i in range(4)]+[f"field2_{i}" for i in range(4)]
        exact = np.r_[result["exact_height"],np.array(result["exact_profiles"]).ravel(),
                      result["exact_field_moments"]["mean"],result["exact_field_moments"]["second"]]
        interval = float(np.median(np.diff(data["times"])))
        real,imag = statistics(observables.real,interval),statistics(observables.imag,interval)
        error = np.array(real["se"])
        delta = abs(np.array(real["mean"])-exact)
        scores = np.divide(delta,error,out=np.zeros_like(error),where=error>0)
        row.update(labels=labels,exact=exact.tolist(),real=real,imaginary=imag,
                   standardized_discrepancy=scores.tolist(),
                   sp_outer_strip_fraction=float(np.mean(abs(data["fields"][:,2].imag)>2*np.pi)))
        rows.append(row)
    (args.root/"analysis.json").write_text(json.dumps(rows,indent=2)+"\n")
    for r in rows:
        if "real" not in r:
            print(r["run"],r["status"])
            continue
        i=r["labels"].index("phi_C_cell0")
        j=r["labels"].index("field_3")
        print(r["run"],"C",r["real"]["mean"][i],"+/-",r["real"]["se"][i],
              "z=",r["standardized_discrepancy"][i],"psi",r["real"]["mean"][j],
              "+/-",r["real"]["se"][j],"exact",r["exact"][j],
              "psi z=",r["standardized_discrepancy"][j])
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    axes=axes.ravel()
    color={"lm-etd":"#327798","heun":"#c35c42","euler":"#997044","adaptive-euler":"#688b43"}
    marker={"lm-etd":"o","heun":"s","euler":"d","adaptive-euler":"^"}
    for r in rows:
        if "real" not in r or r["coupling"] != 100 or r["mobility"] != 1:
            continue
        for ax,label in zip(axes[:2],["phi_C_cell0","field_2"]):
            i=r["labels"].index(label)
            ax.errorbar(r["dt"],r["real"]["mean"][i]-r["exact"][i],yerr=r["real"]["se"][i],
                        fmt=marker[r["method"]],color=color[r["method"]],capsize=3,alpha=.8)
    for ax,title in zip(axes[:2],["C density, cell 0","Auxiliary amplitude 2 (entire observable)"]):
        ax.axhline(0,color="black",lw=1)
        ax.set(xlabel="Time step",ylabel="CL − exact",title=title)
    for method in ["lm-etd","heun","adaptive-euler"]:
        axes[0].plot([],[],marker[method],color=color[method],label=method)
    axes[0].legend()
    selected=[r for r in rows if "real" in r and r["coupling"]==100 and r["dt"]==.000625
              and r["seed"]==301 and r["mobility"]==1]
    for r in selected:
        data=np.load(args.root/r["run"]/"samples.npz")
        drift=np.sort(data["drift"][:,0])
        axes[2].step(drift,np.arange(len(drift),0,-1)/len(drift),where="pre",color=color[r["method"]],label=r["method"])
        axes[3].hist(data["fields"][:,2].imag,bins=150,density=True,histtype="step",color=color[r["method"]],label=r["method"])
    axes[2].set(xscale="log",yscale="log",xlabel="Reduced drift magnitude",ylabel="Empirical survival",title="E=100, dt=.000625")
    for sign in [-1,1]:
        axes[3].axvline(sign*2*np.pi,color="black",ls=":")
    axes[3].set(xlabel="Imaginary SP field amplitude",ylabel="Density",title="Nearest SP zeros: Re field=0, Im field=±2π")
    axes[2].legend(); axes[3].legend()
    fig.suptitle("Unchanged finite action: integrator and sampling diagnosis")
    fig.savefig(args.root/"integrator_diagnosis.png",dpi=180)
    fig.savefig(args.root/"integrator_diagnosis.pdf")


# Original workflow: summarize_validation.py



def analyze_brush():
    ap = argparse.ArgumentParser(description='Summarize validate_* outputs and export a scientific validation figure.')
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


# Original workflow: summarize_clfts_pole_resolution.py



def analyze_marginal():
    ap = argparse.ArgumentParser(description='Compare unchanged CL trajectories with the exact positive marginal sampler.')
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
