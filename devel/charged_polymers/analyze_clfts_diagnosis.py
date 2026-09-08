"""Compare reduced CL integrators, including entire field observables."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_clfts_exact import statistics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
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


if __name__=="__main__":
    main()
