"""Locate exact chain/ion Q zeros and check native densities at drift extremes."""
import argparse
import contextlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from brush_clfts import BrushChargedCLFTS
from diagnose_clfts_reduced import densities
from validate_clfts_exact import C, COUNTS, TYPES, enumerate_particles


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root",type=Path)
    args=ap.parse_args()
    rows=[]
    sims={}
    with open(args.root/"pole_native_checks.log","w") as log,contextlib.redirect_stdout(log):
        for path in sorted(args.root.glob("cl_*/summary.json")):
            result=json.loads(path.read_text())
            if result["status"] != "completed":
                continue
            data=np.load(path.parent/"setup.npz")
            e=float(data["coupling"])
            if e not in sims:
                sim=BrushChargedCLFTS(result["params"],random_seed=9991)
                sims[e]=(sim,enumerate_particles(2,e))
            sim,exact=sims[e]
            state=np.load(path.parent/"final_state.npz")["worst_state"]
            delta,h,ratios=densities(state,data["mapping"],float(data["smear"]),data["chain_weights"])
            roots=-2*np.log(np.roots(data["chain_weights"][::-1]).astype(complex))
            shifted=roots+4j*np.pi*np.round((h[0].imag-roots.imag)/(4*np.pi))
            closest=shifted[np.argmin(abs(h[0]-shifted))]
            species=int(np.argmin(ratios))
            nearest=(closest if species==0 else
                     1j*(2*np.pi+4*np.pi*np.round((h[species].imag-2*np.pi)/(4*np.pi))))
            w=np.zeros((5,2),complex)
            w[sim._active_aux,0]=state[:3]
            w[sim._active_aux,1]=-state[:3]
            sim.psi=np.array([state[3],-state[3]])
            ph,_=sim.compute_concentrations(w)
            native=np.array([ph[t] for t in TYPES])
            conditional=np.column_stack([COUNTS/9+delta,COUNTS/9-delta])
            field=np.column_stack([h,-h])
            exponent=exact["log_bond"]-C*np.einsum("cts,ts->c",exact["phi"],field)/2
            weight=np.exp(exponent-exponent.real.max())
            enumerated=np.einsum("c,cts->ts",weight,exact["phi"])/weight.sum()
            scale=1+float(abs(enumerated).max())
            row=dict(run=path.parent.name,worst_drift=result["max_drift_every_step"],
                cancellation=ratios.tolist(),smallest_Q_species=TYPES[species],
                smallest_Q_field=[float(h[species].real),float(h[species].imag)],
                closest_partition_zero=[float(nearest.real),float(nearest.imag)],
                partition_zero_distance=float(abs(h[species]-nearest)),
                P_field=[float(h[0].real),float(h[0].imag)],
                closest_P_zero=[float(closest.real),float(closest.imag)],
                P_zero_distance=float(abs(h[0]-closest)),
                native_density_absolute_error=float(abs(native-enumerated).max()),
                native_density_scaled_error=float(abs(native-enumerated).max()/scale),
                reduced_density_scaled_error=float(abs(conditional-enumerated).max()/scale))
            assert row["native_density_scaled_error"] < 1e-9,row
            assert row["reduced_density_scaled_error"] < 1e-9,row
            rows.append(row)
    report=dict(results=rows,native_checkpoints=len(rows),
        P_zero_representatives=[[float(r.real),float(r.imag)] for r in roots],
        P_zero_period_imag=4*np.pi,
        point_ion_zeros="h = (2n+1) 2 pi i, for the smeared half-difference h")
    (args.root/"pole_checks.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,e in zip(axes,[0,100]):
        path=args.root/f"cl_E{e}_dt0.000625_heun_s301_mob1.0"
        data=np.load(path/"samples.npz")
        h=data["fields"][:,0]
        ax.hexbin(h.real,h.imag,gridsize=90,bins="log",mincnt=1,cmap="Blues")
        lower,upper=ax.get_ylim()
        for root in roots:
            for period in range(-2,3):
                pole=root+4j*np.pi*period
                if lower <= pole.imag <= upper:
                    ax.plot(pole.real,pole.imag,"x",color="#bc3e35",ms=10,mew=2)
        ax.set(xlabel="Re polymer field amplitude",ylabel="Im polymer field amplitude",title=f"E={e}; crosses = exact Q_P zeros")
    fig.suptitle("Sampled polymer fields and analytically located partition zeros")
    fig.savefig(args.root/"polymer_partition_poles.png",dpi=180)
    fig.savefig(args.root/"polymer_partition_poles.pdf")


if __name__=="__main__":
    main()
