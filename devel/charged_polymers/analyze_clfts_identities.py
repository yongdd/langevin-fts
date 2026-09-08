"""Stationarity and pole-cutoff diagnostics from saved CL states.

These are bulk integrals of unevolved generators, not a complete boundary
surface-flux calculation or a sufficient correctness criterion. Excluded
samples contribute zero; cutoff means are never conditionally renormalized.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit

from analyze_clfts_exact import statistics
from diagnose_clfts_reduced import drift
from validate_clfts_exact import C


@njit(cache=True)
def generators(states,roots,mapping,smear,chain_weights,coeff,signs,mobility,coupling):
    n=len(states)
    values=np.empty((n,6),np.complex128)
    distance=np.empty(n)
    ynorm=np.empty(n)
    diffusion=np.empty(4)
    for i in range(3):
        diffusion[i]=mobility[i]*signs[i]/C
    diffusion[3]=-mobility[3]/C
    if coupling==0:
        diffusion[3]=0
    for j in range(n):
        x=states[j]
        f,_,h,_=drift(x,mapping,smear,chain_weights,coeff,signs,mobility,coupling)
        for k,i in enumerate([2,3,0]):
            values[j,2*k]=f[i]
            values[j,2*k+1]=2*x[i]*f[i]+2*diffusion[i]
        nearest=1e300
        for root in roots:
            dy=(h[0].imag-root.imag+2*np.pi)%(4*np.pi)-2*np.pi
            nearest=min(nearest,np.sqrt((h[0].real-root.real)**2+dy*dy))
        for t in range(1,5):
            dy=(h[t].imag)%(4*np.pi)-2*np.pi
            nearest=min(nearest,np.sqrt(h[t].real**2+dy*dy))
        distance[j]=nearest
        # Imaginary directions after rotation to real-noise coordinates.
        ynorm[j]=max(abs(x[0].imag),abs(x[1].real),abs(x[2].real),abs(x[3].real))
    return values,distance,ynorm


def block_summary(values):
    errors=[]
    for n in [10,20,40]:
        blocks=np.array([v.mean(axis=0) for v in np.array_split(values,n)])
        errors.append(blocks.std(axis=0,ddof=1)/np.sqrt(n))
    return dict(mean=values.mean(axis=0).tolist(),se=np.maximum.reduce(errors).tolist())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.run/"summary.json").read_text())
    setup=np.load(args.run/"setup.npz")
    config=tuple(float(setup[k]) if k in ("smear","coupling") else setup[k]
                 for k in ["mapping","smear","chain_weights","coeff","signs","mobility","coupling"])
    roots=-2*np.log(np.roots(setup["chain_weights"][::-1]).astype(complex))
    data=np.load(args.run/"samples.npz")
    values,distance,ynorm=generators(data["states"],roots,*config)
    dt=float(np.median(np.diff(data["times"])))
    # Decompose the observed bias into contributions inside/outside pole neighborhoods.
    bias=np.column_stack([data["profiles"][:,1,0].real-summary["exact_profiles"][1][0],
                          data["states"][:,2].real-summary["exact_field_moments"]["mean"][2]])
    rows=[]
    for epsilon in [0,.001,.003,.01,.03,.05,.1,.3,.6]:
        outside=distance>epsilon
        rows.append(dict(epsilon=epsilon,excluded_fraction=float(np.mean(~outside)),
            generator_real=block_summary(values.real*outside[:,None]),
            generator_imag=block_summary(values.imag*outside[:,None]),
            bias_inside=block_summary(bias*(~outside)[:,None]),
            bias_outside=block_summary(bias*outside[:,None])))
    field_rows=[]
    for cutoff in [.5,1.,2.,3.,5.,10.]:
        mask=ynorm<cutoff
        field_rows.append(dict(cutoff=cutoff,excluded_fraction=float(np.mean(~mask)),
            generator_real=block_summary(values.real*mask[:,None]),
            generator_imag=block_summary(values.imag*mask[:,None])))
    result=dict(run=args.run.name,coupling=float(config[-1]),
        generator_labels=["L_field_2","L_field2_2","L_field_3","L_field2_3","L_field_0","L_field2_0"],
        uncut_real=statistics(values.real,dt),uncut_imag=statistics(values.imag,dt),
        bias_labels=["phi_C_cell0","field_2"],total_bias=block_summary(bias),
        pole_cutoffs=rows,field_cutoffs=field_rows,
        min_saved_pole_distance=float(distance.min()),
        note="Unevolved bulk cutoff diagnostics; not a full boundary-term proof or corrected estimator.")
    np.savez_compressed(args.out/"diagnostics.npz",generators=values,pole_distance=distance,imaginary_norm=ynorm)
    (args.out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(run=args.run.name,uncut_real=result["uncut_real"]["mean"],
          uncut_se=result["uncut_real"]["se"],bias=result["total_bias"],pole_cutoff_point1=rows[6]),indent=2))


if __name__=="__main__":
    main()
