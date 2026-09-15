"""Consolidated cl diagnostics workflows. Run through workflow.py."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from studies.exact import C, CHARGES, TYPES, enumerate_particles, COUNTS
import time
from scipy.special import roots_hermitenorm
from studies.reduced import field_reference, densities, drift
import contextlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.brush_clfts import BrushChargedCLFTS
from numba import njit
from studies.statistics import statistics

# Original workflow: check_clfts_exact_checkpoint.py

def check_checkpoints():
    ap = argparse.ArgumentParser(description='Compare saved finite-lattice CL densities with conditional enumeration.\n\nChecks the final saved fields, not every state or the largest sampled drift.\nUses the independently constructed matrices of validate_clfts_exact.py.')
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


# Original workflow: check_clfts_field_integral.py

def integrate(setup, order):
    mapping, smear, cw, coeff, signs, mobility, coupling = setup
    nodes, weights = roots_hermitenorm(order)
    weights /= np.sqrt(2*np.pi)
    axes = [nodes]*4
    waxes = [weights]*4
    if coupling == 0:
        axes[3],waxes[3] = np.array([0.]),np.array([1.])
    mesh = np.array(np.meshgrid(*axes[:3],indexing="ij")).reshape(3,-1).T
    wm = np.prod(np.array(np.meshgrid(*waxes[:3],indexing="ij")),axis=0).ravel()
    std = np.r_[np.sqrt(abs(1/(2*C*coeff))),np.sqrt(coupling/(C*np.pi**2))]
    rotation = np.r_[np.where(signs>0,1.+0j,1j),1j]
    moments = np.zeros(14,dtype=complex)
    for node,weight in zip(axes[3],waxes[3]):
        x = np.column_stack([mesh,np.full(len(mesh),node)])*std*rotation
        fields = smear*(x[:,:3]@mapping.T+x[:,3,None]*CHARGES)
        # Four possible occupancies of cell 0 for the grafted P chain.
        terms = np.exp(-.25*fields[:,0,None]*np.array([-2,0,2,4]))*cw
        qp = terms.sum(axis=1)
        p0 = (terms*(4+np.array([-2,0,2,4]))/9).sum(axis=1)
        q = np.cosh(.25*fields[:,1:])
        ion_product = q[:,0]**2*q[:,1]*q[:,2]*q[:,3]
        integrand = qp*ion_product
        row = np.empty((len(mesh),14),dtype=complex)
        row[:,0] = integrand
        row[:,1] = p0*ion_product
        multiplicities = np.array([2,1,1,1])
        for t in range(4):
            product = qp.copy()
            for j in range(4):
                product *= q[:,j]**(multiplicities[j]-(j==t))
            row[:,t+2] = COUNTS[t+1]/9*np.exp(-.25*fields[:,t+1])*product
        row[:,6:10] = integrand[:,None]*x
        row[:,10:14] = integrand[:,None]*x*x
        moments += (wm*weight)@row
    return moments[0],moments[1:]/moments[0]


def check_field_integral():
    ap=argparse.ArgumentParser(description='Independent Gauss-Hermite integration of the reduced original field contour.\n\nIntegrates the entire product Q_P Q_C^2 Q_SP Q_SM Q_S, without logarithms or\ndivision by Q. Uniform Gaussian modes cancel in normalized observables.')
    ap.add_argument("--setup",type=Path,required=True)
    ap.add_argument("--order",type=int,default=32)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    data=np.load(args.setup)
    config=tuple(data[k] for k in ["mapping","smear","chain_weights","coeff","signs","mobility","coupling"])
    exact=enumerate_particles(2,float(config[-1]))
    fm=field_reference(exact,config)
    reference=np.r_[exact["mean"][:,0],fm["mean"],fm["second"]]
    start=time.perf_counter()
    partition,answer=integrate(config,args.order)
    result=dict(order=args.order,coupling=float(config[-1]),
        labels=[f"phi_{t}_cell0" for t in TYPES]+[f"field_{i}" for i in range(4)]+[f"field2_{i}" for i in range(4)],
        field_integral_real=answer.real.tolist(),field_integral_imag=answer.imag.tolist(),
        particle_reference=reference.tolist(),max_error=float(np.max(abs(answer-reference))),
        partition_real=float(partition.real),partition_imag=float(partition.imag),
        wall_seconds=time.perf_counter()-start)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if not np.isfinite(answer).all():
        raise SystemExit(1)


# Original workflow: check_clfts_poles.py



def check_poles():
    ap=argparse.ArgumentParser(description='Locate exact chain/ion Q zeros and check native densities at drift extremes.')
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


# Original workflow: analyze_clfts_identities.py

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


def check_identities():
    ap=argparse.ArgumentParser(description='Stationarity and pole-cutoff diagnostics from saved CL states.\n\nThese are bulk integrals of unevolved generators, not a complete boundary\nsurface-flux calculation or a sufficient correctness criterion. Excluded\nsamples contribute zero; cutoff means are never conditionally renormalized.')
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
