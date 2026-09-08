"""Independent Gauss-Hermite integration of the reduced original field contour.

Integrates the entire product Q_P Q_C^2 Q_SP Q_SM Q_S, without logarithms or
division by Q. Uniform Gaussian modes cancel in normalized observables.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import roots_hermitenorm

from validate_clfts_exact import C, CHARGES, COUNTS, TYPES, enumerate_particles
from diagnose_clfts_reduced import field_reference


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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
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


if __name__=="__main__":
    main()
