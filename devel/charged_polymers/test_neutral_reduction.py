"""Validation A: charged mode with all-zero charges + uniform radii must
reproduce the neutral compressible run with global Gaussian smearing EXACTLY
(same seed -> identical trajectory)."""
import os
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS","1"); os.environ.setdefault("OMP_NUM_THREADS","2")
import numpy as np
from polymerfts import lfts
from lfts_charged import ChargedLFTS

A_SMEAR=0.15
base = {
    "nx":[16,16,16], "lx":[2.0,2.0,2.0],
    "chain_model":"discrete", "ds":1/20,
    "segment_lengths":{"A":1.0,"B":1.0},
    "chi_n":{"A,B":12.0}, "zeta_n":50.0,
    "distinct_polymers":[{"volume_fraction":1.0,
        "blocks":[{"type":"A","length":0.5},{"type":"B","length":0.5}]}],
    "langevin":{"max_step":10,"dt":1.0,"nbar":10000},
    "recording":{"dir":"/tmp/charged_valA","recording_period":5,
                 "sf_computing_period":10000,"sf_recording_period":100000},
    "saddle":{"max_iter":200,"tolerance":1e-6},
    "compressor":{"name":"am","max_hist":20,"start_error":1e-1,"mix_min":0.1,"mix_init":0.1},
    "platform":"cuda", "verbose_level":1,
}
np.random.seed(0)
w_init={"A":np.random.normal(0,1.0,16**3),"B":np.random.normal(0,1.0,16**3)}

def run(charged):
    p=dict(base)
    p["recording"]=dict(base["recording"], dir=base["recording"]["dir"]+("_c" if charged else "_n"))
    if charged:
        p["charges"]={"A":0.0,"B":0.0}
        p["radiuses"]={"A":A_SMEAR,"B":A_SMEAR}
        p["bjerrum_e"]=500.0
    else:
        p["smearing"]={"type":"gaussian","a_int":A_SMEAR}
    sim=(ChargedLFTS if charged else lfts.LFTS)(params=p, random_seed=12345)
    sim.run(initial_fields={k:v.copy() for k,v in w_init.items()})
    return sim

s_n=run(False)
s_c=run(True)
import scipy.io as sio, glob
fn=sorted(glob.glob("/tmp/charged_valA_n/fields_*.mat"))[-1]
fc=sorted(glob.glob("/tmp/charged_valA_c/fields_*.mat"))[-1]
dn=sio.loadmat(fn,squeeze_me=True); dc=sio.loadmat(fc,squeeze_me=True)
key=[k for k in dn if k.startswith("w_") and not k.startswith("__")]
md=max(np.abs(np.asarray(dn[k],dtype=float)-np.asarray(dc[k],dtype=float)).max() for k in key)
print("compared:",fn.split("/")[-1],"keys:",key,"max diff =",md)
print("psi max|.| =", np.abs(s_c.electro.psi).max())
assert md < 1e-10, "NEUTRAL REDUCTION FAILED"
print("NEUTRAL REDUCTION PASSED")
