# Finite-model CL convergence diagnosis — 2026-09-07

**Further investigation:** [CL_POLE_RESOLUTION.md](CL_POLE_RESOLUTION.md) adds
an exactly equivalent positive marginal sampler. Twelve real-Langevin runs
recover the exact averages. Six original CL trajectories pass the tested
low-order stationarity identities despite biased means; excluding near-zero
observations does not remove that bias. The original CL path remains unchanged.

**The current CL dynamics fails the exact finite-model sampling benchmark.**
The discrepancy persists with much smaller steps, longer trajectories, a
different integrator, adaptive steps and independent seeds. The action,
conditional densities, projected noise and original one-step update pass
independent checks. Actual partition zeros occur within the sampled field
region and generate the observed drift extremes. These results strongly
support incorrect CL convergence in this fixture, rather than an FFT error
or an incorrect particle reference. They are numerical evidence, not a formal
proof of the infinite-time, zero-step limit or a complete boundary-term analysis.

No production FFT, CL drift or existing calibrated test parameters were changed.
The new implementations are diagnostic Python scripts only. This is the same
two-cell canonical model described in [EXACT_CL_VALIDATION.md](EXACT_CL_VALIDATION.md),
not the PRL model and not a continuum calculation.

## Exactly reduced dynamics

Write each field as its spatial mean plus `x*(1,-1)`. In this canonical model,
uniform species fields multiply single-particle partitions by constants that
cancel from normalized densities. Uniform auxiliary modes have decoupled
Gaussian dynamics; only their differences affect the spatial observables.
Eliminating those uniform modes therefore leaves the same marginal stochastic
dynamics, without changing interactions, charges, smearing or the bond matrix.

There are four remaining complex amplitudes: native active auxiliary indices
0, 3, 4, followed by psi. These are called `field_0` through `field_3` in the
saved analysis. In particular, `field_2` is the amplitude of native auxiliary
mode 4, not the frozen native mode 2.

The reduced runner explicitly sums the eight conformations of the four-bead
grafted chain, grouped by the occupancy of cell 0. Point-particle densities
use their exact two-cell tanh expression. The noise variance of a half-difference
is the native per-cell variance divided by two, namely `2 dt / C` before the
mobility factor. The imaginary noise directions and mobilities are preserved.

| Check | Maximum discrepancy |
|---|---:|
| Native versus reduced conditional densities; 64 complex fields, real/imaginary std=5 | 4.81e-14 |
| Native versus reduced drift at those fields | 7.36e-14 |
| Partition cancellation ratio | 1.00e-15 |
| Projected noise variance | 0 |
| Actual `BrushChargedCLFTS.run` step with the same supplied noise increments | 4.97e-15 |

The native tests call `check_total_partition()` and compare all species.
The one-step comparison reads the native post-update checkpoint, including psi.
Additional checks cover E=0 and psi mobility 0.1.

## A second independent reference: the original field integral

Gauss-Hermite quadrature directly integrates the original field contour after
the uniform modes have been integrated out. Its integrand is
`Q_P * Q_C**2 * Q_SP * Q_SM * Q_S` times the Gaussian field factors.
The implementation forms density numerators without dividing by Q and does
not evaluate log Q. It is an entire integrand, including at partition zeros.

Besides densities, it integrates each field amplitude and its square. Their
particle references follow by completing the Gaussian square for every one
of the 256 particle configurations. If `dphi` is the microscopic half-difference,
the conditional Gaussian centers are

```
mu_i = -sum_t A[t,i] * smear * dphi_t / (2 h_coef_mu2[i]),
mu_psi = E / pi**2 * sum_t z_t * smear * dphi_t.
```

The corresponding analytic covariances of the stored complex fields are
`1/(2 C h_coef_mu2[i])` and `-E/(C pi**2)`. Averaging the centers and their
squares plus these covariances over the positive particle weights gives the
exact field moments. Thus the comparison also uses entire observables that
do not themselves contain a partition denominator.

| E | Hermite order per active dimension | Maximum error across densities and field moments |
|---|---:|---:|
| 100 | 20 | 1.479e-3 |
| 100 | 32 | 2.684e-10 |
| 100 | 48 | 2.788e-15 |
| 0 | 32 | 2.443e-15 |

This independently confirms the particle-to-field mapping, including the
finite-grid self terms and the canonical multiplicities.

## Stochastic results

Twenty-one additional trajectories and four quadrature jobs completed:

- E=100: original LM/ETD and additive-noise Heun, dt=.0025/.000625, seeds
  301/401, total time 10,000 with 1,000 discarded.
- E=0/25: both integrators at dt=.000625, seed 301, the same duration.
- E=100: Heun with psi mobility .1/10 at dt=.000625, seed 301, the same duration.
- E=100: dt=.00015625, time 20,000 with 2,000 discarded; LM/ETD seed 501 and
  Heun seeds 501/601.
- E=0: dt=.000625, time 50,000 with 5,000 discarded; Heun seed 501 and LM/ETD seed 601.
- E=100: adaptive Euler at maximum dt=.000625/.00015625, seed 701, time 20,000
  with 2,000 discarded. The step is chosen from the current drift before noise
  is drawn: `dt_actual <= dt_max * min(1, 10/max|drift|)`. Noise scales with
  `sqrt(dt_actual)`. Observations are taken at fixed physical times, not weighted
  equally per adaptive step. No clipping, rejection or stabilizing force is added.

All observations are spaced by .1 Langevin time. Errors are the maximum of
10/20/40-block and autocorrelation estimates, as in the earlier benchmark.
Runs that share a seed are paired comparisons, not independent replicates.
Different fields use successive PCG64 draws within the reduced runner;
cross-run seeds are separated. The reduced stochastic trajectories need not
match the native trajectories point for point to have the same noise law.

At E=100, dt=.00015625, with psi mobility 1:

| Method, seed | C density in cell 0 ± SE | Auxiliary amplitude `field_2` ± SE |
|---|---:|---:|
| Exact | .20718911 | .65699013 |
| LM/ETD, 501 | .21321148 ± .00020726 | .56404950 ± .00185565 |
| Heun, 501 | .21314436 ± .00021742 | .56421569 ± .00185824 |
| Heun, 601 | .21312338 ± .00020704 | .56384330 ± .00188818 |
| Adaptive Euler, 701 | .21331872 ± .00019318 | .56490673 ± .00190322 |

The C-density discrepancies are about 27–32 estimated SE; the entire auxiliary
field mean differs by about 48–50 estimated SE. The various integrators agree
with each other much better than they agree with the exact reference. The
adaptive fine-step run reaches the full physical duration in 128,130,102 steps.
Changing psi mobility by factors of ten also does not restore agreement at
the tested settings. These findings cannot be explained simply by the original
LM/ETD time step or by statistical noise in a single short trajectory.

**The longer E=0 controls also fail this stronger comparison.** At dt=.000625,
the exact C density .19243640 is compared with .19302000 ± .00006632 (Heun,
seed 501) and .19307754 ± .00007713 (LM/ETD, seed 601), differences of 8.8 and
8.3 estimated SE. The earlier 2,000-time-unit controls did not resolve this
smaller bias. Their reported measurements remain unchanged, but their apparent
agreement must not be used as evidence that this finite E=0 CL model is valid.
This does not invalidate the neutral-reduction test: that checks equality of
two code paths, not correctness of their equilibrium distribution.

## Actual partition zeros and drift poles

For the smeared polymer field half-difference h, the chain partition is

```
Q_P(h) = c0 exp(h/2) + c1 + c2 exp(-h/2) + c3 exp(-h),
```

where the positive c coefficients sum the eight bond-weighted conformations.
Factoring out `exp(h/2)` gives a cubic polynomial in `exp(-h/2)`. Its exact
roots locate partition zeros at

```
h = 0.9614601145 ± 2.7421589828 i,
h = 1.26847297   + 2 pi i,
```

and their repeats by `4 pi i`. Each point-particle partition also has zeros
at `h=(2n+1) 2 pi i`. These are zeros of the mathematical partition functions,
independent of FFT evaluation. They produce poles in derivatives of log Q.

The strongest recorded drift states lie close to P or SP zeros. For example:

- E=0, Heun seed 301: polymer h=.96143145+2.74221821i, only 6.58e-5 from a
  Q_P zero; drift magnitude 10,167.
- E=100, fine adaptive Euler: polymer h=.96145934−2.74216402i, only 5.09e-6
  from a Q_P zero; drift magnitude 131,319. Its time step is reduced accordingly.

At the maximum-drift state of each of the 21 trajectories, the native solver
still passes its unchanged partition check. Comparing native densities with
full conditional enumeration gives a maximum scaled error of 1.98e-10, using
`max|difference| / (1+max|reference density|)`. Densities can be of order 1e5
near these poles, so absolute errors are larger there (up to 1.73e-5 in the
fine adaptive example); they should not be described as uniformly 1e-14.
The independent reduced calculation and the native solver agree despite their
different evaluation methods, and both encounter the same true zeros.

The connection between sampled drift poles and possible incorrect CL
convergence is established in the CL literature; merely having a pole does
not prove failure in every model. See [Aarts et al., JHEP 05 (2017) 044](https://arxiv.org/abs/1701.02322)
and [Nagata, Nishimura & Shimasaki, PRD 94 (2016) 114515](https://arxiv.org/abs/1606.07627).
Here the independent exact averages establish a sampling discrepancy, while
the pole analysis identifies a concrete mechanism consistent with it. A full
boundary-term calculation or mathematical classification of invariant measures
has not been performed. Finite-sample error bars and tail plots alone are not
such a proof.

## Consequences and remaining work

The deterministic implementation checks support the existing action-to-drift
and chain computations in this fixture. They do not certify that its CL
stationary distribution represents the intended field integral. The tested
changes in integrator, step size and mobility do not supply a correction.
No production patch is justified by tuning the output to the reference.

Before accepting a production CL result, a sampling formulation or a clearly
bounded parameter regime must pass independent equilibrium checks. For this
finite fixture, direct quadrature already supplies a correct field-integral
reference; it is not a scalable replacement for a 3D production sampler.
This benchmark does not prove that every CL parameter regime fails, and it
does not establish behavior at the physical PRL parameters. The separate PRL
model/ensemble/boundary mismatches in [VALIDATION_STATUS.md](VALIDATION_STATUS.md)
also remain. PRL reproduction is not complete.

## Code and evidence

- [diagnose_clfts_reduced.py](diagnose_clfts_reduced.py): reduction, native
  comparisons, fixed and adaptive integrators, exact field moments.
- [check_clfts_field_integral.py](check_clfts_field_integral.py): quadrature.
- [check_clfts_poles.py](check_clfts_poles.py): analytic poles and native checks
  at drift extremes.
- [analyze_clfts_diagnosis.py](analyze_clfts_diagnosis.py): correlated errors and plots.

Raw trajectories, parameters, source snapshots, RNG states, maximum-drift
states, SLURM scripts and job statuses are under
[/home/yongdd/polymer/dh_salt_runs/exact_cl_diagnosis_20260907](/home/yongdd/polymer/dh_salt_runs/exact_cl_diagnosis_20260907).
The original 19 trajectories use snapshot v1; the two adaptive trajectories
use v2. A subsequent deterministic check adds the explicit noise-variance
assertion without changing either integrator. `production_unchanged_check.json`
confirms the production source hashes are unchanged from the campaign launch.

Numba 0.67.0 and llvmlite 0.49.0 were installed only into that evidence
directory's `python_deps`, not into the shared conda environment. Example:

```bash
conda activate polymerfts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_MAX_ACTIVE_LEVELS=0 OMP_STACKSIZE=1G
ulimit -s unlimited
export PYTHONPATH=/home/yongdd/polymer/dh_salt_runs/exact_cl_diagnosis_20260907/python_deps
python devel/charged_polymers/diagnose_clfts_reduced.py --out /shared/check --checks-only
python devel/charged_polymers/analyze_clfts_diagnosis.py \
  /home/yongdd/polymer/dh_salt_runs/exact_cl_diagnosis_20260907
```

Use the saved per-configuration `run.sh` scripts with SLURM for long runs.
All 25 campaign jobs completed with exit 0:0; this describes execution status,
not successful CL equilibrium validation. Plots are `integrator_diagnosis.png`
and `polymer_partition_poles.png`, with PDF copies.
