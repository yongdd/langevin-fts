# Historical research log

Original notes preserved verbatim before the 2026-09-08 consolidation.
Some early conclusions were superseded by exact-reference checks. Use the two
current reports for conclusions and workflow.py for executable commands.
Local links and old script names in the records below are historical.


---

# Source: CL_CONVERGENCE_DIAGNOSIS.md

SHA-256: `bcae002a98920735e2c5bae44c5367127e169295f089ca1f7ccb4e19972a0b64`

<!-- BEGIN CL_CONVERGENCE_DIAGNOSIS.md -->
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

<!-- END CL_CONVERGENCE_DIAGNOSIS.md -->


---

# Source: CL_POLE_RESOLUTION.md

SHA-256: `48a67817acdde7d6158f69beb96baa0c17fee6e9f9d44b1657cd272e0b612833`

<!-- BEGIN CL_POLE_RESOLUTION.md -->
# Same-action sampling control and CL pole diagnostics — 2026-09-07

**1D extension:** [ONED_PRL_REPRODUCTION.md](ONED_PRL_REPRODUCTION.md) replaces
exhaustive marginal construction by joint particle/real-field Gibbs updates.
It tests 4/16/32 cells and separately recalculates PRL Fig. 1 parameter cases.

**Exact Gaussian marginalization restores agreement with the finite-model
reference; the original CL trajectories remain biased.** The new sampler
changes the field representation while preserving the physical action. Twelve
real-Langevin runs at E=0/25/100 agree with all 19 reference observables within
1.36 estimated standard errors. Six saved CL trajectories also show that their
bias is not a direct averaging artifact from a few samples near partition zeros.

This extends [CL_CONVERGENCE_DIAGNOSIS.md](CL_CONVERGENCE_DIAGNOSIS.md). It is a
two-cell diagnostic, not a production CL repair or PRL reproduction. Production
FFT, CL drift, physical parameters and existing calibrated tests are unchanged.

## A positive marginal of the identical action

Use the four complex half-difference amplitudes of the earlier reduction:
three active auxiliary fields (native indices 0, 3, 4) and psi. For each of
the 256 particle configurations, their action is quadratic plus linear in the
fields. Integrate the two repulsive auxiliary amplitudes and psi analytically.
Retain only the attractive amplitude `x=field_0` on its real contour.

Let `a_i` be the quadratic coefficients, `l_i = smear * dphi @ A[:,i]`,
`q = smear * dphi @ charges`, and `C=9/4`. The integrated configuration weight
apart from the retained field is

```
B_config = bond_weight * exp[C * sum_{i=1,2} l_i^2/(4 a_i)
                            - C E q^2/(2 pi^2)],
p(x) ∝ exp(-C a_0 x^2) * sum_config B_config exp(-C l_0 x).
```

Here `a_0=1.25` and `a_1,a_2<0`. Grouping configurations with the same integer
occupancy difference `n_P(cell0)-n_S(cell0)` reduces the sum to five terms.
All coefficients are positive. Thus **p(x) has no zero for real x**, and its
logarithmic derivative is finite along the sampled path. This does not assert
that its analytic continuation has no complex zeros.

The real Langevin drift is `-2 a_0 x - <l_0>_x` with noise variance `2 dt/C`.
The sampler uses additive-noise Heun. Densities and the eliminated field moments
are conditional averages at x; x and x² are observed directly. Coefficients
come from the bond weights and interaction parameters, without fitting to the
exact means. No saddle approximation, interaction removal, clipping or
parameter adjustment is used.

The last real Gaussian integral reproduces the original normalized particle
weights to `5.56e-17`; numerical real quadrature reproduces all 19 observables
to `2.56e-15`. A finite-difference check of the marginal log-density derivative
at 64 random fields of std=5 has maximum error `1.58e-10`.

## Sampling comparison

The 12 runs cover E=0,25,100; dt=.005,.00125; and seeds 901,1001. Each runs to
Langevin time 20,000, discards 2,000 and records 180,000 observations spaced by
.1. Error bars use the maximum of 10/20/40-block and autocorrelation estimates,
as in the original benchmark. Seeds distinguish two replicates per setting;
the same seeds across parameters are paired comparisons, not independent runs.
The 19 observables are also correlated, not 19 independent statistical tests.

Every run completed, conserved each species' mass to `2.23e-16`, and agreed
with the reference within 1.353 estimated SE for every measured observable:
height, ten species/cell densities, four field means and four field squares.

At E=100, total time 20,000 with 2,000 discarded:

| Sampling method, seed | dt or maximum dt | C density, cell 0 ± SE | `field_2` mean ± SE |
|---|---:|---:|---:|
| Exact particle sum | — | .20718911 | .65699013 |
| Original CL LM/ETD, 501 | .00015625 | .21321148 ± .00020726 | .56404950 ± .00185565 |
| CL Heun, 501 | .00015625 | .21314436 ± .00021742 | .56421569 ± .00185824 |
| CL adaptive Euler, 701 | .00015625 | .21331872 ± .00019318 | .56490673 ± .00190322 |
| Positive real marginal, 1001 | .00125 | .20719224 ± .00005738 | .65695408 ± .00218514 |

`field_2` means native auxiliary mode 4 in this reduced numbering. It is an
entire field observable, so its original CL discrepancy does not arise from
dividing the observable itself by a small Q.

The new real sampler also resolves the smaller E=0 discrepancy: at dt=.00125,
seed 1001, C density is `.19244016 ± .00007421`, against exact `.19243640`.
Original long CL runs gave approximately `.19302–.19308` with SE `.00007`.

[Comparison figure (PNG)](/home/yongdd/polymer/dh_salt_runs/exact_cl_pole_resolution_20260907/sampling_comparison.png)
and [PDF](/home/yongdd/polymer/dh_salt_runs/exact_cl_pole_resolution_20260907/sampling_comparison.pdf)
show the original and new E=100 results. All 12 runs and 19 observables are in
`positive_comparison.csv` in the evidence directory.

## The existing CL bias survives exclusion of near-zero observations

For six saved CL trajectories, measure each configuration's minimum distance
to any exact P or point-particle partition zero in its smeared field amplitude
h, including the imaginary-period repeats. Analyze observations saved at fixed
physical times. These distances are not the minima over every integration step.

For an observable O, decompose its measured bias as

```
mean(O - O_exact) = mean[(O - O_exact) 1(distance <= epsilon)]
                 + mean[(O - O_exact) 1(distance > epsilon)].
```

Both terms divide by the original full sample count; they are not conditional
means or a proposed corrected estimator. At epsilon=.1:

| CL run | Near-zero sample fraction | Total C-density bias | Near-zero contribution | Remaining contribution |
|---|---:|---:|---:|---:|
| E=100, Heun 501 | .1789% | .00595525 | .00000299 | .00595226 |
| E=100, LM/ETD 501 | .1733% | .00602237 | .00000265 | .00601972 |
| E=100, adaptive Euler 701 | .2050% | .00612961 | .00000096 | .00612865 |
| E=0, Heun 501 | .0364% | .00058360 | −.00000328 | .00058689 |

The fourth E=100 run (Heun 601) gives the same conclusion. Over 99.94% of the
E=100 C-density bias remains in the contribution from outside these small
neighborhoods. The `field_2` bias also remains almost entirely outside them.
The result holds qualitatively at larger neighborhoods recorded in the data.
This rules out simply removing a few extreme measurements as a remedy. It
does **not** rule out the dynamical influence of brief pole encounters on the
subsequent distribution, or prove which pole causes that distribution.

## Low-order stationarity does not validate the sampled measure

Compute the unevolved holomorphic generators for `O=x_i,x_i²`, i=0,2,3:

```
L x_i = f_i,
L x_i² = 2 x_i f_i + 2 D_i.
```

For auxiliary amplitudes, `D_i=sign_i * mobility_i/C`; for psi it is
`-mobility_psi/C`, or zero at E=0. Negative D here accounts for imaginary noise
in the stored complex coordinates. In all six trajectories, every tested real
and imaginary part of `<L O>` lies within 1.686 estimated SE of zero, despite
their incorrect equilibrium averages. The LM comparison concerns the continuous
generator at its finite time step, not an exact finite-step stationarity identity.

Cutoff bulk averages and cutoffs in the imaginary directions after noise-axis
rotation are saved as diagnostics. They are **not a complete surface-flux or
evolved-observable boundary-term calculation**. A finite cutoff signal alone
does not prove a nonzero limiting boundary term; a small uncut value does not
prove correct convergence.

This limitation is consistent with [Salcedo & Seiler, J. Phys. A 52 (2019)
035201](https://arxiv.org/abs/1809.06888), which discusses incorrect CL limits
that satisfy Schwinger-Dyson identities. Their contour-classification theorem
is for one complex variable; it has not been established for this four-variable
fixture. [Seiler's pole-boundary analysis](https://arxiv.org/abs/2111.01609) also
shows, in its simple models, why unevolved late-time diagnostics can miss pole
effects. Neither paper supplies a completed boundary-term proof for this model.

## Interpretation and evidence

The exact particle sum, original contour quadrature and new positive marginal
sampler now agree on the same finite action. The original CL paths still give
different means. Together with the earlier native drift/noise/update checks,
this further supports a problem with the sampled CL measure in this fixture.
Poles remain a concrete candidate mechanism; this experiment changes several
field coordinates at once and does not isolate one pole or prove all CL fails.

The positive marginal is computationally cheap here because only 256 particle
configurations exist. Building the corresponding conditional sums in a large
3D polymer system would not be cheap. This diagnostic therefore supplies a
working finite-model control, not a scalable replacement for production CL.
The existing CL path and the PRL model/ensemble comparison remain unvalidated.

New code:

- [validate_clfts_positive_marginal.py](validate_clfts_positive_marginal.py):
  exact marginal, deterministic checks and real Langevin sampling.
- [analyze_clfts_identities.py](analyze_clfts_identities.py): generators,
  analytic pole distances and decomposition of measured bias.
- [summarize_clfts_pole_resolution.py](summarize_clfts_pole_resolution.py):
  aggregate checks, CSV and standalone figures.

Evidence is under
[/home/yongdd/polymer/dh_salt_runs/exact_cl_pole_resolution_20260907](/home/yongdd/polymer/dh_salt_runs/exact_cl_pole_resolution_20260907).
SLURM jobs 333564–333575 (12 sampling runs) and 333584–333589 (six analyses)
all completed successfully. The initial six analysis submissions failed on an
unsupported Numba `np.r_` expression before producing results; their logs are
preserved separately. Replacing it by an explicit array initialization and
checking compiled/Python agreement fixed the diagnostic only. No trajectories
or physical parameters were changed to resolve that compile failure.

Per-run `run.sh` files contain exact commands, setup paths and environment
settings. Numba/llvmlite remain confined to the earlier campaign's `python_deps`.
`analysis.json`, source snapshots, RNG states, SLURM logs and production hash
checks preserve the evidence needed to reproduce this comparison.

<!-- END CL_POLE_RESOLUTION.md -->


---

# Source: EXACT_CL_VALIDATION.md

SHA-256: `1567ff4e51331ea366f464988b3f041a6d26736631d27099be8b7372ba8f522c`

<!-- BEGIN EXACT_CL_VALIDATION.md -->
# CL versus exact finite-lattice particles — 2026-09-07

**Follow-up:** [CL_CONVERGENCE_DIAGNOSIS.md](CL_CONVERGENCE_DIAGNOSIS.md) extends
this campaign with 21 trajectories and direct field quadrature. Smaller steps,
longer sampling and alternative/adaptive integrators retain the discrepancy.
Longer E=0 controls also resolve a smaller bias that the initial campaign below
could not distinguish. The historical results below are preserved.

[CL_POLE_RESOLUTION.md](CL_POLE_RESOLUTION.md) subsequently recovers the same
exact averages with a positive real marginal obtained by analytic integration
of repulsive fields. It also shows that simple stationarity checks and removal
of near-zero observations cannot certify or repair the original biased CL data.

This benchmark tests the existing native discrete CL path against a positive
particle partition sum. No FFT implementation, production drift, or existing
test parameter is changed. It does not reproduce the PRL model or establish a
continuum limit.

## Model and independent reference

The unit-length, reflecting 1D box has two or four equally weighted cells. It
contains one four-bead P chain, two monovalent C particles, one trivalent SP,
one monovalent SM anion, and one neutral S particle. P has valence −1 per bead.
The first P bead is fixed at cell 0; its other three beads are free. Thus the
system has nine beads and zero total charge. There is only one graft anchor;
the annealed/quenched lateral distinction does not arise in this 1D fixture.

With N_ref=4, V=1 and C=sqrt(nbar)=9/4, the canonical particle multiplicities
`C V f_p / alpha_p` are exactly 1, 2, 1, 1, 1. The volume fractions are
4/9, 2/9, 1/9, 1/9, 1/9. chi_PS N=2, zeta N=4, all smearing radii=0.1.
At this coarse spacing the smearing is not continuum-resolved; both reference
and CL use the same finite cosine multipliers.

Let n_t(j) be the bead occupancy of species t in cell j. The microscopic
volume fraction is `phi_t(j) = n_t(j) / (C N_ref dv)`, with `dv=1/M`.
Using the smeared density phi_bar, the particle interaction energy is

```
U = C V [ chi_PS N * mean(phi_bar_P * phi_bar_S)
          + (zeta N / 2) * mean((sum_t phi_bar_t - 1)^2)
          + (E / 2) * mean(c * G c) ],
c = sum_t z_t phi_bar_t,   G = (-laplacian)^(-1) on nonzero modes.
```

All self terms implied by this finite field action are included. No separate
self-energy subtraction, local incompressibility, or salt reservoir is added.
The bond multiplier is exp(−k² ds/6), smearing exp(−a² k²/2), and G's multiplier
is 1/k² except at k=0, where it is zero. The reference constructs these matrices
explicitly with cosine modes rather than calling the production transforms.

Each configuration has positive weight `product(bond weights) * exp(-U)`.
Enumerating labelled C particles supplies their correct occupancy multiplicity;
the canonical 2! is a configuration-independent factor and cancels from averages.
The 2-cell reference sums all 2^8=256 configurations; the 4-cell reference sums
all 4^8=65,536. Profiles are therefore obtained without Monte Carlo uncertainty.

Two preliminary identities check the mapping:

- At complex fields with real and imaginary standard deviation 5, enumerating
  noninteracting chains/particles with those external fields agrees with the
  native conditional densities: maximum error 3.09e-14 (4 cells, CPU) and
  7.74e-16 (2 cells, CUDA). Native partition consistency also passes.
- Eliminating the quadratic auxiliary and electrostatic fields reproduces U
  above: maximum per-CV error 4.45e-15 in the checked configurations.

## Stochastic comparison

The initial campaign has 22 parameter-specific SLURM jobs:

- 2 cells, E=0/25/100, dt=.02/.01/.005, stabilization=0, seeds 71/72.
- 2 cells, E=25, dt=.01, stabilization=.05, seeds 71/72.
- 4 cells, E=25, dt=.01, stabilization=0, seeds 71/72.

Each trajectory lasts 2000 Langevin units, discarding the first 200. Profiles,
bare and stabilized drift maxima, and partition cancellation ratios are saved
every 0.1 units. Native partition checks and species mass checks remain active.
Failures preserve the offending auxiliary fields and psi; a finite trajectory
or a successful process exit alone is not treated as a correctness test.

Two follow-up jobs use 2 cells, E=25/100, dt=.0025, stabilization=0 and seed
173, lasting 1000 units and discarding 100. These extend the time-step check;
their shorter physical duration must be considered when comparing error bars.
The original seeds 71/72 are reported separately: the existing psi generator
uses seed+1, so these trajectories share one generator seed across different
field components. They are not pooled as independent replicates. Seed 173's
streams do not overlap those seed assignments. Production RNG code is unchanged.

The cancellation diagnostic is `|Q(W)| / Q(Re W)` using the same finite bond
operator. It is invariant under uniform real field shifts, unlike raw |Q|.
A small ratio indicates cancellation near possible drift poles; a finite-sample
histogram alone does not establish a CL correctness criterion.

For each observable the analysis reports 10/20/40-block errors, autocorrelation
times, estimated effective sample counts, and imaginary means. The reported SE
is the largest of those block estimates and the autocorrelation estimate.
These are empirical uncertainty estimates, not rigorous bounds. Height and all
species cell densities are compared with the exact reference. Distinct cells
and species have correlated errors, so the largest standardized deviation is
a diagnostic rather than an independent multiple-testing confidence level.

## Results

All 24 trajectories completed, including their native partition and
species mass checks. Maximum species mass error was 3.54e-13. The six E=0
controls agree with the reference: every checked real observable differs by
at most 1.51 estimated SE. Charged trajectories do not show comparable agreement.

For the two-cell fixture at dt=.005 with stabilization off, the monovalent
counterion volume fraction in cell 0 is:

| E | Exact | CL seed 71 ± SE | CL seed 72 ± SE |
|---|---:|---:|---:|
| 0 | .19243640 | .19278850 ± .00033426 | .19282432 ± .00035266 |
| 25 | .20787987 | .20933511 ± .00027945 | .20926531 ± .00024669 |
| 100 | .20718911 | .21294327 ± .00058985 | .21359478 ± .00081007 |

These E=25/100 discrepancies are 5.21–9.76 estimated SE. Their counterion
autocorrelation estimates correspond to about 337–594 effective samples per
trajectory, rather than treating all 18,000 stored values as independent.
At E=100 the mean heights are .408940 ± .002489 and .409227 ± .002591,
versus exact .39815593. E=25 seed 71 instead has a large height error estimate
(.409159 ± .009108), illustrating the sensitivity of observables to rare events.
The dt=.01 runs with stabilization=.05 also retain counterion discrepancies;
that tested stabilization value does not resolve the comparison.

The finer-step follow-ups (dt=.0025, seed 173, stabilization off) give:

| E | Observable | Exact | CL ± SE | Difference / SE |
|---|---|---:|---:|---:|
| 25 | C density, cell 0 | .20787987 | .20926778 ± .00057137 | 2.43 |
| 25 | SP density, cell 0 | .12889864 | .12650501 ± .00078950 | 3.03 |
| 100 | C density, cell 0 | .20718911 | .21308439 ± .00112541 | 5.24 |
| 100 | Mean height | .39815593 | .40964106 ± .00355063 | 3.23 |

The E=100 counterion discrepancy persists at the smallest tested time step
with disjoint seed assignments. Its central value does not move toward the
exact result over dt=.02 to .0025 within the observed variation. These data
still do not provide a controlled dt→0 extrapolation. In particular, the two
follow-ups have half the production time of the original trajectories, and
rare large density excursions make empirical error estimates uncertain.

The four-cell fixture is a separate spatial discretization, with its own exact
reference. Its height agrees within the estimated uncertainty, but P density
in cell 1 does not: exact .47375505, versus .45115654 ± .00333992 and
.44947711 ± .00337511 at dt=.01. A height-only acceptance check would miss this.

At all 24 final checkpoints, independent enumeration in the saved complex
fields reproduces the saved native densities to at most 9.67e-15. This checks
the chain/particle density calculation at those states, not every intermediate
state or the rarest drift events. It provides no reason to change native FFT
code to resolve these stochastic discrepancies.

Charged trajectories also sample stronger drift excursions and smaller
partition cancellation ratios than the E=0 controls. For example, E=25,
dt=.005, seed 71 reaches a sampled bare drift of 455.45 and a cancellation
ratio of 6.96e-4. These observations motivate investigation of rare events and
drift poles; they do not establish the cause or prove a tail asymptotic law.

**The charged CL comparison is not validated at the tested settings.**
Finite time steps and empirical errors in distributions with rare large events
do not prove an incorrect dt→0 CL limit. Conversely, finite trajectories,
conserved mass, small imaginary averages and consistent chain partitions do
not justify accepting these averages or claiming PRL reproduction. The next
question is to distinguish integration bias, insufficient exploration of rare
events, and possible incorrect CL convergence for this finite model before
using target PRL parameters.

## Reproduction and evidence

[validate_clfts_exact.py](validate_clfts_exact.py) implements the reference and
the run; [analyze_clfts_exact.py](analyze_clfts_exact.py) analyzes saved samples.
[check_clfts_exact_checkpoint.py](check_clfts_exact_checkpoint.py) independently
checks conditional densities at final saved fields.

```bash
python devel/charged_polymers/validate_clfts_exact.py \
  --out /shared/check --sites 4 --deterministic-only
python devel/charged_polymers/analyze_clfts_exact.py \
  /home/yongdd/polymer/dh_salt_runs/exact_cl_20260907
python devel/charged_polymers/check_clfts_exact_checkpoint.py \
  /home/yongdd/polymer/dh_salt_runs/exact_cl_20260907
```

Long runs must use the per-configuration `run.sh` SLURM scripts in
[/home/yongdd/polymer/dh_salt_runs/exact_cl_20260907](/home/yongdd/polymer/dh_salt_runs/exact_cl_20260907).
That directory preserves the runner snapshot, source hashes, parameters,
checkpoints, raw samples and job identifiers. Initial submission failed during
conda shell activation before simulations began; `run.sh` now enables nounset
after conda activation, and those environment failures are separately recorded.
Simulation jobs 333502–333525 all finished COMPLETED with exit code 0:0;
this is execution status, not a successful CL correctness comparison.
The numerical results are in `analysis.json`, conditional-density comparisons
in `checkpoint_checks.json`, and plots in `exact_cl_validation.png`/`.pdf`.
`source_unchanged_check.json` confirms that all ten recorded production/runner
source hashes match the launch versions, including the native FFT solvers.

<!-- END EXACT_CL_VALIDATION.md -->


---

# Source: FIG23_PRL_REPRODUCTION.md

SHA-256: `132681a79f61d51d3bceb4b1db11a56c6199e864e46f5c9de4cd8683f0188a43`

<!-- BEGIN FIG23_PRL_REPRODUCTION.md -->
# PRL Figs. 2–3 recalculation at fixed parameters

2026-09-07. **Contraction and re-expansion occur, but quantitative reproduction remains incomplete.**
These results check smoothed-LDA SCFT, not the existing CL dynamics or the paper's full SI implementation.
See the later Fig. 2–3 SCFT section of [ONED_PRL_REPRODUCTION.pdf](ONED_PRL_REPRODUCTION.pdf);
the editable LaTeX supplement is [FIG23_PRL_REPRODUCTION.tex](FIG23_PRL_REPRODUCTION.tex).

## Model and scope

- N=100, b=1 nm, v=1 nm³, σ=0.1 nm⁻², χ=0.5, ℓB=0.7 nm, all Born radii=0.25 nm.
- Polymer charge −1, canonical monovalent C totaling 10 nm⁻², GC z:1 salt, point ions.
- **c1=0 fixed**. No valence-dependent background salt, reference height, or correlation coefficient was fitted.
- Standard grid L=200 nm, Nz=1600. Wall-centered half-Gaussian graft of width 0.5b,
  Gaussian bonds, hard bottom wall and reflecting top. This is not assumed identical to the paper's delta graft.
- For each z+=1,2,3, 19 standard points span rho_b=10⁻⁷…50 M. Validated solutions exist at **57/57 points**.
  Every reported point was calculated; interpolation across failures was not used to fill results.
- Upward/downward continuation, smaller concentration steps near difficult points, and grid/box controls
  used **one SLURM job per parameter value**.
- The exact correspondence between current LDA and the paper's nonlocal Green function/I_ex treatment is unverified.
  The paper also neglects u_P^ex in brush calculations; that omission alone cannot explain the discrepancy.

## Fig. 2

Two heights were computed explicitly.

$$h_m=2\int z\rho_P(z)dz/(\sigma N),\qquad
h_{eq}=\sigma N/\max_z\rho_P(z).$$

The second is an equimolar slab estimate using max rho_P as the reference density.
The paper/SI does not specify the Gibbs dividing-surface extraction procedure, so neither is assumed equal to its height.
Each h0 was obtained by solving **exactly rho_b=0**. The moment h0 is
50.91070746 nm,
and the equimolar h0 is 38.44857944 nm.
A rho_b=10⁻¹⁰ M control gives a moment height of about 50.910706 nm, consistent with the salt-free limit.

Minimum moment heights among the 19 standard points:

| z+ | rho_b [M] | h_m [nm] | log10(h_m/h0) |
|---|---:|---:|---:|
| 1 | 50 | 21.91363454 | -0.366095 |
| 2 | 0.1 | 19.51183502 | -0.416511 |
| 3 | 0.0316228 | 14.11156674 | -0.557234 |

These are sampled minima, not exact minima over continuous concentration. The monovalent minimum is at the scan limit.
Multivalent contraction and re-expansion occur, but contraction starts at substantially different low-salt concentrations than in the paper.
Neither height definition yields simultaneous quantitative agreement for all three valences.

Twenty-four points were digitized from the paper's theory lines in the original raster, excluding experimental symbols.
Coordinates, color masks, line-selection guides, and image hashes are saved in `paper_fig23_digitized.json`.
The overlay in `paper_fig2_digitization_check.png` was checked visually against the original.
Digitization uncertainty is about ±0.01 in log10(h/h0), not a statistical error. Maximum differences at exactly matching concentrations:

| z+ | Maximum moment-height difference | Maximum equimolar-height difference |
|---|---:|---:|
| 1 | 0.019317 | 0.091799 |
| 2 | 0.317223 | 0.315491 |
| 3 | 0.473847 | 0.436211 |

## Fig. 3: analytical formulas, SCFT, and units

Differentiating the SI function u(x)=1−x erfcx(x/√π) gives Γ from Eq. (7) and potential from Eq. (9) directly.
The calculation uses κ²=4πℓB z+(z++1)rho_b and rho_b[nm⁻³]=0.6022 rho_b[M].
These analytical formulas are independent of SCFT. The SI S34 assumptions of rho_b≫rho_P and small density changes
do not automatically hold in low-salt SCFT.

In SCFT, Γ(z)=1−z+(rho_+(z)−rho_b)/rho_P(z) was averaged with rho_P weights over rho_P>c max(rho_P),
using c=0.5 and 0.8. The canonical C contribution in the core was saved separately.
Both psi(0) and psi(0)−psi(L) were recorded and normalized by h_m/(σNv).
Wall/far-end values use first/last cell centers (reference grid z=0.0625, 199.9375 nm).
Since psi(L)≠0 in the finite canonical box, the sign of psi(0) alone cannot establish charge inversion.

Among the 19 standard points, the divalent core Γ minimum is about −0.0140, but the relative potential remains negative.
The trivalent core Γ minimum is about −0.5314; the maximum normalized relative potential is about +0.7834.
This shows the qualitative separation of divalent overcharging from potential reversal,
and trivalent potential reversal. The trivalent potential peak nevertheless differs from the paper's value of about 2.5.
These are values at standard concentration points, not extrema over a continuous function.

**The monovalent potential in the paper's Fig. 3 differs from unit-consistent Eq. (9) by a factor of about 0.6022.**
For z+=1, the normalized potential is −1/(2v rho_b), independent of Γ, a, and ℓB.

| rho_b [M] | Eq. (9), M→nm⁻³ applied | Paper (approx.) | Eq. (9)×0.6022 |
|---|---:|---:|---:|
| 0.3 | −2.7676 | −1.6598 | −1.6667 |
| 1 | −0.8303 | −0.4985 | −0.5000 |
| 3 | −0.2768 | −0.1701 | −0.1667 |

The factor is consistent with the M→nm⁻³ conversion, but its cause is unconfirmed without the authors' raw data/plotting code.
Additional multivalent discrepancies remain. **Calculated curves were not rescaled to fit the paper.**
Only a separate diagnostic curve with this factor explicitly labeled was added to `fig3_units_check.pdf`.
The printed SI S42 lacks the rho_b denominator visible in main-text Eq. (9), so potential calculations use the main-text equation.

Forty Fig. 3 points were digitized. Uncertainties are about ±0.02 for Γ and ±0.05 for potential, not statistical errors.
Overly steep points and those obscured by axes/reference lines were excluded. Coordinate overlays were checked visually.

## Numerical reliability

Failures in initial pilot/independent runs were addressed only in the separate research runner.

1. Derivatives of canonical PB normalization were included. The Jacobian is tridiagonal plus rank one,
   solved with Sherman–Morrison. The equations were unchanged. For fields with std≈5, finite-difference
   Jacobian relative error was 3.77e−10, and the solution difference from the existing PB solver was 4.10e−11.
2. When a bead slice in the existing **real-space** float64 chain fell below its 1e−300 floor,
   densities could be wrong despite final normalization enforcing total mass. In a low-salt example,
   the SCFT residual was about 2e−10 but the independent log-space density difference was 0.209798 nm⁻³.
3. The research runner uses the np.longdouble exponent range and stepwise scaling.
   One Q is used at every contour position, without per-bead mass normalization.
   Independent all-log/log-sum-exp bond calculations validate it; the difference on the failing field was 6.99e−14.
4. When the relative inner high-salt PB tolerance failed the final absolute residual check,
   the problem was solved more tightly with `--pb-tolerance 1e-10`. Final acceptance criteria were not relaxed.

`prl_scft.py`, the Fig. 1 runner, existing CL drift, and production C++/CUDA/FFT were unchanged in this work.
SHA-256 hashes of the 33 protected files match those at the start.
The research runner does not validate a modification to the production path.

| Check | Acceptance threshold | Maximum error in accepted extended runs |
|---|---:|---:|
| Full SCFT residual | 1e-07 | 6.0219e-08 |
| Independent chain density | 1e-10 | 2.3334e-13 |
| Contour Q | 1e-10 | 1.9792e-12 |
| Polymer integral | 1e-10 | 3.5527e-15 |
| C integral | 1e-10 | 3.5527e-15 |

Preserved records include 215 validated extended runs, 249 successful summaries,
and 47 failed summaries. Successful summaries include initial float64 controls,
so their count differs from the extended-run count. Canceled/resubmitted jobs are tracked in SLURM records and `jobs.jsonl`.
For 57 standard concentration pairs checked in both scan directions, the maximum moment-height difference is
2.8511e-07 nm.
For the same pairs, the largest full polymer density profile difference is 1.9004e-08 nm⁻³,
and the largest potential profile difference is 2.7111e-08.
This agreement checks solution consistency, not a global free-energy minimum.

Selected grid/box checks: reference L=200, Nz=1600; refined grid L=200, Nz=3200;
larger box L=400, Nz=3200. Height changes are in nm.

| z+ | rho_b [M] | Finer grid Δh | Larger box Δh |
|---|---:|---:|---:|
| 1 | 0 | 0.0008016 | 0.0104897 |
| 1 | 1e-07 | 0.0008016 | 0.0099457 |
| 2 | 0.1 | 0.0022102 | -0.5040686 |
| 3 | 0.01 | 0.0034460 | -0.3023044 |
| 3 | 10 | 0.0019476 | -0.0000511 |

At low salt, box dependence can be much greater for potential than height. For example, at zero salt, L=200→400
changes the wall–far-end potential difference by about −2.1744. These selected checks do not establish L→∞ convergence.

## Raw data and rerunning

Raw data: [/home/yongdd/polymer/dh_salt_runs/fig23_prl_20260907](/home/yongdd/polymer/dh_salt_runs/fig23_prl_20260907).

- `comparison.json`, `fig23_primary.csv`: actual run names, initial states, directions, and observables for every accepted point.
- Each run's `run.sh`, `slurm.out`, `summary.json`, `profiles.npz`, and failed checkpoints.
- `method_checks.json`, `low_salt_chain_diagnosis.json`, `protected_source_sha256.json`.
- `check_chain_gradient.py`, `chain_gradient_checks.json`: density versus directional derivatives of the common log Q.
- `paper_fig23_digitized.json`, digitization overlays, original PDF/SI.
- `source_v1/`, `source_final/`, `jobs.jsonl`, and submission helper `submit.py`.

Since early and final runs use different numerical algorithms, acceptance requires
`chain_solver=extended`, `pb_solver=rank1`, `reference=log`, and `status=completed`.
Failed or unvalidated results from the initial algorithm are excluded from the final scan.
Additional PB tolerance/iteration-cycle options are specified in each run's params and run.sh.

Reanalysis/LaTeX build (no new simulations):

```bash
source /home/yongdd/anaconda3/etc/profile.d/conda.sh
conda activate polymerfts
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python devel/charged_polymers/analyze_prl_fig23.py \
  /home/yongdd/polymer/dh_salt_runs/fig23_prl_20260907
python devel/charged_polymers/build_prl_comparison_pdf.py \
  --data-root /home/yongdd/polymer/dh_salt_runs/oned_prl_20260907 \
  --fig23-data-root /home/yongdd/polymer/dh_salt_runs/fig23_prl_20260907
```

To rerun a simulation, change `--out` in its run.sh to a new directory and submit through SLURM.
Preserve existing result directories. Tectonic compiles the PDFs, with equations typeset in TeX.
The current report includes this SCFT section; the combined PDF appends the preserved historical report.

Key remaining tasks: establish matching height definitions, ensembles, and boundaries; compare nonlocal Green function/I_ex treatments;
and confirm potential-figure normalization. Fig. 4 and the original CL correctness issue are not resolved here.

Source: [Duan et al., PRL 134, 048101 (2025)](https://doi.org/10.1103/PhysRevLett.134.048101),
main-text Figs. 2–3, Eqs. (7), (9), SI S33–S42. Comparisons use the saved published PDF.

<!-- END FIG23_PRL_REPRODUCTION.md -->


---

# Source: ONED_PRL_REPRODUCTION.md

SHA-256: `67b9daa4d54bf02aaf7ce40159afea2d2cb6b54a9cf7c634f87d2e699b6a9dc8`

<!-- BEGIN ONED_PRL_REPRODUCTION.md -->
# 1D positive sampling and PRL Figs. 1–3 checks — 2026-09-07

[Current results PDF](ONED_PRL_REPRODUCTION.pdf) · [Combined with the historical comparison report](PRL_COMPARISON.pdf)

**Latest direct comparison:** LDA-SCFT and mean-field SCFT with correlations disabled were solved again at the actual MC bath concentrations.
Sampling is closer to mean field than LDA, but does not agree completely. Heights, charges, potentials, and model differences are documented in
[SAMPLING_SCFT_COMPARISON.md](SAMPLING_SCFT_COMPARISON.md) and the opening comparison section of the PDF.

**Latest addition: actual sampling of an N=100 brush.** The earlier Fig. 2–3 calculations used SCFT and did not fulfill the sampling request.
New results are in [POSITIVE_BRUSH_RESULTS.md](POSITIVE_BRUSH_RESULTS.md); the sampled finite action is in
[POSITIVE_BRUSH_SAMPLING.md](POSITIVE_BRUSH_SAMPLING.md).
The section following the direct comparison preserves actual Monte Carlo averages, statistical errors, the probability distribution, and validation.
This is a separate planar model with transverse fluctuations removed; it does not validate the existing CL or quantitative 3D PRL reproduction.

The current PDF compiles [ONED_PRL_REPRODUCTION.tex](ONED_PRL_REPRODUCTION.tex) with Tectonic (XeTeX), using `amsmath` for equations.
[PRL_COMPARISON.tex](PRL_COMPARISON.tex) combines the current results with the historical report.
After the N=100 sampling section, the current document retains the earlier small-system/SCFT checks.
The combined PDF adds the nine-page historical report.
[Detailed Fig. 2–3 records](FIG23_PRL_REPRODUCTION.md) are maintained separately.
Original PDF and TeX files are preserved in `prl_comparison_figs/archive/`.

After editing, recompile with Tectonic in the `polymerfts` environment. The English report requires no local Korean fonts.

```bash
conda activate polymerfts
cd devel/charged_polymers
tectonic --keep-logs --synctex ONED_PRL_REPRODUCTION.tex
tectonic --keep-logs --synctex PRL_COMPARISON.tex
```

To refresh numerical tables and vector figures from the raw data, run from the repository root:
`python devel/charged_polymers/build_prl_comparison_pdf.py --data-root /home/yongdd/polymer/dh_salt_runs/oned_prl_20260907`.
The builder preserves the editable TeX body and replaces PDFs only after both compilations and missing-character/overflow checks succeed.
It does not rerun simulations.

The records below predate the N=100 sampling addition.
**The small 1D real-field/particle sampler works. PRL Fig. 1 reproduces qualitative trends, but quantitative reproduction with the current correlation approximation remains incomplete.**
These calculations use different models: the 1D sampler validates the existing finite 9-bead action, while PRL calculations use separate N=100 point-ion/incompressible SCFT code.

## 1. 1D real-field sampling without exhaustive enumeration

The earlier two-cell marginal retaining only a real field summed all particle configurations.
Here the repulsive auxiliary field and psi are integrated exactly, while **particle positions and the real field in the attractive channel are retained together**.
This Gibbs sampler draws the real field from a Gaussian at fixed particles and updates each particle using its exact conditional probabilities at fixed field.
It uses no finite-time-step Langevin integration. It is neither a field-only sampler with all particles integrated out nor the same dynamics as existing CL.

The equivalence of the finite action follows as below. Let $\boldsymbol{x}$ be the real field with its uniform mode removed,
$\boldsymbol{\ell}$ the corresponding attractive-channel combination of smeared densities, and $a>0$ the positive field coefficient.
Write the spatial mean as $\overline{f}=M^{-1}\sum_{i=1}^{M}f_i$ and the bond weight as $B(\boldsymbol R)$:

$$
\begin{aligned}
U_{\mathrm{rep}}(\boldsymbol R)
&=U_{\mathrm{particle}}(\boldsymbol R)+\frac{C}{4a}\overline{\ell(\boldsymbol R)^2},\\
P(\boldsymbol R,\boldsymbol x)
&\propto B(\boldsymbol R)\exp\!\left[-U_{\mathrm{rep}}(\boldsymbol R)
-C\overline{a x^2+\ell(\boldsymbol R)x}\right].
\end{aligned}
$$

Integrating $\boldsymbol x$ cancels the added quadratic term exactly, recovering the original particle weight.
The conditional field mean and covariance are

$$
\mathbb E[\boldsymbol x\mid\boldsymbol R]=-\frac{\boldsymbol\ell(\boldsymbol R)}{2a},
\qquad
\operatorname{Cov}(\boldsymbol x\mid\boldsymbol R)
=\frac{M}{2Ca}\left(I_M-\frac{\boldsymbol 1\boldsymbol 1^{\mathsf T}}{M}\right).
$$

The $a$ in this section differs from the PRL Born radius. Particle updates include every interaction, self term,
neighboring bead bond weight, and the fixed first bead. Each update sums M candidate positions, without enumerating all $M^8$ configurations.

A separate particle-only Gibbs control also integrates out the attractive field. Both samplers' energies were checked against direct evaluation of the original density action;
particle conditional probabilities were independently computed for real fields with std=5. Existing FFT and CL drift were unchanged.

Conditions follow [EXACT_CL_VALIDATION.md](EXACT_CL_VALIDATION.md): N_ref=4, one 4-bead chain, two C(+1), one each of SP(+3), SM(-1), and S,
L=1, chiN=2, zetaN=4, smearing radius=.1, and reflecting bonds. The first bead is fixed in the first cell.

- M=4/16/32 and E=0/100: real-field/particle seeds 1101,1201 and particle-only seed 2101, totaling 18 runs.
- Three additional M=4, E=25 runs directly match the historical native CL conditions.
- Each run: 2,000,000 sweeps, discard the first 200,000, save every 10 sweeps, yielding 180,000 samples. A sweep updates eight free particles and the conditional real field.
- Recorded quantities include height, species densities in every cell, local conditional field moments, and the mean and square of the sampled first cosine mode. Particle-only controls evaluate field moments analytically.
- SE is the larger of the existing 10/20/40-block and autocorrelation estimates. Species, positions, and moments are correlated; parameter comparisons sharing a seed are also not independent replicates.

Four-cell results were compared against the exact sum of 65,536 configurations. Every measured quantity across nine runs was within 2.781 estimated SE of its exact value.
For 16 and 32 cells, the two samplers were compared without exhaustive enumeration; there is no independent exact-sum reference at these sizes.
The largest difference between samplers across all runs is 3.049 combined SE. These descriptive statistics cover many correlated positions/observables;
they do not mean that every individual 2-SE interval passed. The maximum energy identity error is 1.25e-14, conditional probability error 7.78e-16,
and every sample conserves each species' mass.

P density in cell 1 at M=4, E=25:

| Calculation | Mean ± estimated SE |
|---|---:|
| Exact particle sum | .47375505 |
| Real-field/particle Gibbs, seed 1101 | .47113827 ± .00095049 |
| Real-field/particle Gibbs, seed 1201 | .47464444 ± .00083619 |
| Particle-only Gibbs, seed 2101 | .47479012 ± .00115897 |
| Historical native CL dt=.01, seed 71 | .45115654 ± .00333992 |
| Historical native CL dt=.01, seed 72 | .44947711 ± .00337511 |

Historical CL values were read from their original files without overwriting them. This compares the four-cell CL at its existing dt;
the four-cell dt→0 limit was not newly investigated.

Mean height $\langle z\rangle/L$ at $E=100$:

| 1D cells | Real-field/particle seed 1101 | Real-field/particle seed 1201 | Particle-only seed 2101 |
|---|---:|---:|---:|
| 4 | .360268 ± .000306 | .360473 ± .000374 | .360899 ± .000341 |
| 16 | .332038 ± .000318 | .331891 ± .000292 | .332504 ± .000319 |
| 32 | .327538 ± .000321 | .328409 ± .000294 | .328293 ± .000331 |

The exact four-cell height is .36049837. Changes from 16→32 cells exceed statistical errors, so continuum convergence is unverified.
The spatial representation of the anchor fixed in the first cell also changes with the grid.
This small system has eight free particles; it does not validate performance for an actual N=100 brush or the PRL model.

## 2. Recalculation of PRL Fig. 1

The target is Fig. 1 of [Duan, Agrawal & Wang, PRL 134, 048101 (2025)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.048101).
Neutral and z+=1/2/3 cases use N=100, b=1 nm, v=1 nm³, sigma=.1 nm⁻², a=.25 nm, lB=.7 nm, chi=.5, and Ib=.3 M.
Primary calculations use canonical monovalent counterions + GC salt and L=200 nm.
The graft source retains the existing .5b Gaussian width; bonds use a truncated bottom and reflecting top.
This numerical wall/graft representation is not assumed identical to the paper's delta source.

Mean field and the existing smoothed-LDA correlation approximation were solved separately.
For divalent/trivalent LDA, existing mixing failed to bring all residuals below 1e-7 even after 30,000 iterations.
A new driver solved the same equations using Newton–Krylov in $V=W/N$ and the shared self-energy $U$.
Equal Born radii give the exact relation $u_K=z_K^2U$. Equations, correlation strength, and acceptance tolerances were unchanged.

All seven cases converged at Nz=1600 and 3200. Final densities were recomputed with an independent scaled direct chain;
checks cover contour partition consistency, polymer/counterion mass, and final W/Poisson/self-energy/volume residuals.
Recorded final fields remain well away from the existing exponential clipping bounds.

### Numerical comparison with the paper's figure

The original Fig. 1 raster (1612×958) was extracted from the PDF. Colored line centers were read at `z/b=2,5,8,10,12`, where curves are separated.
Estimated reading uncertainty is about .004 in phi, not a statistical SE. This compares five interior positions,
not digitization of the complete curves or comparison with the authors' raw data.

Nz=3200 results at `z/b=5`:

| Curve | Paper phi (approx.) | Computed phi |
|---|---:|---:|
| Neutral | .5065 | .5066 |
| z+=1 mean field | .3144 | .3210 |
| z+=2 mean field | .3562 | .3491 |
| z+=3 mean field | .3824 | .3705 |
| z+=1 with correlations | .3392 | .3692 |
| z+=2 with correlations | .5817 | .5562 |
| z+=3 with correlations | .7379 | .7231 |

Maximum absolute differences across five positions for the correlated curves are .0300, .0298, and .0192, respectively.
At those positions, changes from Nz=1600→3200 are at most `1.98e-5` in all cases.
Thus neither the observed grid error nor digitization uncertainty explains the discrepancy.
Multivalent ions produce a denser brush than the neutral case, but exact curve reproduction is incomplete.

### Ensemble, box, and correlation-closure controls

A GC salt-only control without canonical counterions was run as a separate parameter case.
Its mean-field dashed curves agree with the paper to within .0027 at the five interior positions.
This motivates confirming the actual box/ensemble conditions behind Fig. 1; it does not replace validation of the SI's canonical counterion model.
Even for GC-only, maximum differences in the correlated curves remain .0212/.0088/.0228 for z+=1/2/3.
Selecting GC-only therefore does not match all correlated curves.

LDA controls enlarging the canonical box to L=400 nm at fixed dz=.125 nm change $2\langle z\rangle$
from 30.9787→31.3638 nm for $z_+=1$ and 14.3395→14.3667 nm for $z_+=3$.
This moment estimator is not identified with the Gibbs dividing-surface height in Fig. 2.
The separate Fig. 2 concentration scan is documented in the supplement.

The current nonlocal backend was also run separately. It omits the SI's Iex and retains Neumann Green-function boundaries and approximate smearing near walls;
it is therefore not labeled a full-SI result. For z+=1,2, it converged but maximum differences at the five positions were .0315/.0301,
leaving the LDA discrepancy unresolved. Direct Newton failed for z+=3.
A subsequent run continuously switching from LDA to the nonlocal equations converged at a final nonlocal fraction of 1, with all residuals at most 4.61e-9.
Its maximum interior difference decreased to .0081, without quantitatively reproducing all three valences.
Separate Green-function quadrature/grid limits were not investigated. The initial failure and all continuation checkpoints are preserved.

## 3. Outputs and remaining scope

**Additional update:** fixed-parameter Fig. 2–3 concentration scans and unit checks are documented separately in
[FIG23_PRL_REPRODUCTION.md](FIG23_PRL_REPRODUCTION.md) and the later SCFT section of
[ONED_PRL_REPRODUCTION.pdf](ONED_PRL_REPRODUCTION.pdf).
Run counts and validation numbers below refer to the earlier 1D/Fig. 1 campaign.

Of 49 SLURM submissions, 46 completed successfully (21 1D sampling and 25 PRL calculations).
Records of three initial failed PRL iterations are preserved; subsequent Newton/continuation runs converged for the same target equations.
No jobs from this campaign remain active. The maximum independent final chain density difference was 5.67e-15; contour partition error was 4.96e-14.
Sample counts, finiteness, final residuals, and protected-source hashes were checked.
Compilation checks for the three new Python files and `git diff --check` also passed.

- [1D comparison PNG](/home/yongdd/polymer/dh_salt_runs/oned_prl_20260907/positive_1d_comparison.png)
  / [PDF](/home/yongdd/polymer/dh_salt_runs/oned_prl_20260907/positive_1d_comparison.pdf).
- [PRL Fig. 1 comparison PNG](/home/yongdd/polymer/dh_salt_runs/oned_prl_20260907/prl_fig1_checked.png)
  / [PDF](/home/yongdd/polymer/dh_salt_runs/oned_prl_20260907/prl_fig1_checked.pdf).
- [Correlation-closure/ensemble comparison PNG](/home/yongdd/polymer/dh_salt_runs/oned_prl_20260907/prl_closure_comparison.png)
  / [PDF](/home/yongdd/polymer/dh_salt_runs/oned_prl_20260907/prl_closure_comparison.pdf).
- [All raw data](/home/yongdd/polymer/dh_salt_runs/oned_prl_20260907): SLURM commands/logs,
  checkpoints, profiles, RNG states, `comparison.json`, `prl_comparison.csv`,
  `paper_fig1_digitized.json`, paper/SI text, and source snapshots.
- [sample_positive_1d.py](sample_positive_1d.py): exact conditional Gibbs updates and checks.
- [run_prl_fig1_checked.py](run_prl_fig1_checked.py): strict final residual checks and Newton driver.
- [compare_oned_prl.py](compare_oned_prl.py): digitization, statistical comparisons, and standalone figures.

Existing production FFT, CL drift, `prl_scft.py`, and result files were unchanged during this campaign.
Additions comprise research/validation Python and documentation. Quantitative PRL reproduction requires resolving differences in actual box/ensemble conditions
and Green-function boundaries/approximations. These calculations do not establish which difference causes the full discrepancy.

<!-- END ONED_PRL_REPRODUCTION.md -->


---

# Source: POSITIVE_BRUSH_RESULTS.md

SHA-256: `55d6e36bb7b86862bfb019c890b37e8d633fd82ff84dc557191d03d51faf8922`

<!-- BEGIN POSITIVE_BRUSH_RESULTS.md -->
# N=100 brush: actual 1D sampling results — 2026-09-07

The earlier Fig. 2–3 scan used SCFT. These Monte Carlo calculations sample particle configurations and the real field in the attractive channel.
Samples of this finite 1D model were obtained; the PRL's 3D fluctuations and quantitative curves have not been reproduced.

[Distribution and implementation](POSITIVE_BRUSH_SAMPLING.md) · [LaTeX PDF](ONED_PRL_REPRODUCTION.pdf)

## Runs and validation

- Standard: 3 valences × 5 nonzero salt concentrations × 2 initial states + 2 salt-free runs = 32.
- Two particle-only controls for trivalent salt at .01/1 M; 8 grid/area controls. Total: 42 brush trajectories.
- Each brush: 200,000 warmup sweeps + 1,000,000 measurement sweeps. Scalars stored every 100 sweeps; profiles every 1,000 sweeps.
- Exception: z+=2, .01 M initially had split Rhat=1.09546 and relaxation from the compact start, so both seeds received 1,000,000 additional measurement sweeps. Tables use this fixed extension interval. The original interval was not pooled as an independent seed.
- Eighteen separate pure periodic salt baths. Activities were adjusted only during warmup and fixed during measurement.
- Exact 2-cell/2-bead sums were compared with 12 runs: two methods, two seeds, three valences.
- Maximum deviation from exact values: 2.6697 SE. Salt partition truncation bound < 3e-40.
- Maximum absolute detailed-balance/identity error: 1.288e-13; large-grid scaled energy error: 4.310e-15.
- Integer P/S occupancy and total charge are conserved in all trajectories. Maximum accumulated Q update error: 2.981e-12.
- Per-run height ESS: 10.06–79.36; maximum standard paired split Rhat 1.0391.

SE uses the larger of autocorrelation and 10/20/40-block estimates. Combining independent seeds also accounts for between-run scatter.
Height relaxes slowly: a million sweeps are not a million independent samples. Errors exclude finite-grid and model bias.

## Sample averages

h = 2∫z⟨ρP⟩ dz/(σN). Γ is the salt-excess ratio in the core above 0.5 of the mean polymer density peak.
Ψ = (⟨ψwall⟩−⟨ψfar⟩)h/(σNv). Nonlinear observables are computed from mean profiles, with block-jackknife errors.
Bulk density uncertainty and h/ψ covariance are included. Actual net charge (including C and anions) is recorded separately in analysis.json.

| z+ | Target salt [M] | Measured bulk [M] | h [nm] | Γ0.5 | Ψ |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 ± 0 | 58.5607 ± 0.3510 | 1.0000 ± 0.0000 | -30.8028 ± 0.4466 |
| 1 | 0.001 | 0.000995724 ± 9.4e-06 | 57.5166 ± 0.3555 | 0.9583 ± 0.0003 | -20.7814 ± 0.3666 |
| 1 | 0.01 | 0.0100481 ± 2.4e-05 | 54.6272 ± 0.2566 | 0.8576 ± 0.0011 | -13.1627 ± 0.3883 |
| 1 | 0.1 | 0.10016 ± 7.5e-05 | 43.8775 ± 0.1733 | 0.6312 ± 0.0012 | -4.7413 ± 0.2464 |
| 1 | 1 | 0.999995 ± 0.00023 | 28.4894 ± 0.2127 | 0.5417 ± 0.0016 | -0.4192 ± 0.1540 |
| 1 | 10 | 9.99608 ± 0.0011 | 23.5720 ± 0.1412 | 0.5580 ± 0.0042 | 0.1359 ± 0.1261 |
| 2 | 0.001 | 0.00101316 ± 7.4e-06 | 54.3083 ± 0.2232 | 0.8688 ± 0.0013 | -15.1694 ± 0.3467 |
| 2 | 0.01 | 0.00997499 ± 2e-05 | 46.0193 ± 0.2106 | 0.6495 ± 0.0034 | -7.1467 ± 0.2663 |
| 2 | 0.1 | 0.10019 ± 7.9e-05 | 33.0426 ± 0.1861 | 0.4067 ± 0.0038 | -1.8721 ± 0.2038 |
| 2 | 1 | 0.999946 ± 0.00019 | 24.8579 ± 0.1702 | 0.3984 ± 0.0024 | -0.2052 ± 0.1293 |
| 2 | 10 | 10.0017 ± 0.00084 | 23.0538 ± 0.1651 | 0.4186 ± 0.0061 | 0.0452 ± 0.1358 |
| 3 | 0.001 | 0.00100569 ± 9.4e-06 | 50.2135 ± 0.2903 | 0.7774 ± 0.0029 | -11.8280 ± 0.3145 |
| 3 | 0.01 | 0.0100335 ± 2.1e-05 | 39.2816 ± 0.3541 | 0.4856 ± 0.0056 | -4.5453 ± 0.3697 |
| 3 | 0.1 | 0.0998522 ± 5.3e-05 | 28.5975 ± 0.3301 | 0.3098 ± 0.0040 | -0.5994 ± 0.2435 |
| 3 | 1 | 1.00008 ± 0.00019 | 23.7963 ± 0.1887 | 0.3292 ± 0.0033 | 0.0296 ± 0.1224 |
| 3 | 10 | 10.0031 ± 0.00078 | 22.7973 ± 0.2936 | 0.3500 ± 0.0152 | -0.0342 ± 0.1406 |

## Grid/cross-sectional-area checks

| run | h [nm] | height ESS |
|---|---:|---:|
| check_area20_z3_r0.01_s6101 | 38.1583 ± 0.7654 | 20.9 |
| check_area20_z3_r0.01_s6201 | 37.5512 ± 0.6061 | 34.8 |
| check_area80_z3_r0.01_s6101 | 39.7251 ± 0.2355 | 49.7 |
| check_area80_z3_r0.01_s6201 | 39.5296 ± 0.3090 | 42.9 |
| check_grid800_z1_r0_s6101 | 58.7606 ± 0.3331 | 45.6 |
| check_grid800_z1_r0_s6201 | 58.8128 ± 0.9427 | 10.1 |
| check_grid800_z3_r0.01_s6101 | 39.1419 ± 0.3389 | 35.4 |
| check_grid800_z3_r0.01_s6201 | 39.2180 ± 0.2270 | 53.6 |

## Interpretation and remaining scope

Parameters: N=100, b=1 nm, σ=.1 nm⁻², v=1 nm³, χ=.5, ℓB=.7 nm, a=.25 nm. Standard A=40 nm², L=200 nm, M=400.
This explicit finite action has discrete bead positions and integer cell occupancy. It is not claimed identical to the existing CL or 9-bead validation action.
A=20/80 nm² uses 2/8 chains, respectively, to retain σ. Area sets 1D fluctuation strength and is more than a mesh-accuracy parameter.
Height decreases at the sampled concentrations. This restricted 1D measure has not quantitatively reproduced contraction/re-expansion or charge/potential reversal from the paper's 3D correlations.
Remaining checks include dilute-limit tails, mesh/box/area dependence at all concentrations, longer sampling, and comparison with full 3D theory/simulation.
Correct convergence of the original CL-FTS also remains unproven. Production FFT/C++/CUDA and the existing CL drift were unchanged in this work.

## Rerunning

Activate conda `polymerfts`. Numba was loaded from the existing local dependency directory:
`/home/yongdd/polymer/dh_salt_runs/exact_cl_diagnosis_20260907/python_deps`.
Add this directory and `devel/charged_polymers` to PYTHONPATH. Run calculations longer than one minute through SLURM.
Full run commands are preserved in `/home/yongdd/polymer/dh_salt_runs/positive_brush_20260907/jobs/*.sh`; job IDs are in `jobs.jsonl`.
The main runner is `run_positive_brush.py`: first use `--bulk` to create a bath summary, then pass it to the brush via `--bath`.
`--rho 0` needs no separate bath. The runner stores `checkpoint.npz` and RNG state and supports `--resume`.
Particle-only controls use `--mode particles`; exact-sum/detailed-balance checks use `check_positive_brush.py`.
Analysis: `python devel/charged_polymers/analyze_positive_brush.py /home/yongdd/polymer/dh_salt_runs/positive_brush_20260907 --prl-root /home/yongdd/polymer/dh_salt_runs/fig23_prl_20260907`.
This document is generated from analysis.json. Raw data are in samples.npz and summary.json.

<!-- END POSITIVE_BRUSH_RESULTS.md -->


---

# Source: POSITIVE_BRUSH_SAMPLING.md

SHA-256: `f204ba918b49387fbbb9660e65f275fc2d81c1ea92dadcaa0ef364d412897eb6`

<!-- BEGIN POSITIVE_BRUSH_SAMPLING.md -->
# Positive 1D brush sampling: implementation record

Actual N=100 stochastic results are recorded separately in
[POSITIVE_BRUSH_RESULTS.md](POSITIVE_BRUSH_RESULTS.md). The equations below
define exactly which finite model was sampled; they are not a claim of
equivalence to the existing CL code or to the full 3D PRL measure.

The requested continuation is stochastic sampling. The earlier Fig. 2–3
LDA SCFT sweep did not carry out that request. This document records the
replacement calculation; the SCFT files are retained as separate references.

The initial implementation keeps the previously requested one-dimensional
restriction. Its explicit finite-volume action has polymer bead coordinates
z only, and integrates the electrostatic Gaussian field exactly. Transverse
charge fluctuations are absent, so it is not the full three-dimensional PRL
measure even when N, b, sigma, v, chi and charge radii have their PRL values.
The transverse area A determines the strength of fluctuations and must be
reported and varied. No LDA self-energy is inserted into the sampler.

For M cells, dz=L/M, K=A dz/v an integer, p_j polymer occupancies and
s_j=K-p_j solvent occupancies, the positive weight is

$$
\pi\propto B(\mathbf R)\,e^{-U_e-U_\chi}
\prod_j\frac{1}{s_j!c_j!}
\frac{a_+^{n_{+,j}}a_-^{n_{-,j}}}{n_{+,j}!n_{-,j}!},
\qquad U_\chi=\frac{\chi}{K}\sum_j p_j(K-p_j).
$$

Polymer bead paths are labeled and B contains Gaussian bonds and the graft
source. The C occupancies have fixed total n_C=n_chain N. Salt occupancies
obey sum n_- = z_+ sum n_+. The a's are dimensionless single-cell activities,
not measured bulk concentrations. Solvent positions and the pressure
constraint have been summed explicitly through s_j=K-p_j and its factorial.

If F_ej is the cumulative Gaussian charge shape of a particle in cell j at
edge e, Q_e=sum_j F_ej[-p_j+c_j+z_+ n_+,j-n_-,j], then

$$U_e=\frac{2\pi\ell_B\,dz}{A}\sum_e Q_e^2$$

with zero normal electric field at the boundaries. The periodic bulk
reference instead uses (Q_e-mean Q)^2. Each update preserves exact charge
neutrality. This is an explicitly discretized planar electrostatic action;
Gaussian smearing, boundary conditions and self terms must stay the same
in its checks and in sampling.

An optional attractive real field x gives an exactly equivalent joint
particle/field sampler:

$$
e^{(\chi/K)\sum_j(p_j-\bar p)^2}
\propto\int_{\sum x=0} dx\,
e^{-K\sum_j x_j^2/(4\chi)+\sum_j p_jx_j},
\quad
x\mid p\sim\mathcal N\left(2\chi(p-\bar p)/K,\;(2\chi/K)\Pi\right).
$$

The fully integrated particle sampler provides a second implementation of
the same finite action. They must agree within sampling uncertainty.

Implemented checks: exact small-cell grand partition sums; detailed balance of
polymer/solvent exchanges, ion redistribution, neutral salt insertion and
deletion; direct versus incrementally updated electrostatic energy; exact
local P/S occupancy and total charge; independent seeds, equilibration,
autocorrelation, grid and area sensitivity. Bulk activities must be calibrated
against a separately sampled pure-salt reference before labeling an axis
rho_b [M]. Completion, uncertainties and remaining model limits are recorded
in the results document, not inferred from the number of sweeps.

The sampler mixes same-cell neutral salt insertions/deletions with proposals
that place the ions independently across the box. For the latter birth,
the reverse move selects one cation and z anions uniformly without replacement;
the Metropolis factor is

$$R_+=\exp(-\Delta U_e)\frac{(a_+a_-^z)M^{z+1}}
{(N_++1)\prod_{k=1}^z(N_-+k)}.$$

The full proposal ratio includes repeated-cell combinatorics; these cancel
against the occupancy factorials. This avoids waiting for rare same-cell
coincidences in dilute multivalent salt. Detailed balance of both proposal
types is tested independently against the complete finite-state weight.

Production FFT/C++/CUDA, the existing CL drift, and the earlier nine-bead
sampler are protected. Evidence directory:
`/home/yongdd/polymer/dh_salt_runs/positive_brush_20260907/`.

<!-- END POSITIVE_BRUSH_SAMPLING.md -->


---

# Source: RESULTS_BRUSH_CL.md

SHA-256: `dc04ad011228bb7bc90d2c22f011c70c5e92f28b4ebbb5a59a27ef3ee01ba4e9`

<!-- BEGIN RESULTS_BRUSH_CL.md -->
# Fluctuation Effects in a Charged Brush: First CL-FTS Results (Stage 2 Complete)

**Results report, 2026-08-16** — completion of the stage-2 campaign of
the PRL-2025 reproduction plan (Duan/Agrawal/Wang,
`references/2025_prl_duan.pdf`). Full pipeline: AM mean-field solver
(RESULTS_BRUSH_STATUS.md §3) → warm-started charged CL-FTS
(`BrushChargedCLFTS` + `dh_salt_runs/test_brush_cl.py`) → dt→0 and
seed-resolved fluctuation observables.

## 1. System and pipeline

Salted-regime grafted PE brush in an all-reflecting 8×8×64 box
(Lz = 8): z_P = −0.2 (α = 0.2), N = 100 discrete, φ_P = 0.05, z₊:1
salt (cation vf 0.05), χ = 0, ζN = 20, a = 0.2, E = 2500. CL at
dt ∈ {0.005, 0.0025}, n̄ ∈ {1e4, 1e5}, seeds {12345, 12346, 12347},
250k steps (50k equil), warm-started from the converged SCFT saddle
(`pb_z*_fields.npz`) with self-calibrated per-species re-gauge shifts.
Warm-start consistency: the CL initial state reproduces the SCFT
Γ_eff/h₁ to 4+ digits before noise acts.

Mean-field baseline (E = 2500): Γ_eff = 0.527 / 0.415 / 0.373 and
h₁ = 0.808 / 0.702 / 0.663 for z₊ = 1/2/3 — all Γ_eff > 0 (no
overcharging at mean field, consistent with the PRL's Eq. 7 statement;
values track their 1/(z₊+1) = 0.50/0.33/0.25 ordering).

## 2. Main result: fluctuations SWELL the salted-regime brush

Δh₁ = h₁(CL) − h₁(SCFT), dt→0 extrapolated, seed-averaged (n = 3),
n̄ = 1e4:

| z₊ | Δh₁ (dt→0) | significance |
|---|---:|---:|
| 1 | +7.1(3.4)×10⁻⁵ | 2.1σ |
| 2 | +8.9(2.2)×10⁻⁵ | 4.0σ |
| 3 | **+10.9(2.0)×10⁻⁵** | **5.3σ** |

- **Unanimous sign: 18/18 runs** (3 valences × 2 dt × 3 seeds) show
  h₁(CL) > h₁(SCFT); sign-test p = 3.8×10⁻⁶ independent of any error
  model.
- **Monotone valence dependence**: Δh₁(z₊=3)/Δh₁(z₊=1) ≈ 1.5.
- **1/√n̄ scaling**: n̄ = 1e5 values are 2.7–3.2× smaller (expected
  √10 ≈ 3.16) — a genuine loop-order fluctuation effect.
- **dt-robust**: Δh₁ is nearly dt-independent (e.g. z₁:
  +6.85e−5 → +6.98e−5 for dt 0.005 → 0.0025), so no integrator-bias
  contamination for this observable.

Relative effect: Δh₁/h₁ ~ 1.6×10⁻⁴ at n̄ = 1e4 — small in this
weak-charge/strongly-smeared corner, as expected at one-loop order with
α = 0.2 and aκ ≈ 0.4.

## 3. Secondary observable: effective brush charge

ΔΓ_eff = Γ(CL) − Γ(SCFT) is positive (less counterion excess in the
brush) in 14/18 runs (sign-test p = 0.015), with dt→0, seed-averaged
values +0.4/+1.6/+2.8 ×10⁻⁴ for z₊ = 1/2/3 — same ordering as Δh₁ but
only ~1σ per point against seed scatter (Γ is noisier than h₁; its
blocking errors are dominated by slow collective modes). Consistent
picture: the net fluctuation correction in this regime is repulsive —
slightly fewer condensed counterions, slightly taller brush.

Finite-dt lesson: the raw dΓ at dt = 0.005 UNDERESTIMATES the dt→0
value (integrator bias partially masks the physical effect); Δh₁ is the
clean observable.

## 4. Interpretation and relation to the PRL

The PRL's collapse-and-reexpansion physics is driven by ion
correlations at strong coupling (α = 1, multivalent, thin double
layers, ~point-like charges). This campaign measures the same
CL−mean-field difference in the *opposite*, weak-coupling corner
(salted regime, α = 0.2, aκ ≈ 0.4 smearing), where the net fluctuation
correction turns out to be **swelling**, not collapse — yet it already
carries the valence ordering (z₊ = 3 strongest). Candidate mechanism:
the chain's charge-fluctuation self-energy (repulsive in this regime)
outweighing smearing-suppressed ion-correlation attraction; an E- and
a-scan would separate the channels. Walking α up and a down along the
obstacle ladder of RESULTS_BRUSH_STATUS.md (§2.2–2.3: finer nz,
log-domain propagators) should continuously connect to the PRL's
collapse regime.

## 5. Data

`dh_salt_runs/`: `pb_z{1,2,3}.json` (+`_fields.npz`) mean-field;
`clbrush_*` (dt 0.005, seed 12345, n̄ 1e4/1e5), `clbrush2_*`
(dt 0.0025, seed 12345), `clbrush3_*` (seeds 12346/12347, both dt,
n̄ 1e4). All runs finite; |Q| stays ~e⁻⁴⁰ due to the CL random walk of
the gauge (pressure-mean) mode — harmless for densities (canonical
invariance) but pin it in future long runs.

<!-- END RESULTS_BRUSH_CL.md -->


---

# Source: RESULTS_BRUSH_STATUS.md

SHA-256: `3bd440f2773bec743924bd7822ee379b0530bfe51c1743f48c3119f3726cd831`

<!-- BEGIN RESULTS_BRUSH_STATUS.md -->
# Charged Brush CL-FTS (Stage 2): Infrastructure Validated, Mean-Field Solver COMPLETE

**Status report, 2026-08-14 (updated same day)** — stage 2 of the
PRL-2025 reproduction plan (Duan/Agrawal/Wang,
`references/2025_prl_duan.pdf`): grafted polyelectrolyte brush with
fluctuating ψ in a wall box. The infrastructure is built and
quantitatively validated; the strongly-charged brush exposed a ladder of
numerical obstacles — **all of which are now solved at the mean-field
level**: the AM-based charged-brush SCFT solver
(`brush_scft_charged.py`, `BrushChargedSCFT`) converges the salted-regime
brush to err ~ 1e-7 across an E-continuation ladder up to E = 2500 with
clean profiles (ringing ≤ 1e-6). See §3 for the solution set and the
first z₊ = 1 vs 3 mean-field physics.

Code: `brush_clfts.py` (`BrushChargedCLFTS`), `electrostatics.py`
(`ElectrostaticsReflecting`). Runs/probes: `dh_salt_runs/`
(`test_brush_wallsalt_oneloop.py`, `test_brush_charged.py`,
`probe_brush_saddle_*.py`, `bw_*.json`, `brush*_*.json`).

## 1. What is validated

### 1.1 Wall-box (DCT) ψ sector — one-loop DH: PASSED
All-reflecting 16³ box, 1:1 salt, χ=0. ⟨∂H/∂E⟩ vs the exact lattice
Gaussian sum over DCT-II modes (k_d = πn_d/L_d), dt→0 extrapolated:
**E=25: ratio 0.9996; E=6.25: 1.0022** (an initial E=6.25 outlier was a
rare-event seed; reruns normal). This validates the DCT transform pair,
the ETD integrator in the cosine basis, the per-mode noise
normalization, and H_elec — per-mode equipartition carries over from
the periodic case exactly as predicted.

### 1.2 Grafting + wall infrastructure: WORKS
`grafting_points` + delta-sheet `q_init` (1/dz at the z=0 cell) through
the complex CPU solver with all-reflecting BCs: mean φ_P =
volume_fraction, material conservation ⟨Σφ⟩=1, laterally uniform
wall-localized profile. NOTE: **CUDA is broken** for complex +
non-periodic (`CudaPseudo::upload_fourier_basis` memcpys
`d_negative_k_idx`, which exists only for periodic BC, unconditionally
for complex T) — mainline fix deferred; use `cpu-mkl` (10–15 ms/step at
8×8×32 with N=100).

## 2. Obstacle ladder (each root-caused)

### 2.1 Pressure-mode time-step limit with POLYMERIC species (fixed)
All brush runs at dt ≥ 0.02 oscillated/diverged — **including the
neutral brush** (charge irrelevant). Root cause: the w₊ relaxation
stability limit is dt·ζN·(density response) < O(1). All previous bulk
validation systems were single-segment species whose response carries a
factor ds = 1/N (dt=0.2, ζN=100 → effective 0.2: stable); an N=100
chain responds coherently (O(1)) → dt=0.05·ζN=100 = 5: unstable. The
graft delta sheet (φ≈3 in the wall cell) worsens the local factor ~3×.
**Fix: dt ≲ 0.002 at ζN=100, or ζN=20 with dt=0.005** — neutral brush
then converges smoothly and monotonically. This limit applies to the CL
Langevin dt of ANY charged-polymer run containing real chains, not just
brushes.

### 2.2 Double-layer grid resolution
At E ≥ ~1500 with Σz²φ̄·ds ≈ 2.4×10⁻³, κ⁻¹ = 1/√(E·Σz²φ̄·ds) approaches
dz = 0.25: the double layer becomes 1–2 cells wide while ψ swings by
O(10–100) code units across it → Gibbs ringing on the N=100 propagator →
negative densities → NaN. The physical PRL regime (κ⁻¹ ~ 0.03–0.2 R₀)
needs nz ≥ 128 for Lz = 8 (dz ≤ 0.0625), cost ~4× (still CPU-feasible:
~40 ms/step).

### 2.3 Propagator dynamic range at strong charging (open)
The per-chain Donnan energy is α·N·βeφ_phys: for α ≥ 0.5 at brush-like
compositions this is 20–100 kT → |Q_grafted| ~ e⁻²⁰ and propagators
spanning e³⁰⁺ across the box — double precision exhausts, seen as
growing negative-density ringing even on otherwise-converged
trajectories. The PRL's full-charge (α=1) system sits squarely in this
regime. Options: (a) start in the **salted regime** (high salt, α ≤
0.2: α·ψ_Donnan ≲ few kT — where the PRL's scaling analysis lives
anyway); (b) contour-renormalized / log-domain propagator bookkeeping
for large per-chain actions; (c) grand-canonical analytic Boltzmann
ions (the PRL's own formulation) to reduce stiffness.

### 2.4 Naive drift relaxation has a residual-invisible k=0 drift (open)
In the salted regime (α=0.2, φ_P=0.05, salt=0.05, E=625, E-ramped) the
deterministic saddle relaxation converges beautifully for ~24k
iterations (res 0.15 → 0.0095 monotone, h₁ ≈ 0.65, ψ stationary) —
**but ln|Q| drifts linearly the whole time** (10⁻⁸ → 10⁻²²) until
precision dies. The residual monitor std(λ) is blind to the spatially
uniform (k=0) component of the force, which never converges under plain
drift. Consequence: the warm start / mean-field reference needs a
proper saddle solver — Anderson mixing as in `scft.py`/`lfts_charged.py`
(which the naive drift loop here deliberately avoided), plus a correct
k=0/normalization treatment for grafted chains.

### 2.5 ψ-update damping with chains
The `Electrostatics` Newton preconditioner S_scr = Σz²φ̄ĥ² neither
includes the 1-segment ds factor (overdamps ions ×N) nor the
chain-coherent polymer response (α²φ_P·N — underdamps the polymer
channel ×N). For brush work use S_scr = α²φ_P·N·ĥ² + Σ_ions z²φ̄·ds·ĥ²
(implemented in the probe scripts; not yet folded into the class).

## 3. SOLVED: the AM-based charged-brush SCFT solver

`brush_scft_charged.py` (`BrushChargedSCFT(ChargedLFTS)`) resolves
obstacles 2.3–2.5 with four ingredients (each verified necessary):

1. **Anderson mixing** via `ChargedLFTS.find_saddle_point` (bc flows
   through LFTS → PropagatorSolver natively; grafted polymers
   re-registered with `grafting_points` + q_init). Replaces 30k+
   non-converging drift iterations with 30–650 AM iterations per
   continuation rung (err ≤ 1e-6).
2. **Chain-coherent ψ Newton preconditioner**: S_scr per species =
   z²φ̄·N_i·ds·ĥ² (N-segment chains respond ~N× a single segment; the
   segment-blind S_scr limit-cycles through the polymer channel).
3. **Adaptive per-species re-gauging** (obstacle 2.3): canonical
   densities are invariant under per-species constant shifts of W_t; the
   frozen zero-mode k=0 components otherwise park ln Q_P at ~ −50
   (double-precision exhaustion → the E ≳ 400 NaN wall). Shifts are
   updated each iteration to hold ln Q_solver ≈ 0; the Hamiltonian is
   corrected by +Σ_p vf_p·s_t. With this, ln Q₀ stays 0.000 across the
   whole ladder.
4. **Gaussian graft source** (width `graft_width` ≈ a, default 0.2)
   instead of a one-cell delta sheet, plus nz = 64 (dz = 0.125): kills
   the propagator ringing that produced NEGATIVE Q_P at E ≳ 450
   (min φ_P improves from −0.02 to −6×10⁻⁶).

**E-continuation ladder** [25, 100, 250, 400, 625, 1000, 1600, 2500],
salted regime (α = 0.2, φ_P = 0.05, salt = 0.05, ζN = 20, a = 0.2,
8×8×64, Lz = 8): every rung converges (z₊=1: 39–654 iters; z₊=3:
31–134 iters).

**First mean-field physics (E = 2500)** — consistent with the PRL's
mean-field (dashed-line) phenomenology:

| | z₊ = 1 | z₊ = 3 |
|---|---:|---:|
| h₁ (first moment) | 0.808 | 0.663 |
| ψ(wall) | −36.3 | −9.2 |
| φ_P(wall) | 0.287 | 0.364 |

Trivalent counterions screen far more efficiently (fewer needed +
e^{−3ψ·ds} response): the double layer is ~4× shallower and the brush
mildly contracts but **remains swollen — no collapse at mean field**,
exactly the PRL's point that collapse requires correlations.

Record runs: `dh_salt_runs/test_brush_pb.py` → `pb_z{1,2,3}.json`
(+ `pb_z*_fields.npz` for CL warm-start handoff).

## 4. Forward path

1. Port the same four ingredients into `BrushChargedCLFTS` (Gaussian
   graft source; per-species shifts frozen at their SCFT values during
   CL; ζN = 20, dt ≤ 0.005 per §2.1) and warm-start from
   `pb_z*_fields.npz`.
2. Salted-regime CL at n̄ = 1e4–1e5: z₊ = 1 vs 3 ion partitioning;
   the CL−SCFT difference is the correlation contribution (bulk
   counterpart already resolved at 120σ in RESULTS_MULTIVALENT.md).
3. Ladder toward PRL conditions: larger α with finer nz (thin double
   layers), stronger E.

The bulk multivalent correlation results (RESULTS_MULTIVALENT.md) are
unaffected and remain the validated stage-1 anchor.

<!-- END RESULTS_BRUSH_STATUS.md -->


---

# Source: RESULTS_FIG1_REPRO.md

SHA-256: `ced1eb4e3c1a190fbbf5798c071bf208ed152ac5b1d335fe55edbf9f77a40849`

<!-- BEGIN RESULTS_FIG1_REPRO.md -->
# Reproduction of PRL-2025 Fig. 1 Physics: Correlation-Induced Multivalent Brush Contraction by CL-FTS

**Results report, 2026-08-17** — first successful reproduction of a
core result of Duan, Agrawal & Wang, *PRL* **134**, 048101 (2025)
(`references/2025_prl_duan.pdf`) with this codebase's charged CL-FTS,
executed entirely inside the CL-stable operating envelope established
in RESULTS_BRUSH_STATUS.md / RESULTS_BRUSH_CL.md.

## 1. Claim being reproduced

The qualitative core of their Fig. 1 (and the Γ mechanism of Fig. 3):
at fixed conditions, **ion correlations selectively contract the brush
neutralized by multivalent counterions and drive extra counterion
condensation (Γ_eff toward zero), while mean-field electrostatics shows
neither**. (Their full collapse *below the neutral brush* lives at ~7 kT
per-trivalent-ion correlation strength — outside our current stable
envelope; see §4.)

## 2. Design: strong correlations inside the CL-stable envelope

All stability rules from the stage-2 campaign respected (ζN = 20,
dt = 0.005, n̄ = 1e4, AM warm start, re-gauge shifts, Gaussian graft):

- **N = 50** (l_B/R₀ = E/(4πN²√n̄) — halving N quadruples l_B at fixed E),
  α = 1 (fully charged backbone, as in the PRL), φ_P = 0.05.
- **High salt** (z₊:1, cation vf 0.02): keeps the Donnan potential small
  so the per-chain action stays ~7 (|Q| ≈ e⁻⁷; no precision wall even
  at full charge).
- **a_ion = 0.05 R₀** (a_P = a_S = 0.15), **E = 10⁵** → per-ion
  correlation u = z₊²l_B/(2√π a) ≈ **1.6 kT for z₊ = 3** vs 0.18 kT for
  z₊ = 1 (the 9× valence contrast that drives selectivity).
- Box 10×10×144 (Lxy = 1.0, Lz = 3.0): lateral dx = 2a_ion, dz = 0.021;
  κ⁻¹ ≈ 0.04 marginally resolved; ψ sampling accelerated with
  psi_dt_scaling = 20 (ETD exact for the linear part at any dt).
- Runs: mean-field baselines (`pb2_z{0,1,3}.json`, err < 1e-6, clean
  profiles) + CL warm-started 250k-step runs, 3 seeds for z₊ = 1, 3 and
  a neutral control (`clfig1_*.json`).

## 3. Result

| system | h₁(MF) | Δh₁ = h₁(CL) − h₁(MF) | ΔΓ_eff |
|---|---:|---:|---:|
| neutral (control) | 0.467 | −0.0008 | — |
| z₊ = 1 | 1.259 | −0.0025 ± 0.0007 | −0.0126 ± 0.0010 |
| z₊ = 3 | 0.857 | **−0.0222 ± 0.0017 (13σ)** | **−0.0509 ± 0.0016 (>30σ)** |

- **Valence-selective contraction**: Δh₁(z₊=3)/Δh₁(z₊=1) = **8.7 ≈ z₊² = 9**
  — the quantitative signature of the ion-correlation mechanism.
- **Correlation-driven condensation**: Γ_eff(z₊=3) drops 0.234 → 0.184
  (toward the overcharging direction of their Fig. 3); z₊ = 1 barely
  moves (0.668 → 0.655).
- **Controls**: the neutral run bounds all non-electrostatic systematics
  (finite dt, sampling) at 4% of the z₊ = 3 signal. Seed scatter is
  small (three independent noise realizations agree to ~10%).
- Contrast with RESULTS_BRUSH_CL.md: in the weak-correlation corner
  (α = 0.2, a = 0.2, u ≪ kT) the net fluctuation effect was a tiny
  *swelling* (+1e-4); at u₃ ≈ 1.6 kT the correlation attraction
  dominates and the sign flips to *contraction*, 200× larger — the
  crossover into the PRL's physics.
- **E-causality control (completed)**: z₊ = 3 at E = 2.5×10⁴ (u₃ =
  0.4 kT, coupling ÷4) gives Δh₁ = −0.0062 and ΔΓ = −0.0182 — the
  contraction shrinks 3.6× and the condensation 2.8× (vs the 4×
  coupling reduction; single seed). Together with the z₊² valence ratio
  and the null neutral control, all three causality legs point to the
  ion-correlation mechanism (`pb2_z3_E25k.json`,
  `clfig1_z3_E25k.json`).

![Fig-1-style reproduction](fig1_reproduction.png)

## 3b. Second rung: u₃ = 6.5 kT (2026-08-17)

N = 25, a_ion = 0.025, E = 5×10⁴ (16×16×144 box, Lxy = 0.8, Lz = 2.5;
`pb3_*`/`clstrong_*`): MF h₁ = 0.433 (neutral) / 0.988 (z₊=1) /
0.671 (z₊=3), all converged with positive profiles.

| observable | z₊ = 1 | z₊ = 3 |
|---|---:|---:|
| Δh₁ (CL − MF) | −0.009 | **−0.0755 ± 0.0010** (11% contraction) |
| Γ: MF → CL | 0.586 → 0.526 | 0.233 → **0.119** |
| lateral rel. std of column density | 0.12 | **0.17** (neutral: 0.015) |

Ladder so far (z₊ = 3): u₃ = 0.4 / 1.6 / 6.5 kT →
Δh₁ = −0.006 / −0.022 / −0.076 (≈3.5× per 4× coupling, mildly
sublinear); Γ has been driven halfway to overcharging. The lateral
inhomogeneity indicator jumps to ~10× the neutral baseline for charged
brushes (z₊ = 3 strongest) — a fluctuating-lateral-structure precursor
of their Fig. 4 (snapshot run `snap_z3_cols.json` records instantaneous
column-density maps for direct inspection). Below-neutral collapse
(gap 0.24) is not yet reached at this rung; the remaining paths are a
further coupling increase (a_ion → 0.0125, heavier grid) and/or the
paper's own Θ-solvent (χ ≈ 0.5 per monomer), which softens the neutral
reference and lowers the collapse threshold — the latter requires
extending the mean-field solver to saddle the real exchange field.

## 3c. CORRECTED results after the complex/non-periodic transform fix (2026-08-18)

Mainline commit c0ab6e0c revealed and fixed a solver bug that had
real-projected the complex propagator on every bond convolution
(imaginary part silently zeroed in the DCT/DST path on all platforms) —
all §3/§3b CL numbers were therefore approximations. Full GPU reruns of
the u₃ = 6.5 kT rung with the corrected solver (`clfix_*`; the CUDA port
gives ~12× at the solver level, though end-to-end step times on shared
nodes were comparable):

| observable (z₊=3, 3 seeds) | Im-projected (§3b) | **corrected CL** |
|---|---:|---:|
| h₁ (MF 0.671) | 0.596 | **0.581 ± 0.007** (Δh₁ = −0.091) |
| Γ_eff (MF 0.233) | 0.119 | **0.094 ± 0.007** (ΔΓ = −0.139) |
| lateral rel. std | 0.17 | **0.52** |
| max\|ψ\| during run | ~1000–6100 | ~2000–6000 (finite, 3 seeds coherent) |

z₊ = 1: Δh₁ = −0.013, ΔΓ = −0.083; neutral control: +0.003 (≪ the z₊=3
signal). **Every conclusion survives and strengthens**: the projection
was damping the fluctuations (and with them the correlation physics) —
the corrected contraction is 20% larger, Γ is now **60% of the way to
overcharging** (0.233 → 0.094), and the valence selectivity
(z₃/z₁ ≈ 7) persists. Lateral fluctuations triple, consistent with the
correct CL sampling of the full complex measure. The §3 first-rung and
E-control numbers remain projected-solver values pending rerun; the
qualitative claims they support (z² ordering, E-causality) are
re-established at this rung by the corrected data.

## 3d. Θ-solvent campaign and the coupling saturation (2026-08-19)

Adding the paper's Θ solvent (χN = 12.5 = 0.5·N) with the corrected
solver, quasi-saddle (n̄ = 1e9) references replacing the AM solver for
χ ≠ 0 (`theta_*`, `final_*`; all warm-started from the χ = 0 AM saddles
`pb3_*`/`pb4_*` at nz = 144/288):

| rung | u₃ | Δh₁(z₊=3) | Γ(z₊=3): MF → CL | h₁(CL) vs neutral CL |
|---|---:|---:|---:|---:|
| good solvent, E=5e4 | 6.5 kT | −0.091 | 0.233 → 0.094 | 0.581 vs 0.437 |
| Θ, E=5e4 | 6.5 kT | −0.122 | 0.207 → 0.082 | 0.508 vs 0.346 |
| Θ, E=1e5, nz=288 | 13 kT | −0.137 | 0.206 → 0.081 | 0.493 vs 0.344 |

Θ amplifies the contraction by ~34% at fixed coupling (softer neutral
reference, as in the PRL). But doubling E (u₃ 6.5 → 13 kT) barely moves
either observable: **the correlation effect saturates in E at fixed
a_ion and salt**. Two mechanisms, both consistent with the PRL
literature: (i) the screening length κ⁻¹ has reached a_ion, so the
smeared interaction ĥ² cuts off the correlation-peak modes (the same
cutoff physics quantified in RESULTS_LPF2008.md — more E buys nothing
without smaller a); (ii) counterion condensation at u₃ ≫ kT is already
nearly complete at this salt point (Γ pinned at ~0.08), and the
paper's overcharging (Γ < 0) lives in a specific BULK-correlation
window of ρb (their Fig. 3), not at arbitrary coupling at fixed salt.
Below-neutral collapse and Γ < 0 are therefore NOT yet reached
(remaining gap 0.149 at 90σ resolution).

Next probes: a salt scan at this rung (`scan_s*` — simultaneously the
h(ρb) axis of their Fig. 2, hunting the collapse valley/overcharging
window), and, orthogonally, a_ion → 0.0125 with nxy = 32 (8× grid; the
cutoff lever).

## 3e. Salt scan: the nonmonotonic h(ρb) of their Fig. 2 (2026-08-20)

Seven-point salt scan (cation vf 0.001–0.04, z₊ = 3, Θ, E = 1e5,
nz = 288; `pb5_*`/`scan_*` + the s = 0.02 rung), each point warm-started
from its own mean-field saddle:

| salt | h₁(MF) | h₁(CL) | Γ(CL) |
|---:|---:|---:|---:|
| 0.001 | 0.847 | 0.472 | +0.006(3) |
| 0.002 | 0.824 | 0.463 | +0.005(3) |
| 0.003 | 0.803 | **0.453** (valley) | +0.008(4) |
| 0.005 | 0.767 | 0.457 | +0.010(3) |
| 0.010 | 0.703 | 0.489 | +0.026(6) |
| 0.020 | 0.630 | **0.493** (reexpanded) | +0.081(7) |
| 0.040 | 0.559 | 0.473 | +0.132(13) |

(`fig2_reproduction.png`.) Findings:

1. **Nonmonotonic h(ρb) — the PRL's headline collapse-and-reexpansion —
   is reproduced in CL**: valley at s ≈ 0.003, reexpansion to s ≈ 0.02,
   then the salted-regime decrease; the valley-to-peak amplitude 0.040
   is ~5–10σ against seed/block errors. The mean-field curve is
   MONOTONIC over the same range (0.85 → 0.56) — the nonmonotonicity is
   purely a correlation effect, exactly their claim.
2. **Γ_eff plateaus at 0⁺ over the whole low-salt window**
   (0.005–0.010 with residual downward equilibration drift; second-half
   block averages reach +0.003–0.005): condensation saturates at
   complete neutralization but does NOT cross into overcharging at this
   smearing (a_ion = 0.025). Their Γ < 0 requires the bulk-correlation
   drive that our ĥ² cutoff caps — the a_ion → 0.0125 lever.
3. The collapse plateau (h ≈ 0.45–0.47) stays ABOVE the neutral brush
   (0.344): the condensed-ion-laden brush is bulkier than the bare
   neutral chain at this coupling. Below-neutral collapse needs the
   same stronger-correlation lever.
4. s = 0.08 diverged as predicted by the resolution rule
   (κ⁻¹ = 0.015 < 2Δz); the scan boundary is a numerics limit, not
   physics. Lateral inhomogeneity grows toward low salt (0.9 at
   s = 0.001), consistent with their Fig. 4 living at ~mM salt.

## 3f. The a_ion = 0.0125 ladder: stability matrix and the overcharging frontier (2026-08-23)

Halving the ion smearing radius (a_ion = 0.0125, 32×32×288 grid,
lateral dx = 2a) to release the ĥ² cutoff exposed a NEW stability
boundary — and its cure:

| attempt | knobs | outcome |
|---|---|---|
| a2 | u₃ = 26 kT (E=1e5), n̄=1e4, dt=0.005, s_ψ=20 | diverged |
| a2b | u₃ = 13 kT (E=5e4) | diverged |
| a2c | n̄ = 1e5 (10× weaker noise) | diverged |
| a2c | dt = 0.0025 | diverged |
| neutral control | same a_ion, uncharged polymer | **stable** |
| **a2d** | **psi_dt_scaling 20 → 1** | **stable** |

The destabilizer was the ψ acceleration factor: at small a_ion the
ions respond to high-k ψ roughness, and s_ψ = 20 (introduced to speed
ψ sampling at large E) drives the condensed-layer/ψ feedback into
runaway — independent of noise amplitude, timestep, and coupling.
**Rule: small ion radii require s_ψ = 1** (at the cost of longer ψ
correlation times).

Physics at the valley point (s = 0.003, u₃ = 13 kT): the single-seed
150k run showed Γ = −0.019 ± 0.021, but the **confirmation campaign
(3 additional seeds × 300k steps, a2e_*) settles it**: pooled
SECOND-HALF (equilibrated) Γ = **+0.0200 ± 0.0032** (four seeds agree
at +0.011…+0.026; the early negative excursion was an equilibration
transient). VERDICT: **no overcharging within the CL-stable envelope**
— correlations drive the brush to essentially complete neutralization
(Γ: 0.21 mean-field → ~0.02, a regime the PRL notes is impossible at
mean field) but not beyond it. h₁ = 0.444 ± 0.010 (4 seeds) stays
above the neutral 0.347: below-neutral collapse is likewise not
reached. The envelope is bounded on one side by the ĥ² cutoff
(a ≥ 0.025 saturates the coupling) and on the other by ψ-sector
stability (a = 0.0125 requires s_ψ = 1 and still carries slow
collective modes); reaching the PRL's Γ < 0 / below-neutral regime
within CL-FTS likely needs either the grand-canonical analytic-ion
formulation (their Eq. S24) or a dedicated strong-coupling
stabilization for the ψ sector — recorded as future work.

## 3g. Verdict on their theory: LDA vs exact CL at matched parameters
(2026-08-28)

With the PRL correlation SCFT reimplemented and validated
(RESULTS_PRL_SCFT.md), THEIR theory was applied to THIS campaign's
microscopic system (Theta rungs; unit mapping lB = E/(4 pi N^2
sqrt(nbar)), Born radii a_B = sqrt(pi) a_smear, sigma N v = phi_P Lz,
rho_b = s rho0, reflecting chain wall, CL graft sheet width 0.2 R0;
driver `dh_salt_runs/cl_compare.py`). Mean-field cross-validation:
3-9% absolute agreement (residual = zeta_N=20 compressibility and the
closed-box canonical ions vs GC — model-tier, cancels in differences).

Differential verdict (z+=3, corr - MF, per the fixed-grid differential
protocol):

| u3      | LDA dh1 | CL dh1 (3 seeds) | ratio | LDA dGamma | CL dGamma |
|---------|--------:|-----------------:|------:|-----------:|----------:|
| 6.5 kT  | -0.227  | -0.126           | 1.8x  | -0.22      | -0.125    |
| 13 kT   | -0.376  | -0.137 +/- 0.002 | 2.7x  | -0.39 (Gamma=-0.22, overcharged) | -0.126 (Gamma=+0.08) |

**The variational-Gaussian LDA lacks the coupling saturation of the
exact system**: doubling u3 doubles the LDA response but barely moves
the CL one (the h^2-cutoff saturation of 3d). Its overshoot grows
~linearly with coupling (1.8x -> 2.7x), and at u3 = 13 kT it predicts
strong overcharging where three independent CL seeds pin Gamma > 0.
Figure: `cl_vs_scft_fig1.png`. This completes the user-ordered final
step (CL-FTS vs the reimplemented SCFT) at the Fig-1 level: the paper's
theory is directionally right and quantitatively reliable only at weak
coupling (u <~ few kT); its strong-coupling amplitudes (deep collapse,
overcharging) are artifacts of the Gaussian approximation relative to
the exact sampling of the same model.

## 3h. Hard-wall correction and the final verdict (2026-08-28/29)

User-caught: the campaign's reflecting boxes lack the wall depletion
layer (profiles peaked AT the wall — unphysical for a grafted brush and
unlike the PRL). Fixed: `brush_clfts.py` now supports absorbing-z
(hard-wall) chains + off-wall graft sheet (`--wall hard`, sheet at 1 b);
psi stays Neumann (independent Electrostatics). Verified: neutral MF
profile now phi(wall)=0.0002 rising to its max at 0.47 R0. The full
E=5e4 and E=1e5 rungs were rerun hard-wall (hw5_*/hw_*, MF quasi-saddle
+ 1-3 CL seeds each); MF cross-validation vs the mapped PRL-SCFT: 1-6%.

Hard-wall verdict (z+=3, differentials):

| u3     | LDA dh1        | CL dh1 (3 seeds)  | LDA Gamma | CL Gamma |
|--------|----------------|-------------------|-----------|----------|
| 6.5 kT | -0.190 (2.0x)  | -0.095 +/- 0.011  | -0.060 (overcharged) | +0.089 +/- 0.011 |
| 12 kT  | -0.291 (2.3x)  | -0.129 +/- 0.026  | -0.248 | +0.092 +/- 0.005 |
| 13 kT  | NO SOLUTION (electrolyte runaway) | -0.131 +/- 0.023 | runaway | +0.088 +/- 0.010 |

The LDA runaway threshold was located by a deterministic u3 scan
(`lda_threshold.py`): physical solutions up to u3 = 12 kT, runaway
between 12 and 13 (u3* ~ 12.5). The u3 = 12 kT matched CL campaign
(E = 9.28e4, `u12_*`) pins the verdict at the strongest physical-LDA
point: exact contraction saturates (-0.095 -> -0.129 -> -0.131 across
6.5/12/13 kT) while the LDA deepens toward its fold (-0.190 -> -0.291
-> no solution), and the LDA's overcharging (Gamma to -0.25) is
contradicted by three-seed-robust CL positives throughout.

With the physical wall the Gaussian LDA fails EARLIER than with
reflecting walls: at u3 = 6.5 kT it already predicts overcharging where
the exact ensemble is robustly positive, and at 13 kT its electrolyte
sector has no physical solution at all (the saturated-condensation
runaway of RESULTS_PRL_SCFT.md 3b becomes the only branch), while CL
remains well-behaved and saturating. Figure `cl_vs_scft_fig1.png`
(hard-wall data). The differential CL verdict is wall-robust
(reflecting: -0.126/-0.137; hard: -0.095/-0.131).

## 4. What is and is not reproduced

Reproduced (this work): correlation-induced, valence-selective brush
contraction + enhanced condensation, absent at mean field — the
mechanism and ordering of the PRL's Fig. 1/Fig. 3, at ~2.6% contraction
amplitude set by our u₃ ≈ 1.6 kT.

Not yet: their full quantitative regime — collapse *below the neutral
height* and Γ < 0 (overcharging proper) need u₃ ~ 7 kT (their Born
radius 2.5 Å ≈ 0.025 R₀ with l_B = 0.7 nm), i.e. another ~4× in
coupling. The path is the established ladder: a_ion 0.05 → 0.025 with
nz 144 → 288 (grid cost ×2, still CPU-feasible), E ↑, plus a Θ-solvent
χN if the below-neutral comparison is wanted. Stability-wise nothing
new is required — the high-salt design keeps the per-chain action
bounded independent of u₃.

## 5. Data

`dh_salt_runs/`: `pb2_z{0,1,3}.json` (+`_fields.npz`),
`clfig1_z{0,1,3}_s*.json`, figure
`devel/charged_polymers/fig1_reproduction.png`. Drivers:
`test_brush_pb.py`, `test_brush_cl.py` (with `--a_ion`,
`--psi_dt_scaling`, lateral warm-start broadcast).

## 3i. Hard-wall salt scan: Fig. 2/3 rerun (2026-08-29)

The §3e/3f reflecting-wall campaigns rerun with the corrected hard-wall
box (`--wall hard`: absorbing z faces + graft sheet at 1b), same grid
convention as the §3h verdict campaign (16×16×288, lxy = 0.8, lz = 2.5,
E = 1e5, u₃ ≈ 13 kT, a_ion = 0.025, 3 seeds/point, 60k steps;
`hs_*` + the existing `hw_*` s = 0.02 rung):

| salt | h₁(MF) | h₁(CL, 3 seeds) | Γ(CL) |
|---:|---:|---:|---:|
| 0.001 | 0.954 | 0.637 ± 0.016 | +0.008(9) |
| 0.002 | 0.938 | 0.674 ± 0.011 | +0.011(10) |
| 0.003 | 0.924 | 0.634 ± 0.011 | +0.012(9) |
| 0.005 | 0.900 | 0.665 ± 0.035 | +0.025(12) |
| 0.010 | 0.856 | 0.681 ± 0.036 | +0.050(14) |
| 0.020 | 0.803 | 0.672 ± 0.032 | +0.088(16) |
| 0.040 | 0.751 | 0.692 ± 0.013 | +0.144(27) |

Hard-wall neutral CL reference: h₁ = 0.623. Findings (honest update of
the §3e claims):

1. **The correlation collapse vs mean field is large and robust**
   (Δh₁ = −0.32 at s = 0.001, shrinking to −0.06 at 0.04); the MF curve
   stays monotone-decreasing while CL is already collapsed at low salt.
2. **The reflecting-era "5–10σ nonmonotonic valley" does NOT survive
   the wall fix at this statistics.** The CL curve is nearly flat:
   valley 0.634 ± 0.011 at s = 0.003 vs reexpanded 0.692 ± 0.013 at
   0.04 is only ~3σ, and the s = 0.002 point (0.674 ± 0.011) breaks a
   smooth descent. Verdict downgraded: weak/marginal nonmonotonicity,
   not the deep collapse-and-reexpansion of their Fig. 2.
3. **The collapsed brush now sits AT the neutral height** (valley
   0.634 ± 0.011 vs neutral 0.623) — closer to the PRL picture than
   the reflecting-era "plateau far above neutral", but still never
   below it.
4. **Γ rises monotonically with salt and stays positive everywhere**
   (+0.008 → +0.144); the low-salt plateau is complete neutralization
   (Γ = 0⁺), no overcharging — consistent with §3h at matched coupling.
5. Lateral inhomogeneity persists at low salt (instantaneous column
   rel. std 0.36–0.44; box-scale single cluster in 0.8 R₀ boxes) —
   the Fig. 4 precursor survives the hard wall
   (`lateral_snapshots_z3.png`, 6 snapshots/run via `--snaps`).

Data: `dh_salt_runs/hs_*` (+ `launch_hs.sh`); figures
`fig2_reproduction.png`, `lateral_snapshots_z3.png` (both now
hard-wall). The a_ion = 0.0125 rung (`ha2_*`, Fig. 3 orange curve) —
results appended below when complete.

### 3i-b. 150k-step extension: the valley does not exist (2026-08-29)

The 60k campaign's ~3σ ambiguity resolved by rerunning all 7 salt
points at the OLD campaign's statistics (150k steps, production 120k,
24000 samples, 3 seeds; `hsl_*`):

| salt | h₁(CL, 150k) | Γ(CL, 150k) |
|---:|---:|---:|
| 0.001 | 0.640 ± 0.016 | +0.009(2) |
| 0.002 | 0.642 ± 0.006 | +0.006(3) |
| 0.003 | 0.643 ± 0.012 | +0.010(2) |
| 0.005 | 0.660 ± 0.005 | +0.017(4) |
| 0.010 | 0.662 ± 0.015 | +0.041(7) |
| 0.020 | 0.674 ± 0.003 | +0.070(6) |
| 0.040 | 0.683 ± 0.006* | +0.137(6)* |

(*2 seeds: seed 12347 hit a late-time ψ excursion after 120k steps —
the s = 0.04 point sits at the κ⁻¹ ≈ 2Δz resolution boundary, same
mechanism that killed s = 0.08 in the reflecting era; the two
surviving seeds agree to 0.0003.)

**Verdict, final:** the hard-wall exact ensemble has NO valley and NO
salted decrease — h₁(s) is monotone nondecreasing: a low-salt plateau
at 0.640–0.643 (the neutral floor + ~0.02) followed by smooth
reexpansion. The 60k zigzag (s = 0.002 at 0.674) was seed noise, and
the reflecting-era "5–10σ valley at s = 0.003" was an artifact of the
unphysical wall convention, not of statistics. Γ is monotone positive
(+0.006 → +0.137), low-salt complete-neutralization plateau confirmed
at 4× statistics. The PRL Fig. 2 nonmonotonic h(ρb) is therefore NOT
reproduced by the exact ensemble at matched coupling in this window —
only the collapse-to-neutral-floor and reexpansion halves survive.

### 3i-c. Their theory on our system, salt axis: the Fig. 2 shape is the LDA (2026-08-29)

To attribute the Fig. 2 disagreement, THEIR Gaussian-LDA theory (Part I
implementation, hard wall, mapped parameters) was run over the same
salt window at u₃ = 12 kT (`lda_salt_scan.py`, `lda_*/ldb_*`):

| salt | LDA h₁ | LDA Γ | exact CL h₁ (150k) |
|---:|---:|---:|---:|
| 0.001 | RUNAWAY | — | 0.640 |
| 0.002 | **0.401** (below neutral 0.589) | −0.109 | 0.642 |
| 0.003 | RUNAWAY | — | 0.643 |
| 0.005 | RUNAWAY | — | 0.660 |
| 0.010 | 0.471 | −0.228 | 0.662 |
| 0.020 | 0.502 | −0.248 | 0.674 |
| 0.040 | 0.554 | −0.224 | 0.683 |

**On the same microscopic system and window, the LDA reproduces the
paper's Fig. 2 structure** — deep below-neutral collapse, salt-driven
reexpansion, overcharging (Γ < 0 everywhere it converges) — and has NO
solution over most of the low-salt regime (electrolyte runaway; the
salt-axis face of the u₃* ≈ 12.5 kT fold of §3h; convergence near the
fold is fragile — s = 0.002 holds a marginal solution while 0.003/0.005
run away even under a 120k-iteration adiabatic ramp). The exact
ensemble on the same system shows none of this. Conclusion: the
difference between our Part II Fig. 2 and the paper's is not a
reproduction failure but the Gaussian-approximation error itself.
Figure: `fig2_reproduction.png` (three-curve overlay).

### 3i-d. CORRECTION: the §3e/3i scans were the wrong axis for Fig. 2 (2026-08-30)

All previous salt scans (reflecting AND hard-wall) had the brush
counterions THEMSELVES trivalent (`charges["C"] = zp`), so the brush is
multivalent-collapsed already at s → 0 and added salt can only screen:
a monotone axis BY CONSTRUCTION. The PRL Fig. 2 axis is different —
monovalent counterions + 1:1 background salt, with multivalent ions
arriving ONLY via the added salt: extended (osmotic, low salt) →
collapse (multivalent condensation) → reexpansion (salted). The §3i-b
"no valley — refuted" verdict therefore over-reached: it refutes the
valley on OUR axis (where their LDA also shows no valley in-window),
but does not test the paper's axis. New campaign `f2_*`
(--z_counter 1 --salt_bg 0.005, trivalent salt vf 1e-4 … 2e-2,
6 pts × MF + 3 seeds, hard wall, E = 1e5): results below.

### 3i-e. Hard-wall a_ion = 0.0125 rung (Fig. 3 orange, 2026-08-30)

Halved ion smearing (32×32×288, psi_dt_scaling = 1, E = 5e4, 150k
steps, 2 seeds; `ha2_*`):

| salt | Γ(MF) | Γ(CL) | h₁(CL) |
|---:|---:|---:|---:|
| 0.003 | +0.281 | **−0.017 ± 0.017** | 0.63–0.70 |
| 0.020 | +0.238 | +0.043 ± 0.019 | 0.66–0.68 |

At doubled per-ion coupling the low-salt point sits AT zero
(−0.017 ± 0.017): complete neutralization, and at most MARGINAL
overcharging — a sign flip within 1σ, vs the LDA's robust −0.25 at
comparable coupling. The reflecting-era +0.020(3) moves down slightly
under the hard wall. More seeds would be needed to resolve a genuine
sub-zero dip; the exact ensemble clearly saturates at neutrality
rather than overshooting into the LDA's deep overcharging.

### 3i-d results: the true axis reproduces the Fig. 2 mechanism (2026-08-30)

`f2_*` campaign (monovalent counterions, salt_bg = 0.005, trivalent
salt scanned; hard wall 3D 16×16×288, E = 1e5, 60k steps, 3 seeds;
Γ recomputed charge-weighted — the JSON gamma_eff is wrong for
z_counter ≠ zp):

| s_tri | h₁(MF) | h₁(CL) | Γ_cw(CL) |
|---:|---:|---:|---:|
| 0.0001 | 1.098 | 1.071 ± 0.012 | +0.53 |
| 0.0003 | 1.093 | 1.078 ± 0.013 | +0.54 |
| 0.001 | 1.077 | 1.050 ± 0.009 | +0.49 |
| 0.003 | 1.036 | 0.990 ± 0.008 | +0.41 |
| 0.01 | 0.927 | 0.828 ± 0.013 | +0.22 |
| 0.02 | 0.843 | 0.745 ± 0.005 | **+0.17 (min)** |
| 0.04 | 0.769 | 0.709 ± 0.010 | +0.19 |
| 0.06 | 0.737 | 0.701 ± 0.009 | +0.22 |

Findings:
1. **The Fig. 2 mechanism IS reproduced by exact CL on the correct
   axis**: extended osmotic brush (1.07) → trivalent collapse → valley
   floor 0.70 (slope → 0 between 0.04 and 0.06: −0.008 ± 0.013).
   Collapse depth 35% (paper: ~64%). CL sits below MF at every point
   (correlation-enhanced collapse), gap growing to 0.10.
2. **Γ is nonmonotonic with a positive minimum** at s ≈ 0.02
   (+0.17) — the paper's Fig. 3 valley SHAPE, without the sign flip.
3. The reexpansion upturn lies just past the model's ionic-resolution
   boundary (κ⁻¹ ≲ 2Δz beyond s ≈ 0.06); within the stable window we
   resolve the collapse and the valley floor.
4. The valley floor (0.70) stays ABOVE the neutral brush (0.623) —
   the paper's below-neutral undershoot remains exclusive to the
   Gaussian LDA (§3i-c overlay `lda_overlay.png`).
Figure: `fig2_reproduction.png` (h₁ + Γ panels).

### 3i-f. Valence series + their theory on the true axis: final Fig. 2 (2026-08-30)

Full valence series on the true axis (hard wall, 3D, E = 1e5; z₊=2:
`f2z2_*` 2 seeds, z₊=1 control: `f2z1_*`; low-salt x-extension to
vf 1e-6). All valences share one extended plateau within ±2%
(h₀ = 1.068 = mean over all valences' points at s ≤ 1e-4):

| s | z₊=1 CL | z₊=2 CL | z₊=3 CL |
|---:|---:|---:|---:|
| 1e-6…1e-4 | 1.059 | 1.042–1.065 | 1.071–1.081 |
| 0.003 | 1.059 | 1.026 | 0.990 |
| 0.02 | 1.009 | 0.870 | 0.745 |
| 0.06 | 0.914 | 0.773 | 0.701 |

Valence-ordered collapse (z₊=3 > 2 > 1; z₊=1 tracks MF = pure
screening, no correlation collapse) — the paper's central mechanism,
in the exact ensemble.

Their theory on THIS system, same axis (`lf2_u12/u13_*`, mapped
PRLBrushSCFT — note its counterion sector is GC-reservoir vs our
canonical cloud; MF cross-validation on this axis is 6–12%, worse
than the old axis's 1–6%):

- u₃ = 12: NO solution (runaway) for s ≤ 0.003 — the 6 sampled points
  spanning 3.5 decades of the low-salt window; solutions at 0.01–0.06
  sit at log(h/h₀) = −0.32 … −0.23 (below its neutral, marginally so
  at the 0.06 endpoint) with Γ = −0.17 … −0.25.
- u₃ = 13 (the CL campaign's coupling): loses s = 0.01 as well.

Final Fig. 2 verdict (figure `fig2_reproduction.png`, paper axes):
exact CL reproduces the mechanism (plateau → ordered collapse →
floor, nonmonotonic Γ) at depth −0.18, above neutral; their theory on
the same system overshoots to −0.32/below-neutral/Γ<0 where it exists
and has no solution at all over most of the paper's own x-range. The
paper's −0.58 deep valley is unreachable for the exact ensemble and
unreachable for their own theory on our system — it lives in their
stronger-brush regime (N=100) where only their approximation has been
evaluated.

### 3i-g. Ensemble mapping + corrected ionic environment: final attribution (2026-08-31)

Two mapping corrections complete the Fig. 2 story:

1. **x-axis = MEASURED reservoir concentration.** The canonical box's
   nominal SP vf is NOT the experiment's bulk ρb: counterion leakage
   raises the bulk monovalent to 0.034 vf (N=25) / 0.075 (N=50) — 7-30×
   the nominal salt_bg — and the brush depletes bulk trivalent by up to
   1300× at low s (N=50 nominal 3e-3 → measured 2.3e-6). Plotted vs
   measured bulk, the N=50 stronger brush DOES collapse ~1 decade
   earlier than N=25 (the nominal-vf axis had hidden it). N=50 probe:
   stable (|Q| ~ 1e-33 with re-gauge, ψ finite), extended h=1.39 fits
   lz=3.

2. **The mapped LDA had NO monovalent background** — `set_bulk()`
   recomputes Ib from (multivalent, c1) and discards the constructor
   `Ib_molar`; c1 was 0. **The §3i-f "no solution over 3.5 decades"
   was therefore a MAPPING ARTIFACT** (unscreened electrolyte), not a
   property of their theory on this system. With c1 = measured bulk
   (0.034 vf → set_c1(139.5)): no runaway anywhere, MF cross-validation
   improves to ~4% at low salt, and their theory produces its FULL
   phenomenology on our system: extended → deep valley (log(h/h0) =
   −0.32 at ~0.1 M, BELOW neutral) → reexpansion; Γ: +1.0 → 0 (at
   ~0.03 M) → −0.24 (overcharging window) → recovery. (The OLD-axis
   runaway/threshold results of §3g-3h stand — that system genuinely
   has no monovalent cations, c1=0 was correct there.)

**Final matched-ionic-environment attribution (fig2_reproduction.png):**
on identical systems and identical reservoir axes, their Gaussian
theory collapses ~1 decade EARLIER (onset ~3e-3 vs ~3e-2 M), ~1.75×
DEEPER (−0.32 vs −0.18), undershoots the neutral brush, and
overcharges — the exact ensemble does none of these. The remaining
distance to the paper's original curve (onset 1e-5 M, depth −0.58) is
brush strength (their N=100, denser graft): the N=50 exact probe moves
the onset left as expected while staying shallow. Every qualitative
gap between our exact Fig. 2 and theirs is now accounted for as
approximation (onset advance, depth, sign of Γ) plus system strength
(absolute onset position). Data: `lf2_c1fix_*`, `n50_*`; drivers
`lda_f2_scan.py` (--ib_molar → set_c1).

### 3i-h. N=50 strong-brush three-way judgment: FINAL (2026-09-04)

Full campaign complete (`n5f_*`: 8 salt points x (quasi-saddle MF + 2
CL seeds) + neutral pair; tall box lz=6, 32x32x704, u3=12.9 kT, 90k
steps; `l50b_*`: their theory at measured c1=0.040 vf background):

| s_nom | h1(MF) | h1(exact) | bulk rho_b [M] | G_cw(exact) |
|---:|---:|---:|---:|---:|
| 1e-5 | 1.876 | 1.672(37) | 1.4e-5 | +0.67 |
| 3e-4 | 1.845 | 1.673(4) | 2.3e-4 | +0.66 |
| 1e-3 | 1.775 | 1.583(8) | depleted | +0.62 |
| 3e-3 | 1.577 | 1.468(10) | depleted | +0.53 |
| 6e-3 | 1.431 | 1.320(9) | depleted | +0.40 |
| 1e-2 | 1.334 | 1.111(5) | 9.2e-3 | +0.23 |
| 2e-2 | 1.171 | 0.941(18) | 0.15 | +0.11 |
| 4e-2 | 1.017 | 0.831(8) | 0.60 | +0.08 |

Exact neutral: 0.564 (CL) / 0.573 (MF) — chain fluctuations negligible
at nbar=1e4, as designed. Their theory (measured-c1): valley
log(h/h0) = -0.44 at ~0.07-0.14 M, overcharging to -0.25, reexpansion.

**FINAL VERDICT (fig2_threeway.png)**: doubling the brush strength
does not change the structure of the disagreement —
1. exact collapse deepens (-0.18 -> -0.30) but remains 40-50% of the
   Gaussian prediction at every measurable x;
2. exact stops 0.17 above its own neutral floor (-0.47); their theory
   reaches its neutral line; the paper's N=100 below-neutral undershoot
   is the next increment of the same uncontrolled feedback;
3. exact Gamma stays positive everywhere (+0.67 -> +0.08); their
   theory overcharges to -0.25;
4. onset-region canonical boxes titrate ALL trivalent (bulk below
   detection) — open-ensemble x for those points is an upper limit.
Low-salt extended-plateau MF/CL absolute heights carry a residual
upward equilibration drift (60k->90k averages still rising ~5%); the
collapse-region points are drift-free. Mapped-theory MF cross-check:
1.652 vs 1.876 (12%) — the GC-vs-canonical residual documented in
3i-g remains the dominant mapping systematic.

### 3i-i. The c-interpolation: exact sampling = HALF the Gaussian correction (2026-09-04)

Scaling their self-energy by c in F = F_MF + c*F_1loop (`c50_*`,
`lda_n50c_scan.py`; c-stage h,Gamma recorded during the ramp) produces
a family of curves interpolating MF (c=0) -> full PRL SCFT (c=1).
**The exact CL-FTS curve lies on c = 0.5 within error bars across the
entire collapse region** (x=0.15 M: exact -0.250 vs c=0.5 -0.254;
x=0.60 M: -0.299 vs -0.29; fig2_cfamily.png). Interpretation: at
u3 = 12.9 kT the non-Gaussian (higher-loop) terms cancel ~half of the
one-loop correlation free energy. Consequences:
1. NO physical parameter setting makes their theory match exact
   sampling — after the full mapping (3i-g table) the only dial that
   closes the gap is the unphysical truncation weight c ~ 0.5.
2. c* = 0.5 is an empirical, coupling-dependent number (must -> 1 as
   u3 -> 0); it quantifies the approximation error directly.
3. Practical prescription for their method at strong coupling:
   half-weighted self-energies reproduce the exact equation of state
   of this system remarkably well.

### 3i-j. c*(u3): the confirmed fraction of the Gaussian correction (2026-09-05)

For each coupling rung, c* is the self-energy weight at which
PRL-SCFT(c) reproduces the exact CL h1 (old-axis s=0.02 system;
c-ramps `cstar_u*.json`, new weak rungs u3=3, 4.5 `u3_*`/`u4p5_*`):

| u3 [kT] | exact h1 | c* |
|---:|---:|---:|
| 3.0 | 0.749(8) | 0.59 ± 0.11 |
| 4.5 | 0.718(19) | 0.64 ± 0.12 |
| 6.5 | 0.708(6) | 0.51 ± 0.04 |
| 12.0 | 0.674(18) | 0.46 ± 0.06 |
| 13.0 | 0.672(23) | 0.46 ± 0.07 (ramp @12) |

**c* ~ 0.5-0.6 across the ENTIRE tested range, mildly decreasing with
coupling — the naive weak-coupling recovery (c* -> 1) is NOT visible
even at u3 = 3 kT.** Readings: (i) 3 kT per trivalent ion is still not
weak — the one-loop-exact limit lies below the couplings where the
collapse physics exists at all; (ii) at the weak end part of the
exact-vs-MF difference may not have the LDA's functional form (the
chain charge-fluctuation self-energy that produced net SWELLING in the
alpha=0.2 corner acts oppositely), so c* conflates form error with
magnitude error there. Practical summary: over the whole
collapse-relevant window their Gaussian correction should be halved
(c ~ 0.5); nowhere in this window is the full correction (c=1)
supported by exact sampling. Figure: `cstar_vs_u3.png`.

**3i-j addendum — two-observable c* (2026-09-06).** c* was defined by
matching a single observable (h1); extracting it independently from
the condensation Gamma (same runs, same c-ramps) gives:

| u3 [kT] | c*_h | c*_Gamma |
|---:|---:|---:|
| 3.0 | 0.59(11) | 0.62(11) |
| 4.5 | 0.64(12) | 0.62(5) |
| 6.5 | 0.51(4) | 0.43(2) |
| 12.0 | 0.46(6) | 0.24(1) |
| 13.0 | 0.46(7) | 0.26(1) |

At weak coupling the two agree (c* ~ 0.6 is observable-independent —
a genuine effective theory). At strong coupling they SPLIT (0.46 vs
0.24): no single rescaling fixes both observables — the Gaussian
truncation carries a FORM error, not just a magnitude error. Reading:
at matched c the LDA over-condenses relative to how much it contracts;
equivalently the exact ensemble converts condensation into collapse
more efficiently — consistent with non-Gaussian ion-bridging
attraction that a single-ion self-energy cannot represent by
construction. c* ~ 0.5 therefore stands as an h-calibrated practical
prescription, with the c*_h/c*_Gamma split as the quantitative marker
of where (u3 > ~6 kT) the theory stops being fixable by rescaling.

### 3i-k. CL correctness diagnostics at the strongest rung (2026-09-06)

Does CL formally converge to the right measure at u3 = 13 kT? Standard
lattice-QCD criteria applied to a dedicated rerun of the hw rung
(s=0.02, 6000 production samples recording instantaneous h1, Gamma,
max|psi-drift| = max|lap(psi)/E + rho_c|, max|psi|; `cl_diag_run.py`,
`cldiag_u13.json`, figure `cl_diag_tails.png`):

| quantity | tail form (CCDF fit, 1e-3..1e-1) | reading |
|---|---|---|
| max\|psi-drift\| | **EXPONENTIAL-like** (exp resid 0.14 < power 0.19) | the Nagata-Nishimura-Shimasaki drift criterion PASSES: no indication of boundary-term contamination |
| inst. h1 | power-ish, alpha ~ 3.5 | mean and variance finite; ratio-estimator tails, averages safe |
| inst. Gamma | power-ish, alpha ~ 2.5 | variance marginal — quantifies why Gamma is the slow observable |
| max\|psi\| | steep power, alpha ~ 4.8 | benign excursions |

Consistency: the diagnostic run's h1 = 0.704 sits inside the hw-seed
scatter (0.672 +- 0.032 x sqrt(3)); half-run mean drift -0.04 (h1) /
+0.09 (Gamma) is within the block scatter used for the campaign error
bars (slow collective modes, already priced in via multi-seed errors).

**Verdict**: the key formal criterion (exponential drift-magnitude
decay) is satisfied at the strongest coupling used anywhere in this
comparison, upgrading "CL as ground truth" from continuity argument
(one-loop 0.1% at weak coupling, LPF2008 at moderate) to a directly
checked property at the operating point. Caveats kept: one run, one
coupling; observable tails are heavy enough that Gamma error bars
require the multi-seed/block treatment we already use.

<!-- END RESULTS_FIG1_REPRO.md -->


---

# Source: RESULTS_LPF2008.md

SHA-256: `3f6caed0eb09e9ab555cb9600bfc5ff53db218e29aac3f1f0464b762773b8ebe`

<!-- BEGIN RESULTS_LPF2008.md -->
# Reproduction of Lee–Popov–Fredrickson (2008): Polyelectrolyte Complexation by CL-FTS

**Reference**: J. Lee, Y. O. Popov, G. H. Fredrickson, *J. Chem. Phys.* **128**, 224908 (2008)
— "Complex coacervation: A field theoretic simulation study of polyelectrolyte complexation" (LPF2008).

**Date**: 2026-08-02. **Code**: `devel/charged_polymers/clfts_charged.py` (`ChargedCLFTS`).

## 1. Purpose

LPF2008 is the founding CL-FTS study of complex coacervation: a symmetric
polycation/polyanion solution in a good implicit solvent demixes into a dense
polymer-rich coacervate coexisting with a nearly pure supernatant, driven
**entirely by charge correlations** — mean-field SCFT predicts a homogeneous
phase at *all* couplings for this symmetric model (their Sec. III.B).
Reproducing (i) their charge structure factor $S_{cc}(k)$ (their Fig. 5) and
(ii) the fluctuation-induced coacervation transition (their Figs. 4, 6) is
therefore a stringent end-to-end test of the fluctuating-$\psi$ sector of
`ChargedCLFTS`: any error in the electrostatic coupling, the ETD $\psi$
integrator, or the CL sampling would destroy both.

## 2. Model mapping

LPF2008 use discrete Gaussian chains ($N$ beads, every bead carrying charge
$\pm 1$), an Edwards excluded-volume term $\tfrac{u_0}{2}\rho^2$, and point
charges, with dimensionless parameters

$$B = \frac{u_0 N^2}{R_g^3}, \qquad C = \frac{n N R_g^3}{V}, \qquad E = \frac{4\pi l_B N^2}{R_g},$$

and lengths in units of $R_g = b\sqrt{N/6}$. Our conventions (lengths in
$R_0 = b\sqrt{N}$, Helfand compressibility $\zeta N (\sum\phi - 1)^2$, smeared
charges $\hat h = e^{-a^2k^2/2}$) map exactly onto theirs in the canonical
ensemble:

| LPF2008 | This code |
|---|---|
| $B$, $C$ | $\zeta N = B\,C$ |
| $E$ | $E_\text{code} = 6\,C\,E$ |
| $C$ | $\bar n = 216\,C^2$ |
| $L/R_g$ | $L_x/R_0 = (L/R_g)/\sqrt 6$ |
| $N$ beads | discrete chain, $ds = 1/N$ |
| good solvent | $\chi N = 0$ |
| point charges | smearing $a \to 0$ (see Sec. 5) |

The Edwards $\tfrac{u_0}{2}\rho^2$ and Helfand $\tfrac{\zeta N}{2}(\rho-1)^2$
penalties differ only by terms linear in $\rho$ (a chemical-potential shift)
and a constant, so at fixed mean density they generate identical fluctuation
physics. The $\chi N = 0$ exchange mode has zero interaction eigenvalue and is
frozen at its (vanishing) saddle; only the pressure-like mode $W_+$ and the
electrostatic potential $\psi$ fluctuate.

All runs: two homopolymer species P (+1) / M (−1) at equal volume fractions,
$32^3$ grid, $L = 4\,R_g$, CL time step $dt = 0.1$, ETD $\psi$ integrator,
CUDA platform, homogeneous ($W=0$, $\psi=0$) initial state.

## 3. Stage A — charge structure factor $S_{cc}(k)$ (their Fig. 5)

Parameters: $B = 1$, $C = 12$, $N = 512$, $E \in \{900, 3600, 19600\}$
(their Fig. 5 couplings), $a = 0.03$, $10^5$ CL steps (9000 samples).

$S_{cc}$ is measured through the analytic $\psi$ correlator
$A(k) = \mathrm{Re}\,\langle \hat\psi_k \hat\psi_{-k} \rangle$
(the CL-valid estimator; $\langle|\hat\psi_k|^2\rangle$ is *not* an
observable), converted to their plotted dimensionless combination

$$G(kR_g) \equiv 4\pi l_B R_g^2\, S_{cc}(k) = (kR_g)^2\left[1 + \frac{(kR_g)^2 \tilde L^3}{E}\,A(k)\right], \qquad \tilde L = L/R_g = 4 .$$

**Result** (`lpf_scc.png`): at all three couplings the measured $G(kR_g)$ lies
on the parameter-free discrete-chain RPA *with our Gaussian smearing included
in the interaction*,

$$G_\text{RPA} = \frac{X}{1 + X/(kR_g)^2}, \qquad X = \frac{E\,C\,\omega(k)\,\hat h^2(k)}{N^2},$$

with $\omega(k)$ the exact discrete-chain double sum. Peaks:
$G = 54.8,\ 92.6,\ 163.3$ at $kR_g = 9.4,\ 12.6,\ 15.7$ for
$E = 900,\ 3600,\ 19600$.

![Charge structure factor $G(kR_g) = 4\pi l_B R_g^2 S_{cc}$ at $B=1$, $C=12$ (their Fig. 5 conditions). Symbols: CL-FTS; solid: parameter-free smeared discrete RPA; dashed: point-charge RPA.](lpf_scc.png)

This reproduces both LPF2008 conclusions for the dense regime ($C = 12$):

1. **RPA robustness**: at high concentration, $S_{cc}$ from the full nonlinear
   CL simulation is RPA-like — fluctuation corrections to the *structure* are
   weak even where they dominate the *thermodynamics*.
2. **Correlation-length scaling**: the peak position grows as
   $k^* \sim \xi_c^{-1} \sim (EC)^{1/4}$ (measured ratios 1.33 and 1.25 vs
   ideal 1.41 and 1.53, limited by discrete $k$-shells and the smearing
   cutoff at the highest $E$).

Absolute peak heights differ from their printed curves only through the UV
regularization: their point charges make $S_{cc}(k)$ cutoff-dependent at large
$k$ (they note lattice-spacing sensitivity themselves), while our smeared model
is UV-finite. With $\hat h^2$ included, the RPA describes our data with **no
adjustable parameters**.

## 4. Stage B — fluctuation-induced coacervation (their Figs. 4, 6)

Parameters: $E = 14400$, $C = 12$, $N = 384$ (their Fig. 6 conditions), scan
over $B \in [0.3, 4.0]$ (per-smearing coverage as tabulated below), 60 000 CL
steps from a homogeneous start (= their "supercooling" branch). Order
parameter: $\Delta\rho$ = standard deviation of the $4^3$-block coarse-grained
total polymer density $\mathrm{Re}[\phi_P + \phi_M]$, averaged over the second
half of the run.

**Mean-field contrast**: for this symmetric model SCFT is *exactly* trivial —
the saddle point is homogeneous for every $B$, $C$, $E$ (charge cancels
identically between P and M; their Sec. III.B). Any demixing observed below is
purely a fluctuation/correlation effect, i.e. exists only because $\psi$ (and
$W_+$) fluctuate. A partial-saddle L-FTS with slaved $\psi$ cannot produce it.

**Result** (`coac_scan.png`). Late-time $\Delta\rho$ (second half of the run):

| $B$ | $a=0.03$ | $a=0.0125$ | $a=0.005$ |
|----:|---------:|-----------:|----------:|
| 4.0 | 0.065 | — | — |
| 2.0 | 0.063 | 0.087 | — |
| 1.5 | 0.064 | 0.090 | 0.103 |
| 1.2 | 0.066 | 0.094 | 0.108 |
| 1.0 | 0.068 | 0.098 | 0.115 |
| 0.8 | 0.071 | 0.106 | 0.127 |
| 0.5 | 0.080 | 0.135 | **0.716** |
| 0.45 | — | 0.146 | — |
| 0.4 | — | 0.172 | — |
| 0.35 | — | **0.844** | — |
| 0.3 | — | **0.975** | **1.120** |

(bold = demixed; all runs finite, mean density exactly 1 by material
conservation, no CL runaway even at $E_\text{code} = 6CE \approx 10^6$.)

- Homogeneous branch: $\Delta\rho \approx 0.09$–$0.17$, flat in time.
- Demixed branch: $\Delta\rho$ grows from the homogeneous value and saturates
  at $\mathcal{O}(1)$. At $a = 0.0125$, $B = 0.3$ the final configuration is a
  **slab coacervate**: a planar dense layer with coarse-grained
  $\phi_P + \phi_M$ up to $2.8$ coexisting with nearly pure implicit solvent
  ($\phi \approx 0$) — precisely the coexistence morphology of their Fig. 4.
- Near the boundary ($a = 0.005$, $B = 0.5$) the time series shows an
  induction period followed by nucleation and growth over
  ~15 000–25 000 steps — the classic supercooled kinetics implied by their
  hysteresis protocol.

Boundary locations (homogeneous-start = supercool branch):

| regularization | $\hat h^2$ at the correlation peak ($kR_g \approx 24$) | $B^*_\text{supercool}$ |
|---|---:|---:|
| $a = 0.03$ | 0.04 | $< 0.5$ (no demixing seen down to $B{=}0.5$) |
| $a = 0.0125$ | 0.58 | $0.38 \pm 0.03$ |
| $a = 0.005$ | 0.92 | $0.65 \pm 0.15$ |
| LPF2008 point charges | 1 | $\approx 1$ (their Fig. 6) |

![Coacervation scan at $E=14400$, $C=12$, $N=384$. Top left: late-time demixing order parameter vs $B$ for three smearing radii; top right: time series at $a=0.005$ showing nucleation-and-growth at $B=0.5$; bottom: final coarse-grained density slices showing slab coacervate coexisting with nearly pure solvent.](coac_scan.png)

**Cutoff dependence** (expected, and quantified here): the transition location
depends monotonically on the short-range regularization of the Coulomb
interaction, converging toward the point-charge value as $a \to 0$. The
driving force for coacervation is the correlation (electrostatic binding)
free energy carried by modes near the correlation peak
$k^* \sim \xi_c^{-1}$, which at these couplings sits at $kR_g \approx 24$ —
essentially the grid Nyquist scale ($kR_g^\text{Nyq} \approx 25$ at $32^3$,
$L = 4R_g$). Smearing multiplies the interaction there by
$\hat h^2 = e^{-a^2k^2}$: at $a = 0.03$ this removes 96% of the driving
force (no demixing anywhere in the scanned range $B \ge 0.5$), at
$a = 0.0125$ it removes ~40% ($B^* \approx 0.38$), at $a = 0.005$ only ~8%
($B^* \approx 0.65$). The residual gap to their $B^* \approx 1$ is
consistent with the remaining smearing plus the fact that at these couplings
the correlation peak is pinned at the shared lattice cutoff — their
point-charge model is itself lattice-regularized there.

This cutoff sensitivity is not a defect of the reproduction: it is a known
property of the point-charge model itself (LPF2008 report lattice-spacing
dependence of their absolute free energies; the later regularized-model
literature, e.g. Villet & Fredrickson, was motivated by exactly this). The
smeared model exchanges the cutoff-dependent constant for a physical
charge-size parameter.

## 5. Conclusions

1. `ChargedCLFTS` reproduces the LPF2008 charge structure factor
   quantitatively (parameter-free smeared RPA, all three couplings) and the
   qualitative dense-regime conclusions (RPA robustness, $\xi_c$ scaling).
2. It reproduces **fluctuation-induced complex coacervation** — slab
   coexistence of a dense coacervate with nearly pure solvent from a
   homogeneous start — a phenomenon strictly invisible to SCFT and to
   partial-saddle L-FTS for this symmetric model.
3. The transition location $B^*(a)$ moves monotonically toward the
   point-charge value as $a \to 0$
   ($<0.5 \to 0.38 \to 0.65$ vs their $\approx 1$), tracking the
   smearing suppression $\hat h^2$ of the correlation-peak modes —
   connecting our UV-finite smeared model to their cutoff-regularized point
   charges and quantifying, with a single code, the UV sensitivity of the
   point-charge model that motivated the later regularized-charge
   literature.

**Scripts/data**: `/home/yongdd/polymer/dh_salt_runs/`
(`test_lpf2008_scc.py`, `analyze_lpf.py`, `lpf_E*.json`, `lpf_scc.png`;
`test_lpf2008_coacervation.py`, `coac_B*.json` ($a{=}0.03$),
`coac2_B*.json` ($a{=}0.0125$), `coac3_B*.json` ($a{=}0.005$),
`coac_scan.png`).

<!-- END RESULTS_LPF2008.md -->


---

# Source: RESULTS_MULTIVALENT.md

SHA-256: `612e2bb23c31e898dcf5faaa8f23d4b7deb0e98de6b296bfd1aee31112398682`

<!-- BEGIN RESULTS_MULTIVALENT.md -->
# Multivalent (3:1) Electrolyte: Beyond-Gaussian Ion Correlations from Charged CL-FTS

**Validation report, 2026-08-14** — first multivalent-ion (z = +3) runs of
`ChargedCLFTS`. Bulk-electrolyte step toward reproducing the correlation
physics of Duan, Agrawal & Wang, *PRL* **134**, 048101 (2025)
(`references/2025_prl_duan.pdf`): their polyelectrolyte-brush
collapse/overcharging is driven by valence-specific ion correlations
*beyond* the Gaussian (Debye–Hückel one-loop) level. Here we isolate
exactly that contribution in the simplest periodic setting the current
code supports, with the compressible (ζN) + per-species-smearing model
throughout.

Script: `dh_salt_runs/test_multivalent_oneloop.py`; data
`dh_salt_runs/mv*_*.json`, `mvs*.json`; figure `multivalent_oneloop.png`.

## 1. Design: matched-Debye A/B pair

Two systems with **identical Debye constant** and identical Gaussian
(one-loop) description — all χ = 0, ζN = 100, a = 0.2 (all species),
discrete 1-segment ions, 16³ grid, L = 4, E ∈ {6.25, 25}:

| system | composition | Σ z²φ̄ | frozen aux modes |
|---|---|---|---|
| `11` | P(+1) 1/2, M(−1) 1/2 | 1 | 1 |
| `31` | SP(+3) 1/12, SM(−1) 1/4, S(0) 2/3 | 1 | 2 |

At the Gaussian level the ψ sector depends on the ions only through
$S_{\rm scr}(k) = \sum_i z_i^2 \bar\phi_i \hat h_i^2(k)$, equal for both
systems, so the one-loop lattice prediction (the χ→0, B→∞ limit of the
formula validated in `RESULTS_ONELOOP_DH.md`)

$$\left\langle \frac{\partial H}{\partial E}\right\rangle_{1\text{-loop}}
 = -\frac{1}{2\sqrt{\bar N} V}\sum_{k\ne 0}
   \frac{\kappa/E}{\kappa + u_2}, \qquad
   \kappa = \frac{k^2}{2E},\;\; u_2 = \frac{S_{\rm scr}(k)}{2},$$

is **identical** for `11` and `31`. The variational Gaussian theory of the
PRL (Wang PRE 2010 lineage) likewise predicts zero difference for fixed
uniform densities. Any measured 31−11 difference is therefore a pure
beyond-Gaussian valence effect. Runs are **seed-paired** (identical noise
streams for both systems), which cancels nearly all sampling noise in the
difference.

## 2. One-loop validation at $\bar N = 10^4$ (including z = +3)

100k CL steps, 10k equilibration, dt ∈ {0.2, 0.1}, linear dt→0
extrapolation (seed 12345):

| system | E | dt→0 measured | prediction | ratio |
|---|---:|---:|---:|---:|
| 11 | 6.25 | −5.06413×10⁻² | −5.06277×10⁻² | 1.0003 |
| 31 | 6.25 | −5.06406×10⁻² | −5.06277×10⁻² | 1.0003 |
| 11 | 25 | −1.24288×10⁻² | −1.24333×10⁻² | 0.9996 |
| 31 | 25 | −1.24274×10⁻² | −1.24333×10⁻² | 0.9995 |

Agreement ≤ 0.05% at every point — the multivalent sector (z² factors in
screening, electroneutrality with z = 3, **two** simultaneously frozen
zero-eigenvalue modes) is validated with no adjustable parameters. The
internal identity `dHdE_from_A` (computed from the analytic ψ correlator)
reproduces the direct estimator exactly, as it must.

## 3. Resolved beyond-Gaussian valence effect

Four seed-paired runs per dt at E = 25, $\bar N = 10^4$:

| dt | ⟨dH/dE⟩₃₁ − ⟨dH/dE⟩₁₁ |
|---:|---:|
| 0.2 | +1.279(6)×10⁻⁶ |
| 0.1 | +1.359(5)×10⁻⁶ |
| **dt→0** | **+1.440(12)×10⁻⁶** (≈120σ) |

- Seed-to-seed scatter of the paired difference is ~5×10⁻⁹ — the pairing
  cancels the common Gaussian fluctuations and isolates the systematic
  valence effect.
- Sign: the 3:1 salt is *less negative* — its effective screening is
  **stronger** than its own one-loop description (larger effective
  $u_2$ → smaller per-mode contribution). This is the CL manifestation of
  correlation-induced counterion condensation around the trivalent
  cations (Agrawal & Wang, *JCTC* **18**, 6271 (2022)) — the bulk
  precursor of the overcharging (Γ < 0) mechanism behind brush collapse
  in the PRL.
- Magnitude: 0.012% of the total ⟨dH/dE⟩, but 0.37% of its
  correlation (screening) part at E = 25, growing steeply with coupling
  (0.11% of the correlation part at E = 6.25: extrapolated diff
  +6.7×10⁻⁷). Loop counting predicts the relative effect scales as
  $1/\sqrt{\bar N}$.

**Shell-resolved structure** (panel c of `multivalent_oneloop.png`): the
difference of the analytic ψ correlators $A_{31}(k) - A_{11}(k)$,
normalized by $|A^{\rm pred}(k)|$, is largest at the longest wavelength
(+1.18(1)% at $|k|R_0 = 1.57$) and decays monotonically (+0.44% by
$|k|R_0 = 3.9$): the valence effect is a **long-wavelength enhancement of
screening**, exactly the k → 0 (Debye-constant renormalization) behavior
expected from ion condensation.

![Multivalent one-loop validation and valence effect](multivalent_oneloop.png)

## 4. Strong-fluctuation stability boundary ($\bar N = 10^2$)

$\bar N = 10^2$ (the value implied by the PRL's model parameters
v = b³ = 1 nm³, N = 100 — an artifact of their compact-monomer choice,
not of real PEs) is **outside the CL operating range**:

| dt | outcome (E = 25, both systems) |
|---:|---|
| 0.2, 0.1, 0.05 | diverge (ψ/W₊ hot-spot runaway within ~10²–10³ steps) |
| 0.02 | completes 100k steps, but biased: ratio 1.09 (11) / 1.02 (31), Im⟨dH/dE⟩ ~ 10⁻⁴ |

χ = 0 means there are no real auxiliary fields, so `alpha_ds` is inert
here; nothing stabilizes W₊/ψ excursions. Consistent with the project
guideline (user): **CL-FTS operates at $\bar N \sim 10^4$–$10^5$;
$10^5$ is comfortably stable.** PRL-reproduction work should be designed
at that fluctuation strength — physically equivalent to a realistic
monomer volume (v ≈ 0.1 nm³ ⇒ $\bar N \approx 10^4$) rather than the
paper's v = b³.

## 5. Conclusions and next step toward the PRL

1. The multivalent sector of `ChargedCLFTS` is quantitatively validated
   at the Gaussian level (≤0.05%, parameter-free, first z = +3 runs).
2. CL-FTS resolves a systematic beyond-Gaussian valence-specific
   correlation effect — enhanced long-wavelength screening in the 3:1
   salt — at 120σ significance via seed-paired A/B runs. This observable
   is *identically zero* in both the one-loop theory and (for fixed
   uniform densities) the PRL's variational Gaussian theory: it is the
   physics class their approximation truncates.
3. Next (stage 2 of the PRL plan): inhomogeneous setting — grafted
   chains (`q_init` plumbing into `ChargedCLFTS`) + wall boundary
   conditions for the ψ sector (DCT-based ETD/Poisson) — to observe
   correlation-driven ion partitioning/overcharging (Γ < 0) directly, at
   $\bar N = 10^4$–$10^5$ with the compressible + smeared model
   unchanged.

<!-- END RESULTS_MULTIVALENT.md -->


---

# Source: RESULTS_ONELOOP_DH.md

SHA-256: `cfe30d319bc8cadafa9c54539b0464e63a6f0b906a70ee9aa6ba97b62aae7b23`

<!-- BEGIN RESULTS_ONELOOP_DH.md -->
# One-Loop Debye–Hückel Thermodynamics from Charged CL-FTS

**Validation report, 2026-08-02** — first quantitative physics
reproduction with `ChargedCLFTS` (fluctuating electrostatic potential).
This observable is generated ENTIRELY by $\psi$ fluctuations and is
identically zero in the partial-saddle L-FTS prototype: it is the test
that motivated the CL implementation.

## 1. System and observable

Symmetric $\pm 1$ salt of single-segment discrete ions ($ds = 1$,
$\bar\phi_\pm = 0.5$), $\chi N = 0.5$, $\zeta N = 100$, smearing
$a = 0.2$, grid $16^3$, $L = 4$, $\bar n = 10^4$,
$\alpha_{ds} = 0.05$. Coulomb couplings
$E \in \lbrace 2.5, 6.25, 12.5, 25, 50 \rbrace$
(Debye constant $\kappa_D^2 R_0^2 = E$, so screening lengths
$0.14\,R_0$–$0.63\,R_0$).

The measured quantity is the analytic CL observable

$$\left\langle \frac{\partial H}{\partial E} \right\rangle
  = -\frac{\langle H_{\rm exp}\rangle}{E}, \qquad
  H_{\rm exp} = \frac{1}{2EV}\int \psi\,\nabla^2\psi\; d\mathbf x ,$$

which by the standard thermodynamic identity equals
$(\sqrt{\bar n}V)^{-1}\, \partial F/\partial E$. Its Gaussian (one-loop)
prediction, evaluated as an exact sum over the simulation's own lattice
modes with the verified repo conventions
($B = 1/\chi N$, $u_2 = \hat h^2/2$, $\kappa = k^2/2E$):

$$\left\langle \frac{\partial H}{\partial E} \right\rangle_{\rm 1\text{-}loop}
 = -\frac{1}{2\sqrt{\bar n}\,V} \sum_{k \neq 0}
   \frac{(\kappa/E)\,(B-u_2)}{(B-u_2)(\kappa+u_2) + u_2^2}.$$

This is the smeared, finite-lattice generalization of the classic
Debye–Hückel limiting law (continuum, unsmeared:
$F_{\rm elec}/V = -\kappa_D^3/12\pi$; the Gaussian-fluctuation theory of
Wang, *PRE* **81**, 021501 (2010)). There are **no free parameters**.

## 2. Protocol

Two time steps ($dt = 0.2$ and $0.1$), 100k CL steps each (10k
equilibration), 5 runs per $dt$; linear $dt \to 0$ extrapolation
$O(0) \approx 2\,O(0.1) - O(0.2)$. Errors from 20-block blocking.
Script: `dh_salt_runs/test_oneloop_dh.py`.

## 3. Results

| $E$ | meas $dt{=}0.2$ | meas $dt{=}0.1$ | extrap $dt\to0$ | prediction | ratio |
|----:|----:|----:|----:|----:|----:|
| 2.5  | $-1.2757\times10^{-1}$ | $-1.2744\times10^{-1}$ | $-1.2730(4)\times10^{-1}$ | $-1.2721\times10^{-1}$ | 1.001 |
| 6.25 | $-5.0753\times10^{-2}$ | $-5.0672\times10^{-2}$ | $-5.0591(2)\times10^{-2}$ | $-5.0575\times10^{-2}$ | 1.000 |
| 12.5 | $-2.5187\times10^{-2}$ | $-2.5142\times10^{-2}$ | $-2.5098(1)\times10^{-2}$ | $-2.5098\times10^{-2}$ | 1.000 |
| 25   | $-1.2452\times10^{-2}$ | $-1.2431\times10^{-2}$ | $-1.2409(1)\times10^{-2}$ | $-1.2413\times10^{-2}$ | 1.000 |
| 50   | $-6.1276\times10^{-3}$ | $-6.1186\times10^{-3}$ | $-6.1095(3)\times10^{-3}$ | $-6.1130\times10^{-3}$ | 0.999 |

**Agreement is $\leq 0.1\%$ at every coupling.**

![One-loop DH validation](oneloop_dh.png)

*(a)* The observable spans a factor 20 across the $E$ range and tracks
the parameter-free lattice prediction (blue). The gray dashed line is
the free-field ($u_2 = 0$) part — the smeared per-mode self-energy
$-N_{\rm modes}/(2E\sqrt{\bar n}V)$, which dominates the total. *(b)*
Measured/predicted for the ETD integrator (red, $0.999$–$1.001$) vs the
earlier semi-implicit integrator (gray crosses, $0.25$–$0.90$ — see §4).
*(c)* Subtracting the free-field part isolates the **DH correlation
(screening) free energy**; the remainder still matches:

| $E$ | correlation meas | correlation pred | ratio |
|----:|----:|----:|----:|
| 2.5  | $6.64(4)\times10^{-4}$ | $7.59\times10^{-4}$ | 0.87 |
| 6.25 | $5.97(2)\times10^{-4}$ | $6.13\times10^{-4}$ | 0.97 |
| 12.5 | $4.961(9)\times10^{-4}$ | $4.953\times10^{-4}$ | 1.00 |
| 25   | $3.877(5)\times10^{-4}$ | $3.837\times10^{-4}$ | 1.01 |
| 50   | $2.889(3)\times10^{-4}$ | $2.855\times10^{-4}$ | 1.01 |

The correlation part is a $\sim 0.5$–$5\%$ fraction of the total; it
agrees to $1$–$3\%$ for $E \geq 6.25$. The residuals (few $\sigma$) are
consistent with the neglected $O(dt^2)$ remainder of the linear
extrapolation and beyond-Gaussian corrections at $\bar n = 10^4$; the
largest relative deviation sits at $E = 2.5$, where the correlation part
is the smallest fraction of the signal.

## 4. Two integrator lessons captured by this test

1. **Semi-implicit $\psi$ integration silently biases fluctuation
   thermodynamics.** The scheme
   $(1 + 2\kappa\,dt)\hat\psi' = [\psi + dt\,c + \eta]^{\wedge}$ is
   stable but suppresses the per-mode stationary variance by
   $1/(1+2\kappa\,dt)$. Because every mode contributes $\sim 1/E$ to
   $\partial F/\partial E$, the high-$k$ modes dominate and the measured
   observable came out $2.5$–$4\times$ too small at small $E$ (panel b,
   gray). The fix is **ETD** — the exact per-mode Ornstein–Uhlenbeck
   step $\hat\psi' = a\hat\psi + (1-a)E\hat c/k^2 + i\,\sigma
   \sqrt{(1-a^2)/(2\gamma\,dt)}\,\hat\xi$ with $a = e^{-\gamma dt}$,
   $\gamma = k^2/E$ — whose linear-part variance is exact for any $dt$.
2. **The explicit Hamiltonian term must be analytic off the saddle**:
   $H_{\rm exp} = +(1/2EV)\int\psi\nabla^2\psi$, not
   $-\tfrac12\langle c\psi\rangle$ (the two coincide only when the
   Poisson equation holds, i.e., in the L-FTS prototype).

## 5. Conclusion

The fluctuating-$\psi$ sector of `ChargedCLFTS` reproduces the smeared
Debye–Hückel one-loop free energy with sub-percent accuracy across a
20-fold range of Coulomb coupling, with no adjustable parameters. Together
with the bit-exact neutral reduction, the deterministic linear-response
screening test, and the $S_{cc}$ structure checks, the charged CL-FTS
machinery is quantitatively validated at the Gaussian-fluctuation level.
Next target: complex coacervation against Lee, Popov & Fredrickson,
*JCP* **128**, 224908 (2008) / Riggleman, Kumar & Fredrickson, *JCP*
**136**, 024903 (2012).

<!-- END RESULTS_ONELOOP_DH.md -->


---

# Source: RESULTS_PRL_SCFT.md

SHA-256: `5ce73b51566c4d2b8cef288329893bff7d84c3665b855e36e7cbd131e5629bbd`

<!-- BEGIN RESULTS_PRL_SCFT.md -->
# Reimplementation of the PRL-2025 Correlation SCFT (their own theory, our code)

Date: 2026-08-25. Paper: Duan, Agrawal & Wang, PRL **134**, 048101 (2025)
(`references/2025_prl_duan.pdf` + SI). Code: `prl_scft.py`,
driver `dh_salt_runs/prl_fig1_run.py` (v1) / `prl_fig1_run2.py` (v2).

Context: the CL-FTS campaign (`RESULTS_FIG1_REPRO.md`) reproduced their
*mechanism* but not their amplitudes. Per the decision of 2026-08-25, we
first reproduce **their own SCFT** exactly as published, and only then
compare CL-FTS against it.

## 1. What is implemented

Their electrostatic-correlation-augmented SCFT, in physical units
(nm, kT), 1D wall geometry, at their exact parameters
(N=100, b=1 nm, a=2.5 Å for all species, σ=0.1 nm⁻², l_B=0.7 nm,
Θ solvent χ=0.5, I_b=0.3 M for Fig. 1):

(Section updated 2026-08-26 to the FINAL v9 algorithm; the v1 choices it
replaced — spectral chain sector, under-relaxed ψ — are kept as history
in §3b–3c, which explain why they had to go.)

- **Chain sector**: real-space banded bond convolution
  (`_chain_density_rs`, the default `chain_backend="realspace"`):
  discrete chain (N beads, N−1 Gaussian bonds, 1D kernel variance b²/3,
  truncation 8b), HARD WALL at z=0 via truncated bond integral (zero
  left ghosts — gives the entropic depletion layer of their Fig. 1),
  graft source = Gaussian of width 0.5b centered ON the wall
  (half-Gaussian; the smooth stable analog of their δ(z), finite in the
  discrete formulation), per-bead equal-mass assembly, φ rescaled to
  ∫ρ dz = σN. The repo spectral `PropagatorSolver` remains available
  (`chain_backend="spectral"`) but is only valid for per-chain field
  contrasts < ~37 kT (§3c).
- **Ions**: analytic Boltzmann, all grand-canonical at the bulk z:1 salt
  (volumeless — same as the paper). I_b > 0 required (salt-free PB with
  two Neumann walls is singular; `_solve_pb` raises).
- **Electrostatics**: nonlinear Poisson–Boltzmann, guarded Newton
  (backtracking line search on the residual norm + residual-based
  convergence test, §3c), Neumann walls, bulk gauge ψ(∞)=0. Production
  runs use `psi_exact=True` (exact PB each iteration).
- **Correlations**: LDA self-energies, SI Eq. S33:
  u_K(z) = z_K² l_B/(2a)·[u(aκ(z)) − u(aκ_b)],
  u(x) = 1 − x·exp(x²/π)·erfc(x/√π); κ(z) from the local ionic strength
  SMOOTHED over the Born radius a (UV regularization, §3e).
- **Excluded volume**: ξ eliminated exactly, ξ = −ln(1−φ_P) − χφ_P.
- **Convergence**: α-continuation (charge-fraction ladder from 0.05) with
  a ΔW trust region, then correlation-strength ramp 0 → 1 from the
  converged mean-field state; plain slow mixing (λ≈0.01) suffices once
  the chain backend is smooth (§3c). Ion observables evaluated after
  `_solve_ion_sector` (Anderson-accelerated (ψ,u) equilibration at fixed
  ρ_P — keeps the electrolyte on the physical branch, §3b).

## 2. Fig. 1 conditions — headline result (v1 runs, L=100 nm, Nz=800)

| z₊ | MF h (nm) | corr h (nm) | corr converged? |
|----|-----------|-------------|-----------------|
| 1  | ~84*      | ~71*        | no (err 4e-2)   |
| 2  | ~74*      | **17.98**   | yes (5e-4)      |
| 3  | ~102*     | **22.50**   | yes (5e-4)      |
| neutral | —    | 21.2        | yes (3e-4)      |

*The v1 MF branch above α≈0.5 did **not** converge: at L=100 the
fully-charged mean-field brush (h ~ 74–102 nm, contour Nb = 100 nm) is
pressed against the far wall — and, as it later turned out (§3c), the v1
MF "heights" were dominated by spectral-propagator garbage anyway. The
bigger v2 box did NOT rectify the MF baseline (§3b); only the v6
real-space chain backend did (§3c–3d). The collapsed correlation states
were nonetheless robust throughout (identical h across v1/v2/v6 paths).

**The paper's central claim reproduces with their own theory in our
implementation**: mean field is swollen and nearly valence-blind;
switching on the LDA self-energies collapses the multivalent brushes to
dense plateau profiles (φ ≈ 0.55–0.64) with a sharp edge —
z₊=2 collapses *below the neutral height* (18.0 < 21.2 nm), z₊=3 to
neutral level — while z₊=1 stays swollen. Figure:
`prl_scft_fig1.png`.

## 3. Their Fig. 3 observables: Γ and ψ_S at I_b = 0.3 M

Definitions recovered from the paper/SI (important — two earlier
estimators were wrong for this comparison):

- Their Γ ≡ 1 − z₊(ρ₊ − ρ_b)/ρ_P is a **local brush-interior** quantity
  (cations only). `gamma_eff()` now evaluates it as a density-weighted
  core average (φ > 0.5·max). An integral "excess/bare" estimator
  diverges for collapsed layers (cation Boltzmann factors ~e⁺⁶); a
  net-charge (Gauss-law) estimator is ≈0 by screening. Neither is their Γ.
- Their Fig. 3 curves are the **algebraic** asymptotic Γ of Eq. (7)/(S38):
  ((z₊+1)Γ−1)/(Γ−1) = −z₊(z₊+1)·(κ_b l_B/4)·u′(aκ_b), a function of bulk
  salt only (`gamma_algebraic()`); MF limit Γ = 1/(z₊+1).

Reconstructed from the converged v1 correlation states (u,ψ
re-equilibrated at fixed ρ_P to du < 1e-5):

| z₊ | Γ_SCFT (local) | Γ_algebraic Eq. 7 | ψ_S (kT/e) |
|----|---------------:|------------------:|-----------:|
| 1  | +0.277 | +0.389 | −0.542 |
| 2  | **−0.057** | −0.048 | **−0.147** |
| 3  | **−0.262** | −0.650 | **+0.747** |

This reproduces the **sign structure of their Fig. 3 exactly**:

- z₊=1: Γ>0, no overcharging, ψ_S<0.
- z₊=2: **Γ<0 (overcharging) yet ψ_S<0** — their "collapse–reexpansion
  without charge inversion". Quantitative agreement of the full-SCFT
  local Γ (−0.057) with their asymptotic formula (−0.048).
- z₊=3: **both Γ<0 and ψ_S>0** — charge inversion. The asymptotic
  formula overshoots (−0.65 vs −0.26) exactly where its |Δρ|≪ρ_b
  linearization fails, as expected.

## 3b. Convergence pathology of the mean-field branch — diagnosis (2026-08-26)

The v2 big-box MF branch (and four successive mixing schemes: simple
mixing over lambda 0.001-0.05, settle+AM with relaxed psi, AM with exact
psi, slow capped simple mixing with exact psi) all fail above alpha ~0.5
with a lambda-INDEPENDENT residual — the signature of a real fixed-point
eigenvalue mu > 1. Power iteration on the W-map Jacobian at alpha=0.5
(z+=2, `prl_eigdiag.py`) measured **mu ~ +300-600**, eigenvector smooth
and delocalized over the dilute exterior (peak ~104 nm). Mechanism: the
per-chain field convention squares the chain length into the
Donnan feedback loop, local gain g(z) ~ N^2 f^2 (4 pi lB / kappa^2)
rhoP(z). With |mu| ~ O(500), no fixed-point mixing (nor secant-history
AM from a distant start) can converge — the cure is Newton-Krylov on the
full residual R(W) = F(W) - W (`solve_newton()`), which is insensitive
to mu.

Separately, the v2 z+=3 run exposed an **electrolyte double branch**: with
u mixed on the outer clock while psi is under-relaxed, the (u, rho_ion)
subsystem can run away to a spurious saturated-condensation solution
(up -> -8.25 kT saturated, rho+ ~ 140 nm^-3, PB violated at net charge
~450 nm^-3) while the W-error still converges (psi never enters the error
metric). v1 z3 sat on the physical branch; v2 z3 on the runaway branch —
remarkably both give the same polymer profile (h = 22.50 vs 22.51), but
ion observables (Gamma, psi_S) are only meaningful after re-equilibrating
(psi, u) at fixed rhoP (`_solve_ion_sector`), which lands on the physical
branch from either start. Robustness note: the collapsed correlation
states are strong attractors — v1 (L=100, from a clipped MF) and v2
(L=200, from a wildly sloshing MF) agree to 0.01 nm in h for both z2
and z3.

## 3c. ROOT CAUSE FOUND: spectral leakage amplification (2026-08-26)

The 3b diagnosis (huge Jacobian eigenvalue) was itself a symptom. Chasing
why Newton-Krylov, dense-FD Newton, and Levenberg-Marquardt ALL stalled
(no descent even along the gradient at any damping) exposed that the
W-map was effectively DISCONTINUOUS: an O(1) response to 1e-6 field
perturbations, |dR| independent of eps. Component isolation:

- Guarded-Newton rewrite of `_solve_pb` (backtracking line search on the
  residual norm + residual-based convergence; the old blind damping
  `step=min(1, 2/|dpsi|)` capped at 60 iterations could exit unconverged,
  and a non-converged Newton iterate is chaotically input-sensitive).
  After the fix the PB sector is perfectly linear (slope constant over
  eps = 1e-8..1e-4) — but the map was still discontinuous.
- The remaining source is the CHAIN sector, i.e. the repo spectral (DCT)
  discrete propagator. Minimal isolated repro (grafted N=100 chain,
  Nz=800, L=200, reflecting BCs, cpu-mkl):
  * W = 0 and W = 50 = const: clean (tail ~1e-14).
  * W = step 100 (z<60) -> 50 (z>60): **phi(far wall) = 23, phi_min =
    -3.9** — pure garbage 100 nm beyond the chain's contour length.

  Mechanism: the ~1e-16 double-precision floor of the spectral
  representation seeds the low-field exterior each step and grows by
  e^{+dw} per bead. Once the per-chain field contrast exceeds
  ln(1e16) ~ 37 kT, the seed outgrows the physical solution
  (here: contrast 50 kT -> junk ~ 1e-16*e^50 >> 1). Our brush crosses
  37 kT at alpha ~ 0.5 — exactly the mysterious "stiffness threshold" of
  v1-v5. The measured mu ~ +500 was the junk's sensitivity, not physics.
  This limitation is intrinsic to spectral propagators in double
  precision (the repo's tests at field std ~ 5 never enter this regime);
  a per-step floor clamp in the C++ solvers would be the upstream fix.

**Cure**: `_chain_density_rs` — real-space banded bond convolution
(truncated Gaussian kernel, half-width 8b, `correlate1d(mode='reflect')`
which matches the DCT-II cell-centered mirror), per-step max
renormalization, per-bead equal-mass assembly (exact for grafted chains).
Transport is local so there is no spectral floor to amplify; the step
repro gives tail ~1e-135 and phi >= 0. Validation: weak-field agreement
with the repo solver (h 21.97 vs 22.09 at alpha=0.2, profile differences
at the graft cell from the half-bond convention), and the W-map is now
SMOOTH: |dR|/eps = 2.2964 constant over eps = 1e-6..1e-2 — the true
Jacobian gain is ~2.3, entirely benign. The alpha=0.5 state sits at
residual 6e-4 with plain mixing: **the "instability" never existed; it
was numerical junk all along.**

v6 production (`prl_fig1_run6.py`): real-space backend + exact-PB psi +
plain slow mixing, full MF ladder + correlation ramp, L=200, Nz=1600,
with `_solve_ion_sector` re-equilibration before ion observables.

## 3d. v6 results: quantitative Fig. 1 agreement (2026-08-26)

v6 (real-space backend, exact-PB psi, plain lambda=0.01 mixing, L=200,
Nz=1600) converges every stage in ~2000 iterations, smoothly and
monotonically. Box-model cross-check with identical ingredients
(discrete-Gaussian elasticity + exact Donnan + lattice Theta solvent)
reproduces the SCFT h to ~1 nm (35.3/31.2/28.8 vs 36.2/32.1/29.7),
i.e. the implementation is internally exact.

| z+ | MF h | corr h | err | Gamma_loc | Gamma Eq.7 | psi_S |
|----|------|--------|-----|-----------|------------|-------|
| 1  | 36.15 | 31.41 | 5e-4 | +0.230 | +0.389 | -0.814 |
| 2  | 32.09 | **17.98** | 5e-4 | **-0.057** | -0.048 | -0.147 |
| 3  | 29.74 | **14.04** | 5e-4 | **-0.321** | -0.650 | +0.749 |
| neutral | -- | 21.16 | 3e-4 | | | |

Against the paper's actual Fig. 1 (crop inspected): their MF dashed
family phi(0)=0.32-0.38 decaying by ~45-50b and nearly valence-blind —
ours 0.31-0.44 to 45-52 nm, matching curve-for-curve; their neutral
phi(0)~0.51 ending ~28b — ours 0.58/~30; their z2 solid plateau ~0.59 to
~19b — ours 0.58-0.65 to ~18-19 (essentially exact); their z3 solid
~0.74 to ~14b — ours ~0.8 to ~15; their z1 solid slightly compacter than
dashed — same here. (The earlier impression that their MF reached "90b"
came from our junk-era plots, not the paper.) The corr h ordering z3<z2
now also matches the paper (v1/v2's z3>z2 was a branch/convergence
artifact).

## 3e. LDA UV instability at z+=3 full coupling, and its physical cure

At full correlation strength the z+=3 collapsed state initially refused
to converge below ~1e-2, with a 2*dz checkerboard (grid-scale sawtooth,
amplitude ~0.08) in the plateau. Cause: the POINTWISE LDA feedback loop
phi -> I0 -> u(a*kappa(I0)) -> phi has no wavelength cutoff; past the
z+=3 coupling threshold its local gain exceeds what b-scale chain
connectivity stabilizes, so the shortest representable wavelength blows
up first. This is a UV pathology of the LDA itself, not of the solver.
Physical regularization: an ion samples the ionic strength within its
Gaussian charge spread, so I0 is smoothed over the Born radius a before
computing kappa (their full nonlocal G_s carries this intrinsically;
Eq. S33 is its homogeneous limit). With that one change every stage
converges to 5e-4 (z3 c=1.0: 33k iterations), and the smooth z1/z2
states are numerically unchanged (z2: h=17.97, Gamma=-0.057).

Final z3: h=14.04, phimax=0.762 vs the paper's ~14b, ~0.74 —
essentially exact. Corr height ordering z3 < z2 < neutral < z1 matches
their Fig. 1.

## 3f. Salt sweeps: their Figs. 2-3 observables (2026-08-26)

Salt continuation (`prl_fig23_run.py`, `fig23_z{1,2,3}.npz`, 71 points,
all but two converged to 5e-4; figure `prl_scft_fig23.png`):

- **h(rho_b)** (their Fig. 2): z1 monotone salted decrease (51 -> 25 nm,
  no valley) — matches their z1. z2: shallow valley at ~0.06 M then
  reexpansion (17.85 -> 20.4 nm at 1 M). z3: collapsed plateau
  (~13.3 nm) then reexpansion from ~0.05 M. The collapse-and-reexpansion
  structure of their Fig. 2 is reproduced; NOT reproduced is the
  low-salt osmotic swelling and the full collapse transition, for two
  identified reasons: (i) the GC-PB box cutoff (kappa^-1 < L/5 stops the
  sweep at ~1e-5..1e-4 M; the osmotic regime needs L ~ several hundred
  nm), and (ii) the down-branch rides the METASTABLE collapsed solution
  (first-order-like transition; an up-sweep from a swollen state would
  bracket the equilibrium jump). Both are defined follow-ups.
- **Gamma(rho_b)** (their Fig. 3a): z1 stays positive; z2 crosses zero
  into an overcharging window (min -0.13 at ~0.6 M); z3 deep
  overcharging with minimum near ~0.3 M. The algebraic Eq.-7 curve
  brackets/tracks the full-SCFT local Gamma; at low salt Gamma_alg ->
  1/(z+1) (their MF limit) while Gamma_loc -> 0 (osmotic neutrality —
  the |drho| << rho_b linearization fails there, as expected).
- **psi_S(rho_b)** (their Fig. 3b): sign inversion ONLY for z+=3
  (crossing at ~1.3e-4 M); z1, z2 negative throughout — their central
  Fig.-3 asymmetry (charge inversion decoupled from overcharging),
  reproduced in full.

Caveat: the two z3 points at rho_b >= 0.5 M converged suspiciously fast
(12/154 iters, h frozen at 16.27, Gamma_loc -1.2/-2.3) — treat as
unequilibrated; rerun with a fresh ramp if they matter.

**v7 rerun (hard wall, `fig23v7_z{1,2,3}.npz`, figure updated)**: all 71
points converged to 5e-4 in W, and the ion observables were recomputed
for every point by AA re-equilibration at fixed rhoP (`_solve_ion_sector`
from u=0 — lands on the physical branch; all 71 ion sectors converged).
Final structure: z1 monotone 52 -> 26 nm; z2 valley at ~0.06 M
(h_min=18.9) then rise to 21.6; z3 collapsed ~14.1 nm with sharp
reexpansion at ~0.3-0.5 M up to 22.7 nm — i.e. back to the neutral
height (22.3), their high-salt "neutral regime". Gamma: z2 window min
-0.132 @ 0.63 M; z3 min -0.643 @ 0.5 M with recovery beyond (the earlier
-78 was the runaway-branch estimator artifact, now gone). psi_S:
inversion only for z3 (max +0.61 near 0.01 M, returning to ~0 at high
salt); z2 approaches 0 from below without crossing.

## 3g. Wall boundary condition fix: the near-wall depletion layer (v7)

User-caught discrepancy: the paper's Fig. 1 profiles are SMALL near the
grafting wall (entropic depletion over ~1-2 b), while ours peaked AT the
wall. Cause: the real-space backend used an even ('reflect') mirror at
z=0, which has no wall entropy loss; the paper's model has an
impenetrable wall (SI: chains tethered at the z=0 plate, propagators
with a hard wall). Fix: the bond convolution's left ghost cells are ZERO
(truncated bond integral -- the exact discrete-chain hard-wall rule),
and the graft source is a Gaussian of width 0.5b centered at z=b (per
user instruction: a single-point source is numerically unstable, keep it
slightly broadened). v7 results (all converged, ~+1 nm shifts from the
depletion layer):

| z+ | MF h | corr h | Gamma_loc | psi_S |
|----|------|--------|-----------|-------|
| 1  | 36.97 | 32.34 | +0.235 | -0.453 |
| 2  | 33.04 | **19.08** | -0.056 | -0.060 |
| 3  | 30.75 | **14.87** | -0.324 | +0.537 |
| neutral | -- | 22.26 | | |

Against the paper: z2 plateau 0.58 to 19.1 (theirs ~0.59 to ~19b), z3
plateau 0.72-0.73 to 14.9 (theirs ~0.74 to ~14b), neutral 0.51/~28b
(same), near-wall rise over ~1.5b -- now matching in every visible
feature. (The salt sweeps were subsequently rerun on the hard-wall
states -- see the v7-rerun paragraph at the end of §3f; a final v9
sweep rerun is noted in §3h.)

## 3h. Graft-source placement (v8, v9 = FINAL)

The v7 graft source (Gaussian centered at z=b) leaves the tethered
bead-1 density peak INSIDE the depletion zone: a +sigma/(sqrt(2pi)*0.5b)
~ +0.07 bump rides on the dilute MF plateau (decomposition:
`graft_bump_decomposition.png` -- bump = bead-1 contribution, invisible
in the paper only because their delta(z) sits exactly on the wall axis).
v8 (center 0.5b) shrinks but does not remove it. **v9 (FINAL): Gaussian
of width 0.5b centered ON the wall** -- the wall clips it to a
half-Gaussian, the smooth stable analog of their delta(z); bead-1
density is then monotone-decaying from the wall and no interior bump can
form. No divergence: in the discrete hard-wall formulation (truncated
bond integral) the propagator at the wall face is finite (finite
per-bond entropy loss), unlike the continuum Dirichlet limit where a
delta-on-wall graft is singular. Verified: near-wall profile monotone
for z < 2 nm; all states converge at unchanged rates.

v9 numbers (all err 5e-4): neutral h=21.81; z1 MF 36.52 -> corr 31.89
(Gamma +0.233, psi_S -0.601); z2 32.57 -> **18.65** (Gamma -0.056,
psi_S -0.098); z3 30.28 -> **14.46** (Gamma -0.324, psi_S +0.663).
Paper agreement unchanged (z2 ~19b, z3 ~14b). Salt sweeps rerun on the
v9 states: `fig23v9_z{1,2,3}.npz` (submitted 2026-08-26; the quoted §3f
v7-sweep structure is expected to carry over with ~0.2 nm shifts).

## 3i. Full-figure campaign: their Figs. 2, 3, 4 (2026-08-26/27)

**Fig. 2** (`prl_scft_fig2.png`; up-branch `fig2low_z*.npz` L=800 nm from
low-salt swollen starts + the v9 anchored sweeps): with their
normalization (h0 = common osmotic height; empirically their neutral
line at log(h/h0) = -0.38 with our neutral 21.8 gives h0 = 52, and our
z1 osmotic height IS 51.2), the **valley depths and locations match
quantitatively**: z2 valley -0.445 @ 0.053 M (theirs -0.45), z3 valley
-0.573 @ 1.3e-4 M (theirs -0.58); z1 monotone salted decrease, no
valley (theirs too); high-salt recovery to the neutral line for z2/z3.
NOT matched: the low-salt collapse ONSET — their z2 stays swollen to
~1e-4..1e-3 M while ours is already collapsed at 5e-6 M (z3: theirs
~1e-6..1e-5, ours < 3e-6): our G_s-LDA implementation over-binds at low
bulk salt (the u(a kappa_b) -> 1 reference makes the brush interior a
deep self-energy trap regardless of bulk dilution). No hysteresis: the
swollen-start correlation ramp converges smoothly into the same
collapsed states as the down-sweep at every reachable salt.

CAUSE IDENTIFIED (SI Eq. S25): their numerics solve the FULL nonlocal
Green-function equation
  -div[eps grad G] + 2 I0(r) G + int dr'' 2 Iex(r,r'') G(r'',r') = delta,
iterating G itself in their mixing loop (their SI numerical section
lists G among the updated fields). Iex is the NONLOCAL chain-
connectivity contribution of the polymer charges. Our implementation is
the S33 LDA of the LOCAL part only: it counts polymer charges in I0 as
if they were free screening ions, overestimating kappa_local and hence
the self-energy trap depth -- ion over-binding and premature collapse
at low bulk salt, exactly the observed direction. Where bulk salt
dominates the screening (0.3 M Fig. 1, the Fig. 2 valley/reexpansion,
Fig. 4 layering) the LDA is valid and agreement is quantitative; their
own scaling results are likewise restricted to rho_b >> alpha rho_P
(S34). Full agreement at low salt would require implementing the
nonlocal S25 solver (contour-pair kernel Iex included) -- a
substantially larger project, noted as the natural next step.

**Fig. 3** (`prl_scft_fig3.png`, panels formatted as theirs): their
plotted Gamma curves are reproduced by our algebraic Eq.-7 evaluation —
z3 minimum -1.27 @ 0.2 M vs their ~-1.3 @ ~0.15 M, z2 minimum -0.20
(theirs ~-0.25), z1 positive throughout with the shallow dip; the
full-SCFT local Gamma tracks the same shapes at reduced amplitude.
psi_S (normalized by sigma N v / h): positive bump ONLY for z+=3, z1/z2
negative approaching zero — their central asymmetry. Amplitude of the
z3 psi_S bump is smaller in our SCFT wall value (+0.4 normalized) than
their curve (+2.5, which follows their Donnan estimate Eq. 9).

**Fig. 4(c,d)** (`prl_scft_fig4cd.png`; 1D, z+=3, rho_b = 1 mM,
sigma = 0.1): reducing ONLY the cation Born radius (new `aplus` param)
a+ = 2.5 -> 2.0 -> 1.5 A gives smooth collapsed layers (phi = 0.76,
0.80) for the first two and, at 1.5 A, **stationary oscillatory layers
through the collapsed film — period ~1.1 nm ~ b, amplitude ±0.06,
converged to 5e-4** — their predicted normal-direction microphase
separation, obtained as a genuine converged solution. (Retrospective:
the §3e grid-scale UV instability was this physical layering mode below
threshold; at a+ = 1.5 A it is supercritical with a wavelength safely
above the Born-smoothing scale.)

**Fig. 4(a,b)** (`prl_scft_fig4ab.png`; NEW 2D code `prl_scft2d.py` —
x-periodic × z-hard-wall, same physics promoted to 2D, sparse-LU
guarded-Newton PB). Two-stage story:
- First runs used effectively ANNEALED grafting (global-mass
  normalization only) and showed dramatic lateral dewetting with
  phi > 1 — an artifact. Fixed by the quenched-grafting division of
  SI Eq. S30 (q~(r;1) = src/q_dagger(r;1)), which pins the lateral
  graft distribution to uniform sigma; verified rho_bead1 ∝ src exactly.
- With quenched grafting at their conditions (sigma = 0.03, z+=3,
  1 mM): the uniform collapsed film is LINEARLY stable (noise decays at
  all correlation strengths), but finite-amplitude stripe seeds at 15,
  20, 30 nm spacing ALL converge (err 1e-3) to laterally structured
  states with contrast 1.5-1.8: **an array of pinned dense domes
  (phi ~ 0.75, height ~8 nm, width ~13 nm, near-bare gaps) — the 2D
  cross-section of their pinned micelles**, coexisting with the
  metastable uniform film (first-order lateral transition with a
  nucleation barrier). Spacing selection among 15-30 nm is nearly
  degenerate in this pilot; picking the equilibrium spacing needs a
  free-energy evaluator (not in the 2D pilot) — future work, as is the
  3D hexagonal arrangement.

## 3j. Fig. 2 SOLVED: canonical monovalent counterions + Anderson mixing
(2026-08-27)

The remaining Fig.-2 discrepancy (no swollen plateau, collapse at all
reachable low salt) had a MODEL cause, not (primarily) the LDA: their
ensemble is SEMICANONICAL with a separate species of MONOVALENT
counterions (SI: n_C fixed, z_C = +1) — our GC-only implementation let
the multivalent salt cations neutralize the brush at any dilution, so
z^2-strong correlations collapsed it everywhere. Physically, their
collapse onset rho*_b is the 1:z ION-EXCHANGE threshold: at low salt
the entropic cost of importing multivalent ions from a dilute reservoir
keeps the brush neutralized by its own monovalent counterions (weak
z^2=1 correlations -> osmotic swollen); their S34 ("counterions
negligible at rho_b >> alpha rho_P") delimits exactly where our old
model was valid — matching our old high-salt agreement.

Implementation: `counterions=True` — canonical monovalent cloud (total
|zP| sigma N per area) inside the guarded-Newton PB (canonical
normalization rank-1 Jacobian term left to the line search), u_C = u_m
(same valence/radius), valence-corrected I0. Control experiment: the
polymer-exclusion I0 bracket (`include_polymer_I0=False`) does NOT
restore the swollen branch — the counterion species is the dominant
factor.

Also: with the smooth real-space map, the repo ANDERSON MIXING now
works (it was abandoned during the spectral-junk era and never retried
— user's catch): benchmark 270 vs 12560 iterations (46x), 36 vs 296 s
for identical states. The AM-first driver finished the z2/z3 campaigns
in 4-9 min each vs hours.

Final Fig. 2 (`prl_scft_fig2.png`, counterion curves spliced with the
v9 high-salt tails where counterions are negligible): common osmotic
h0 = 51-52 nm (matches their implied common h0 ~ 52); z1 plateau +
monotone salted decrease; z2 S-collapse crossing neutral at ~8e-3 M,
valley -0.448 @ 0.1-0.2 M (theirs -0.45); z3 crossing ~6e-4 M, valley
-0.559 (theirs -0.58), recovery to neutral. Nested ordering identical
to theirs. Residual gap: multivalent onsets ~1 decade above theirs
(G_s-LDA truncation; their nonlocal G with Iex is the remaining
refinement).

CONSISTENCY CHECK (Fig-1 conditions with counterions, `fig1_ccheck.py`):
z2 gives h = 23.6, Gamma = +0.11 (GC-only: 18.65 / -0.056; paper: ~19b
with overcharging); z3 gives h = 17.2 (GC-only 14.46; paper ~14). So at
0.3 M our counterion model has the 1:z ion exchange only PARTLY
completed, while the paper's states there are fully exchanged. This is
the SAME ~1-decade lag as the Fig.-2 onsets — one root cause: the
G_s-LDA truncation under-favors multivalent condensation relative to
their nonlocal G by about a decade in rho_b. Unified picture:
- GC-only calculation == the fully-exchanged limit == their high-salt
  states -> quantitative Fig. 1/3/4 agreement (kept as the Fig. 1
  comparison of record);
- counterion model reproduces the exchange transition itself (Fig. 2's
  full S-structure) with the onset lag;
- the single remaining refinement (nonlocal S25 G_s with Iex) should
  reconcile both simultaneously.

## 3k. Final figure set at tol = 1e-6 (2026-08-27, user-requested)

All production figures regenerated at W-residual tolerance 1e-6
(user instruction; the AM solver reaches it in O(100) iterations per
point) with PRL-matched axis limits. Key facts:

- **Fig. 2** (`fig2fine4_z{1,2,3}.npz`, 72/77/80 points, 10-20
  pts/decade, single semicanonical model — the earlier kinks came from
  (i) splicing GC-only high-salt tails onto counterion curves and
  (ii) AM stopping-scatter at tol 5e-4 exciting the soft height mode;
  both eliminated): max|d2 log h| = 0.0025 (z1). Full S-curves;
  valleys z2 -0.43, z3 -0.56; recovery to neutral; onsets ~1 decade
  above theirs (documented LDA lag).
- **Fig. 3** rebuilt from the same runs (Gamma_loc, Eq.-7 Gamma, psi_S
  saved per point after AA ion-sector equilibration).
- **Fig. 4 model choice**: the paper-facing comparison uses GC-only
  (= fully-exchanged limit) at their face-value rho_b = 1 mM, rerun at
  tol 1e-6 (`fig4gc_a*.npz`); the semicanonical model at face-value
  1 mM sits pre-exchange (no layers/micelles — consistent with the
  lag), and at lag-corrected 10 mM restores collapse/micelles
  (2D contrast 1.5, spacing 20 nm) but weakens the (c,d) layering
  because shifting rho_b also shifts the reference screening kappa_b —
  the lag-correction is exchange-faithful but not layering-faithful.
  All three treatments recorded (`fig4cdC_*`, `fig4cdC10_*`,
  `fig4abC_*`).

## 3l. The onset-lag mystery SOLVED: it is the box, not the theory
(2026-08-28)

Two decisive experiments after implementing the nonlocal G_s:

1. **Stage A (exact inhomogeneous G_s, S25 local part)**: implemented as
   `selfenergy="nonlocal"` — per transverse wavenumber q, the exact 1D
   Green function of [(q^2-d2/dz2)/4 pi lB + 2 I0(z)] via half-line
   continued-fraction recursions (O(Nz) per q, overflow-free), Gaussian
   spread fixed analytically (|h(k)|^2 = e^{-k^2 a^2/pi}, derived from
   their u(x) signature; homogeneous limit reproduces S33 to ~1% at
   dz=0.125 — built-in validation; one bug found: discrete-delta
   normalization G = A^{-1}/dz). VERDICT: the z2 sweep with exact G_s
   overlaps the LDA curve (neutral crossing 6.7e-3 vs 8.4e-3 M, valley
   identical): **the LDA is an excellent approximation here; stage A
   moves the onset by only ~0.1 decade** (`prl_scft_stageA_z2.png`).
   Their m_G lazy-update trick (SI: update G every m_G iterations;
   ours: u_every=10) adopted for the slow-mixing fallback, ~10x.
2. **Box-length contrast (L = 400/800/1600 nm, z2 LDA semicanonical)**:
   neutral crossing = 1.77e-2 / 8.5e-3 / 3.9e-3 M — **exactly
   proportional to 1/L** (c* L ~ 0.4 n_C). The canonical monovalent
   counterion cloud dilutes into the reservoir volume, so the 1:z
   ion-exchange threshold is set by the counterion reservoir
   concentration n_C/L, NOT by bulk thermodynamics. Extrapolation: their
   z2 crossing (~1e-3 M) corresponds to L ~ 6-7 um.

CONCLUSION: the "~1-decade onset lag" attributed in 3j to the G_s-LDA
truncation is in fact an ENSEMBLE-GEOMETRY effect — the semicanonical
onset position is controlled by the unreported reservoir size. The
paper and SI contain NO statement of box size, grid, or discretization
(searched: box, Lz, system size, grid, lattice, domain, simulation
cell, discretiz) — a genuine reproducibility gap uncovered by this
reproduction. Corollaries: (i) the LDA is fully rehabilitated; (ii)
Iex (stage B) is NOT implicated in the Fig-2 onset; (iii) our Fig-2
curves can be brought onto theirs by the single knob L (a fit, not a
derivation) — L ~ 6.5 um for z2 if desired.

## 3m. Final graph-aligned Fig. 2/3: GC monovalent background c1
(2026-08-28)

Reformulation (user-driven): the canonical counterion cloud in a finite
box is physically a MONOVALENT RESERVOIR; its transparent form is a GC
1:1 background salt at concentration c1 (`set_c1`) — the experimental
residual/buffer level. Consequences: (i) box-independence restored;
(ii) L shrinks to the chain scale (L = 100 nm = 10 R0 validated: h
shift 0.1% vs L=200, wall density 3e-9) since c1 >= 6 mM keeps
kappa_b^-1 < 4 nm at ALL salts; (iii) the paper's full 8-decade range
(1e-7..1 M) becomes reachable. Together with AM + per-point parallel
warm starts (30-job chunks) a full 92-point sweep at tol 1e-6 runs in
MINUTES (vs 6+ h serial).

Calibration: the ion-exchange crossing responds as ~c1^1.2 (z2) and
~c1^2.7 (z3) — no single c1 aligns both with the paper (their z2/z3
crossing ratio 33 vs this theory tier's ~14; the ~0.4-decade residual
is the irreducible signature of physics beyond G_s-LDA+background, e.g.
their Iex). Graph-reproduction choice (user goal): per-valence c1,
LABELED on the figure — z1: 10 mM (insensitive), z2: 6 mM (crossing
9.2e-4 vs their ~1e-3), z3: 50 mM (crossing 2.1e-5 vs their ~3e-5).
Valleys: z2 -0.42 (theirs -0.45), z3 -0.49 (theirs -0.58; the 50 mM
background screening shallows it — documented trade-off). One recovery-
branch spike at 0.22 M (z3) repaired by adiabatic recompute from its
left neighbor. Figures `prl_scft_fig2.png` / `prl_scft_fig3.png` rebuilt
on the paper's axes; data `fig2c1_z1_*`, `fig2c1f_z2_*`,
`fig2c1g_z3_*.npz`.

## 3n. Final-code rerun of the complete figure set (2026-08-28)

At user request, ALL figure data was regenerated on the final code
version (single code state: real-space hard wall, wall-centered graft,
guarded PB, Born-regularized LDA, c1/counterion machinery present, AM,
tol 1e-6), removing the code-era patchwork:
- Fig 1 (`fig1F_z*.npz`): neutral 21.81; z1 36.52->32.19; z2
  32.57->18.90; z3 30.28->14.46 — confirms the earlier v9 values
  (differences < 0.3 nm from the tighter tolerance alone).
- Fig 3 SCFT curves (`fig3F_z*_*.npz`): GC sweep rerun, all points
  converged, observables via AA ion-sector re-equilibration.
- Fig 2 (c1-aligned) and Fig 4 data were already final-code.
- Fig 3 psi_S finding recorded: the paper's orange curve is their
  Eq.-9 Donnan estimate by shape, but its amplitude (+2.5) is NOT
  reproducible from their own plotted Gamma via Eq. 9 (would give +5);
  our Eq.-9 (+3.9) and SCFT wall value (+0.9) bracket it. Their curve
  originates in unpublished numerics — reproducibility gap alongside
  the unreported box/discretization.
Master builder: `dh_salt_runs/make_all_figs.py` regenerates all four
figures from the canonical npz sets.

FIG-4 DIFFERENCE STATEMENT (user-requested, recorded): the visual
difference of Fig. 4(a,b) from the paper is DIMENSIONALITY — their
hexagonal pinned-micelle lattice is a 3D calculation; ours is a 2D
(x,z) pilot in which micelles can only appear as stripe cross-sections
by construction. The lateral microphase separation itself (metastable
uniform film; converged dome arrays phi~0.75, spacing 15-30 nm,
near-bare gaps) is reproduced; the hexagonal arrangement requires a 3D
extension (not undertaken). Fig. 4(c,d) matches in substance (layering
at a+ = 1.5 A) and differs only in presentation (1D profiles vs their
density renders).

## 4. Open items

1. ~~Clean MF baselines, neutral, z1~~ (done, v6, §3d), ~~side-by-side
   figure~~ (done: `prl_scft_fig1.png` + Part I of PRL_COMPARISON.tex),
   ~~Fig. 2/3 sweeps~~ (done, §3f).
2. Low-salt completion of the Fig.-2 curve: bigger box (L ~ 400-800 nm)
   for the osmotic regime + two-sided (swollen-start) continuation to
   bracket the equilibrium collapse transition; rerun the two frozen z3
   high-salt points.
3. CL-FTS vs this reimplemented SCFT (the user-ordered final step):
   same-model comparison now meaningful since both codes share the
   microscopic parameter set.
4. Possible upstream repo issue: spectral propagator garbage at
   per-chain field contrast > ~37 kT (see §3c and memory note
   `spectral-leakage-strong-fields`); a per-step floor clamp in the C++
   solvers would fix it.

## 5. Data

All in `~/polymer/dh_salt_runs/` (outside the repo). Current (v9,
FINAL): `fig1v9_z{0,1,2,3}.npz` + logs `prlv9_z*.out`; salt sweeps
`fig23v9_z{1,2,3}.npz` (v7 sweeps `fig23v7_*.npz` kept, ion observables
re-equilibrated in-place). Historical: v1 `fig1_z*.npz`, v2
`fig1v2_z*.npz`, v6 `fig1v6_*.npz` (+ per-cscale z3 states), v7/v8
`fig1v{7,8}_z*.npz`. Drivers: `prl_fig1_run6.py` (ladder + corr ramp),
`prl_fig23_run.py` (salt continuation), `prl_eigdiag.py` (Jacobian power
iteration), plot scripts `plot_fig1v9.py` etc. Figures (repo,
gitignored): `prl_scft_fig1.png`, `prl_scft_fig23.png`,
`graft_bump_decomposition.png`.

<!-- END RESULTS_PRL_SCFT.md -->


---

# Source: SAMPLING_SCFT_COMPARISON.md

SHA-256: `64c70123d315f6750759b14cbaac970a7e8e34ab94905028f89d6940078524b3`

<!-- BEGIN SAMPLING_SCFT_COMPARISON.md -->
# Direct comparison of 1D sampling and PRL-type SCFT

The samples are closer to mean-field SCFT with correlations disabled than to smoothed-LDA SCFT with correlations.
This is not a comparison of sampling methods for the same action.

## Comparison method

- MC: the completed finite planar sampler with N=100, A=40 nm², L=200 nm, M=400.
- LDA: the existing smoothed-LDA SCFT using PRL parameters. Equivalence to the full nonlocal theory in the paper has not been validated.
- MF: the same SCFT with correlation self-energies disabled (uK=0), retaining the mean Coulomb potential and canonical C.
- SCFT was recomputed at the measured MC bulk concentrations for 15 nonzero salt points. The MF salt-free reference was solved separately.
- Retained parameters: N=100, b=1 nm, σ=.1 nm⁻², v=1 nm³, χ=.5, ℓB=.7 nm, a=.25 nm, and canonical C with Nσ=10 nm⁻².
- Concentration means bulk cation molarity, distinct from ionic strength. 1 M = .6022 nm⁻³.
- Height is h=2∫zρP dz/(σN). Normalized curves use each model's independently computed h0, without matching to a shared h0.
- SCFT M=1600 densities were averaged over four cells onto MC M=400. Γ uses the same .5 peak core definition.
- SCFT potentials were interpolated to z=.25/199.75 nm and compared as differences at the same locations as MC, not as absolute gauges.
- MC error bars are 1 SE and exclude model bias and SCFT approximation errors.

## Height [nm]

| z+ | Target ρb [M] | Matched ρb [M] | MC ± SE | LDA SCFT | MF SCFT | MC−LDA | MC−MF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 58.561 ± 0.351 | 50.911 | 59.602 | +7.650 | -1.041 |
| 1 | 0.001 | 0.000995724 | 57.517 ± 0.355 | 49.698 | 58.383 | +7.819 | -0.867 |
| 1 | 0.01 | 0.01004805 | 54.627 ± 0.257 | 46.342 | 54.827 | +8.285 | -0.199 |
| 1 | 0.1 | 0.1001598 | 43.878 ± 0.173 | 37.104 | 43.745 | +6.774 | +0.133 |
| 1 | 1 | 0.9999946 | 28.489 ± 0.213 | 25.785 | 27.795 | +2.704 | +0.694 |
| 1 | 10 | 9.996082 | 23.572 ± 0.141 | 22.366 | 22.490 | +1.206 | +1.082 |
| 2 | 0.001 | 0.00101316 | 54.308 ± 0.223 | 44.068 | 55.637 | +10.241 | -1.329 |
| 2 | 0.01 | 0.009974988 | 46.019 ± 0.211 | 29.379 | 46.943 | +16.640 | -0.924 |
| 2 | 0.1 | 0.10019 | 33.043 ± 0.186 | 19.511 | 33.037 | +13.532 | +0.006 |
| 2 | 1 | 0.9999455 | 24.858 ± 0.170 | 21.180 | 23.931 | +3.678 | +0.927 |
| 2 | 10 | 10.00168 | 23.054 ± 0.165 | 21.930 | 22.015 | +1.124 | +1.038 |
| 3 | 0.001 | 0.001005687 | 50.213 ± 0.290 | 29.942 | 52.821 | +20.272 | -2.607 |
| 3 | 0.01 | 0.01003352 | 39.282 ± 0.354 | 14.121 | 40.366 | +25.161 | -1.084 |
| 3 | 0.1 | 0.09985221 | 28.597 ± 0.330 | 14.899 | 28.395 | +13.698 | +0.203 |
| 3 | 1 | 1.000078 | 23.796 ± 0.189 | 20.005 | 22.881 | +3.791 | +0.916 |
| 3 | 10 | 10.00312 | 22.797 ± 0.294 | 21.842 | 21.897 | +0.956 | +0.901 |

## Charge and potential: trivalent salt

| Target ρb [M] | MC Γ ± SE | LDA Γ | MF Γ | MC Ψ ± SE | LDA Ψ | MF Ψ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.7774 ± 0.0029 | 0.4179 | 0.8396 | -11.8280 ± 0.3145 | -5.6654 | -14.4799 |
| 0.01 | 0.4856 ± 0.0056 | 0.0480 | 0.5089 | -4.5453 ± 0.3697 | -0.2740 | -5.2131 |
| 0.1 | 0.3098 ± 0.0040 | -0.3966 | 0.3238 | -0.5994 ± 0.2435 | 0.7841 | -0.8601 |
| 1 | 0.3292 ± 0.0033 | -0.3357 | 0.3292 | 0.0296 ± 0.1224 | 0.0964 | -0.0930 |
| 10 | 0.3500 ± 0.0152 | 0.2238 | 0.3349 | -0.0342 ± 0.1406 | -0.0063 | -0.0093 |

## Interpretation

The large multivalent contraction and re-expansion in LDA-SCFT are absent from MC. MC heights and densities are much closer to MF.
MC and MF nevertheless differ: finite A, solvent entropy from integer cell occupancy, grid, charge smearing, and actual 1D fluctuations all matter.
A 1D mean density ρ(z) differs from restricting charge fluctuations to 1D. The PRL correlation theory includes a 3D Green function.
Current MC samples a planar Coulomb action retaining only transverse wave number q_parallel=0. More iterations cannot restore excluded transverse correlation modes.
These results motivate investigating model differences first; this comparison does not establish an MC algorithm error or failure of the original CL.
The paper neglects uP^ex but retains the individual-charge term uP^0. That omission alone is not an established cause of the discrepancy.
[Paper and Eqs. 3–5](https://arxiv.org/html/2404.09103v3), [saved SI](../../references/2025_prl_duan_si.pdf).
PRL squares in the plots are digitized theory curves. The paper uses a Gibbs dividing-surface height; our height is a first moment.

## Numerical checks

- Maximum full SCFT residual 8.762e-08, independent log-chain density error 2.017e-13, contour Q error 1.566e-12.
- Matching nominal concentrations to the actual MC bath changes LDA height by at most 0.053068 nm.
- Maximum profile rebinning integral error 4.547e-13. Polymer/C integrals were each checked against 10 nm⁻².
- The failed initial MF salt-free attempt is preserved. A retry from the existing smooth LDA solution converged without changing residual thresholds or equations.

Raw data: `/home/yongdd/polymer/dh_salt_runs/sampling_scft_20260907`. `cases.json` records every MC–SCFT mapping.
`comparison.json`, `comparison.csv`, and `comparison_profiles.npz` contain numbers and full profiles. `jobs.jsonl` and each `run.sh` preserve execution details.
[Height plot](prl_comparison_figs/sampling_scft_20260907/mc_scft_height.pdf) · [Charge/potential plot](prl_comparison_figs/sampling_scft_20260907/mc_scft_charge.pdf) · [Density profiles](prl_comparison_figs/sampling_scft_20260907/mc_scft_profiles.pdf)

<!-- END SAMPLING_SCFT_COMPARISON.md -->


---

# Source: THEORY.md

SHA-256: `7e81364d9c14893de3c855d822b94492d0962642cdfec4f51f3188930ed16401`

<!-- BEGIN THEORY.md -->
# Charged-Polymer FTS: Theory and Implementation Plan

> Working notes (devel). Formulation of polyelectrolyte solutions within this
> codebase's multi-monomer field theory, following the smeared-charge model of
> Wang [1] and Villet–Delaney–Fredrickson [2,3], written in the conventions of
> *Macromolecules* **2025**, 58, 816 (the neutral multi-monomer theory this
> library implements). **Compressible (finite $\zeta N$) model with
> per-species density smearing** — the fully UV-regularized formulation.

## 1. Model

Species $i = 1, \dots, S$ (here: polymer P, solvent S, counter-ion C, salt
ions SP/SM). Each species is a (possibly very short) bead-spring chain of the
discrete or continuous model already implemented.

**Smeared densities.** Every species carries a normalized Gaussian shape
function
$$h_i(\mathbf r) = \frac{1}{(2\pi a_i^2)^{3/2}} e^{-r^2/2a_i^2},
  \qquad \hat h_i(\mathbf k) = e^{-a_i^2 k^2/2},$$
of radius $a_i$ (params `radiuses`, in $R_0$ units). $a_i$ is the
**smearing length** (Fredrickson school: width of the "shape function" or
form factor of the species [2,3]); for charged species Wang [1] identifies
it with the ion's **Born radius**, since the Gaussian self-energy
$u_{\rm self} = z_i^2 l_B/(2\sqrt{\pi}\, a_i)$ is the Born solvation
energy of an ion of that size. ALL interactions are written in terms of the
smeared volume fractions
$$\bar\phi_i \equiv h_i * \phi_i .$$
Smearing regularizes both the contact (excluded-volume/χ) and the Coulomb
self-interactions, making the fluctuating field theory UV-convergent
[1,2,3]. The point-particle limit $a_i \to 0$ is singular for a compressible
fluctuating model and must not be taken. ($a_i = $ `None` ⇒ $h_i = \delta$;
acceptable only for testing or in mean-field.)

Three interaction channels:

1. **Flory–Huggins**: $\chi N_{ij}$ pairs (here only P,S — all pairs
   involving ions are zero). No separate `chi_monomers` machinery is needed
   in the compressible model: the $\chi N$ matrix is simply $S \times S$
   with zero rows/columns for the ions (see §3).
2. **Compressibility (Helfand)**: instead of the incompressibility
   constraint, a finite penalty
   $$\beta U_{\zeta} = \frac{\rho_0 \zeta N}{2 N} \int
     \Big( \sum_i \bar\phi_i(\mathbf r) - 1 \Big)^2 d\mathbf r$$
   with the code's existing `zeta_n` parameter. The incompressible theory is
   the $\zeta N \to \infty$ limit. In a fluctuating simulation the smeared
   $\bar\phi_i$ (not raw $\phi_i$) must enter here, or the contact penalty
   is UV-divergent.
3. **Coulomb**: segment of species $i$ carries per-segment valence $z_i$;
   the smeared charge density is
   $$c(\mathbf r) = \sum_i z_i\, \bar\phi_i(\mathbf r)
     \qquad (\text{same } h_i \text{ as above}),$$
   and
   $$\beta U_C = \frac{l_B \rho_0^2}{2} \int\!\!\int
      \frac{c(\mathbf r)\, c(\mathbf r')}{|\mathbf r-\mathbf r'|}
      \, d\mathbf r\, d\mathbf r',
      \qquad l_B = \frac{e^2}{4\pi \varepsilon k_B T}$$
   (Bjerrum length; uniform dielectric $\varepsilon$, no dielectric
   contrast in this first version).

**Global electroneutrality** is still required (compressibility does not fix
charge). With per-segment valences $z_i$ and equal segment volumes:
$$\sum_i z_i\, \bar\phi_i^{\,\rm tot} = 0$$
over overall volume fractions (NO division by chain length — that would be
per-chain counting, wrong for per-segment valences). The counter-ion
fraction is NOT a free parameter: for P with $z_P = +1$ and fraction
$\phi_P$, one needs $\phi_C = \phi_P z_P/|z_C|$ (plus balanced salt
$z_{SP}\phi_{SP} + z_{SM}\phi_{SM} = 0$). *The current `Lamella.py`
placeholder sets $\phi_C = 0$ with charged P — not electroneutral; the
$k=0$ Coulomb mode diverges. The driver must compute $\phi_C$ from
$\phi_P$.*

## 2. Units and code conventions

Same as the neutral theory: lengths in $R_0 = a_{\rm Ref} N_{\rm Ref}^{1/2}$,
fields per **reference chain** ($w_{\rm code} = N\,w_{\rm per\,segment}$),
densities in $\rho_0$. Chain number density
$C = \rho_0 R_0^3 / N = \sqrt{\bar N}$.

Nondimensionalize with $\mathbf x = \mathbf r/R_0$. The Coulomb energy in
$k_BT$ becomes (note the $C^2$ — Coulomb is quadratic in $\rho_0$, unlike
the χ/ζ block where $\chi N, \zeta N \propto \rho_0$ leave one $C$):
$$\beta U_C = \frac{C^2 E_0}{8\pi}\int\!\!\int
   \frac{c(\mathbf x)\,c(\mathbf x')}{|\mathbf x-\mathbf x'|}\,
   d\mathbf x\, d\mathbf x'
 = \frac{C^2 E_0}{2} \sum_{\mathbf k} \frac{|\hat c_{\mathbf k}|^2}{k^2},
 \qquad E_0 \equiv \frac{4\pi l_B N^2}{R_0}.$$
For the per-chain-normalized Hamiltonian used throughout this codebase, the
convenient coupling is
$$\boxed{\;E \;\equiv\; C\,E_0 \;=\; \frac{4\pi l_B N^2 C}{R_0}
   \;=\; 4\pi l_B\, \rho_0 N R_0^2\;}$$
(density-DEPENDENT: $E = \sqrt{\bar N}\,E_0$). Caveat: the literature's
"electrostatic strength" [2,3] is the density-independent $E_0$, usually in
$R_g = R_0/\sqrt6$ units — convert when comparing. New scalar param:
`"bjerrum_e"` $= E$ (or `"bjerrum_length"` $= l_B/R_0$ with $E$ computed
internally from `nbar`).

## 3. Field theory: Hubbard–Stratonovich structure

Two independent HS blocks (the compressible model has NO separate pressure
constraint):

**(a) χ+ζ block — the existing compressible SPT, over ALL $S$ species.**
The quadratic form of channels 1+2 is
$$\tfrac12\, \bar{\boldsymbol\phi}^{\mathsf T}
  \big( \zeta N\, \mathbf J + \boldsymbol\chi N \big) \bar{\boldsymbol\phi}
  \;-\; \zeta N\, \mathbf 1^{\mathsf T}\bar{\boldsymbol\phi} + \tfrac12 \zeta N,$$
with $\mathbf J$ the all-ones matrix — exactly the matrix
$u = \zeta N\,\mathbf J + \chi N$ that `SymmetricPolymerTheory`'s
compressible branch (`zeta_n` set) already diagonalizes, including the
linear term (its `vector_large_s`). Ion rows of $\chi N$ are zero. Spectrum
for the 5-species example:
- one large positive eigenvalue $\approx S\zeta N$ (the "pressure-like"
  mode) — imaginary-type;
- the χ-split P–S pair: one negative (exchange, **real** Langevin field),
  one positive (imaginary-type);
- pure ion-composition modes orthogonal to $\mathbf 1$ with no χ:
  eigenvalue **0** — ideal mixing in those channels; the corresponding
  auxiliary fields are identically zero and SPT's existing zero-eigenvalue
  handling (Gram–Schmidt + warning) applies. This is physical, not an
  error: those composition fluctuations are free.

Because $\zeta N\,\mathbf J$ lifts the would-be singularity of the χ-only
matrix, **the full $S$-species SPT can be used directly** — the
chi-subset wrapper of the incompressible draft is no longer needed.

**(b) Coulomb block — new.** HS on the positive-definite Coulomb quadratic
form introduces the electrostatic potential $\psi(\mathbf x)$:
$$e^{-\beta U_C} = \int \mathcal D\psi\;
  \exp\!\left[ - C \int d\mathbf x\, \frac{|\nabla\psi|^2}{2E}
               \;+\; i\, C \int d\mathbf x\; c(\mathbf x)\,\psi(\mathbf x) \right]
  \Big/ \mathcal N,$$
which reproduces $\exp[-\tfrac{C^2E_0}{2}\sum_k |\hat c_k|^2/k^2]$ exactly
because $E = CE_0$ (Gaussian integral over $\psi$). $\psi$ is the
per-reference-chain potential $N\beta e\varphi_{\rm phys}$. Coupling with
$+i$ to a real density and a positive-definite kernel ($4\pi l_B/k^2$,
analogous to $\lambda > 0$): $\psi$ is an **imaginary-type field**,
partial-saddled like the positive-eigenvalue SPT fields — see §5.

## 4. Hamiltonian and single-chain problem

The per-chain effective Hamiltonian is
$$\frac{H[\{w_k\}, \psi]}{C\,k_BT V/R_0^3}
 = \underbrace{h_{\rm const} + \sum_k \left[ A_k \langle w_k\rangle + B_k \langle w_k^2\rangle\right]}_{\text{neutral compressible (existing SPT coefficients)}}
 \;+\; \frac{1}{V}\int \frac{|\nabla \psi|^2}{2E} d\mathbf x
 \;-\; \sum_p \frac{\bar\phi_p}{\alpha_p} \ln Q_p[\{W_i\}]$$
with the $h$-coefficients of the existing compressible SPT. The one-body
potential of each species picks up the smearing convolution on EVERY term
(the interactions couple to $\bar\phi_i = h_i*\phi_i$, so the functional
derivative w.r.t. $\phi_i$ convolves back with $h_i$):
$$\boxed{\;W_i(\mathbf x) \;=\; h_i * \Big[ W_i^{\rm SPT}
   \;+\; i\, z_i\, \psi \Big](\mathbf x)\;}$$
where $W_i^{\rm SPT}$ is the usual `matrix_a` mapping of the auxiliary
fields (now including the pressure-like ζ mode; ion rows couple only to
that mode and to $\psi$). No explicit $N$ multiplies the electrostatic
term: $\psi$ is already per-reference-chain and $z_i$ is per-segment — the
code's $w = N w_{\rm seg}$ convention is carried entirely by $\psi$. The
smearing is a k-space multiplication
$\hat h_i(k)\,[\widehat{W^{\rm SPT}_i} + i z_i \hat\psi](k)$ — cheap, per
species, applied once per saddle iteration; the propagator solvers are
untouched.

**Rotated contour / stored real field.** As with the positive-eigenvalue
SPT modes, the code stores the REAL array $\psi_s = -i\psi$ (the saddle
lies on the imaginary $\psi$ axis). Under this substitution the one-body
term becomes $+z_i \psi_s$ (the $i$ disappears — this is what the code
adds to $W_i^{\rm SPT}$), and the explicit quadratic term flips sign:
$$\frac{1}{V}\int \frac{|\nabla\psi|^2}{2E}\, d\mathbf x
  \;\longrightarrow\;
  -\frac{1}{V}\int \frac{|\nabla\psi_s|^2}{2E}\, d\mathbf x
  \;=\; -\frac{1}{2V}\int c\,\psi_s\, d\mathbf x$$
(the last equality at the Poisson-consistent $\psi_s$). This mirrors the
negated $B_k$ coefficients of the imaginary SPT fields. Note the
$+c\psi_s$ coupling itself lives inside $-\ln Q$ via $W_i$; the explicit
term recorded in $H$ is only the (negative) gradient piece.

Concentrations: standard $\phi_i$ from propagators; the smeared densities
that enter all interaction terms and the Poisson source are
$\bar\phi_i = h_i * \phi_i$, $c = \sum_i z_i \bar\phi_i$ (again diagonal
in k).

## 5. Forces and saddle conditions

- SPT auxiliary fields: unchanged from the neutral compressible theory —
  functional derivatives evaluated with the SMEARED densities
  $\bar\phi_i$ in place of $\phi_i$. Negative-eigenvalue (exchange) fields
  receive Langevin noise; positive-eigenvalue fields are partial-saddled by
  the existing compressor loop; zero-eigenvalue fields stay zero.
- Electrostatic field:
$$\frac{1}{C}\frac{\delta H}{\delta \psi(\mathbf x)}
  = -\frac{\nabla^2 \psi}{E} \;-\; c(\mathbf x)
  \qquad\Longrightarrow\qquad
  \text{saddle: } -\nabla^2\psi^* = E\, c(\mathbf x),$$
the (smeared) **Poisson equation**; with the ideal-gas ions responding
through their Boltzmann factors this is the fluctuating-field
generalization of Poisson–Boltzmann. In k-space the solve is diagonal and
exact:
$$\hat\psi^*(\mathbf k) = \frac{E\,\hat c(\mathbf k)}{k^2}, \qquad k \ne 0,$$
and the $k=0$ mode is fixed by electroneutrality ($\hat c(0) = 0$ holds
exactly per configuration in the canonical ensemble once §1's constraint
holds; set $\hat\psi(0)=0$ as gauge).

**Debye sanity check** (1-segment ions, $z=\pm1$, ideal): linearizing gives
$\kappa^2 R_0^2 = E\,\bar\phi_{\rm ion}/N = 4\pi l_B \rho_0
\bar\phi_{\rm ion} R_0^2$ — the physical Debye constant. Any factor error
in $E$ or $W_i$ shows up here first; keep this as a unit test.

**CL-FTS (the real implementation): $\psi$ fluctuates.** L-FTS holds
imaginary-type fields at partial saddle, so it can never capture $\psi$
fluctuations — the charged theory belongs in CL-FTS, where $\psi$ evolves
with complex Langevin dynamics exactly like the pressure field $W_+$
(rotated storage, imaginary-direction noise, "+$\Lambda$" drift):
$$\psi \;\leftarrow\; \psi + \left(\frac{\nabla^2\psi}{E} + c\right)
  \Delta t\, s_\psi \;+\; i\,\mathcal N(0,\sigma)\sqrt{s_\psi},$$
whose drift fixed point is the Poisson equation. Three discrete-scheme
facts (all found the hard way):
- The $\nabla^2/E$ part is stiff at high $k$: explicit Euler is UNSTABLE
  ($k^2\Delta t/E$ exceeds the stability limit), and semi-implicit Euler,
  while stable, SUPPRESSES the stationary variance by $1/(1+2\kappa\Delta
  t)$ per mode — this silently wrecks fluctuation thermodynamics (the
  one-loop $\langle\partial H/\partial E\rangle$ came out $2$–$4\times$
  too small). The correct treatment is **ETD (exact Ornstein–Uhlenbeck
  integration per k-mode)**:
  $$\hat\psi' = a_k\hat\psi + (1-a_k)\frac{E\hat c}{k^2}
    + i\,\sigma\sqrt{\frac{1-a_k^2}{2\gamma_k\Delta t}}\;\hat\xi,
    \qquad a_k = e^{-\gamma_k \Delta t s_\psi},\;\; \gamma_k = k^2/E,$$
  which reproduces the exact per-mode stationary variance of the linear
  part for ANY $\Delta t$; the $c[\psi]$ screening stays explicit
  (per-step rate $\le \sum_i z_i^2\bar\phi_i\,\Delta t$). The $k=0$ mode
  is gauged to zero every step.
- The EXCHANGE field's explicit mass is $\sim 1/\chi N$, so small-$\chi$
  systems (e.g. the pure-salt validation) need $\Delta t < \chi N$;
  violating this diverges regardless of the charge sector.
- The explicit Hamiltonian term must be evaluated ANALYTICALLY off the
  saddle: $+(1/2EV)\int \psi\nabla^2\psi$ (equal to
  $-\tfrac12\langle c\psi\rangle$ only when Poisson holds).

At $E=0$ the $\psi$ stiffness is infinite: $\psi$ stays pinned at zero.

**L-FTS prototype (mean-field $\psi$): scheme per saddle iteration** —
after each propagator solve, (i) build $\bar\phi_i$, $c$ (k-space
$\hat h_i$ multiplications), (ii) update $\psi$ toward the Poisson
solution (below), (iii) take the usual Anderson-mixing step on the
imaginary SPT fields. Both residuals enter the stopping criterion. $\psi$
stays OUT of `aux_fields_imag_idx` and the AM compressor state.

**Stabilized $\psi$ update (important).** Replacing $\psi$ by the bare
solution $E\hat c/k^2$ every iteration is an undamped fixed-point sweep
whose linearized gain is
$$g(k) \;=\; \frac{E}{k^2}\sum_i z_i^2\,\bar\phi_i\,\hat h_i^2(k),$$
which exceeds 1 at long wavelengths for realistic $E$ (e.g. $E=25$,
$k_{\min}=2\pi/L$) — the joint saddle loop then diverges. Use instead the
preconditioned Newton update with the ideal (local) screening estimate
$S_{\rm scr}(k) = \sum_i z_i^2 \bar\phi_i \hat h_i^2(k)$:
$$\hat\psi \;\leftarrow\; \hat\psi
  \;+\; \frac{E\,\hat c - k^2\,\hat\psi}{\,k^2 + E\,S_{\rm scr}(k)\,},
  \qquad \hat\psi(0)=0 .$$
The fixed point is unchanged ($-\nabla^2\psi = E c$, monitored via the
residual $-\nabla^2\psi/E - c$), but the screening term in the denominator
damps exactly the modes the bare sweep amplifies; for ideal 1-segment ions
the update is a true Newton step. $S_{\rm scr}$ only sets the convergence
rate, so the segment-fraction estimate suffices for polymeric charges.

## 6. Discretization notes

- All new operations are diagonal in k-space: $\hat h_i(k)$ multiplications
  and the $E/k^2$ Poisson solve. Use the existing FFT objects and the
  deformation-vector $|\mathbf k|^2$ tables (`Pseudo`) — oblique cells work
  automatically.
- **No real-space $\nabla\psi$ is ever needed.** The electrostatic energy
  is evaluated by Parseval, and at the saddle the substitution
  $\hat\psi = E\hat c/k^2$ eliminates the gradient entirely:
  $$\frac{1}{2E}\int |\nabla\psi|^2 d\mathbf x
    = \frac{E}{2}\,\frac{V}{M^2}\sum_{\mathbf k \ne 0} w_t\,
      \frac{|\hat c_{\mathbf k}|^2}{k^2}
    = \frac12 \int c\,\psi\, d\mathbf x$$
  (unnormalized `rfftn`; $w_t$ = 2 interior / 1 edge modes, as in
  `wtmd.py`). Only $h_i * \psi$ requires an inverse transform. A real-space
  gradient (component-wise $ik_\alpha \hat\psi$) first becomes necessary
  for the deferred Maxwell-stress term. The identity above gives the
  MAGNITUDE; on the rotated contour the term entering the recorded $H$ is
  $-\tfrac{1}{2V}\int c\,\psi_s$ (see §4) — the code evaluates it as
  `-0.5*mean(c*psi)`.
- Smearing widths must satisfy $a_i \gtrsim dx/2$ **per species** — the
  binding constraint is $\min_i a_i$; below that the regularization becomes
  grid-dependent (same issue class as the bond-function study). With finite
  $\zeta N$ this applies to ALL species, not just charged ones.
- Per-species radii compose automatically: the effective $i$–$j$ pair
  kernel is $\hat h_i(k)\hat h_j(k) = e^{-(a_i^2+a_j^2)k^2/2}$, i.e. an
  effective pair smearing width $\sqrt{a_i^2+a_j^2}$ — Gaussian widths add
  in quadrature, so mixed radii (small ions, fat polymer segments) need no
  special treatment. Consequence for validation: with unequal radii the
  RPA/Debye–Hückel comparison must keep the per-species $\hat h_i^2$
  factors separate (they only factor out as a common $\hat h^2$ when all
  radii are equal); the $k \to 0$ Debye limit is radius-independent
  ($\hat h_i(0) = 1$), which is why it is the safe unit test.
- Charge–charge structure factor $S_{cc}(k)$ from $\hat c(k)$: record
  alongside the existing $S(k)$.
- Stress with electrostatics (Maxwell term from
  $\partial_L \int |\nabla\psi|^2/2E$) and with the $\hat h_i(k)$ cell
  dependence: defer; raise on `box_is_altering=True`.

## 7. Parameter dictionary (target)

```python
"zeta_n": 100.0,                     # compressible Helfand penalty (existing param)
"charges":  {"P": 1.0, "S": None, "C": -1.0, "SP": +1, "SM": -1.0},
"radiuses": {"P": 0.025, "S": 0.025, "C": 0.025, "SP": 0.025, "SM": 0.025},
"bjerrum_e": 10000.0,                # E = 4 pi l_B rho_0 N R0^2   (NEW)
"chi_n": {"P,S": 50},                # full-matrix SPT; ion rows zero
# NO "chi_monomers" needed; NO incompressibility constraint
# molecules: counter-ion fraction computed from electroneutrality, not free
```

`None` charge = neutral species (no $\psi$ coupling). `radiuses` now apply
to ALL interactions (χ, ζ, Coulomb) of that species — in the compressible
fluctuating model neutral species need smearing too.

## 8. Implementation (devel-only; CL-FTS is the primary implementation)

**Where the code lives (2026-08-02)**: entirely in THIS folder — mainline
`src/` is untouched.
- `electrostatics.py`: per-species smearing kernels (real AND complex
  fields), coupling $E$, electrostatic Hamiltonian, electroneutrality
  check, plus the $\psi$ Newton update used by the L-FTS prototype.
- `clfts_charged.py` — **the primary implementation**:
  `ChargedCLFTS(polymerfts.clfts.CLFTS)` with $\psi$ as a fully
  fluctuating CL field (§5): overrides `compute_concentrations` (complex
  $W_i = h_i*(W^{\rm SPT}_i + z_i\psi)$) and `run` (co-evolves $\psi$
  semi-implicitly, adds $-\tfrac12\langle c\psi\rangle$ to $H$, saves
  $\psi$ in checkpoints). $\psi$ noise uses a separate PCG64 stream
  (seed+1) so the neutral reduction stays bit-exact.
- `lfts_charged.py` — mean-field-$\psi$ PROTOTYPE kept for validation
  only: its deterministic saddle makes the sampling-free Debye–Hückel
  linear-response test possible (`test_debye_huckel.py`), which pins the
  $E$ normalization/kernels shared with the CL class.

L-FTS samples only the REAL exchange fields and holds imaginary-type
fields at partial saddle, so $\psi$ there is mean-field
(Poisson–Boltzmann level) — that is why CL-FTS is the real home of the
theory. The compressible choice simplifies everything versus the
incompressible draft:

1. `polymer_field_theory.py`: **unchanged** — the compressible branch
   (`zeta_n`) over the full $S$-species matrix already produces the right
   eigen-system, including the zero-eigenvalue ion modes (existing
   handling).
2. Relation to the existing global `Smearing` class: that class implements
   the SPECIAL CASE of one common radius applied to all fields — the
   prototype bypasses it (adapter), never applying both to the same field.
3. Ions as short chains work today (`length=0.01` ⇒ 1 discrete segment);
   no C++ changes — everything new is k-space Python.
4. Validation ladder:
   a. $E \to 0$, uniform $a_i$: must reproduce the existing neutral
      compressible run with global smearing exactly.
   b. $\zeta N \to \infty$, $E \to 0$: approach the incompressible neutral
      results.
   c. Salt-only (no polymer): $S_{cc}(k)$ against smeared Debye–Hückel/RPA,
      $S_{cc}^{-1} = [S_{cc}^{\rm ideal}]^{-1} + E/k^2$ where
      $S_{cc}^{\rm ideal}$ is the ideal structure factor of the SMEARED
      charge density (contains the $\hat h_i^2$ factors — with a bare ideal
      $S$ one must write $\hat h^2 S/(1 + (E/k^2)\hat h^2 S)$; the
      $\hat h^2$ cannot be dropped). $k \to 0$ recovers the Debye
      $\kappa^2$ of §5.
   d. Literature anchor: polyelectrolyte solution structure of Refs. [2,3]
      (mind the $E_0$ vs $E$ and $R_g$ vs $R_0$ conversions of §2).

**Validation status (2026-08-02)**:
- (a) Neutral reduction PASSED bit-exactly for BOTH classes
  (`test_neutral_reduction.py` for `ChargedLFTS`,
  `test_neutral_reduction_clfts.py` for `ChargedCLFTS`): $z=0$ with
  uniform $a_i$ reproduces the neutral run with global smearing to the
  last bit over 10 Langevin steps.
- (c) Debye–Hückel PASSED. Deterministic linear response
  (`test_debye_huckel.py`, via the L-FTS prototype's saddle): perturbing
  $w_-$ by $\varepsilon\cos(k_0x)$ reproduces
  $|\hat c| = \varepsilon\hat h^2/(1+E\hat h^2/k_0^2)$ and
  $\hat\psi = E\hat c/k_0^2$ to $10^{-6}$ for
  $E \in \lbrace 0, 6.25, 25\rbrace$ over six $k_0$ — pins the $E$
  normalization, $\hat h^2$ pair kernel and $W_i$ assembly without any
  sampling.
- Stochastic $S_{cc}$: with the exact single-mode Gaussian model (orig
  variables: $H_k = Bw^2 + \kappa\psi_o^2 - u_2(w+i\psi_o)^2$,
  $B = 1/\chi N$, $u_2 = \hat h^2/2$, $\kappa = k^2/2E$), the
  parameter-free predictions match: L-FTS
  $r(k) = \kappa^2(B-u_2)/[(\kappa+u_2)(B\kappa+Bu_2-u_2\kappa-2u_2^2)]$
  agrees with 20k-step runs to 1–3% except the lowest-$k$ shell; CL-FTS
  with the ANALYTIC estimator $\langle c_k c_{-k}\rangle$ (NOT
  $\langle|c_k|^2\rangle$, which is not an observable in CL) follows
  $r(k) = (\kappa-B)(B-u_2)/[(B-u_2)(\kappa+u_2)+u_2^2]$ including its
  characteristic sign flip at $k = \sqrt{2EB}$; the strongest-signal
  shells agree to a few % ($E=25$, $k_{\min}$: $-3.08$ vs $-3.07$).
  Residual deviations were pinned down quantitatively: (i) a finite-$dt$
  integrator bias — a synthetic single-mode replica of the exact discrete
  scheme reproduces the $dt{=}0.2$ run values and converges to the
  continuum prediction as $dt \to 0$, and real $dt{=}0.05$ runs move onto
  the synthetic curve (e.g. $k{=}4.71$: run $-2.164$ vs synthetic
  $-2.172$, continuum $-2.31$); (ii) shells where the Gaussian signal
  crosses zero are dominated by a nonlinear (beyond-Gaussian) background
  that does not shrink with statistics — they are not usable for this
  comparison.
- **One-loop DH thermodynamics PASSED (the CL-only validation)**: the
  analytic observable $\langle\partial H/\partial E\rangle = -\langle
  H_{\rm exp}\rangle/E$ measured over $E \in \lbrace 2.5, 6.25, 12.5,
  25, 50\rbrace$ (two $\Delta t$, linearly extrapolated) matches the
  exact-lattice Gaussian prediction $-(1/2\sqrt{\bar n}V)\sum_{k\ne0}
  (\kappa/E)(B-u_2)/\det(k)$ — the smeared generalization of the
  Debye–Hückel limiting law $-\kappa_D^3/12\pi$ — to $\le 0.1\%$ at every
  coupling (ratios 0.999–1.001). This free energy comes entirely from
  $\psi$ FLUCTUATIONS and is identically zero in the partial-saddle
  L-FTS. (Script: dh_salt_runs/test_oneloop_dh.py.)
- Pitfalls recorded the hard way: $\chi N = 2$ IS the mean-field spinodal
  of the symmetric 1-segment salt (RPA breaks); cosine amplitudes on the
  cell-centered grid carry an $e^{ik\,dx/2}$ phase — compare $|\hat c|$;
  the exchange field's explicit mass $1/\chi N$ caps the CL time step at
  $\Delta t < \chi N$; semi-implicit $\psi$ integration biases per-mode
  variances (use ETD, above).

## References

1. Z.-G. Wang, *Phys. Rev. E* **81**, 021501 (2010) — fluctuation theory of
   electrolytes, self-energy and smearing.
2. M. V. Villet, G. H. Fredrickson, *J. Chem. Phys.* **141**, 224115 (2014) —
   CL-FTS with Gaussian-regularized (UV-convergent) models.
3. R. A. Riggleman, R. Kumar, G. H. Fredrickson, *J. Chem. Phys.* **136**,
   024903 (2012) — polyelectrolyte complexation with smeared charges
   (sharper primary reference for the charged CL-FTS model).
4. K. T. Delaney, G. H. Fredrickson, *J. Phys. Chem. B* **120**, 7615 (2016) —
   perspective on fully fluctuating polyelectrolyte FTS.
5. *Macromolecules* **2025**, 58, 816 — the neutral multi-monomer framework
   (this library).

<!-- END THEORY.md -->


---

# Source: VALIDATION_STATUS.md

SHA-256: `b8151fdf7d875c8d4eeffb88aca33cade6a27dcae1488e5f2c08580d89cd3c54`

<!-- BEGIN VALIDATION_STATUS.md -->
# CL-FTS implementation and PRL comparison validation — 2026-09-07

**Current verdict: charged CL still disagrees with exact references; quantitative PRL reproduction is incomplete.**

## Latest comparison — actual MC and PRL-type SCFT

Smoothed-LDA SCFT and MF SCFT with correlations disabled were solved directly at 15 measured MC bulk concentrations.
A salt-free MF reference was added. MC heights and densities are closer to MF than LDA, but this does not imply statistical agreement.
For trivalent salt near .01 M, h is MC 39.28±.35 nm, LDA 14.12 nm, and MF 40.37 nm.
This compares models: MC excludes transverse fluctuations, while LDA approximates local 3D correlations.
Numbers, definitions, and checks are in [SAMPLING_SCFT_COMPARISON.md](SAMPLING_SCFT_COMPARISON.md).

## Latest addition — actual 1D sampling of an N=100 brush

The earlier Fig. 2–3 scan used SCFT and did not fulfill the user's sampling request.
A separate positive particle/real-field Monte Carlo implementation samples incompressible P/S,
canonical monovalent C, and grand-canonical z:1 salt.
It was validated against exact sums for small systems. N=100 runs cover three valences, varying salt concentration, independent initial states,
particle-only controls, and grid/cross-sectional-area checks. Each salt activity was determined using separate bulk MC.

This samples an explicit finite planar action, not an action claimed identical to the existing CL.
Area A sets 1D fluctuation strength and transverse charge fluctuations are excluded, so this does not establish full 3D PRL reproduction.
Autocorrelation, SE, additional relaxation, and remaining scope are recorded in [POSITIVE_BRUSH_RESULTS.md](POSITIVE_BRUSH_RESULTS.md);
equations are in [POSITIVE_BRUSH_SAMPLING.md](POSITIVE_BRUSH_SAMPLING.md).
New opening PDF sections also distinguish actual samples from earlier SCFT results.
Production FFT/C++/CUDA and the existing CL drift were unchanged during this work.

## Additional runs — 1D sampler and PRL Fig. 1

Real-field/particle Gibbs sampling was run on 4, 16, and 32 cells without exhaustive enumeration.
Four-cell results were compared with exact sums; larger grids were compared with particle-only sampling.
PRL Fig. 1 mean-field and correlated curves were recalculated at Nz=1600/3200.
The neutral curve agrees with the paper, but current LDA curves differ in interior phi by up to about .02–.03.
Grid changes at the same positions are below 2e-5. GC-only mean-field controls agree well with the paper's dashed curves;
ensemble and box effects are recorded separately. The current nonlocal backend is also not validated as a full-SI implementation.
[ONED_PRL_REPRODUCTION.md](ONED_PRL_REPRODUCTION.md) records numbers, figures, failures, and scope.

## Additional investigation — another representation of the same action recovers exact averages

For the two-cell model, the repulsive auxiliary field and psi were integrated out exactly, leaving one real field to sample.
The physical action is unchanged, with no approximation or parameter tuning.
Across 12 runs covering E=0/25/100, two dt values, and two seeds, all 19 observables agree with exact values within 1.36 estimated SE.
At E=100, the C density `.20719224 ± .00005738` recovers the exact `.20718911`.

Stored samples from existing CL were also analyzed. At E=100, about .2% lie within .1 of a partition zero;
contributions excluding those samples still retain more than 99.94% of the C density bias.
The checked low-order `<L O>=0` conditions pass despite incorrect means.
This strengthens the evidence for a sampled-distribution problem in the small system.
It does not establish the dynamical role of zeros or a complete boundary term.

The new real sampler uses exhaustive enumeration for the small system and is not a replacement for 3D production code.
Existing CL and PRL reproduction remain unvalidated. FFT and production CL drift were unchanged.
Equations, results, and raw data are in [CL_POLE_RESOLUTION.md](CL_POLE_RESOLUTION.md).

## Latest causal investigation — CL convergence failure for the same model

Twenty-one further trajectories and four direct field integrals were added.
Exhaustive particle sums and the original field integral agree in densities and field moments within `2.8e-15`.
An independent implementation of the same stochastic field-difference equations matches native densities, forces, noise variances,
and actual one-step updates.

At `E=100`, `dt=.00015625`, Langevin time 20,000, exact counterion density is `.20718911`,
versus `.21321148 ± .00020726` for existing LM/ETD, `.21314436 ± .00021742` for Heun,
and `.21331872 ± .00019318` for adaptive Euler.
Field means with no Q denominator also deviate by about 50 SE.
Smaller time steps, longer sampling, alternative integrators, and drift-adaptive steps did not resolve the discrepancy.

Large drifts occur near **actual partition zeros** of the chain or SP.
At each trajectory's maximum-drift state, native densities and independent calculations agree with a maximum scaled error of `1.98e-10`.
The evidence strongly indicates incorrect CL convergence in this small model.
A rigorous limiting proof and complete boundary-term analysis remain outstanding.

**Long runs also reveal a discrepancy in the same small model with charge disabled.**
The earlier short E=0 control's statistical agreement must not be generalized into a validation pass.
This differs from the existing neutral-reduction test, which checks equivalence of two code paths.

[CL_CONVERGENCE_DIAGNOSIS.md](CL_CONVERGENCE_DIAGNOSIS.md) records equations, checks, raw data, and interpretation limits.
This follow-up added/modified only validation Python and documentation; production FFT and CL drift were unchanged.
A validated sampling method or an established range of applicable parameters is still needed.

## Earlier additional validation — exact finite particle systems versus CL

An independent reference enumerates all particle configurations of the same finite-grid action:
256 configurations for two cells and 65,536 for four cells.
These small canonical validation systems are neither the PRL model nor continuum calculations. All 24 SLURM calculations completed.

- Six `E=0` controls agree with reference densities for every species/position and with reference heights within estimated statistical errors.
- With charge enabled, monovalent counterion densities differ by **5.2–9.8 estimated SE** even at `dt=.005`. Errors account for block means and autocorrelation.
- At `dt=.0025` with another seed, E=100 counterion density is `.21308439 ± .00112541` versus exact `.20718911`, a difference of about 5.24 SE.
- Four-cell runs can match mean height while missing P density at some positions.
- Mass/partition checks passed. Independent chain/particle densities recomputed at final fields also agree with native results within about `1e-14`.

Chain-calculation consistency must therefore be distinguished from the accuracy of CL sample averages.
Further smaller-dt comparisons and drift/partition cancellation diagnostics are in [EXACT_CL_VALIDATION.md](EXACT_CL_VALIDATION.md).
At this stage, the data alone did not distinguish time discretization, undersampled rare events, and incorrect CL convergence.
This work added validation/analysis Python and records; existing FFT and CL drift were unchanged.

The following records describe the earlier validation of 2026-09-06.

Strong-field error detection, a stable reference chain calculation, and quenched grafting were added.
Attempted automatic corrections to the existing CPU/CUDA discrete spectral solver were withdrawn at the user's request.
Strong-field errors in native FFT/DST remain unresolved; follow-up is recorded as item 65 of [TODO.md](../../TODO.md).
MF/CL comparisons for the same model and MF recalculations at PRL parameters were run.
Existing result files were not overwritten with recalculated results.

The paper is [Duan, Agrawal & Wang, PRL 134, 048101 (2025)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.048101).
Grafting/ensemble checks use S29–S32 of the saved [SI](../../references/2025_prl_duan_si.pdf).
The detailed pre-change audit is [CLFTS_PRL_AUDIT.md](/home/yongdd/polymer/dh_salt_runs/audit_20260906/CLFTS_PRL_AUDIT.md).

## 1. Implemented changes

- **Partition checks:** eight CPU/CUDA, continuous/discrete, normal/reduce-memory implementations replace absolute errors of magnitudes with **complex relative errors**. Contour Q values are compared with the Q actually used for density normalization. The existing `1e-7` tolerance is unchanged. Zero/nonfinite Q also fails.
- **Strong-field reference:** [stable_grafted_chain.py](stable_grafted_chain.py) adds NumPy/CuPy Gaussian bond matrices. Chain propagation stores magnitudes and logarithmic scales separately; densities use one partition function. Complex propagators and densities are not clipped. Hamiltonians use `log Q` to avoid underflow of Q itself. Only rounding errors in the positive real Gaussian kernel are removed during kernel construction.
- **Quenched grafting:** each lateral anchor is normalized by its own Q and uses `mean(log Q_anchor)`, distinct from annealed grafting with a uniform source and global Q. Zero width fixes the anchor to one grid cell; finite width averages its z location over the source.
- **CL diagnostics/restart:** H and structure functions use consistent pre-update fields, densities, and Q; checkpoint densities correspond to stored post-update fields. Separate psi RNG state, graft/model settings, shifts, and log Q are saved. `initial_params_json` preserves integer graft keys. At `E=0`, psi and electric energy are fixed to zero. ETD coefficients use accurate `expm1` at small arguments.
- **PRL MF comparison:** removed the invalid `psi -= psi[-1]` shift at fixed salt fugacity. PB nonconvergence fails; `solve()` reports final self-energy and Poisson residuals as well as W. Cell-centered integrals consistently use `sum * dz`. Direct chain calculations are selectable. Existing `height()` is explicitly a **`2 <z>` moment estimator**.
- Added an assertion where the Debye–Hückel test previously printed failure but exited successfully.
- **Mixed-valence Γ:** the external production driver `dh_salt_runs/test_brush_cl.py` now multiplies C and SP excess densities by `z_counter` and `zplus`, respectively. Mean/block Γ and error bars use the same corrected expression; species block profiles and sample counts are now saved.

**Existing FFT/DST propagation paths were restored to their state before automatic correction.**
Relative partition checks and the Python direct backend remain. The direct calculation is a validation alternative.
Default brush settings remain `spectral`/`annealed` for compatibility.
The new backend supports single-block linear homopolymers with one segment species;
it does not optimize large 3D production performance or support branched/multiblock chains.

## 2. Deterministic validation results

| Check | Result |
|---|---|
| Installed `ctest -L basic` | 56/56 passed, 36.7 s |
| New `TestPartitionFunctionCheck`, complex nonperiodic mirror | 2/2 passed |
| Existing CL neutral reduction | Final w difference 0 |
| Existing Debye–Hückel checks | 18 combinations passed |
| Continuous Gyroid SCFT | 221 iterations, residual `8.04e-9`, H `-0.674220654` |
| Quenched CL force–H finite differences, real/imaginary field std=5 each | Maximum error `9.99e-11` |
| CPU/CUDA densities at identical fields | Maximum difference `5.62e-16` |
| Fixed-anchor column density | Maximum deviation from 1: `2.56e-16` |
| Annealed anchors at the same lateral field | Column CV `1.084`; the ensembles differ |
| 4-step uninterrupted run versus 2-step checkpoint restart | Maximum field/density/psi difference `5.33e-15`; identical psi RNG state |
| Stored density versus recomputation from stored fields | Maximum difference `5.59e-16` |
| Initial H time-step consistency | Error 0 |
| E=0, periodic/wall | Removes preexisting nonzero psi and returns finite H/densities |
| Fixed-fugacity Donnan check | Potential `-0.2161641104`, maximum charge error `1.39e-17` |
| Nonconvergence with fixed incorrect self-energy | Reports failure after the iteration cap |

The strong-field counterexample uses N=50, nz=704, L=6, a **Dirichlet image** bond kernel with absorbing ends,
and per-chain W=100 for z<3. Its original field, grid, and chain parameters were not weakened.

| Calculation | Q / log Q | Mean height | Mass error |
|---|---|---|---|
| Existing spectral CPU | Q=`-1.797e-18` | 4.183237 | 2.55% |
| Existing spectral CUDA | Q=`-1.599e-18` | 4.201706 | 1.70% |
| New direct CPU/CUDA | log Q=`-62.9319184541`, Q=`+4.667e-28` | 3.076019224 | At most `5.6e-15` |

Updated partition checks reject both erroneous spectral cases and accept zero-field controls.
Adding a uniform `1000+20j` to per-bead potentials changes direct-backend densities by at most `5.4e-14`.
This distinguishes **small Q** from a **spatially inaccurate propagator**.

## 3. Connecting MF and CL for the same model

[validate_brush_sampling.py](validate_brush_sampling.py) uses a canonical five-species model in 2D:
8×32, L=(1.2,2), N=10, E=25, zetaN=20, chiPSN=10, with C=1 and SP=3 valences.
MF and CL use identical new quenched grafting and truncated-bottom/reflecting-top bonds.
Dynamic stabilization is disabled with `alpha_ds=0`. **These benchmark numbers are not the PRL physical conditions.**

The full active-field MF residual is `5.14e-12`; MF `<z>=0.4518632443`.

| nbar | dt | seed | Production Langevin time | CL mean height ± estimated SE |
|---|---|---|---|---|
| 1e4 | .05 | 51 | 400 | .451938 ± .001787 |
| 1e4 | .025 | 51 | 400 | .451050 ± .001566 |
| 1e4 | .05 | 52 | 1900 | .452344 ± .000832 |
| 1e4 | .025 | 52 | 1900 | .452869 ± .000809 |
| 1e8 | .05 | 51 | 400 | .451842 ± .000180 |
| 1e8 | .025 | 51 | 400 | .451745 ± .000153 |

SE is the larger of the 20-block estimate and the initial-positive-autocorrelation-sum estimate.
All six runs have mass errors `<3.34e-16` and contour partition errors `<2.56e-15`.
Long-run imaginary means are statistically consistent with zero. The two dt values agree within current errors;
increasing nbar suppresses fluctuations and approaches MF.
This checks consistency under these conditions; **it does not establish dt→0 extrapolation or correct strong-coupling CL convergence**.
Drift samples, complex block profiles for all species, and final checkpoints are preserved.

Refining the MF grid to 16×64 gives `<z>=0.4476186900`, a **0.94%** change.
The fixed-cell anchor representation also changes with the grid; these are not continuum values.
Besides the validation code, the external production driver's effective-charge calculation was corrected to use species valences.
Mean Γ for 26 existing N=50 results was recalculated from original profiles and saved as separate copies.
For example, `n5f_0p01_s12345.json` changes from Γ=`−0.07406685` to `+0.23989397`.
Historical mixed-valence results lack species block profiles, so corrected error bars cannot be reconstructed.
Recalculated copies mark `gamma_blocks` and `gamma_err` as `null` and retain old values in metadata.
Original files are unchanged. Correcting valences does not validate historical trajectories or agreement with PRL observable definitions.
See the [regression/recalculation summary](/home/yongdd/polymer/dh_salt_runs/gamma_correction_20260906/summary.json)
and [recalculated results](/home/yongdd/polymer/dh_salt_runs/gamma_correction_20260906/corrected).

## 4. Mean-field recalculation at PRL parameters

N=100, b=1 nm, v=1 nm³, sigma=0.1 nm⁻², lB=0.7 nm, chi=0.5, Ib=0.3 M, L=200 nm, nz=1600.
Correlations are disabled; the existing reference code's graft source remains Gaussian with width 0.5 b.
Historical MF fields serve only as initial guesses; the corrected equations were solved to convergence again.

| Salt valence | Monovalent counterions | `2 <z>` (nm) | max phiP |
|---|---|---|---|
| 1 | Canonical, total 10 nm⁻² | 35.4650 | .32444 |
| 2 | Canonical, total 10 nm⁻² | 33.0479 | .35251 |
| 3 | Canonical, total 10 nm⁻² | 31.3686 | .37428 |
| 3 | None; GC salt only | 30.2437 | .38689 |

Polymer and canonical C integrals each retain 10 nm⁻².
Maximum final W residual is `6.13e-8`; maximum PB residual is `2.51e-9`.
For identical W/source/boundaries, existing real-space convolution and the new CL direct backend differ in density by at most `2.95e-15`
and in single-bond values by `1.78e-15`. **The same-model comparison restricted to chain calculations passed.**

The new CL backend was tested here for chain statistics. This does not mean that the complete point-ion/semicanonical MF equations
underlying the table are implemented in `BrushChargedCLFTS`.
Values are recalculated references, not errors against digitized paper curves or a PRL reproduction success rate.

## 5. Key remaining requirements for PRL reproduction

1. **Microscopic model:** current charged CL assigns segment volume to ions, with finite compressibility, canonical particle numbers, and Gaussian smearing of all monomer potentials. A separate action/drift is needed for the SI's point ions, incompressible P/S, canonical P/C, and GC salt/solvent.
2. **Physical coupling:** the PRL rho0=1/v convention gives R0=10 nm, `nbar=100`, `E=87964.5943`, and Gaussian radius/R0=`0.01410474`. The present E=25 CL results cannot be extrapolated to this coupling and UV resolution.
3. **Graft/wall:** annealed and quenched grafting differ under fluctuating lateral fields. A DST Dirichlet image kernel also differs from a truncated discrete bond. The paper's delta graft needs a defined finite-volume regularization; narrowing a one-bead delta can cause first-cell volume-occupancy problems.
4. **Correlation theory:** the default smoothed LDA in `prl_scft.py` is not the SI's full nonlocal Green-function variational theory. The pre-change audit also found an asymmetric self-energy Jacobian. There is no basis yet for interpreting `c*F1loop` or `c≈0.5` fitting as an exact free-energy derivation or CL validation.
5. **CL/statistics:** the target 3D model needs checks of dt, grid/box, multiple seeds, drift tails, and partition poles. Even for identical microscopic models, quantitative agreement between the PRL approximation and FTS is not automatic.

## 6. Rerunning and outputs

Select the new brush path explicitly as below. `validate_brush_sampling.py` first solves MF for the chosen conditions,
rather than continuing directly from existing annealed/wall results.

```python
params.update(
    chain_backend="direct",
    grafting="quenched",
    bond_boundaries=["reflecting"] * (len(params["nx"])-1) + ["hard-reflecting"],
    graft_width=0.0,
    graft_center=0.0,
)
```

```bash
conda activate polymerfts
ulimit -s unlimited
export OMP_STACKSIZE=1G OMP_NUM_THREADS=1 OMP_MAX_ACTIVE_LEVELS=0
# Set CUDA_VISIBLE_DEVICES to an idle GPU identified with nvidia-smi.
python devel/charged_polymers/validate_clfts.py --output /shared/validation/deterministic.json
```

Calculations longer than one minute used a separate SLURM job per parameter.
Rerunnable `run.sh` files and raw data are in
[/home/yongdd/polymer/dh_salt_runs/validation_20260906](/home/yongdd/polymer/dh_salt_runs/validation_20260906).
CL jobs: 332819–332822, 332827–332828. Refined MF: 332829. Final PRL MF: 332835–332838.
All are COMPLETED / exit 0. C++ was rebuilt and installed; final source hashes and test logs are preserved.

- [Deterministic checks JSON](/home/yongdd/polymer/dh_salt_runs/validation_20260906/deterministic.json)
- [CL statistics/autocorrelation analysis](/home/yongdd/polymer/dh_salt_runs/validation_20260906/sampling_analysis.json)
- [Validation figure PNG](/home/yongdd/polymer/dh_salt_runs/validation_20260906/validation_summary.png)
- [Validation figure PDF](/home/yongdd/polymer/dh_salt_runs/validation_20260906/validation_summary.pdf)

Existing user work in `RESULTS_FIG1_REPRO.md`, `fig2_reproduction.png`, and `PRL_COMPARISON.*` was preserved.
Those changes did not revalidate historical long trajectories or PRL correlation curves under the new model. No commit had been made at that stage.

## 7. Additional Fig. 2–3 checks — 2026-09-07

[FIG23_PRL_REPRODUCTION.md](FIG23_PRL_REPRODUCTION.md) records a fixed-parameter smoothed-LDA concentration scan.
Validated solutions were obtained at 57 points: three valences × 19 concentrations.
Checks include directly solving the salt-free reference, two height definitions, bidirectional continuation, grid/box effects,
independent log-space chains, and log Q derivatives. Production FFT/C++/CUDA and CL drift were unchanged during this work.

Contraction/re-expansion and the qualitative distinction between effective-charge and potential reversal appear,
but the low-salt contraction onset in Fig. 2 and potential magnitude in Fig. 3 differ from the paper.
In particular, the monovalent Fig. 3 potential differs by about a factor of 0.6022 from Eq. (9) evaluated with M→nm⁻³ conversion,
requiring confirmation of figure units/normalization.
The paper also explicitly neglects `u_P^ex` for brushes, so that omission alone is not an established cause of disagreement.

The later Fig. 2–3 SCFT section of the [current PDF](ONED_PRL_REPRODUCTION.pdf) contains these results.
The [combined PDF](PRL_COMPARISON.pdf) appends the original report marked as historical.
**Existing CL validation and quantitative PRL reproduction remain incomplete.**

<!-- END VALIDATION_STATUS.md -->
