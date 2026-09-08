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
