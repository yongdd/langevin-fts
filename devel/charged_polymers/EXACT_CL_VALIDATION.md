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
