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
