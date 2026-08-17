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
