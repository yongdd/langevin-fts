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
