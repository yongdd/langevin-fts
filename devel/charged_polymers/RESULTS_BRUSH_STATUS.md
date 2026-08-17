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
