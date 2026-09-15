---
title: "PRL reproduction: SCFT and planar Monte Carlo"
subtitle: "Consolidated results, code map and execution protocols"
date: "2026-09-15; brush, electrolyte and McCarty bulk Gaussian calculations"
---

# Scope and current conclusion

This report collects the PRL reproduction work for Duan, Agrawal and Wang,
*Physical Review Letters* **134**, 048101 (2025), including checked SCFT calculations
and the requested N=100 planar Monte Carlo comparison. The results are **partial reproduction,
not a completed quantitative reproduction of the full paper**.

- Fig. 1: neutral and mean-field profiles, correlated profiles, grid refinement,
  ensemble and nonlocal-closure controls were recalculated. Correlated curves retain
  discrepancies larger than the observed grid and raster-reading errors.
- Figs. 2–3: all 57 standard smoothed-LDA concentration points passed numerical checks.
  Multivalent contraction/re-expansion and distinctions between overcharging and
  potential reversal appear, but onset and potential magnitude differ from the paper.
- Actual N=100 MC was subsequently run. Its finite planar action excludes transverse
  charge fluctuations; it lies closer to mean-field SCFT than correlated LDA. More
  samples cannot recover the missing transverse modes.
- Fig. 4 was not revalidated in these checked campaigns. Earlier 2D/scaling claims
  are retained as history and are not included as a successful reproduction result.

The separate [CL-FTS validation report](CLFTS_VALIDATION.pdf) contains native CL
correctness tests, exact finite-particle benchmarks, the reduced CL diagnosis, and
the small nine-bead sampling controls. Those checks do not become PRL reproduction
merely because the models contain charged polymers.

The electrolyte section adds the 2026-09-14 deterministic campaign using
the PRL self-energy. It compares the 2022 PRL bulk coexistence curve and Galvani
potential, tests charged-wall limits, and records incomplete interface and
condensation attempts. These electrolyte results do not upgrade the status of
the Duan brush reproduction or the native CL sampler.

The final section adds the 2026-09-15 implementation of salt-free McCarty
polyampholyte bulk Gaussian thermodynamics. It contains computed sequence
responses and coexistence curves, including a comparison with the paper's
RPA formulas. It does not implement the complete Duan conformational
self-consistency or reproduce McCarty's CL/MD sampling.

# Code used and current organization

Directory organization updated 2026-09-14: models are in `models/`, tests in
`tests/`, and this report and its build script in `reports/`. All code paths in
prose are relative to `devel/charged_polymers`; `workflow.py` commands are unchanged.


The original runs predate the cleanup. Their commands, source snapshots and hashes
remain in their evidence directories. The current implementations are accessed
through `workflow.py`; numerical parameters and algorithms were retained during
consolidation. The pre-cleanup repository revision is `d395ed1a`.

| Calculation | Original script | Current entry / implementation |
|---|---|---|
| Fig. 1, MF/LDA/nonlocal | `run_prl_fig1_checked.py` | `workflow.py prl fig1`; `studies/prl_runs.py:run_fig1` |
| Figs. 2–3 and matched concentrations | `run_prl_fig23_checked.py` | `workflow.py prl salt`; `studies/prl_runs.py:run_salt` |
| Earlier mean-field cross-check | `validate_prl_meanfield.py` | `workflow.py prl meanfield`; `studies/prl_runs.py:run_meanfield` |
| Planar MC and periodic salt bath | `run_positive_brush.py` | `workflow.py sample brush`; `studies/brush_sampling.py:run` |
| Independent MC weight checks | `check_positive_brush.py` | `workflow.py check brush`; `studies/brush_sampling.py:check` |
| Fig. 1 digitization and controls | `compare_oned_prl.py` | `workflow.py analyze fig1-finite`; `studies/prl_analysis.py` |
| Figs. 2–3 digitization and scan | `analyze_prl_fig23.py` | `workflow.py analyze prl-scan`; `studies/prl_analysis.py` |
| MC/SCFT matched comparison | `compare_sampling_scft.py` | `workflow.py analyze matched`; `studies/prl_analysis.py` |
| MC statistics and profiles | `analyze_positive_brush.py` | `workflow.py analyze brush`; `studies/brush_analysis.py` |
| Related electrolyte benchmarks | New 2026-09-14 campaign | `workflow.py prl electrolyte` and `analyze electrolyte`; `studies/literature_scft.py`, `models/prl_electrolyte.py` |

`models/prl_scft.py:PRLBrushSCFT` remains the underlying approximate SCFT model.
`models/positive_brush_1d.py` remains the explicit finite-action MC kernel.
`models/stable_grafted_chain.py` supplies the independent scaled direct-chain reference.
`studies/prl_runs.py` contains the shared coupled Newton solve, extended-range
positive convolution, independent log-chain reference, and canonical PB Jacobian.
These different chain calculations are retained as independent numerical checks.

Since the 2026-09-14 maintenance, analysis commands emit numerical summaries
and link here for model definitions, protocols and interpretation. The shared
note formatter is `reports/__init__.py`; this Markdown is the editable report
source. Reanalysis does not automatically replace its recorded results.

# Execution protocol

## Environment, units, and scheduling

Run from the repository root. Lengths below are nm, densities are number densities
in nm$^{-3}$ unless labeled M, and energies are in thermal units. In particular,
$\rho_b[\mathrm{nm}^{-3}]=0.6022\rho_b[\mathrm M]$;
the salt axis denotes cation molarity, not ionic strength.

```bash
source /home/yongdd/anaconda3/etc/profile.d/conda.sh
conda activate polymerfts
ulimit -s unlimited
export OMP_STACKSIZE=1G OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_MAX_ACTIVE_LEVELS=0
# Numba/llvmlite used by the historical sampling campaigns:
export PYTHONPATH=/home/yongdd/polymer/dh_salt_runs/exact_cl_diagnosis_20260907/python_deps
```

The installed `polymerfts` extension is required by SCFT/reference imports. Numba
and llvmlite were provided by the recorded local dependency directory, not silently
installed into conda. `nvidia-smi` identifies an idle GPU before CUDA checks.
Planar MC itself runs on CPU. Every calculation exceeding one minute is submitted
through SLURM, **one job per concentration, valence, closure, grid, or seed**.
Use the commands below as the body of a job script with the environment above;
submit with `sbatch --cpus-per-task=1 --mem=8G --time=24:00:00 job.sh`.
Resource requests must suit the local partition. Replays use fresh output directories.

## Common SCFT parameters and Fig. 1 sequence

Use $N=100$, $b=1$, $v=1$, $\sigma=0.1$, $\chi=0.5$, $\ell_B=0.7$,
and equal Born radii $a_P=a_+=a_-=0.25$.
Primary runs use canonical monovalent counterions totaling $\sigma N=10$
nm$^{-2}$, GC point salt, incompressible P/S, a half-Gaussian graft of width
$0.5b$, a truncated Gaussian bond at the bottom wall and a reflecting top.
The standard box is $L=200$; Fig. 1 uses $I_b=0.3$ M.

1. Start from the archived MF/correlated warm profile. Interpolate `W`, `psi`,
   `uP`, `up`, and `um` onto the target cell-centered grid. The `--warm` filename
   in each saved `run.sh` identifies the actual dependency.
2. Solve neutral MF and charged $z_+=1,2,3$ MF/LDA cases at `--nz 1600` and
   `--nz 3200`, one job per case. The equal-radius relation $u_K=z_K^2U$ reduces
   the outer solve to coupled variables $V=W/N$ and $U$.
3. The shared Newton–Krylov solve retains `f_tol=3e-8`, `rdiff=1e-5`, at most
   80 outer iterations and 50 inner iterations, with Armijo line search.
   Final acceptance is evaluated from the full SCFT state, not the Newton flag alone.
4. For the difficult nonlocal trivalent control, switch from LDA to the current
   nonlocal closure with fractions 0.1, 0.2, 0.4, 0.6, 0.8, 1.0. Save every stage.
   This changes the continuation path; the final fraction is exactly one.
5. Run GC-only controls and selected $L=400$, fixed-spacing controls separately.
   Do not change background salt or height scale to fit the paper.

Example using a retained warm profile (the actual archived path can be substituted):

```bash
python devel/charged_polymers/workflow.py prl fig1 \
  --zplus 3 --closure lda --nz 3200 --L 200 \
  --warm /path/to/saved/profiles.npz \
  --coupled-newton --out /path/to/new/fig1_z3_lda
```

The original Fig. 1 bundle contains 25 successful PRL runs and three failed
initial attempts. The mixed historical bundle also contains 21 small-system
sampling runs, which are documented in the CL-FTS report rather than counted
as PRL calculations. Failed attempts remain in the evidence but are excluded
from converged curves.

## Figs. 2–3 concentration scan and acceptance

Keep `c1=0`, canonical C, and all physical parameters above. For each of the
three valences, scan the 19 saved points from $10^{-7}$ to 50 M in
`fig23_prl_20260907/rho_grid.json`. The report's run catalog preserves the exact
floating-point concentrations; endpoints alone do not define the grid. The grid is
$\rho_b=10^{k/2}$ M for integers $k=-14,-13,\ldots,3$, plus 50 M;
`rho_grid.json` preserves their actual floating-point encodings.
Solve zero salt directly and also solve the neutral reference.

Use both upward and downward continuation. An accepted previous `profiles.npz`
provides the next warm state. Difficult transitions use additional intermediate
concentrations and, where recorded, up to eight Newton cycles. Each attempt's
`--direction`, warm file, and cycle count are retained; failed points are not
replaced by interpolation.

The accepted scan requires `--chain-solver extended --pb-solver rank1
--reference log`. The chain uses one partition function for all contour
positions. The canonical C normalization derivative is included in a tridiagonal
plus rank-one PB Jacobian. High-salt cases requiring tighter inner convergence
use `--pb-tolerance 1e-10`; final acceptance thresholds are unchanged.

```bash
python devel/charged_polymers/workflow.py prl salt \
  --zplus 3 --rho-molar 0.01 --closure lda --L 200 --nz 1600 \
  --warm /path/to/previous/profiles.npz --direction up \
  --chain-solver extended --pb-solver rank1 --reference log \
  --out /path/to/new/scan_z3_r0.01
```

Acceptance requires all W/Poisson/self-energy/volume residuals below $10^{-7}$,
independent chain density and contour partition errors below $10^{-10}$,
and polymer/C integral errors below $10^{-10}$. States approaching the legacy
exponential clipping bounds are rejected (maximum checked exponent must be
below 290). Selected grid checks double `nz`; box checks double both `L` and
`nz`, keeping the original spacing. No tolerance, field strength or calibrated
validation parameter is weakened to obtain a pass.

The final scan includes 57 accepted standard points, 57 up/down pairs, and
215 accepted extended-range calculations including intermediate/control states.
There are 249 successful and 47 failed summaries overall; early float64 controls
explain the differing counts. SLURM cancellation/resubmission history is retained.

## Actual N=100 sampling: bath first, then brush

Standard MC uses four N=100 chains in $A=40$ nm$^2$, $L=200$, $M=400$,
$\Delta z=0.5$, and integer cell capacity $K=A\Delta z/v=20$.
The finite action, update probabilities and self terms are given in the method
chapter below. It is a planar particle/real-field model, not native CL or the
full 3D PRL action.

1. **Independent exact checks.** Run the 2-cell/2-bead grand-partition sum and
   detailed-balance/energy tests. Tiny production controls use valences 1/2/3,
   modes `joint`/`particles`, seeds 1101/1201, 20,000 warmup sweeps, 1,000,000
   measurement sweeps, scalar stride 5 and profile interval 100 scalar samples.
2. **Bulk calibration.** For each target salt concentration and valence, run a
   pure periodic salt bath. Chemical potential is adjusted only during calibration,
   then frozen. The final standard bath jobs use 20,000 warmup and 200,000
   measurement sweeps, scalar stride 100, and the recorded calibration history.
   There are 18 selected bath references, including area/grid controls.
3. **Brush sampling.** Pass the frozen bath `summary.json` via `--bath`. Use
   targets 0.001, 0.01, 0.1, 1, and 10 M for each valence, plus two salt-free runs.
   Seeds 4101/4201 start extended/compact, respectively. Each standard run uses
   200,000 warmup sweeps and 1,000,000 measurement sweeps. Store scalars every
   100 sweeps and profiles every 1,000 sweeps. A sweep makes N times the chain
   count random single-bead proposals, together with ion and real-field updates.
4. **Documented extension.** At divalent 0.01 M, initial split Rhat=1.09546 and
   compact-start relaxation triggered another 1,000,000 measurement sweeps for
   each seed, resuming `checkpoint.npz` with `--burn 0`. Published means use the
   extension interval. Original intervals are preserved, not treated as extra seeds.
5. **Controls.** Particle-only checks at trivalent 0.01/1 M use seed 5101.
   Grid/area controls use seeds 6101/6201: M=800 at A=40, A=20 with two chains,
   A=80 with eight chains, and salt-free M=800. All retain grafting density 0.1.
   These yield 42 selected brush trajectories. Area changes fluctuation strength;
   it is not just a discretization parameter.
6. **Statistics.** Use the larger of autocorrelation and 10/20/40-block estimates,
   account for between-run scatter, and report ESS and split Rhat. Nonlinear
   observables use mean profiles and block jackknife; include core-boundary
   changes, shared bath uncertainty and potential–height covariance. Error bars
   exclude model bias and finite-grid errors.

Representative commands with actual campaign seeds and settings:

```bash
python devel/charged_polymers/workflow.py sample brush \
  --bulk --valence 3 --rho 0.01 --seed 3303 \
  --burn 20000 --sweeps 200000 --stride 100 \
  --output /path/to/new/bath_z3_r0.01
python devel/charged_polymers/workflow.py sample brush \
  --valence 3 --rho 0.01 --seed 4101 --initial extended \
  --bath /path/to/new/bath_z3_r0.01/summary.json \
  --burn 200000 --sweeps 1000000 --stride 100 --profile-every 10 \
  --output /path/to/new/brush_z3_r0.01_joint_s4101
```

Submit the brush job only after its bath has completed and passed its checks.
For seed 4201 use `--initial compact`. For salt-free runs use `--rho 0` without
a bath. `--mode particles` removes the attractive field analytically; all other
terms of that finite model remain. Checkpoints store model parameters and PCG64
state; `--resume` rejects parameter mismatches.

## Direct MC/LDA/MF comparison

Use the **measured** bath concentration of each MC group, not its nominal target.
Solve 15 nonzero LDA states and 15 nonzero MF states, plus the MF salt-free reference,
with the same checked salt runner. Matched jobs use three Newton cycles and inner
PB tolerance $10^{-10}$. Preserve the failed first MF zero-salt attempt and its
successful retry from a smooth LDA warm state; 31 final states pass.

Average every four SCFT density cells (M=1600) onto one MC cell (M=400), preserving
integrals. Compute the same half-peak core charge ratio. Interpolate potentials
to the same cell centers, $z=0.25$ and $199.75$ nm, and compare their difference.
Normalize heights with each model's directly calculated salt-free height.
`sampling_scft_20260907/cases.json` links every MC group and SCFT state; it also
records the first failed zero-salt attempt. Selection uses the successful retry.

# Evidence and rerunning

All paths below are relative to `/home/yongdd/polymer/dh_salt_runs/`.
The repository copy of `reports/metadata/campaigns.json` records summary hashes, parameters,
script paths and script hashes for the checked campaigns. Raw samples and
checkpoints remain in their original directories.

| Evidence directory | Contents and selection |
|---|---|
| `oned_prl_20260907/` | Fig. 1 runs, closure/ensemble controls, digitized points; `comparison.json` separates PRL and small-system records |
| `fig23_prl_20260907/` | 19-point grid, full scan history, `comparison.json`, `fig23_primary.csv`, up/down pairs and method checks |
| `positive_brush_20260907/` | Frozen baths, 42 selected brush trajectories, 12 final tiny controls, checkpoints and `analysis.json` |
| `sampling_scft_20260907/` | Exact bath matches, `cases.json`, `comparison.csv`, conservative mapped profiles and execution audit |

For an exact historical replay, retain the recorded code snapshot or repository
revision and use the saved `run.sh` with a fresh output path. For the reorganized
code, substitute `python .../workflow.py legacy OLD_SCRIPT.py` for the historical
Python script invocation, preserving its arguments. The legacy name adapter
routes to the consolidated implementation; it does not claim byte identity to
historical source. Numerical refactor checks are documented in the README.

Analysis commands do not run simulations, but may read large sample arrays:

```bash
python devel/charged_polymers/workflow.py analyze prl-scan \
  /path/to/fig23_prl_20260907
python devel/charged_polymers/workflow.py analyze brush \
  /path/to/positive_brush_20260907 --prl-root /path/to/fig23_prl_20260907
python devel/charged_polymers/workflow.py analyze matched \
  /path/to/sampling_scft_20260907 \
  --mc-root /path/to/positive_brush_20260907 \
  --scft-root /path/to/fig23_prl_20260907
python devel/charged_polymers/workflow.py reports
```

The PDF builder compiles the two curated Markdown reports through Pandoc and
LaTeX/Tectonic. It requires no simulation data or Numba to rebuild the committed
report edition. Figures and numeric results in this edition are frozen to the
recorded evidence, not silently recomputed during typesetting.
Successful builds remove intermediates. Add `--keep-build` to the reports
command to retain TeX, logs and build metadata; failed builds retain diagnostics.


# PRL Fig. 1: profiles and closure controls

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


![Fig. 1 profiles and digitized paper values; checked N=100 SCFT, not Monte Carlo.](figures/oned_prl_20260907/prl_fig1_checked.pdf){width=95%}



![Ensemble and nonlocal-closure controls for Fig. 1. The present nonlocal closure is still not the full SI theory.](figures/oned_prl_20260907/prl_closure_comparison.pdf){width=95%}



# PRL Figs. 2–3: fixed-parameter SCFT results

2026-09-07. **Contraction and re-expansion occur, but quantitative reproduction remains incomplete.**
These results check smoothed-LDA SCFT, not the existing CL dynamics or the paper's full SI implementation.
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

During the 2026-09-07 scan campaign, the SCFT model, Fig. 1 runner, existing
CL drift and production C++/CUDA/FFT were protected; all 33 recorded file hashes
matched their launch values. Later folder moves and helper extraction are
tracked separately in `reports/metadata/relocations.json`, including dependency
hashes. Current files are not claimed byte-identical to the historical sources.
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


![Fig. 2: computed heights and digitized theory lines at fixed physical parameters.](figures/fig23_prl_20260907/fig2_checked.pdf){width=95%}



![Fig. 3: analytical formulas and SCFT core charge and relative potential.](figures/fig23_prl_20260907/fig3_checked.pdf){width=95%}



![Potential normalization diagnostic. The 0.6022 factor is shown explicitly and is not applied to the reported calculation.](figures/fig23_prl_20260907/fig3_units_check.pdf){width=95%}



# N=100 planar MC: finite action and update probabilities

Actual N=100 stochastic results are recorded in the following results chapter. The equations below
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


# N=100 planar MC: measured results and convergence controls

The earlier Fig. 2–3 scan used SCFT. These Monte Carlo calculations sample particle configurations and the real field in the attractive channel.
Samples of this finite 1D model were obtained; the PRL's 3D fluctuations and quantitative curves have not been reproduced.


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

$$h=\frac{2}{\sigma N}\int z\langle\rho_P(z)\rangle\,dz,\qquad
\Psi=\frac{h}{\sigma Nv}\bigl(\langle\psi_{\rm wall}\rangle-\langle\psi_{\rm far}\rangle\bigr).$$

$\Gamma$ is the salt-excess ratio in the core above half the mean polymer density peak. Nonlinear observables are computed from mean profiles, with block-jackknife errors.
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


![Actual planar MC heights with statistical errors and the SCFT comparison.](figures/positive_brush_20260907/sampling_fig2.pdf){width=95%}



![Actual planar MC core charge and relative potential.](figures/positive_brush_20260907/sampling_fig3.pdf){width=95%}



![Initial-state, autocorrelation, and grid/area controls for the planar MC campaign.](figures/positive_brush_20260907/sampling_diagnostics.pdf){width=95%}



# Direct comparison at the measured MC bath concentrations

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
[Paper and Eqs. 3–5](https://arxiv.org/html/2404.09103v3), [saved SI](../../../references/2025_prl_duan_si.pdf).
PRL squares in the plots are digitized theory curves. The paper uses a Gibbs dividing-surface height; our height is a first moment.

## Numerical checks

- Maximum full SCFT residual 8.762e-08, independent log-chain density error 2.017e-13, contour Q error 1.566e-12.
- Matching nominal concentrations to the actual MC bath changes LDA height by at most 0.053068 nm.
- Maximum profile rebinning integral error 4.547e-13. Polymer/C integrals were each checked against 10 nm⁻².
- The failed initial MF salt-free attempt is preserved. A retry from the existing smooth LDA solution converged without changing residual thresholds or equations.

Raw data: `/home/yongdd/polymer/dh_salt_runs/sampling_scft_20260907`. `cases.json` records every MC–SCFT mapping.
`comparison.json`, `comparison.csv`, and `comparison_profiles.npz` contain numbers and full profiles. `jobs.jsonl` and each `run.sh` preserve execution details.
[Height plot](figures/sampling_scft_20260907/mc_scft_height.pdf) · [Charge/potential plot](figures/sampling_scft_20260907/mc_scft_charge.pdf) · [Density profiles](figures/sampling_scft_20260907/mc_scft_profiles.pdf)


![Matched MC, LDA and mean-field heights, including each model’s own salt-free normalization.](figures/sampling_scft_20260907/mc_scft_height.pdf){width=95%}



![Matched core charge and relative potential with MC uncertainties.](figures/sampling_scft_20260907/mc_scft_charge.pdf){width=95%}



![Conservatively mapped polymer and ion profiles for the matched comparison.](figures/sampling_scft_20260907/mc_scft_profiles.pdf){width=95%}


# Remaining requirements for quantitative reproduction

Confirm the paper's height extraction, canonical/GC ensemble and boundary/graft
conventions; reproduce its full transverse-mode Green function and self-energy
approximations; resolve the potential normalization with the authors' data; and
perform independent grid, box, area and equilibrium checks for that same model.
The positive planar sampler has a correct finite action, but omits transverse
fluctuations required for the full PRL comparison. Fig. 4 remains outside the
checked reproduction. Historical 2D and scaling claims are in the archived log.

Primary reference: [Duan, Agrawal and Wang, PRL 134, 048101 (2025)](https://doi.org/10.1103/PhysRevLett.134.048101),
main paper and supplementary information stored in the evidence directories.

\clearpage

# Related electrolyte reproduction: literature Topics 1--4

This section records a new deterministic campaign on 2026-09-14. The term
"PRL-SCFT" here means the correlation-augmented framework, including its
electrolyte limit. It does not mean that the existing grafted-chain constructor
can describe every system in the literature collection. The new electrolyte
module adds finite ion volumes and appropriate bulk/wall boundary conditions.
The original brush model and native C++/CUDA/FFT implementations were unchanged.
No CL sampling was performed in this campaign.

## Applicability by system

| Topic and system | Result of this first SCFT attempt | Additional work for the full target |
|---|---|---|
| 1: homogeneous smeared electrolyte | PRL self-energy agrees with independent Fourier integration | Spatially varying dielectric response is not covered |
| 1: counterions at one charged wall | Numerical PB solution matches the finite-cell analytical solution | Strong-coupling correlations and two-wall attraction are not certified by this PB test |
| 2: JCTC-2022 multivalent salt at a charged plate | Two dilute-branch potential profiles agree closely with the paper | Condensed-layer construction and free-energy branch selection are missing |
| 2: PRL-2022 bulk vapor/liquid coexistence | Fig. 1 and the bulk Galvani potential in Fig. 3(a) are closely reproduced | Molecular-volume and exact temperature conventions should still be confirmed |
| 2: PRL-2022 nonlocal interface | A symmetric profile at the displayed $T/T_c=0.9$ converges; its shape is close after translation alignment | Figure density/GDS conventions, lower temperatures, surface tension and asymmetric profiles remain open |
| 3: McCarty polyampholytes | Not reproduced by the brush/local-screening model in this 2026-09-14 campaign | A separate bulk Gaussian implementation was added on 2026-09-15; see the final section for its computed results and limits |
| 3: Grzetic polarizable diblocks | Not reproduced | Drude/permanent-dipole terms, dielectric response and the diblock ordering problem must be implemented |
| 4: Lequieu particle/field comparisons | Not reproduced by this deterministic electrolyte calculation | Match each particle Hamiltonian and ensemble; compare pressure and chemical potential with sampled FTS/MD |

In particular, replacing a polyampholyte by a single mean charge does not
preserve charge sequence. Likewise, a constant-dielectric brush cannot acquire
the polarizable diblock Hamiltonian by changing salt concentration. Ordinary
SCFT may provide a mean-field baseline for some neutral polymer systems, but
that baseline does not reproduce the fluctuation-dependent MD/FTS measurements.

## Source recovery and model reconstruction

The public source archive of
[arXiv:2206.02030v2](https://arxiv.org/abs/2206.02030v2) contains the previously
missing four-page supplementary material for
[Agrawal and Wang, PRL 129, 228001 (2022)](https://doi.org/10.1103/PhysRevLett.129.228001).
It is now stored as
[the local SI](../refs/Agrawal_2022_Vapor_Liquid_Interface_in_Ionic_Fluids_SI.pdf).
The authors' EPS files supply the plotted theory curves. They are vector
digitizations of published figures, not raw simulation samples or uncertainty
estimates. The analysis code records explicit axis calibrations. No interaction
parameter was fitted to the plotted output.

The bulk reconstruction uses equal Born radii $a$ and equal ion/solvent volumes
$v_{\mathrm{phys}}=4\pi a^3/3$. The SI gives general volume-dependent equations
but no complete numerical parameter table; equal volumes are therefore an
explicit reconstruction choice. Agreement with the figures supports this
choice but does not establish a unique reconstruction of the authors' protocol.
The comparison uses each theory's own critical temperature and does not validate
an absolute critical temperature in kelvin.

The absolute self-energy, including the constant Born term, is

$$
u_i(\kappa)=\frac{q_i^2\ell_B}{2a}
\left[1-a\kappa\,\operatorname{erfcx}
\left(\frac{a\kappa}{\sqrt{\pi}}\right)\right].
$$

This is PRL-2022 Eq. (6), JCTC-2022 Eq. (24), and the same scalar expression
used by the existing brush closure. The extension evaluates it with `erfcx`
to avoid the overflow of an intermediate exponential. Its uniform,
polymer-free result was also checked directly against the existing brush
`_self_energies` method.

For an electroneutral salt, define $n=(c_++c_-)a^3$,
$g=q_+q_-\ell_B/a$, $v=v_{\mathrm{phys}}/a^3$, and
$x=\sqrt{4\pi gn}$. The following expressions omit constant Born contributions
and terms linear in $n$, which cancel from coexistence:

$$
\mu_{\mathrm{corr}}=-\frac{g}{2}x\operatorname{erfcx}(x/\sqrt{\pi}),
\qquad
f_{\mathrm{corr}}=-\frac{1}{4\pi}
\int_0^x t^2\operatorname{erfcx}(t/\sqrt{\pi})\,dt,
$$

$$
\mu=\ln\frac{nv}{1-nv}+\mu_{\mathrm{corr}},
\qquad
P=-\frac{\ln(1-nv)}{v}+n\mu_{\mathrm{corr}}-f_{\mathrm{corr}}.
$$

Here $f$ and $P$ are in $k_BT/a^3$ units, and the constant solvent pressure is
subtracted. Spinodals bracket separate vapor and liquid roots, preventing the
trivial equal-density solution. The two phases must have equal $\mu$ and $P$,
positive homogeneous stability, and a common tangent below the sampled
intermediate free energies. The critical point follows from the first and
second concentration derivatives of $\mu$ vanishing.

For equal radii and volumes in an asymmetric salt, the liquid-minus-vapor
Galvani potential follows independently from equality of both ionic chemical
potentials:

$$
\psi_G=-(q_+-q_-)\,[U(n_l)-U(n_v)],
\qquad U=\mu_{\mathrm{corr}}/(q_+q_-).
$$

## Foundation and bulk results

The independent Fourier integral agrees with the analytical self-energy to a
maximum relative error of $1.31\times10^{-14}$. The uniform legacy brush
closure difference is zero at double precision for the checked state.
The counterion-only PB boundary-value calculation uses a cell of length
$40\mu$, a charged left wall and zero field at the right boundary. It agrees
with the finite-cell analytical density to $2.00\times10^{-12}$ relative error;
its integrated mobile charge is $0.9999999999988$ in neutralizing-charge units.
The infinite single-wall PB profile is shown separately because the finite
cell changes the far-field density.

![Independent self-energy and PB checks. The difference between finite-cell and infinite-wall PB at large distance is a boundary-condition effect.](figures/literature_scft_20260914/foundation_checks.pdf){width=95%}


The bulk calculation gives $g_c=12.81473688$ and
$c_c(2a)^3=0.04204145$. At $T/T_c=0.8$, the scaled total vapor and liquid
concentrations are $0.0011527605$ and $0.25317309$. The 2:1 salt has
$\psi_G=2.76676018$ at this temperature; a symmetric salt has zero Galvani
potential.

Ten selected temperatures between approximately $0.5125$ and $0.9930$ were
compared with original vertices of the Fig. 1 theory curve. Across both
branches, the maximum relative concentration difference is **0.647%**.
The largest equality residual for $\mu$ and $P$ in these comparisons is
$4.98\times10^{-14}$. Seven points of the Fig. 3(a) theory curve give a
maximum absolute Galvani-potential difference of **0.001736** in $k_BT/e$ units.
These are differences from plotted theory curves, not confidence intervals or
a fresh comparison with particle simulations. Near-critical normalization and
finite coordinate precision limit the significance of the final digits.

![Bulk coexistence compared with the authors' original Fig. 1 vector curve. The logarithmic panel exposes the dilute branch.](figures/literature_scft_20260914/bulk_coexistence.pdf){width=95%}


![The bulk Galvani potential compared with the original PRL Fig. 3(a) theory curve. This does not require solving the interfacial profile.](figures/literature_scft_20260914/bulk_galvani.pdf){width=80%}


## Charged plate: agreement and the missing condensed branch

The wall calculation follows the 2:1-salt example of
[Agrawal and Wang, JCTC 18, 6271--6280 (2022)](https://doi.org/10.1021/acs.jctc.2c00607).
It uses $c_b=0.2$ M, Born radius $a=0.16$ nm, excluded-volume radius
$a_0=0.25$ nm, $\ell_B=0.7$ nm, and a 10 nm domain. The last value of
$\ell_B$ is an explicit water-like reconstruction value. The local WKB
self-energy is used without the brush code's extra smoothing of ionic strength.
The potential obeys $\psi'(0)=-4\pi\ell_B\sigma$ and $\psi(L)=0$.
Logarithmic occupancy variables enforce positive ion densities and solvent
volume fraction. Ion chemical-equilibrium equations and Poisson's equation
are solved together.

| Surface charge [$e$/nm$^2$] | Calculated wall potential | Maximum difference from the plotted potential |
|---:|---:|---:|
| -0.0100 | -0.02125196 | $4.71\times10^{-5}$ |
| -0.0244 | -0.04467336 | $2.06\times10^{-4}$ |
| -0.0245 | -0.04470443 | $1.45\times10^{-1}$ |

The first two rows use 1600 intervals. Refinement from 800 to 1600 intervals
changes their wall potentials by $6.51\times10^{-6}$ and
$7.07\times10^{-5}$, respectively. The last row uses 800 intervals and is a
**converged dilute-branch solution that does not reproduce the paper's
condensed/inverted state**. A small residual alone does not choose the
thermodynamically relevant branch.

The paper constructs dense and diffuse regions separately and minimizes the
free energy with respect to the boundary-layer thickness. This first wall
implementation does not perform that construction. Accordingly, it establishes
dilute-branch agreement but makes no successful charge-inversion claim.
The mean-field controls visibly miss the correlation correction even before
condensation.

![Two dilute-branch profiles agree closely; at the condensation threshold the continued dilute branch differs qualitatively from the paper.](figures/literature_scft_20260914/wall_comparison.pdf){width=95%}


The fine-grid accepted wall states have chemical-equilibrium residuals below
$6\times10^{-15}$, unscaled Poisson residuals below $7\times10^{-13}$, and
Gauss-law errors below $2\times10^{-16}$ in the reported charge units.
An initial 800-interval attempt failed the unchanged Poisson acceptance test.
A sparse analytic Jacobian and Newton polishing resolved that numerical issue;
the original failed run remains in the campaign. Directional finite-difference
checks of the Jacobian give relative errors below $2\times10^{-10}$.

## Nonlocal interface: a converged profile with comparison caveats

The experimental interface routine implements
$u=u_s+q^2G_l(z,z)/2$ using the PRL-2022 SI's transverse Fourier integral.
It resolves the three-dimensional fluctuation Green function while the mean
profile depends on one spatial coordinate. Infinite uniform vapor and liquid
extensions supply transparent boundary conditions. A homogeneous Green
function on the same discrete operator is subtracted before the continuum
limit, avoiding a spurious bulk offset. A phase condition fixes translation;
the associated auxiliary force must vanish for acceptance.

The Green-function recurrences match an independently assembled dense inverse
to $6.67\times10^{-16}$. A two-half-space analytical Green function provides
a separate continuum check: its maximum relative self-energy error falls from
$4.74\times10^{-3}$ to $3.04\times10^{-4}$ as the spacing decreases from
$0.1a$ to $0.025a$, consistent with second-order convergence.

At $T/T_c=0.9$, the interface converges with 400, 800 and 1600 intervals over
$80a$. The fine-grid chemical residual is $1.34\times10^{-8}$ and the
auxiliary force has magnitude $1.09\times10^{-9}$. Increasing the box from
$80a$ to $160a$ at fixed spacing changes the total density by less than
$6.64\times10^{-11}a^{-3}$. Increasing the transverse quadrature from 128 to
256 nodes, or the cutoff from $100/a$ to $200/a$, gives still smaller changes.
Grid refinement is the leading observed numerical correction.

Fig. 1 explicitly uses total ion concentration. For Fig. 2(a), the plotted
"ion concentration" is interpreted as **one-species density**,
$c_+=c_-=c_{\mathrm{total}}/2$, consistent with its plateau scale. This is a
discrete normalization interpretation, not a fitted amplitude, and should be
confirmed with the original numerical data. The figure legend's rounded
$T/T_c=0.9$ is used literally.

The direct comparison using the figure's displayed Gibbs dividing surface has
a substantial horizontal discrepancy. Independently aligning each profile's
density midpoint requires a translation of $0.4284(2a)$ and reduces the
maximum difference to $0.001429$ in $c_+(2a)^3$, or **1.78% of the calculated
density change**. Both comparisons are shown. Translation is a free coordinate
choice for an unpinned interface, but this agreement does not resolve the
paper's displayed GDS convention or establish the exact density normalization.

![The direct figure-frame comparison and the separately identified midpoint-aligned comparison. Only translation is removed in the right panel.](figures/literature_scft_20260914/nonlocal_interface.pdf){width=95%}


Two attempts at $T/T_c=0.8$ did not converge: the better attempt retained a
chemical residual of approximately $1.10\times10^{-2}$. Neither failed profile
is used as a reproduction result. Surface tension and the asymmetric
interfacial charge/potential profiles have not been calculated. This remains a
partial interface reproduction, even though the bulk thermodynamics and a
symmetric nonlocal profile have passed numerical checks.

## Execution protocol and replay

The implementation is in `models/prl_electrolyte.py`; the runner, independent
checks and figure comparisons are in `studies/literature_scft.py`. All outputs
are generated through the existing `workflow.py` entry point. The campaign is
stored at:

```text
/home/yongdd/polymer/dh_salt_runs/literature_scft_20260914_180824
```

`campaign.json` records every SLURM job and its arguments. `sources/` preserves
the execution versions and original EPS curves. Each case has `summary.json`
with convergence status, residuals, parameters and source hashes; profile
calculations also have `profiles.npz`. Failed and superseded attempts are kept.
The packaged comparison data are in
[comparison.json](figures/literature_scft_20260914/comparison.json).

Each parameter configuration was submitted as a separate CPU job to partition
`a10`, with one task, one CPU, 2 GB RAM and a 10-minute time limit. No GPU was
requested. NumPy/SciPy ran in the `polymerfts` environment with OpenMP, MKL and
OpenBLAS limited to one thread. There is no warmup, random trajectory or sampling
error because these are deterministic calculations.

The bulk charging integral uses 96-point Gauss--Legendre quadrature. Spinodal
bracketing and equal-pressure root finding are followed by a $10^{-8}$ equality
residual test and stability/common-tangent checks. For walls, 200, 400, 800 and
1600 intervals were used at fixed physical length. Warm starts interpolate the
previous potential and occupancy variables; they do not change the equations.
Wall acceptance requires chemical residual below $10^{-7}$, unscaled Poisson
residual below $10^{-6}$ and Gauss-law error below $10^{-6}$. These limits were
not relaxed after the initial failure.

The interface uses logarithmic Gauss--Legendre integration over transverse
wave number $10^{-5}/a\le q\le100/a$, with the refinements stated above.
Acceptance requires an unforced interior residual below $10^{-6}$, auxiliary
force below $10^{-6}$ and endpoint residual below $10^{-5}$. Krylov Newton
iteration has an explicit-Jacobian hybrid fallback. Convergence at one
temperature does not substitute for the failed lower-temperature calculation.

Representative replay commands, from the repository root after activating
`polymerfts`, are:

```bash
python devel/charged_polymers/workflow.py prl electrolyte \
  --case foundation --out /path/to/new/foundation
python devel/charged_polymers/workflow.py prl electrolyte \
  --case bulk --temperature 0.8 --qp 2 --qm 2 \
  --out /path/to/new/bulk
python devel/charged_polymers/workflow.py prl electrolyte \
  --case wall --sigma -0.01 --cb 0.2 --cells 1600 --length 10 \
  --out /path/to/new/wall
python devel/charged_polymers/workflow.py prl electrolyte \
  --case interface --temperature 0.9 --cells 400 --length 80 \
  --out /path/to/new/interface
python devel/charged_polymers/workflow.py analyze electrolyte \
  --campaign /path/to/campaign --out /path/to/new/figures
```

Use fresh output directories and the saved job scripts for complete sweeps and
warm-start dependencies. Figure extraction additionally requires PyMuPDF;
the numerical solvers require NumPy and SciPy. The analysis expects the saved
campaign labels and original EPS files. It reads completed results and does not
launch simulations. Rebuild this report with `workflow.py reports` after
reviewing and packaging new results.

\clearpage

# McCarty: sequence-dependent bulk phase separation

## Implemented model and approximation level

On 2026-09-15, `models/mccarty.py` and `studies/mccarty.py` added an executable
salt-free bulk model for the five neutral E/K sequences in McCarty et al.,
*J. Phys. Chem. Lett.* **10**, 1644--1652 (2019),
[DOI 10.1021/acs.jpclett.9b00099](https://doi.org/10.1021/acs.jpclett.9b00099).
The primary mathematical comparison is its
[Supporting Information](https://doi.org/10.1021/acs.jpclett.9b00099.s001),
Eqs. S12--S18. This is a deterministic bulk Gaussian approximation.
**It is not full Duan PRL-SCFT, CL sampling, or an exact reproduction of the
McCarty particle model.** In particular, it does not include the conformation
feedback in Duan SI S13--S26 or certify the brush implementation's omitted
nonlocal connectivity term.

The chain has 50 beads and 49 Gaussian bonds. E carries charge -1 and K +1.
There are no grafts, walls, explicit counterions or salt. Lengths are in $b$,
monomer density is $\rho b^3$, and the Gaussian charge/density smearing length
is $a/b=1/\sqrt{6}$. Excluded volume is a soft interaction with
$v/b^3=0.0068$; there is no hard packing-fraction limit. This value is explicitly
reported for SI Fig. S3. Its exact correspondence to the main Figs. 2--3 is
not independently specified in their captions and remains a parameter check.
The separate $0.0069b^3$ value in the main Fig. 5B salt scan is not silently
substituted.

The five Fig. 1 sequences were transcribed from the colored bead diagram.
All contain 25 E and 25 K sites, and their independently calculated SCD values
round to the published values:

| Sequence | SCD | Charge-count check |
|---|---:|---|
| sv10 | -2.10 | 25 E, 25 K |
| sv15 | -4.35 | 25 E, 25 K |
| sv20 | -7.37 | 25 E, 25 K |
| sv25 | -12.77 | 25 E, 25 K |
| sv30 | -27.84 | 25 E, 25 K |

The complete strings are stored in `models/mccarty.py:SEQUENCES` and repeated
in each raw summary. `sequence_blocks()` produces the existing native
`PropagatorSolver` block representation with normalized contour length one
and `ds=1/50`. The present uniform calculation evaluates its Gaussian-chain
response analytically; it does not invoke or modify C++/CUDA propagation or
FFT routines.

## Free energy, sequence response and coexistence

With $b=1$, the ideal intrachain correlation is
$C_{ij}(k)=\exp[-k^2|i-j|/6]$. The response factors are

$$
g_{nn}(k)=\frac{1}{N}\sum_{ij}C_{ij}(k),\qquad
g_{nq}(k)=\frac{1}{N}\sum_{ij}\sigma_j C_{ij}(k),\qquad
g_{qq}(k)=\frac{1}{N}\sum_{ij}\sigma_i\sigma_j C_{ij}(k).
$$

This retains the charge signs and contour separations. A net-neutral chain
has $g_{qq}(0)=0$ but nonzero $g_{qq}(k)$ at finite wave number. Its mean charge
alone is therefore insufficient. Neutrality does not in general make
$g_{nq}(k)$ vanish; the antisymmetric sv30 diblock is a special zero-cross-response
case.

Define the pair smearing factor $h^2=\exp(-a^2k^2)$ and

$$
A=v h^2g_{nn},\qquad C=\frac{4\pi\ell_B h^2}{k^2}g_{qq},\qquad
B=\frac{\sqrt{4\pi\ell_Bv}\,h^2}{k}g_{nq}.
$$

The implemented free-energy density is

$$
f(\rho)=\frac{\rho}{N}\left[\ln(\rho/N)-1\right]
+\frac{v\rho^2}{2}+\frac{1}{4\pi^2}\int_0^\infty dk\,k^2\ln D(k,\rho).
$$

The CLI requires the approximation to be named explicitly when comparing modes:

| Mode | Determinant $D$ | Meaning |
|---|---|---|
| `electrostatic` (default) | $1+\rho C$ | Density interaction at its saddle; Gaussian charge fluctuations using the ideal-chain response |
| `paper-rpa` | $(1+\rho A)(1+\rho C)$ | The separate density/charge expressions printed in McCarty SI S12--S18 |
| `coupled-rpa` | $(1+\rho A)(1+\rho C)-\rho^2B^2$ | Also includes the ideal-chain density/charge cross response |

The default is called PRL-style only in the limited sense of supplementing
a density saddle with electrostatic Gaussian correlations. All modes fix
the underlying chain statistics to the ideal Gaussian reference. A change of
mode is a change of approximation, not a different numerical integrator.

The chemical potential per chain and osmotic pressure are obtained from the
same free energy:

$$
\mu_{\rm chain}=N\frac{\partial f}{\partial\rho},\qquad
\Pi=\rho\frac{\partial f}{\partial\rho}-f.
$$

The chemical-potential reference follows the printed SI formula with ideal
term $\ln(\rho/N)$. A different single-particle self-energy convention can
shift chemical potentials by a constant; it must be matched before comparing
absolute values. It does not change the two coexisting densities. The displayed
excess pressure subtracts the ideal-chain contribution $\rho/N$.

Critical points satisfy $f''=f'''=0$. Coexistence is solved by finding two
spinodals, then bracketing the dilute and dense roots separately at a common
chemical potential. Pressure equality selects the common chemical potential.
Each accepted pair has positive $f''$, distinct densities on opposite sides
of the spinodals, and a common tangent below the intermediate homogeneous
free energies. This avoids the trivial equal-density root.

## Computed results and checks

The campaign contains 83 completed SLURM jobs: 15 critical points (five
sequences, three modes), 60 binodal states (five sequences, two modes, six
couplings), and eight fixed-coupling equations of state. The default
electrostatic Gaussian mode gives:

| Sequence | Critical $\ell_B/b$ | Critical $\rho b^3$ | Paper-RPA critical $\ell_B/b$ |
|---|---:|---:|---:|
| sv10 | 0.286573 | 0.507971 | 0.285714 |
| sv15 | 0.199852 | 0.269901 | 0.199468 |
| sv20 | 0.125158 | 0.199602 | 0.124969 |
| sv25 | 0.070889 | 0.172672 | 0.070794 |
| sv30 | 0.038059 | 0.137835 | 0.038017 |

More strongly patched sequences demix at weaker electrostatic coupling in
these calculations. The density-fluctuation and cross-response corrections
are small for these particular parameters, although this does not make them
identically zero for arbitrary sequences.

![Computed binodal branches. Solid: electrostatic Gaussian; dashed: paper RPA. Lines connect computed states and critical points; these are not CL samples. The logarithmic panel clips densities below its displayed lower bound; full values remain in the data files.](figures/mccarty_gaussian_20260915/mccarty_binodals.pdf){width=95%}

![Computed pressure and chemical potential for sv20 and sv30 at the two main Fig. 2 couplings. These panels show the implemented approximations; quantitative agreement with the published Fig. 2 RPA or FTS curves is not established.](figures/mccarty_gaussian_20260915/mccarty_eos.pdf){width=95%}

![Sequence-dependent critical coupling and the ideal-chain charge response. The response changes despite identical total charge and composition.](figures/mccarty_gaussian_20260915/mccarty_sequence_response.pdf){width=95%}

As a limited external check, the minima of the five published **RPA lines**
in Fig. 3B were read from the source PDF raster. Their estimated critical
couplings are 0.2877, 0.1991, 0.1268, 0.0722 and 0.0395 in sv10--sv30 order.
The largest absolute difference from the computed paper-RPA value is 0.0020,
within an estimated raster-reading uncertainty of about 0.003. This uncertainty
is a reading estimate, not a statistical error bar. Source hashes, PDF page,
crop, axis calibration and color thresholds are preserved in
`critical_raster_comparison.json`. This supports a coarse critical-point
comparison, not a precision comparison of the full binodal or of the FTS points.

The same close comparison has **not** been obtained for the main Fig. 2
equations of state. Visual comparison shows discrepancies in the strong-coupling
pressure and absolute chain chemical potential at the assumed parameters.
The implementation agrees with independent integration of the printed SI
expressions, so parameter and observable-reference conventions must be resolved
before calling those curves reproduced. A chemical-potential reference shift
alone cannot explain a pressure discrepancy. No interaction parameter or
normalization was fitted to force a match.

The ten tests in `tests/test_mccarty.py` check published sequence counts/SCD,
native block serialization, direct $N^2$ pair sums, neutral and large-wave-number
limits, independent adaptive Fourier integration, analytical dilute pair
integrals, thermodynamic finite differences, charge/reversal symmetries,
the cross-response channel, zero-coupling stability, criticality, nontrivial
coexistence, numerical quadrature refinement and invalid-input handling.
All ten passed. Across the 83 production jobs, the largest accepted coexistence
differences are $2.54\times10^{-13}$ for chain chemical potential and
$7.00\times10^{-14}$ for pressure, in the stated reduced units.

These residuals certify numerical consistency of this approximation. They do
not measure error against the molecular model or sampled CL equilibrium.
In particular, at $\ell_B/b=0.33$, sv30's default dilute density is about
$1.43\times10^{-21}b^{-3}$: an illustration of how extreme the Gaussian
dilute branch can become. McCarty's paper explicitly demonstrates the failure
of ideal-chain RPA in dilute solution. This implementation therefore makes
the deterministic comparison runnable without treating it as the completed
physical reproduction.

## Execution protocol and saved data

All jobs used one CPU core, 512 MiB requested memory, no GPU, and a five-minute
SLURM time limit. NumPy/SciPy ran in `polymerfts`; OpenMP, MKL and OpenBLAS were
limited to one thread. Fourier integrals used 384 Gauss--Legendre points over
$0<ka<12$. A separate test doubled the quadrature order to 768 and extended
the cutoff to $ka=16$. There is no spatial simulation mesh, random seed,
warmup or sampling time for these deterministic bulk integrals.

For each sequence and mode, binodal couplings were 1.05, 1.15, 1.35, 1.7,
2.2 and 3.0 times that mode's computed critical coupling. The separate EOS
jobs used sv20/sv30 and $\ell_B/b=0.033,0.33$. Each EOS file tabulates the
analytical functions on 321 logarithmically spaced densities from
$10^{-7}$ to $20$; these are evaluations of the bulk formula, not additional
field simulations. Structure factors are also saved on 200 wave numbers at
$\rho b^3=1.47$.

Raw result directory:

```text
/home/yongdd/polymer/dh_salt_runs/mccarty_gaussian_20260915_045023
```

`campaign.json` records all job IDs and arguments. Each configuration has
`summary.json`, and each bulk configuration has `curves.npz`. `sources/`
contains the calculation-time model, study, test and workflow files. The
maintained report packages `critical.csv`, `coexistence.csv`, three figures,
and a hashed comparison manifest under `figures/mccarty_gaussian_20260915/`.

From the repository root, after activating `polymerfts`:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python devel/charged_polymers/workflow.py test mccarty
python devel/charged_polymers/workflow.py prl mccarty \
  --case critical --sequence sv30 --closure electrostatic \
  --out /path/to/new/critical_sv30
python devel/charged_polymers/workflow.py prl mccarty \
  --case bulk --sequence sv30 --closure electrostatic \
  --bjerrum 0.33 --excluded-volume 0.0068 \
  --out /path/to/new/bulk_sv30
python devel/charged_polymers/workflow.py prl mccarty \
  --case bulk --sequence sv30 --closure paper-rpa \
  --bjerrum 0.33 --excluded-volume 0.0068 \
  --out /path/to/new/rpa_sv30
python devel/charged_polymers/workflow.py analyze mccarty \
  --campaign /path/to/complete/campaign --out /path/to/new/figures
```

The bulk runner refuses to overwrite result files. For a complete sweep,
replay the archived SLURM scripts with fresh output paths, one sequence,
coupling and approximation per job. Analysis verifies the submitted manifest
and calculation source hashes before plotting. Salt partitioning, self-consistent
chain conformational changes, spatial coexistence profiles, and matched
CL/MD sampling remain outside this implementation.
