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
