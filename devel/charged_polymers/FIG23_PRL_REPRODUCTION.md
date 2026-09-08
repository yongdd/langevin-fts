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
