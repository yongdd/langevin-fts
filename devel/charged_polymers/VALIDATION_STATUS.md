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
