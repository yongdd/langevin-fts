# TODO

Open items, grouped by area. Numbers are stable identifiers for discussion; new items go to the end of a group.

## Features

2. Parameter sweep / continuation: reuse the previous solution as the initial guess when building phase diagrams.
3. Grand canonical ensemble: the core functions exist (`get_stress_gce`, `get_total_concentration_gce`); the SCFT/L-FTS high-level API is missing.
4. SCFT preconditioner for Anderson mixing (left preconditioning failed; try right preconditioning or a modified inner product).
5. Extend `scft.py` to cover a singular chi matrix.
6. Option to print a data file in a human-readable text format.
7. Progress bar.
8. Graphical user interface.
9. Operator overloading for device arrays (the old `Array` class was removed in 2026-10; revisit if needed).

## Numerical methods and performance

10. GPU bandwidth test.
11. Check whether two GPUs can be used for `CudaAndersonMixing`, and why two-GPU performance fluctuates on A100.
13. Reduce the number of splits in the scheduler.

## Tests

16. Unit test for Anderson mixing (real and complex).
17. Validation of pseudo-spectral parameters.
18. Syntax check of keys in `Molecules`.
19. `TestComparePropagatorKey`.
20. Tests for the example scripts.
21. Check `get_negative_frequency_mapping()` and `compute_single_segment_stress()`.

## Build and packaging

22. Check whether CUDA can be installed with conda.
24. Figure out why `os.environ` does not take effect with the conda-forge Python + OpenMP combination.

## Documentation

26. Python API reference (Sphinx + autodoc for `PropagatorSolver`, `SCFT`, `LFTS`).
28. Examples guide (detailed README for `examples/scft/` and `examples/lfts/`).

## Open investigations (research notes)

29. Discrete FFT/DCT/DST propagator accuracy at large spatial dynamic range (deferred 2026-09-06).

    Native C++/CUDA propagation is left unchanged by user decision. The experimental
    FFT + selective real-space recovery was withdrawn; do not enable it by default.
    Reproducer: N=50, nz=704, Lz=6, absorbing z boundaries, per-chain W=100 for z<3
    and zero elsewhere; Gaussian graft source centered at 1/sqrt(50), width 0.2.
    The audited CUDA spectral calculation returned Q approximately -1.60e-18 and
    1.70% mass error; an independent same-boundary reference gave Q=+4.66676e-28
    and mean height <z>=3.076019224. Inverse-transform roundoff contaminating small
    propagator tails is the diagnosed issue; it remains unresolved in the native path.
    Before proposing a change, establish error control, threshold/precision convergence,
    complex linearity and forward/backward consistency, density versus d(log Q)/dw,
    long-run CL statistics, and CPU/CUDA performance and memory costs. Distinguish
    the same finite spectral bond operator from a different real-space wall model.
    Reproducers and historical experiment results:
    `/home/yongdd/polymer/dh_salt_runs/audit_20260906/audit_strong_field.py` and
    `/home/yongdd/polymer/dh_salt_runs/fft_repair_20260906/`.

30. Resolve charged CL discrepancies against exact finite-lattice particles (2026-09-07).

    The 24-run benchmark in `devel/charged_polymers/reports/CLFTS_VALIDATION.md` passes
    mass/partition and final-state conditional-density checks, but charged CL
    averages differ from exhaustive positive particle sums. At E=100, dt=0.0025,
    the cell-0 counterion density is 0.21308439 +/- 0.00112541 versus exact
    0.20718911. E=0 controls agree within estimated uncertainty.
    Distinguish integration bias, rare-event sampling, and possible incorrect CL
    convergence before treating target PRL simulations as validated. Preserve the
    finite action and native FFT; changing the model cannot validate this benchmark.

    Follow-up: 21 trajectories and direct original-contour field quadrature are in
    `devel/charged_polymers/reports/CLFTS_VALIDATION.md`. Quadrature agrees with the
    particle sum to 2.8e-15. At dt=0.00015625, longer LM/ETD, Heun and adaptive Euler
    still give C density near 0.2132 versus exact 0.20719. Entire field moments also
    disagree. Large drifts occur near analytically located Q_P or Q_SP zeros;
    native and independent densities agree at the drift extremes. Longer E=0
    controls also reveal a smaller bias. These results strongly support incorrect
    CL convergence in this fixture. A validated sampling formulation or validated
    parameter regime is still needed; no production FFT or drift patch was made.

    Further diagnostic: `devel/charged_polymers/reports/CLFTS_VALIDATION.md` exactly
    integrates the repulsive fields, leaving a positive one-real-field marginal
    of the SAME finite action. Twelve real-Langevin runs recover all 19 reference
    observables within 1.36 estimated SE. Original CL remains biased even though
    its tested low-order stationarity identities pass; near-zero measurements
    do not account directly for the density bias. This supplies a finite-model
    control, not a scalable production repair or a complete pole-boundary proof.
    Original CL validation and PRL reproduction remain open; FFT is unchanged.

    1D follow-up: `devel/charged_polymers/reports/CLFTS_VALIDATION.md` and
    `devel/charged_polymers/reports/PRL_REPRODUCTION.md` implements
    particle/real-field Gibbs updates without exhaustive enumeration, tested on
    4/16/32 cells of the same finite action. It also recalculates PRL Fig. 1 at
    two grid resolutions. Current LDA curves still differ from digitized paper
    values beyond grid/raster errors; ensemble/box and nonlocal-theory differences
    remain. A working positive sampler for this small system does not validate
    the existing CL dynamics or reproduce the complete PRL theory.

31. Resolve the remaining PRL Fig. 2-3 discrepancies (2026-09-07).

    See `devel/charged_polymers/reports/PRL_REPRODUCTION.md` and the added LaTeX
    report pages. Fixed-parameter, canonical-C smoothed LDA gives collapse and
    reexpansion, but the low-salt onset differs substantially from the paper.
    Establish the paper's Gibbs-height prescription, box/ensemble and chosen
    nonlocal Green-function/I_ex treatment before claiming quantitative reproduction.
    The paper explicitly neglects u_P^ex in its brush calculations; its omission
    alone is not an established explanation for the discrepancy.
    Fig. 3's monovalent potential raster differs from Eq. (9), with M converted
    to nm^-3, by approximately the 0.6022 conversion factor. Confirm the figure's
    normalization with original data/code; do not silently rescale computed data.

    The separate Fig. 2-3 Python runner also exposes a NON-FFT low-salt issue:
    legacy real-space bead-slice underflow plus its 1e-300 floor/global mass
    normalization can mask an O(0.2) density error. The research runner uses
    extended-range positive convolution, a common partition function, and an
    independent all-log reference. It also includes the canonical PB rank-one
    normalization derivative. These remain isolated research implementations;
    production FFT, CL drift and `prl_scft.py` were not changed in this campaign.

## Done (removed from the list above)

- 1. Parallel tempering: dropped (maintainer's decision, 2026-10-21).
- 12. cuFFT callbacks (2026-10-20 kernel review): measured slower than the separate
  kernels on an A10, so they are not used.
- 25. `std::map` errors in `omp parallel for` (2026-10-21): `operator[]` inserts a missing
  key, which is a write; two threads doing it race on the tree. `find()`/`at()` only read,
  which the standard allows concurrently. The propagator loops read their maps with
  `at()`/`find()` (the tables are filled before the parallel regions).
- 27. File formats: `fields_*.mat`, SCFT results and `structure_function_*.mat` are
  described in docs/references/Tools.md (section 5), YAML/JSON parameter files in
  docs/references/Parameters.md.
- Check whether CUDA cub works with CUDA 12.0 and later (2026-10-21): the CUDA code is
  C++20 and needs CUDA 12.2+; it builds with 12.2.1, 12.4.1 and configures with 13.0.2,
  and the full test suite passes with 12.4.1.
- Replace raw `new` and `cudaMalloc`/`cudaFree` with owning buffers (2026-10-21):
  `CudaDeviceBuffer`/`CudaPinnedBuffer`, `std::vector` and `std::unique_ptr` (the FFTW/MKL
  plans and `fftw_malloc` buffers stay in the CPU FFT classes' `release()`); see
  docs/internals/DeveloperGuide.md (Memory Management).
- Throw exceptions properly from inside OpenMP blocks: `ParallelErrors` in
  `src/common/ComputationCommon.h` carries the first exception out of every parallel
  region of `ComputationContinuous.cpp` / `ComputationDiscrete.cpp` (2026-10-20, 3007130a).
- Packaging with conda, pip and Docker (conda-recipe/, pyproject.toml, docker/).
- Stress computation tests with complex type (`TestStress1DComplex`, `TestStressLinear2DComplex`).
