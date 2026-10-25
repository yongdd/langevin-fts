# TODO

Open items, grouped by area. Numbers are stable identifiers for discussion; new items go to the end of a group.

## Features

2. Parameter sweep / continuation: reuse the previous solution as the initial guess when building phase diagrams.
3. Direct grand-canonical SCFT/L-FTS solver API: accept prescribed chemical
   potentials (fugacities) and handle density, field means and normalization
   consistently. The native GCE functions, CE SCFT thermodynamic transforms and
   incompressible phase coexistence are already implemented; see
   [Phase Equilibrium](docs/theory/PhaseEquilibrium.md).
4. SCFT preconditioner for Anderson mixing (left preconditioning failed; try right preconditioning or a modified inner product).
6. Option to print a data file in a human-readable text format.
7. Progress bar.
8. Extend the simulation script builder with more phase-specific initial-field
   templates (e.g. hexagonal cylinders, BCC, FCC, SC, A15, Sigma, HCP and perforated
   lamellae), alongside lamella and gyroid. Add compatible box presets and export
   initial-field generation in standalone scripts. Keep L-FTS/CL-FTS limited to
   3D orthogonal boxes.
9. Operator overloading for device arrays (the old `Array` class was removed in 2026-10; revisit if needed).
32. Compressible phase-equilibrium API: vary both composition and total density
    in each phase, using a shared reference density and consistent absolute free
    energies. Match all species chemical potentials and thermodynamic pressure.
    The current coexistence helper varies composition only and explicitly rejects
    `zeta_n`; compressible CE SCFT itself is already supported. Validate homogeneous
    analytical limits, the incompressible limit and coexistence conditions.

## Numerical methods and performance

10. GPU bandwidth test.
11. Check whether two GPUs can be used for `CudaAndersonMixing`, and why two-GPU performance fluctuates on A100.
13. Reduce the number of splits in the scheduler.
33. Make the basis of the `PropagatorSolver` field operations explicit (found 2026-10-26).
    With a physical basis active, `integral()`, `mean()`, `inner_product()`,
    `multi_inner_product()` and `zero_mean()` take an array with as many entries as the
    irreducible basis for an irreducible one and convert it. For the primitive
    orthorhombic settings with the m3 basis (Hall numbers 227-297) both bases have N/8
    points in a different order, so a reduced-basis array, such as the result of
    `get_concentration()`, is reordered too. The orbit weights are then uniform, so means
    and products of two arrays of the same basis are unchanged, but a product of an
    irreducible and a reduced array is wrong (Pmmm, 16^3: 4.48 instead of -5.11). SCFT
    keeps every array in the irreducible basis and is not affected. Take the basis as an
    argument, or accept only the reduced basis and convert in SCFT.

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
    Reproducer locations and historical experiment archives are recorded in
    private research notes; the model parameters and required checks are given above.

    Guard (2026-10-26): SCFT (final fields) and L-FTS (a saddle point whose Hamiltonian
    is not finite) raise `ValidationError` when a partition function with real fields
    is not positive and finite (`TestPartitionGuard`); the native library only reports
    values. Intermediate iterates are not checked: with `rqm4`, the L-FTS/WTMD run of
    `TestCheckpointValidation` passes through Q <= 0 during saddle iterations and still
    converges. With the reproducer Q is about -3e-19 to -7e-19 on CPU-FFTW, CPU-MKL and
    CUDA (the sign of the noise changes with the FFTW plan). The check cannot catch a
    positive Q made of round-off noise: at W = 30-50 the platforms give different Q,
    sometimes positive, and only `check_total_partition()` reports the inconsistency
    (relative errors 1e-3 to 1 against 1e-11 at W = 10).

    Proposed fix (not implemented): apply the same discrete bond operator in real space,
    as a banded matrix per axis, K = S^-1 diag(g) S with S the transform of the axis,
    computed once in extended precision. The round-off of each point is then relative to
    the values within the kernel width instead of to the largest value on the grid, so
    the far tails keep their relative accuracy. For orthogonal boxes the operator
    factorizes over the axes and costs O(N x band) per axis and step. Complex fields
    keep the localization, though not the positivity argument; non-orthogonal cells and
    the stress would keep the FFT. Offer it as an optional method, not the default, and
    pass the checks listed above before using it.

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
- 5. SCFT with singular chi matrices (verified 2026-10-22): the inverse-free
  projected incompressible residual and direct compressible residual are already
  implemented. Coverage also includes invertible chi matrices with a vanishing
  Arora projection denominator; see `TestScftOptions` and `TestScftCompressible`.
- 12. cuFFT callbacks (2026-10-20 kernel review): measured slower than the separate
  kernels on an A10, so they are not used.
- 25. `std::map` errors in `omp parallel for` (2026-10-21): `operator[]` inserts a missing
  key, which is a write; two threads doing it race on the tree. `find()`/`at()` only read,
  which the standard allows concurrently. The propagator loops read their maps with
  `at()`/`find()` (the tables are filled before the parallel regions).
- 27. File formats: `fields_*.mat`, SCFT results and `structure_function_*.mat` are
  described in docs/references/Tools.md (section 5), YAML/JSON parameter files in
  docs/references/Parameters.md.
- 30. Charged CL discrepancies against exact finite-lattice particles: moved to the
  charged-polymer research notes (2026-10-26). The user-facing caveat is the "Checked
  range" paragraph of docs/theory/ComplexLangevinSimulation.md (Section 6) and its
  Section 3.2.
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
