# Changelog

Notable changes to PolymerFTS. The project follows a loose semantic versioning scheme: the
minor version changes when results or the Python interface change, the patch version for
fixes that leave converged results unchanged.

Commit hashes refer to the `main`/`devel` history on GitHub.

## [Unreleased]

### Installation and platforms

- The Conda environment and CI use conda-forge (Miniforge) instead of Anaconda's
  `defaults` channel, on Linux and Windows (2026-10-24). OpenBLAS is selected with
  `libblas=*=*openblas` and supplies LAPACKE through `openblas`; spglib comes from
  conda-forge with its C library; the MKL backend installs `mkl-devel`. The pinned
  versions follow conda-forge for Python 3.14: NumPy 2.5, SciPy 1.18, pandas 3.1,
  Matplotlib 3.11 and OpenBLAS 0.3.34.
- Windows builds with MSVC (2026-10-24): CMake finds conda's libraries under
  `<prefix>/Library`, uses MSVC warning flags, the conforming preprocessor that the CUDA
  headers need and MSVC's LLVM OpenMP runtime (CMake 3.30+; configuration stops with an
  error on older versions), names the module
  `_core.pyd` and defines `FFTW_DLL`, without which `fftw_version` came from MKL's FFTW
  wrappers. LAPACKE uses `std::complex` under MSVC, the package adds the spglib DLL
  directory before importing `_core`, and the tests no longer use variable-length arrays
  or POSIX-only calls. The out-of-memory check of `TestCudaDeviceBuffer` runs as its own
  ctest entry, `TestCudaDeviceBufferOutOfMemory`, which is skipped (exit code 77) when the
  driver backs device allocations with system memory once the device memory is full, the
  default sysmem fallback of WDDM drivers; the construction then cannot run out
  (2026-10-26: before, the skip covered the whole test).
- CI builds and runs the basic tests on Windows with MSVC and Ninja (MKL + FFTW) besides
  Linux (2026-10-26).

### Added

- CL-FTS of charged systems (2026-10-24):
  - The optional `electrostatics` input gives valences per segment (`charges`) and the
    Bjerrum length (discrete chains, N = 1/ds) or the coupling E; unknown keys raise
    `ValidationError`.
  - The charges interact through the smeared charge density with the Coulomb kernel E/k².
    A fluctuating electrostatic potential ψ carries this interaction. Its noise comes from
    the jumped stream of the auxiliary fields (`PCG64.jumped()`), so runs with neighboring
    seeds share no stream. ψ is kept in Fourier space, where its stiff linear part is
    integrated exactly for each wavevector. Like the Leimkuhler-Matthews integrator of the
    auxiliary fields, ψ takes the mean of the current and the previous noise
    (`psi_noise_prev` in checkpoints) when the integrator declares `averaged_noise = True`,
    as `lm` does; `euler` declares fresh noise, and a custom integrator used with
    electrostatics must declare one or the other. Its Gaussian statistics are then exact for
    any time step, although the charge density of each step is explicit. With fresh noise
    alone, as before and still with the Euler integrator, the variance of ψ is biased at
    first order in the time step: in `TestChargedStructureFunctionRPA` the charge structure
    functions from ψ lay 2 % above the RPA, and the field-field form up to 30 % below it.
  - Charged block copolymers and counterions are ordinary polymers and one-segment
    molecules in `polymers`; the input must be electroneutral.
  - Smearing is required with charges, with a positive Gaussian length for each charged
    type, and Gaussian smearing in CL-FTS accepts one length per monomer type.
  - Checkpoints store ψ, its Fourier transform (for bitwise continuation), the previous
    noise and the stream of ψ, with the pending drifts of ψ and charge structure sums.
    `continue_run()` checks all of them, and a checkpoint that lacks one or holds an invalid
    one raises `ValidationError`. The run metadata records the initial state of the ψ
    stream (`initial_psi_rng_state`). A run continued by hand takes ψ through
    `run(..., initial_psi=...)` and warns without it, since ψ then restarts at zero;
    without Coulomb coupling ψ stays zero and a nonzero `initial_psi` is rejected.
  - `psi_drift_max_history` records the largest drift of ψ apart from `drift_max_history`,
    whose maximum the exactly integrated linear part of ψ would dominate.
  - Some composition modes have a zero eigenvalue and therefore no auxiliary field, for
    example two types that differ only in charge. The structure functions along these
    modes come from ψ; without this, S_ij would miss the charge correlations of such
    systems.
  - The structure function files also hold `charge_structure_function`, the structure
    function of the smeared charge density. It comes from two estimators that use ψ:
    - the mixed form (k²/E)⟨ψc⟩;
    - the field-field form (k²/E)²[V N̄^(1/2) ⟨ψψ⟩ + E/k²].

    Their difference has mean zero, so at each wavevector the two are combined with the
    weight of least variance, stored as `charge_structure_function_weight`. The
    field-field form had 20-600 times less variance where screening is strong, and the
    mixed form is better where it is weak. Its pending sums continue after a restart.
    The two forms are also stored alone (`charge_structure_function_mixed`,
    `charge_structure_function_field_field`): in the Gaussian regime the combination
    follows from the linear response of the chains and does not show how well ψ is
    sampled.
  - Limitations: periodic boxes only, no random copolymers.

  `TestClftsElectrostatics` checks the reductions to neutral CL-FTS and to one common
  smearing length (bit for bit), the forces against finite differences of H, the ψ step
  against the exact flow of its force, the stationary variance of ψ without charges and
  with a linear charge response, and bitwise restarts. `TestChargedStructureFunctionRPA`
  compares two systems with the RPA: a charged diblock with counterions, and a
  polyampholyte melt whose charge mode has no auxiliary field. It estimates standard
  errors from block means and requires every tolerance to exceed 2 of them.
  Example: `examples/clfts/ChargedDiblock.py`.

### Interface changes

- `scripts/analysis/renormalization.py` replaces `renormalization_discrete.m` and
  `renormalization_rpa.m` (2026-10-24):
  - It reads a fields file or takes options and needs only NumPy and SciPy.
  - It evaluates z∞ and the bare segment length a_b/a of CL-FTS from Eqs. (29) and (26) of
    Matsen et al., *J. Chem. Phys.* **164**, 014905 (2026), with compressibility and smearing.
  - It prints the χ_bN for a target z∞χ_bN.
  - The 1/k² singularity of both integrands is integrated exactly, so z∞ agrees with the
    bond sum to 10⁻¹³ with 64 Gauss-Legendre points per axis; integrating it numerically
    left errors of 10⁻⁷–10⁻⁶ with 128 points.
  - Smearing lengths must be numbers; the per-type `a_int` of charged CL-FTS is rejected,
    since the formulas have one Γ(k) for all monomer types.

  Discrete chains without compressibility and smearing give the values of the former
  `renormalization_discrete.m`, as `tests/scripts/TestRenormalization.py` checks. Continuous
  chains now use the N → ∞ limit of the same formulas instead of the athermal RPA of
  `renormalization_rpa.m`, so their z∞ changes. For example, a 32³ grid with L = 4R₀ and
  N̄ = 10⁴ gives 0.813 instead of 0.858: the RPA integral also counted a finite term from
  k ≲ 1/R₀.
- `SymmetricPolymerTheory.compute_h_deriv_chin()` no longer needs `chi_n`: call it as
  `compute_h_deriv_chin(w_aux=..., phi=...)`. The older call with `chi_n` first still works
  and checks it against the object's χN, as before; fields passed in its place raise a
  `TypeError` that names the keywords (2026-10-24).
- `SymmetricPolymerTheory` exposes its χN matrix as `matrix_chi`; `SCFT.matrix_chi` is the
  same array, so the matrix is built in one place, and `compute_eigen_system(matrix_chi,
  zeta_n)` takes the matrix instead of the pair dictionary (2026-10-24).
- `parse_chi_n` moved from `polymerfts.utils` to `polymerfts.validation`, next to the other
  input checks; `polymerfts.utils` still imports it for existing scripts (2026-10-24).
- With a space group, the internal `advance_propagator_single_segment()` binding takes and
  returns fields in the reduced basis, like `compute_propagators()`. Before, it expected
  full-grid arrays but read and wrote only their first `n_basis` entries. The C++ method
  `PropagatorComputation::compute_propagators_reduced()` is removed;
  `compute_propagators()` already accepts reduced-basis fields (2026-10-25).
- C++ platform factories implement only `create_backend()` (2026-10-25).
  `AbstractFactory::create_propagator_computation()` and `create_molecules_information()`
  are now written once, in `src/common/AbstractFactory.cpp` and `AbstractFactory.h`,
  instead of three identical copies. The numerical method is resolved and validated there,
  once. The CPU and CUDA backend constructors take the resolved `NumericalMethodInfo`
  instead of the method names.

### Fixed

- Native input checks (2026-10-25):
  - A monomer name that ends with an underscore, which the name check allows, no longer
    stops the propagator key parser with "invalid stoi argument".
  - `Molecules.get_solvent()` and `Polymer.get_n_segment()` raise an error for an index
    out of range instead of reading past the end of their arrays.
  - A polymer that `add_polymer()` rejects no longer leaves its contour lengths in the
    contour length mapping, where every solver built tables for them.
  - NaN box lengths, angles, segment lengths and contour lengths are rejected like
    non-positive ones.
  - A NaN or infinite contour step `ds` is rejected (2026-10-26). The contour length
    mapping compared with `<= 0`, which NaN passes: `PropagatorSolver` then ran with one
    segment per block and returned a plausible partition function. SCFT, L-FTS and CL-FTS
    checked `ds` before.
  - The CPU backend raises an error for `cn-adi2` with complex fields, like the CUDA
    backend, instead of leaving its solver null (the method registry rejects the
    combination first).
- Anderson mixing skips the least-squares solve when the history products are not finite
  and uses no history then, as before. An infinite product had made OpenBLAS print DLASCL
  warnings to stdout. The stderr redirection that was meant to hide them is removed: it
  missed them, and on Windows it could end a process that has no console (2026-10-25).
- CUDA (2026-10-25):
  - The CrysFFT Boltzmann tables are built when the computation is created and when the
    cell changes, not with allocations and synchronous uploads inside the propagator loop.
    Only the monomer types and contour steps of the blocks get tables (2026-10-26). Before,
    every combination did: with three monomer types and five contour steps, about 1 GB on a
    128³ grid with the recursive 3m CrysFFT. A step whose table was not built raises an
    error instead of building it inside the loop.
  - The ObliqueZ spectral step updates the half spectrum in place, in mirror pairs, so each
    stream no longer holds a second spectrum (2026-10-26). The results are bitwise identical.
  - The Anderson mixers no longer make redundant device copies.
  - Every kernel launch of the concentration sum is checked.

  `TestMoleculesValidation`, `TestInputChecks` and `TestCudaCrysTablesAllocation` cover
  these fixes.

- CUDA solvers and transforms (2026-10-25):
  - CN-ADI with reflecting or absorbing boundaries stopped with `cudaErrorInvalidValue` on
    an axis longer than about 4000 points, whose tridiagonal coefficients did not fit in
    shared memory. Such lines now read the coefficients from global memory and agree with
    the CPU solver to 1e-12.
  - The complex-spectrum `backward_stream()` of `CudaFFT` overwrote its input for real
    fields (cuFFT Z2D); it now transforms a copy, like the real-coefficient interface.
  - The CUDA backend raises an error for `cn-adi2` with complex fields instead of leaving
    its solver null; the method registry already rejects this combination.

  `TestCudaTridiagonal` solves a 6000-point line from global memory and
  `TestCudaFFTMixedBC` checks that the backward transform keeps its input.
  `TestObliqueZStress` checks all six stress components and lattice changes, including
  γ alone, of oblique cells with the ObliqueZ CrysFFT on both bases, CPU against CUDA.
- CUDA CrysFFT objects that share tables (2026-10-26): after one object changed the cell,
  its clones (`clone_sharing_tables()`) kept pointers to the dropped tables and built new
  tables for their own, older cell. A clone used before its own cell change read a dropped
  table: wrong propagators, or an illegal memory access with Recursive3m. The solvers
  change the cell of every object before they compute, so their results were not
  affected. The cell now lives with the shared tables, each lookup notices a cell change,
  and tables are built for the shared cell. A cell change also waits for the device before
  another stream reuses the buffers of the dropped tables. `TestCrysFFTSharedTables`
  covers the three engines, and `TestCudaCrysTablesAllocation` counts table builds, which
  buffer reuse hid from its allocation check.

- Console output no longer raises `UnicodeEncodeError` when stdout is redirected to a file
  or pipe whose encoding lacks the Greek letters and subscripts it prints (χN, κ,
  βF/(ρ₀V/N)), such as cp949 or cp1252 on Windows: these streams write such characters as
  backslash escapes (2026-10-25). Under cp949, SCFT stopped at the end of every run;
  under code pages without Greek letters, SCFT, L-FTS and CL-FTS stopped at their start.
  `TestConsoleEncoding` runs a short SCFT with an ASCII stdout.
- Oblique cells with strongly anisotropic grid spacing, for example 4×32 points on a
  120-degree cell with equal lengths, used the wrong |k|² for some modes (2026-10-26).
  The search for the alias of least reciprocal-metric form looked one period around the
  centered index and missed the minimum (28 of the 128 in-plane modes of that example),
  in the Boltzmann factors and the stress of the full-grid path and in the ObliqueZ
  CrysFFT. The search is now exact (`src/common/MinimalFormAlias.h`); on grids with
  comparable spacing it picks the same aliases as before, so results there are unchanged.
  Since 16477b99 the CUDA ObliqueZ kernel also needs these factors to be even in
  (kx, ky), and CUDA and CPU differed by 6e-4 on such grids. `TestMinimalFormAlias`
  checks the search against a brute-force one on 44,000 modes of random oblique and
  triclinic cells; `TestCrysFFTObliqueZCuda` adds two such grids.

### Performance

- The CUDA ObliqueZ CrysFFT runs the DCT along z and the FFT in the plane as one real 3D
  FFT of the half grid after Makhoul's permutation along z; one kernel applies the
  Boltzmann factors of each pair kz, Nz/2 − kz between the two transforms (2026-10-25).
  A diffusion step takes three kernels and two cuFFT calls instead of eight kernels, four
  cuFFT calls and a device copy. The propagators of an AB diblock with P6/mmm and the
  z-mirror basis (continuous chains, RQM4, double precision, RTX 5060 Ti) take 1.2-1.6
  times less time from 24×24×30 to 96³. Relative to the full-grid FFT this path now takes
  1.66 times as long at 24×24×30, 0.89 at 48×48×60 and 0.51 at 96³ (before: 2.26, 1.09
  and 0.69). Results change at round-off (about 1e-15 relative). `TestCrysFFTObliqueZCuda`
  compares the CUDA and CPU engines on random fields.
- CUDA CrysFFT, results unchanged bit for bit (2026-10-26):
  - An object looks up a table it has already used without the lock of the shared tables;
    the stress multipliers took it at every segment.
  - `compute_stress()` builds the stress multipliers before its segment loop, once per
    cell; they were built inside the loop, with allocations and synchronous uploads.
  - The CrysFFT steps transform in place, which saves the device copy of an out-of-place
    Pmmm or recursive 3m diffusion.
  - The Pmmm, recursive 3m and ObliqueZ engines share one table cache
    (`CudaCrysFFTTables.h`), and the z-major DCT routines that ObliqueZ no longer uses
    are removed.
- CUDA, results unchanged bit for bit (2026-10-25):
  - The CrysFFT objects of the streams share their k-space tables (Boltzmann factors,
    stress multipliers and, for Recursive3m, the twiddle factors): each table is built and
    stored once instead of once per stream, also after every cell change.
  - With a space group and a mask, the mask multiplies the propagator in the reduced basis
    after the reduction; it was expanded to the full grid at every step.
  - `update_dw` uploads or expands each field once per monomer instead of once per contour
    step size, and reads fields already on the device without a copy.
  - The discrete-chain CrysFFT steps no longer copy the propagator through work buffers.
  - The reduced-basis expansion and reduction and the CrysFFT steps of the RQM4, RK2 and
    discrete solvers are implemented once in `CudaSolverPseudoBase`; unused device buffers
    are removed.
- Cells with α = β = 90° and γ ≠ 90° use the ObliqueZ CrysFFT with the irreducible basis
  too, when the space group has a z-mirror, for example a hexagonal or monoclinic
  `SpaceGroup` passed to `PropagatorSolver` without a physical basis (2026-10-25). They
  used the full-grid FFT. The transform is the one of the z-mirror basis, which SCFT
  enables on these cells, so SCFT is unchanged; on small CUDA grids it remains slower
  than the full-grid FFT (see above). `TestSpaceGroupPlatformEquivalence` adds a
  monoclinic P2/m cell (γ = 100°, odd n_x). `TestCrysFFTSelector` checks the CrysFFT
  each space group, grid and cell takes: the equivalence tests pass whatever is chosen.

### Tests

Tests that passed with the code they guard reverted, or paths no test reached
(2026-10-26):
- `TestCnAdiLongLines`: CN-ADI lines of 4000 to 6000 points in 1D, 2D and 3D, reflecting
  and absorbing, CUDA against the CPU; the shared-memory-only solver fails it.
- `TestCrysFFTCellChange`: Pmmm DCT and recursive 3m CrysFFT solvers after cell changes
  of one axis alone against fresh solvers, stress included, on every platform; a CUDA
  cell comparison that ignores L_y fails it.
- `TestExamples`: every script under `examples/` builds its simulation, in a copy of its
  directory, with platforms the build lacks replaced by a CPU platform.
- `TestSpaceGroupPlatformEquivalence` checks masks with RK2 and CN-ADI too;
  `TestComputationSafety` checks the messages of the method checks of the factory on
  every platform; `TestInputChecks` runs the reduced-basis single step for discrete
  chains and with `reduce_memory`, on fields rescaled to a standard deviation of 5.
- `TestClftsElectrostatics` recomputes the recorded S_ij (per-type smearing) and the ψ
  drift from the saved state, and checks that a second `run()` starts without previous
  ψ noise.
- `TestChargedStructureFunctionRPA` discards its first block, checks the S_CC bound at
  n = 1..4 again (the error margin at n = 1..3), and requires the polyampholyte charge
  structure functions within 1.5 % of the RPA on average, which the former ψ step fails.
- `TestConsoleEncoding`, `TestCrysFFTObliqueZCuda` and `TestCrysFFTSelector` carry the
  `basic` label; `TestCnAdiLongLines` and `TestCrysFFTCellChange` the `numerical` one.

## [0.10.0] - 2026-10-23

The first release since v0.9.1 (May 2025). The
[release notes](.github/release-notes/v0.10.0.md) summarize changes relative to that
release. The dated development history below also records features introduced,
corrected or removed between releases; those entries do not all describe v0.9.1.

### Release highlights

- CPU-MKL, CPU-FFTW and CUDA backends share the propagator computation layer;
  continuous and discrete chains support space-group reduced bases and CrysFFT paths.
- Canonical SCFT exposes full and relative free energies, molecular chemical potentials,
  grand potentials and reference-species phase-coexistence calculations.
- L-FTS and CL-FTS support custom integrators and named observable callbacks, with
  checkpointed extension state. CL-FTS and the browser script builder remain beta.
- Numerical corrections cover structure functions, compressible SCFT, non-orthogonal
  stress and smearing, propagator aggregation, memory ownership and checkpoint restart.
- User tools include a browser simulation script builder and bidirectional PSCF
  conversion; see [Simulation GUI](docs/getting-started/SimulationGUI.md) and
  [PSCF conversion](docs/getting-started/PscfConversion.md).

### Installation and input migration

- Build and install Python wheels with `python -m pip wheel` and `python -m pip install`;
  remove them with `python -m pip uninstall polymerfts`. Direct CMake installation is no
  longer the documented Python installation path. See
  [Installation](docs/getting-started/Installation.md).
- Unvalidated Docker image definitions have been removed.
- The supplied environment uses Anaconda `defaults` and Python 3.14. MKL and FFTW are
  installed separately and enabled explicitly; CUDA detection defaults to `AUTO`.
  At least one backend is required. Python 3.11+, NumPy 2+, C++20, spglib and LAPACKE
  are required; CUDA builds require toolkit 12.2+ and CMake 3.25.2+.
- Public parameter names are now `polymers`, `relax_cell`, `stress_scale`,
  `blocks[].monomer_fractions`, `space_group.hall_number` and
  `recording.checkpoint_interval`, `sample_interval`, `statistics_save_interval`.
  The previous names remain accepted at the input boundary; supplying both names for
  one option is an error. See [Parameters](docs/references/Parameters.md).
- Replace v0.9.1's `reduce_gpu_memory_usage` with `reduce_memory` explicitly; it is
  not an alias. Memory saving now uses contour checkpoints and recomputation.
- `numerical_method` selects the propagator method;
  `langevin.integrator` selects `lm` (the default) or `euler`. Custom methods and
  callbacks use the contracts in [Extensions](docs/internals/Extensions.md).
- Simulation metadata records source/build information, backend, numerical method,
  input hashes and execution status. Version-2 checkpoints retain custom integrator
  and callback state. v0.9.1 checkpoints are unversioned and require
  `continue_run(path, allow_legacy=True)`; settings and pending measurements cannot
  be verified or fully restored. Version-1 checkpoints from development builds
  support default LM without extensions. v0.9.1 does not restore the complete state
  in version-2 checkpoints; resume them with v0.10.0.

### Detailed numerical and development history

- Stress for non-orthogonal crystal systems (2026-01-18, 9c93e0c1, 68293424), and the
  CrysFFT stress path units and axis components (2026-07-30, a6baa345).
- Angle update sign in SCFT box relaxation was a gradient ascent (2026-07-30, 0f242575);
  runs started at exactly 90-degree angles no longer sit in a dead zone (2026-07-31, d51db6e9).
- Incompressible SCFT mean normalization pins the P-projected mean to the saddle point
  (2026-02-23, 7218692e); the normalization became optional (`fts_consistent_energy`,
  2026-03-22, ce2f3cdf). Compressible L-FTS: jk baseline subtraction and mean(W+) pinning
  (2026-02-14, 926fed2b).
- Anderson mixing solves the history system with SVD (LAPACKE) instead of Gaussian
  elimination (2026-02-12, 68995d48, dfdcde2d). LAPACKE is now a build requirement.
- `dt_scaling` excludes the pressure-field eigenvalue (2026-02-11, 604dfad1).
- Concentration normalization uses `n_segment_total` (2026-01-19, e4b09a3f); non-monotonic
  convergence and the reduce-memory stress computation were fixed (d09d3635).
- Space groups: coordinate-system bug and reduced-basis SCFT (2026-01-21, ad6b94c7),
  tetragonal groups (3ed69773), physical-basis handling with the cell-centered symmetry
  convention (2026-01-31, b467d607), hexagonal/trigonal groups via a hybrid orbit convention
  and a group-invariant k representative (2026-08-01, 6006bb8e), Recursive3m fold for
  odd glide translations (f047b583), discrete chains with a space group read `exp_dw` out of
  bounds (2026-07-30, 5493d009), MKL Pmmm CrysFFT DCT normalization (2026-07-31, d118514c).
- Non-periodic boundary conditions: MKL DST-III dropped the highest mode (2026-07-01,
  0225dd3c), CUDA discrete-chain DCT/DST propagator (cc3e48a4), CUDA glue kernels for
  grids above 65536 points (619028e2), wrong per-axis tables for mixed non-periodic BCs on
  CUDA (2026-08-01, a0d02557), complex-field propagators with non-periodic BCs on all
  platforms (2026-08-18, c0ab6e0c).
- Phase test fixtures (`tests/data/*.mat`) were re-converged under the current cell-centered
  symmetry convention and the expected free energies replaced by converged values
  (2026-10-19, 9e69f9e2). The example `DG.mat`/`SG.mat` were re-converged library outputs
  since 2026-01-21.
- CUDA RK2 stress (2026-10-19): complex fields gathered q(-k) in place through the k -> -k
  map (a data race on grids larger than one kernel launch), and non-periodic 1D/2D boxes
  read the wrong basis tables (V11 = 0). RQM4 and the discrete chain were correct.
- CUDA DCT-II of odd length (2026-10-20): the pre-op kernels applied the Nyquist branch to
  the slot N/2 of odd sequences, so the CUDA CrysFFT (Pmmm and z-mirror bases) was wrong
  whenever a logical grid size is 2 mod 4 (the physical grid is half the logical one), e.g.
  Pmmm 12x12x10 or P6/mmm 24x24x30. Sizes divisible by 4, the CPU platforms and the
  irreducible basis were correct. The 1D DCT-II also packed its spectrum in place into the
  caller's N-double buffer, writing two doubles past its end (1D CUDA runs with reflecting
  boundaries); it now packs into its own scratch buffer. The 1D DCT-III wrote its N/2 + 1
  complex outputs into an N-double work buffer, one or two doubles past its end; the
  buffer now holds them.
- SCFT runs with an mmm orthorhombic group and the m3 basis (Pmmm, Pbca, Pnma, ... whenever
  `m3_crysfft_feasible()` holds: even nx, nz >= 16 and a compatible translation parity,
  e.g. Pmmm 10x10x18 or Pnma 12x12x16) saved scrambled full-grid fields: the
  irreducible-to-reduced conversion was skipped because the two bases have the same size
  there, although their orderings differ (2026-10-20). The iteration itself was correct.
- CPU solvers with a space group but without a CrysFFT physical basis (for example a
  hexagonal cell with the irreducible basis) multiplied the reduced-basis mask over the
  full grid, reading past its end (NaN); the mask is now applied in the basis of the
  propagator after the reduction, as on CUDA (2026-10-20).
- Continuing an L-FTS run from a checkpoint restored the Langevin noise array with a
  squeezed shape when there is a single real field (every AB system), so the continued
  trajectory differed from an uninterrupted run; the continuation also no longer
  re-solves the saddle point of the loaded fields (2026-10-19). CL-FTS, whose noise array
  has one row per field, reshapes it the same way for safety.
- SCFT box relaxation with a space group reduces the stress of lengths tied by the crystal
  system (Cubic, Tetragonal, Hexagonal, Trigonal) by averaging the tied components, as runs without
  a space group already did; the components agree to round-off for symmetrized fields, so
  converged results are unchanged but such runs are not byte-identical (2026-10-19).
- `fts_consistent_energy`: the mean concentrations now include random-copolymer blocks and
  the chain length (f_p / alpha_p); before, random copolymers raised and blends of
  unequal chain lengths pinned the wrong mean (2026-10-19).
- Continuous-chain stress with full propagator storage dropped the chain-end segments of
  every block on the assumption that q(r, 0) = 1 carries no stress. That holds only for
  free ends with periodic or reflecting boundaries; with absorbing walls, a mask or a
  grafting point the stress lost an O(ds) term (absorbing walls: -2.7e-2 relative error
  against a finite difference of ln Q at ds = 0.05, -6.7e-3 at ds = 0.0125; now 9.6e-5 and
  6.3e-7). The reduce-memory mode always kept these terms; both modes now agree
  (2026-10-20).
- A failed construction freed nothing it had allocated, at any level: propagator
  computations, solvers, `Pseudo`, the FFT, DCT/DST and CrysFFT transforms, and the CUDA
  computation box. When a full-storage CUDA run did not fit in GPU memory, the process kept
  every buffer allocated up to the failure (22.5 GB on an A10 for a 128^3 grid), so a retry
  with `reduce_memory` in the same process failed as well. Every constructor now releases
  what it holds when it throws; a sweep that runs out of GPU memory at 210 different points
  (box, transforms, solvers, backend, propagators) leaves at most 4 MiB behind, against up
  to 1.1 GB before. Errors thrown inside the parallel loops of the full-storage mode (out
  of memory while building a CrysFFT cache, a grafted polymer without its `q_init`)
  terminated the process; they now reach Python as exceptions (2026-10-20). The 3D DCT-1
  and DST-1 transforms on CUDA did not check that their cuFFT plan was created. The CUDA
  Anderson-mixing constructors and their history buffers now release their memory too
  (they kept up to 1.4 GB after a failed construction for a 2x128^3 field).
- Reduce-memory mode now gives results identical to full storage (it used separately
  written code): the CPU continuous solvent partition function was computed from
  exp(-w ds/2)^2 with uniform cell weights, ignoring a mask; the CUDA discrete variant
  normalized each block with the polymer found by a key search that ignored the polymer
  index, so two polymer types sharing a propagator pair could be normalized with the
  wrong volume fraction (2026-10-20).

- A masked computation box given a second space group, or none after one, kept the mask in
  the first group's reduced basis, so the next computation read past it (NaN or wrong
  partition functions); changing the box size of a masked box with a space group did the
  same, and the accessible volume was not updated when the box changed (2026-10-21).
  `set_space_group` now reduces the full-grid mask each time and commits only after every
  step succeeded. A computation created without a space group on a box that still held one
  wrote past its buffers (the backend took the reduced size, the solver worked on the full
  grid); a computation now sets the box's group, also to none, and refuses to run if a
  later computation gave the box another one. A group built for another grid is rejected.
- Smearing and non-orthogonal cells (2026-10-21). The kernel Γ(k) took
  k² = kx² + ky² + kz² from the box lengths alone, ignoring the cell angles (in a hexagonal
  cell the (1,-1) mode got twice the k² of (1,0)); it now uses the cell metric with the
  minimal alias, as the propagator solvers do. The L-FTS and CL-FTS noise variance used the
  product of the box lengths instead of the cell volume, so runs in hexagonal, oblique,
  monoclinic or triclinic cells sampled an effective N̄ larger by 1/(volume factor)²
  (1.33 for γ = 120°; σ was 0.93 times too small). Orthogonal cells are unchanged.
- SCFT box relaxation with smearing (2026-10-21): the kernel was built once for the initial
  box, so the relaxed model had the smearing length a_int L/L0, and the stress lacked the
  term from Γ(k(L)). The relaxed period depended on the starting box (a_int = 0.05, χN = 20:
  1.6235 from L0 = 1.7, 1.6084 from L0 = 1.3); the kernel now follows the box and the
  stress includes Σ_i <φ_i (∂Γ/∂θ * w_i)>, and both starts relax to the minimum of F(L),
  1.7138.
- **L-FTS and CL-FTS structure functions were too small by the number of monomer types M**
  (one half for AB, one third for ABC) from 2024-09-01 (042cca91; releases 0.9.0 and 0.9.1)
  to 2026-10-21. The estimator μ_i = Σ_k A⁻¹_ki Ω_k/λ_k was written for the earlier
  auxiliary-field matrix (orthonormal eigenvectors), where it was right; with the symmetric
  theory A = √M O and A⁻¹ = Oᵀ/M, and the estimator needs A_ik. For a disordered AB diblock
  (χN = 6, N̄ = 10⁶) the recorded S(k) was 0.50 of RPA and of the estimator
  n<|W₋(k)|²>/(χN)² - 1/(2χN) used by other codes (lfts_cpu, clfts_gpu); now it agrees with
  RPA (tests/TestStructureFunctionRPA.py). Recorded L-FTS files can be corrected by the
  factor M (with smearing they also carry Γ(k), see below). CL-FTS had three more errors,
  so its files cannot: it multiplied μ(k) by the conjugate of φ(k) instead of φ(-k), which
  differ for complex fields; it paired the
  concentrations and partition functions of one step with the fields of the next in H,
  dH/dχN, S(k) and the saved φ; and with a compressible model it left out the fluctuating
  field of the largest eigenvalue. Compressible AB (ζN = 100, dt = 1): S_AA was 0.43 of RPA,
  now 1.006. The CL-FTS fields are unchanged (to round-off, about 10⁻¹⁴ after 300 steps);
  the recorded H now pairs each field with its own partition function.
- **Compressible SCFT (`zeta_n`) solved the incompressible equations** (2026-10-21): the
  residual χφ - Pw was used for both models, and ζ entered nowhere, so every compressible
  run returned the incompressible structure (Σφ = 1 everywhere) and its free energy was the
  compressible Hamiltonian away from its saddle point (1D AB lamella, χN = 15: residual
  9.68, F = -0.364 for ζN = 5 and -0.322 for ζN = 100; with `fts_consistent_energy`
  residual 2.18, F = 1.886 and -0.060). The residual is now the compressible one,
  E = χφ + ζ(Σφ - 1) - w, with its component along (1, ..., 1), where the stiff ζ term lies,
  scaled by κ/(κ + ζ) (see the field residual entry below); the field means, which have
  no gauge freedom in the compressible model, are pinned to the saddle point. ζN = 5 now
  gives 0.90 <= Σφ <= 1.07 for that lamella, the free energy equals the mean-field value
  computed from the fields, and F and φ approach the incompressible solution as 1/ζ
  (tests/TestScftCompressible.py). L-FTS and CL-FTS were not affected.
- Multi-monomer Hamiltonian for three or more monomer types (2026-10-21): the vector S of
  the linear coefficients S/λ and of the constant was computed from the eigenvectors before
  their sign normalization and √M scaling, so for unequal χ the field part of H was off by a
  term linear in the composition (ABC, χN = 9.5/16.625/9.5, φ = (0.5, 0.3, 0.2): 3.624
  instead of 3.658; compressible ABC: off by about 200). Structures, φ and phase boundaries
  were unchanged (only uniform field modes shift), but absolute H, chemical potentials and
  `fts_consistent_energy` free energies were wrong. AB systems were correct.
- dH/dχN for three or more monomer types (2026-10-21): it differentiated the coefficients at
  fixed auxiliary fields but missed the change of the polymer part through A(χ) (the
  eigenvectors rotate with χ), and blew up when an eigenvector changed sign between χ ± ε
  (χN = 10/30/20: ±4·10⁴ instead of about 0.1), and it left out fields dropped at a zero
  eigenvalue (AB at χN = 0, f = 0.3: 0.25 instead of f(1 - f) = 0.21; B and C chemically
  identical: 10-20 %). It is now the derivative at fixed auxiliary fields Ω including the
  rotation term <φ · (∂A/∂χ) Ω>, computed analytically from the first-order perturbation of
  the eigenvectors in a frame that moves only between real, imaginary and dropped fields
  (docs/theory/PolymerFieldTheory.md, Section 4.5), so nearly equal eigenvalues need no
  care; this is the exact dF/dχN of the L-FTS ensemble, where the imaginary fields sit at
  their saddle point (the derivative at fixed monomer fields would differ from it by
  O(1/C)). Dropped fields add their mean-field term. Checked by dH/dχ_ab = <φ_a φ_b> at the
  field saddle for 2, 3 and 4 monomer types (with nearly equal eigenvalues and dropped
  fields) and by a zero-dimensional three-component model of the L-FTS approximation
  (tests/TestPolymerFieldTheory.py). AB systems at χN ≠ 0 are unchanged.
- Multi-monomer field theory with a composition eigenvalue next to the pressure's zero
  (2026-10-22): for 10⁻¹² < |λ| < about 10⁻⁸ (e.g. χ_AB N = χ_AC N = 15, χ_BC N = 10⁻⁸)
  `SymmetricPolymerTheory` failed with "Invalid inverse of matrix A". eigh mixed a part of
  the uniform pressure vector into that eigenvector (about ε|χN|/|λ|, 1.8·10⁻⁷ there), and
  the zero test of the eigenvalues used 10⁻¹² while the roles of the fields used 10⁻⁸. The
  eigenvectors are now made orthogonal to the pressure vector when they hold such a part
  (matrices without such a part are unchanged bit for bit), and one tolerance, |λ| <= 10⁻⁸,
  decides which fields
  are dropped (tests/TestPolymerFieldTheory.py).
- Smearing with reflecting or absorbing walls (2026-10-21) convolved with a periodic FFT,
  coupling the two walls (a ramp in a reflecting box: 0.89 in the first cell instead of
  0.08). Wall axes are now smeared in the DCT-II basis, i.e. with the even image of the
  field across the wall; across an absorbing wall the propagators are odd, so fields and
  concentrations are even there too. An axis with different walls on its two sides is
  accepted (CN-ADI).
- Stress at alias ties (2026-10-21): the stress table v⊗v took the first of the alias
  representatives that tie for the smallest |k|², on the Brillouin-zone boundary of
  oblique cells and on the Nyquist planes. For a hexagonal field dH/da and dH/db differed
  (1.7·10⁻⁶ relative for a random P6₃/mmc field on 18×18×16), and the forced angle derivatives of an
  orthogonal cell missed the central difference. The table is now averaged over the tied
  representatives (tests/TestStressAliasTies.py). SCFT box relaxation was not affected (it
  averages tied lengths, and the a = b derivative was exact); orthogonal cells without
  angle derivatives are bit-identical.
- Structure functions with smearing (2026-10-21): the fields couple to the smeared densities,
  so the estimator gives Γ(k)·S(k); in a periodic box the recorded S(k) is now divided by
  Γ(k) and is that of the densities (tests/TestStructureFunctionRPA.py compares smeared
  L-FTS and CL-FTS runs with the smeared RPA). With walls Γ is defined in the cosine basis,
  and Γ·S is recorded as it is.
- A solvent in a masked box (2026-10-21) had a nonzero concentration inside the obstacles
  (0.23 in tests/TestPropagatorKeys.py); it is now zero there and its volume average is its
  volume fraction. With a space group the reduced-basis integrals (volume averages, inner
  products, the solvent partition function) had no mask weights (Q_solvent of a masked
  Im-3m box: 0.98946 with the space group, 0.96279 without); they now weight each orbit by
  its count times the mask, and agree with the full grid to 1e-14.
- WTMD (2026-10-21): the volume and |k| of the order parameter ignored the cell angles; a
  continued run lost I1 and the samples since the last update (dH/dψ = I1/I0 was
  underestimated after a restart; checkpoints now carry them); and ψ, computed from the
  fields before a Langevin step, was paired with dH/dχN of the fields after it.
- L-FTS steps whose saddle point failed were skipped (2026-10-21): the step counter moved
  on, so a recording step lost its files and the next recording period held the samples
  of two. The step is now repeated from the same fields with new noise (up to five times);
  runs without failures are unchanged.
- Compressible L-FTS (2026-10-21): the uniform mode of the pressure-like field was pinned to
  the value the first saddle-point solve left (the convergence test, a standard deviation,
  ignores this mode), which shifted H and dH/dχN. Every imaginary field except the
  incompressible pressure field now has its uniform mode at the analytic saddle value
  -(c1 + Σ_j A_jk <φ_j>)/(2 c2), set in every saddle iteration so that the recorded H
  belongs to the returned fields.
- SCFT box relaxation with free angles (2026-10-21): the angles entered the optimizer in
  degrees while their residual is per radian, making the angle steps 57 times smaller
  (Oblique2D hexagonal cylinders from γ = 100°: 1493 iterations with cpu-mkl and 1544 with
  cpu-fftw, now 241; the converged cell is unchanged).
- SCFT `axis_ordering` (2026-10-21) reordered nx, lx and angles but not the initial fields
  and the boundary conditions, which were then read in the wrong order (dF = 9.4·10⁻² for
  the fields and 9.0·10⁻² for walls on the wrong axis in tests/TestScftOptions.py). The
  initial fields, `q_init` and `bc` are given in the order of nx and reordered; the results
  (fields, concentrations, saved files) are in the standard order [a, b, c].
- SCFT field residual (2026-10-21): it projected with P = I - e eᵀ(χN)⁻¹/c,
  c = 1ᵀ(χN)⁻¹1. Its pressure part, (1, ..., 1)Σφ/c, pushed the pressure along the excess
  density for c < 0 (AB with χN < 0; ABC with χ_AB N = χ_BC N = 10, χ_AC N = 60, or with
  χ_AB N = χ_AC N = 10, χ_BC N = 50), grew without bound near c = 0 for either sign (the
  latter with χ_BC N = 39.9 and 40.1, c = +5·10⁻⁴ and -5·10⁻⁴), and in compressible runs
  it was amplified at small ζN (AB f = 0.3, χN = 20, ζN = 0.5); these runs ended in NaN
  or did not converge. The residual is now P⊥(χφ - w) + κ e(Σφ - 1) with the orthogonal
  projector P⊥ = I - eeᵀ/M and κ = max(max|λ|, 1)/2 (λ: the eigenvalues of P⊥ χN P⊥), and
  for compressible runs P⊥E + κ/(κ + ζN) (I - P⊥)E (docs/theory/SelfConsistentFieldTheory.md,
  Section 2.2). It needs no (χN)⁻¹ and the fixed points are unchanged. For incompressible
  AB runs with χN >= 1 it differs from the old residual by a constant that the subtracted
  means remove (κ = χN/2 = 1/c), so the iteration is the same up to round-off. In a sweep
  of 70 runs (1D and 2D, AB and ABC, incompressible and ζN = 0.5 to 100) all converge,
  against 18 before; tests/TestScftOptions.py checks 18 of them. A negative `zeta_n` is
  rejected.
- Aggregation with blocks of different local ds (2026-10-21) failed with `map::at`: the
  keys that contain an aggregated propagator took the ds index of the outer block for
  every dependency. Grafting labels with digits ("G1") failed the same way (the digits were
  read as segment numbers); labels may now contain letters, digits and underscores only.
- Aggregation counted some blocks twice (2026-10-22, ef957694). When identical side chains
  of two or more blocks are attached at different positions of a polymer (for example on
  backbone blocks of different lengths), the right key rewritten for the blocks next to the
  aggregated side chains can itself be merged at its own left key; it then still rewrote the
  blocks beyond it, and so did the merged key, so these blocks stayed under two right keys
  and their concentrations were counted twice (Σφ ≠ 1, `check_total_partition()` false; Σφ =
  1.12 for a bottlebrush with B2-A1 side chains on backbone blocks of 1, 2, 3 and 1 segments
  in a random field). Continuous chains were affected since 59ca6db2 (2024-06, so v0.9.1
  too) and, from 0dea94bf (2026-10-21) on, failed with "Contour length 0.000000 not found"
  instead; discrete chains since b9e0d487 (2025-12). Runs with
  `aggregate_propagator_computation` set to False were correct.
- `set_lattice_parameters` with angles that form no cell (2026-10-21) raised the error after
  it had stored the new lengths and angles, so a caught error left the box inconsistent;
  the cell is now checked before anything changes.
- The CUDA box's device `zero_mean` added the sum to real fields instead of subtracting the
  mean, and the device multi-component inner product and zero mean ignored a space group
  (2026-10-21; no simulation path calls them).
- A masked CUDA box given a space group built for another grid read the mask out of bounds
  before the grid mismatch was reported (2026-10-21): the reduced mask weights were built
  before the group was checked. The checks now run first.

### Interface changes

- Reflecting or absorbing boundaries in a non-orthogonal cell, and a space group with
  reflecting or absorbing boundaries, are rejected (2026-10-21). Both were accepted and
  gave silently wrong results: the wall transforms have no metric cross terms (the
  pseudo-spectral path computed an orthogonal box, CN-ADI already refused), and walls do
  not commute with the group's operations (Ia-3d, 16³, reflecting walls: Q off by less
  than 1 % to tens of percent depending on the field;
  groups whose mirror planes lie on the walls, such as Pm-3m, happened to be exact).
  `Smearing` takes the cell `angles` and has `set_box()` and `stress()`.
- `SymmetricPolymerTheory.compute_h_deriv_chin(chi_n, w_aux, phi)` takes the concentrations
  (2026-10-21): the rotation term needs them whenever the eigenvectors depend on χ (three or
  more monomer types; a call without `phi` is then rejected), and `chi_n` must be the one
  the object was built with. `h_deriv_chin_from_moments()` takes the averages instead. The
  coefficient derivatives `h_const_deriv_chin`, `h_coef_mu1_deriv_chin` and
  `h_coef_mu2_deriv_chin` are removed.
- `SymmetricPolymerTheory.compute_hamiltonian()` takes the computation box (2026-01-26,
  65309e1f, for the reduced-basis integrals of a space group); the tutorials that called it
  without the box are updated (2026-10-21).
- Grafting labels (the values of `initial_conditions` and `grafting_points`, the keys of
  `q_init`) may contain
  letters, digits and underscores only (2026-10-21); other characters are rejected.
- A mask without the symmetry of the space group is rejected (2026-10-21): the reduced basis
  keeps one value per orbit, so such a mask was silently replaced (an off-center obstacle
  in an Im-3m box disappeared, Q off by 4.5 %).
- SCFT accepts a singular χN matrix (2026-10-22): the field residual no longer uses
  (χN)⁻¹, so SCFT no longer rejects every singular χN, which included any zero pair in a
  three-monomer system (det χN = 2 χ_AB χ_AC χ_BC) and chemically identical monomers, nor
  1ᵀ(χN)⁻¹1 = 0. There the auxiliary field of a zero eigenvalue of the projected χN has a
  linear term and acts as a uniform field: free energy and box are the limit of the
  neighbors (χ_AB N = χ_AC N = 10, χ_BC N = 40: the mean of χ_BC N = 40 -+ 10⁻⁴ to 10⁻¹¹).
  A single monomer type is still rejected in the
  incompressible model. An ABC triblock whose B and C are identical gives the free energy of
  the AB diblock to 10⁻¹³, and a zero pair (χ_AC N = 0) the limit χ_AC N → 0
  (tests/TestScftOptions.py).
- SCFT exposes the full canonical Helmholtz free energy as `free_energy_absolute` (2026-10-22),
  F = Σ_p (φ_p/α_p)[ln(φ_p/(α_p Q_p)) - 1] + <½ φᵀ χN φ - w·φ>
  (+ the compressibility penalty). `free_energy_homogeneous` is the full homogeneous
  Flory-Huggins value including ideal mixing entropy. `free_energy`, iteration logs
  and progress callbacks use F minus this homogeneous value; the final summary prints
  both relative and full values. A disordered AB melt with alpha = 1, f = 0.3 and chi N = 8
  has relative F = 0 and full F = 0.68. Full F minus the ideal mixing term equals the L-FTS/CL-FTS Hamiltonian
  including its constant at the saddle point; those Hamiltonians are unchanged.
- CE SCFT results include molecular chemical potentials and grand potentials
  (2026-10-22), with saddle-point field means restored analytically. Reference-species
  exchange chemical potentials and reduced grand potentials are available through
  `SCFTResult.coexistence_quantities()`. `find_binary_coexistence()` locates a common
  tangent using only canonical phase evaluations; `homogeneous_thermodynamics()`
  evaluates homogeneous polymer mixtures analytically. Bulk coexistence requires
  converged incompressible periodic states with free chains. See
  docs/theory/PhaseEquilibrium.md and examples/scft/CanonicalCoexistence.py.
- `fts_consistent_energy` is removed (2026-10-22). It pinned the field means of an
  incompressible run to the saddle point so that the free energy matched the field-theoretic
  Hamiltonian; the free energy no longer depends on the field means, and incompressible
  fields have zero means. A parameter file that sets it gets a warning.

- **CUDA 11 is no longer supported** (2026-10-21): the GPU backend needs CUDA 12.2 or
  newer, because the device code is C++20 (nvcc 11.x has no `-std=c++20`; nvcc 12.0/12.1
  cannot compile `std::source_location`). It also needs CMake 3.25.2+ (which passes
  `-std=c++20` to nvcc), a host compiler with `std::source_location` (GCC 11+, Clang 15+)
  and an NVIDIA driver for the toolkit (R550+ for CUDA 12.4). `POLYMERFTS_USE_CUDA` is now
  `AUTO`, `ON` or `OFF`, `AUTO` by default: the GPU backend is built when that toolchain is
  found, otherwise the configure step warns, with the reason and the remedy, and builds
  without it; `ON` makes a missing piece an error, `OFF` skips the check. A build directory
  of an older version keeps the former BOOL switch, whose `ON` becomes `AUTO`. All checks
  look at the nvcc that compiles the code (`CMAKE_CUDA_COMPILER`, `CUDACXX` or PATH); a
  missing or too old one is not kept in the cache, and `-DCMAKE_CUDA_HOST_COMPILER` or
  `CUDAHOSTCXX` selects its host compiler. The cuFFT library is the one of that toolkit,
  also when a conda environment on `CMAKE_PREFIX_PATH` carries CUDA libraries of another
  version, and the default compute capabilities follow the toolkit (60-90 with CUDA 12.2,
  plus 100 and 120 from 12.8, 75-120 with CUDA 13). pip builds (`pyproject.toml`) and the
  conda recipe require CMake 3.25.2+; the conda recipe now declares spglib and LAPACKE and
  builds the MKL backend (FFTW would make the Apache-2.0 package GPL-bound).
  `POLYMERFTS_INSTALL_HEADERS` now also
  installs the static libraries and exports `polymerfts::polymerfts_backends`, so a
  downstream project can `find_package(polymerfts)`.
- CUDA error checks follow the CUDA C++ Best Practices Guide (2026-10-21): every kernel launch
  is followed by a `cudaGetLastError()` check, and the cuFFT, CUB and runtime calls whose
  return codes were ignored (about 210) are checked. Before, about half the launches were
  checked only by a later `cudaPeekAtLastError()`, so an error was reported at an unrelated
  place and, since the peek does not clear it, again by the next call after the exception
  (for example on a retry from Python). No synchronization was added. The CN-ADI solver
  requests at most the device's opt-in shared-memory limit instead of an unchecked 96000 bytes.
  The check macros are named as is conventional for macros (C++ Core Guidelines ES.32):
  `CUDA_CHECK(...)`, `CUFFT_CHECK(...)` and `CUDA_CHECK_LAUNCH()` after a kernel launch
  (formerly `gpu_error_check`, `cufft_error_check`). Release calls (`cudaFree`,
  `cudaFreeHost`, `*Destroy`) go through `ignore_release_error()`, so a failed release no
  longer leaves its error for the next, unrelated launch check. A failed allocation names
  its size and the calling file, line and function.
- C++ ownership (2026-10-21): device memory is held in `CudaDeviceBuffer<T>` /
  `CudaPinnedBuffer<T>` (`CudaCommon.h`), host arrays in `std::vector` and owned objects in
  `std::unique_ptr`, replacing hand-written allocation and cleanup code (the FFTW/MKL plans
  and the `fftw_malloc` buffers of the CPU FFT classes keep their `release()` functions).
  `createFFT()` returns a `std::unique_ptr<FFT<T>>`, and the computation classes take a
  backend factory returning `std::unique_ptr<ComputationBackend<T>>`. The factory functions
  still return raw pointers that the caller owns. Results are unchanged. The MKL ObliqueZ
  CrysFFT sizes each of its per-thread z buffers from its own requirement (two grids with the
  same Nx*Ny*Nz but a different Nz overflowed the spectrum buffer on one thread). Every owned
  CUDA allocation, including the backend's propagator and checkpoint arrays, is counted, so
  the tests check that every object returns all of it, also after an injected or a real
  allocation failure; GCC and Clang reject `delete` on a buffer. The CPU CN-ADI tridiagonal
  tables hold one entry per grid position along their axis (they were sized to the whole
  grid). The per-monomer tables in the propagator loops are read with `at()`, so a missing
  entry throws instead of being inserted from several threads at once, and the CUDA
  `set_space_group` builds its device table before it changes the box. cuFFT plans, CUDA
  streams and events are counted too (`count_*`/`release_*` helpers), so the tests check
  that every object releases them; a failed `cufftDestroy` no longer clears an unrelated
  error (cuFFT has its own runtime). The CUDA CrysFFT solver step no longer copies the
  bond-length map, and the FFTW ObliqueZ step reads its profiling variable once. The CPU
  CN-ADI solver keeps its work arrays in reused per-thread buffers instead of grid-sized
  variable-length arrays on the stack (a GCC extension; 2 MB each at 64^3, several per
  step), and the duplicate copies of the CN-ADI step functions, unused on both
  platforms, were removed (about 500 lines).
- The Python layer is organised for researchers: each simulation class reads its options
  straight from the `params` dictionary, and the package has fewer files (`compressor.py`
  merged into `lfts.py`; `config.py` and `io.py` into `utils.py`; `result.py` into
  `scft.py`). `LR`, `LRAM`, `load_config`, `save_vtk`, `SCFTResult` and `IterationInfo`
  stay importable from `polymerfts`. The `SystemSpec`/`*Options` layer tried during the
  2026-10 cleanup was removed again before release.
- Python helpers (2026-10-20): `SCFTResult.phi` and `SCFTResult.w` are on the full grid
  also with a space group, and `w` is a dictionary like `initial_fields`, so a result can
  restart `SCFT.run`. `load_config` evaluates numpy expressions such as
  `"4.0*np.sqrt(3)/2"` with a restricted evaluator instead of `eval`; anything outside
  numbers, arithmetic, `np.pi`, `np.e` and a list of elementwise functions, with a
  complex or non-finite value, or nested deeply enough to exhaust the parser, stays a
  string.
  `LFTS.from_file` and `CLFTS.from_file` join `SCFT.from_file`. `save_vtk` writes
  several fields per file, 1D/2D grids, flat arrays with `nx`, and non-orthogonal cells
  (`angles`, also `[gamma]` for a 2D cell as in the params dictionary), places the points
  at the cell centres, and writes vectorized; its second argument is now named `fields`.
  `LR` and `LRAM` return arrays in the shape of their input; `LR` failed for flat fields.
  `LR` and `WTMD` pass `axes` to `np.fft.irfftn` together with `s`; omitting it is
  deprecated since NumPy 2.0 (results unchanged).
- Grid-size limits (2026-10-20, `src/common/GridLimits.h`): grid arrays are indexed with
  32-bit integers, so a computation box or space group of more than 1,073,217,535 points (half of
  INT_MAX - 2^20, because the largest arrays hold two fields), Anderson mixing of more
  than 2,146,435,071 variables, and a CUDA launch of more than 2^20 threads
  (`LFTS_GPU_NUM_BLOCKS` x `LFTS_GPU_NUM_THREADS`) or of more than 1024 threads per block
  are rejected with a clear error.
  Before, the grid-point product could overflow silently and SCFT/L-FTS failed with a
  pybind11 type error. The grid-point product of the Python box factory, the
  multi-component inner products (CUDA kernel and host loop) and the space-group
  offsets are formed in 64 bits, and the 2D DCT-4/DST-4 and DST-3 kernels no longer
  need one thread per row element (rows longer than 1024 points failed to launch).
  The 1D CUDA DCT-II/III (reflecting) launched one block with a thread per point rounded
  up to a power of two, and the DST-II/III (absorbing) a thread per pair of points, so 1D
  reflecting grids longer than 1024 points and absorbing grids longer than 2046 points
  could not run on CUDA; the kernels now loop, and a 1D line that does not fit in the
  shared memory of a block (beyond 4096 points for the DCT-II with 48 KB per block) is
  rejected with a message pointing to the CPU platforms, the default for 1D. Both limits are importable as `_core.MAX_GRID_POINTS` and `_core.MAX_GRID_ARRAY`.
- A space group checks the cell against its crystal system when it is attached to a
  computation box and on every box change (`set_lx`, `set_lattice_parameters`), in C++
  (2026-10-20): cubic needs a = b = c and right angles, tetragonal a = b and right angles,
  orthorhombic right angles, hexagonal (and trigonal in hexagonal axes) a = b with angles
  [90, 90, 120], rhombohedral-axes trigonal a = b = c and alpha = beta = gamma, and
  monoclinic and triclinic groups a cell metric that every rotation preserves (the
  monoclinic unique axis perpendicular to the other two). `scft.py` checked its input
  already; a `PropagatorSolver` with a space group ran on any cell, where the symmetry
  operations are not isometries of the box.
- Removed small features that carried disproportionate code: the ADAM optimizer
  (`optimizer.name` must be `am`), the `validation_runtime` option of L-FTS (SCFT, L-FTS
  and CL-FTS check the partition consistency once per run and raise `ValidationError`
  on failure, as `validation_runtime` did by default) and `configure_logging` (the
  package prints).
  `polymerfts.draw_polymer_architectures` redraws the architecture figures that the
  constructors write. Docstrings were cut to the essentials; the parameter documentation
  moved to docs/references/Parameters.md.
- Complex Langevin FTS (`polymerfts.CLFTS`, `examples/clfts/`) promoted from the private
  development folder; dynamical stabilization follows Willis and Matsen.
- Default `stress_interval` changed from 3 to 1 (2026-03-30, c05e3108).
- Mixed periodic/non-periodic boundary conditions raise instead of silently switching the
  solver (2026-07-30, 7e966860).
- Removed numerical methods ETDRK4, CN-ADI4-LR, CN-ADI4-GR and SDC (2026-01-17, f5ab51c4;
  2026-01-28, 05bd9fc0); `numerical_method` accepts `rqm4`, `rk2` and `cn-adi2`.
- Removed the cell-averaged bond-function feature (2026-01-17, ed66e3c8) and the analytical
  sinc correction (0e626ecd).
- Removed deprecated helpers: the `Array` device-array class, `validate_scft_params`
  (use `validate_system_params`), `PropagatorSolver.set_space_group` (pass `space_group`
  to the constructor), the SCFT context manager, `LFTSResult`,
  `save_config`/`create_template_config` (2026-10-19).
- `SpaceGroup.physical_basis_name()` reports the active physical basis
  (`irreducible`, `pmmm`, `m3`, `z-mirror`). `SpaceGroup.m3_crysfft_feasible()` tells
  whether the recursive 3m CrysFFT works on the grid (even nx, nz >= 16, translation
  parity); SCFT enables the m3 basis only then and otherwise uses the Pmmm basis (or the
  irreducible basis for groups without xyz mirrors), instead of failing in the solver or
  silently running the m3 basis through the full-grid FFT on small grids.
- SCFT checks that the box matches the crystal system of the space group (cubic: equal
  lengths and 90-degree angles; tetragonal: a = b and 90-degree angles; orthorhombic:
  90-degree angles; hexagonal/trigonal on hexagonal axes:
  a = b and gamma = 120; rhombohedral axes: a = b = c and alpha = beta = gamma) and rejects
  box relaxation for rhombohedral-axes settings (2026-10-20). A failed
  `enable_m3_physical_basis()` leaves the `SpaceGroup` unchanged instead of half-updated
  (which corrupted memory in the next `symmetrize`). The physical basis is frozen once a
  computation box or solver holds the group (their reduced-basis maps are snapshots), the
  CrysFFT selector rejects a Pmmm/m3 basis on a non-orthogonal cell and a z-mirror basis
  with alpha or beta != 90 (reachable only when a physical basis is enabled by hand on a
  `SpaceGroup` passed to `PropagatorSolver` or through the C++ API; `SCFT` checks the cell
  before enabling one), and
  `symmetrize_reduced_basis` requires a C-contiguous float64 array like `zero_mean`
  because it works in place (both bindings now refuse dtype conversions, which would have
  modified a copy). `SpaceGroup.uses_hexagonal_axes()` tells the axes setting of a
  trigonal group, so the cell check requires the hexagonal cell for hexagonal-axes
  settings and the rhombohedral cell for rhombohedral-axes settings; `angles` with other
  than three entries is rejected for a 3D box.
- The `SpaceGroup` field conversions (`to_reduced_basis`, `from_reduced_basis`,
  `reduced_to_irreducible`, `irreducible_to_reduced`) keep the layout of their input: a
  1-D input gives a 1-D output, an `(n_fields, n_in)` input an `(n_fields, n_out)` output
  (before, two of them squeezed a single field and the other two always returned 2-D).
  Grid compatibility with the translation parts of a space group is checked per axis: a
  1/2 translation along z alone (P6_3/mmc) no longer forces even nx and ny (2026-10-20).
- CUDA real transforms (`CudaRealTransform`): the lengths each type supports were
  verified against FFTW (DCT-2/3/4, DST-1/4 and the 2D/3D DCT-1 accept any length; the
  1D DCT-1 needs an odd number of points; DST-2 and the 2D/3D DST-3 need even lengths)
  and the constructors now reject unsupported lengths instead of returning wrong
  values; nothing in the library passes such lengths (2026-10-20).
- The CPU stress of an orthogonal 2D box no longer computes the shear term (index 2) that
  the CUDA side and the 3D path skip unless `set_force_off_diagonal_stress(True)` is set
  (SCFT sets it when angles are optimized); the platforms now return identical vectors.
- Parameter checks (2026-10-19): crystal systems are checked against the box dimension,
  `max_iter` must be positive; L-FTS/CL-FTS accept
  `verbose_level` 0 and reject other values than 0-2, reject unknown compressor names and
  WTMD on non-3D grids; `"platform": "auto"` may be given explicitly.
  `PlatformSelector.avail_platforms()` lists `cuda` only when a GPU is visible, so CUDA
  builds fall back to the CPU on nodes without a device.
- Python bindings (2026-10-19): input arrays are made C-contiguous and cast to the expected
  dtype (a strided view was read as if contiguous; `zero_mean`, which works in place,
  rejects arrays that would have to be copied), the complex-field factory reads a real
  `mask`, objects created by a factory keep the objects they point to alive,
  `PlatformSelector.avail_platforms` and the `PropagatorCode` helpers are static methods,
  `get_computation_propagators` lost its misspelled name.
- CL-FTS uses the same idioms as L-FTS (propagator solver accessors, architecture figures,
  material-conservation and partition checks); `calculate_sigma` lives in `utils`.
- CUDA: `compute_single_segment_stress` lives once in `CudaSolverPseudoBase` (as on the CPU
  side); RQM4, RK2 and the discrete solver supply the bond factor and their scratch
  buffers. RK2 now transforms the propagator pair in one batched cuFFT call like the
  other two (2026-10-20).
- The propagator computations exist once, in `src/common/`: `ComputationContinuous`,
  `ComputationDiscrete` and their reduce-memory variants decide what is computed and in
  which order, and every operation on grid data goes through `ComputationBackend`
  (`CpuComputationBackend`, `CudaComputationBackend`). The eight platform classes
  (`CpuComputationContinuous`, `CudaComputationReduceMemoryDiscrete`, ...) were removed:
  the factories create the computations with their backend, and the C++ tests obtain
  them from the factories (`tests/TestComputationFactory.h`). About 10,000 lines of
  duplicated CPU/CUDA code were removed (2026-10-20).
  CUDA stress sums the segment stresses of a block on the device and copies them once per
  block instead of once per segment with a device-wide synchronization. CUDA reduce-memory
  keeps the 2√N recomputation buffers on the device instead of round-tripping each
  recomputed segment through pinned host memory; checkpoints stay in pinned host memory.
  Results are unchanged except for the two fixes above and round-off in complex-field
  normalizations and in discrete-chain solvent concentrations (exp(-w ds) is recomputed
  from w with the global ds instead of taken from the solver).
- CUDA non-periodic boundaries: the DCT/DST transforms and the kernels around them are
  issued on the propagator's stream (`CudaFFTBase::forward_stream`), like the periodic
  cuFFT path; they ran on the legacy default stream, which serialized the streams
  (2026-10-20). Results are unchanged.
- `TestConvergenceOrderLite` uses a Gaussian-smoothed field (white noise on a 0.03 grid is
  outside the asymptotic regime at ds = 0.04-0.01) and now passes: RQM4 order 3.6, RK2 2.0.
- CPU CrysFFT (2026-10-20): the Recursive3m k-cache generator and the ObliqueZ DCT pipelines
  exist once (about 300 lines removed); the ObliqueZ DCT scale factors are the analytic
  constants (FFTW: 2 forward, REDFT01 backward; MKL: none) instead of a least-squares
  calibration run at construction, which changes results only at round-off; the
  per-thread Boltzmann/k caches are owned by the CrysFFT object (a map keyed by thread
  id; each thread remembers its slot, so the steady state takes no lock) instead of
  thread-local maps keyed by the object's address, so they are released when the object
  is destroyed (long sessions that create many solvers no longer accumulate stale
  tables in every worker thread).
- Build: C++20, CMake 3.18+ (3.25.2+ for the GPU backend), spglib and LAPACKE are required at configure time; the CPU
  backends share a `cpu-common` object library; `POLYMERFTS_WERROR` turns warnings into
  errors (on in CI); the LAPACKE header may be `mkl_lapacke.h` (conda-forge's MKL build of
  liblapacke ships no `lapacke.h`); `POLYMERFTS_CUDA_ARCHITECTURES` selects the compute capabilities;
  the installed `_core.so` carries an RPATH to the libraries it was linked against; the
  staged package in `build/python` drops modules that no longer exist in the sources. The generated build no longer depends on `LIBRARY_PATH` or
  `CPLUS_INCLUDE_PATH` of the configuring shell: libraries are linked by absolute path and
  the CUDA toolkit directories are always passed explicitly. Packaging with pip
  (`pyproject.toml`) and conda (`conda-recipe/`) was added (2026-01-09, fe3a54df).

### Performance

- CUDA, 2026-10-20, A10, 64^3, results unchanged except where noted. CN-ADI line solvers
  were launched with 256 blocks for 4096 lines (16 busy blocks): 1.7x faster iterations.
  CrysFFT stress applies one multiplier per group of equivalent axes (one pass instead of
  three for cubic groups, two for tetragonal; stress changes at the 1e-14 level and the
  CPU path does the same): BCC iterations 1.4x (continuous) and 1.8x (discrete) faster.
  Box relaxation: the CrysFFT tables reuse their device buffers and are uploaded on
  their own stream (BCC with box relaxation 80 -> 54 ms per iteration, plain RQM4
  113 -> 107 ms). The segment stress sums all components in one pass with a
  deterministic two-stage reduction (stress changes at the 1e-16 level; iterations
  1.14x faster at 32^3, 1.06-1.09x at 64^3). The last RQM4 half-step factor is applied
  inside the Richardson combination, real-field DCT/DST transforms run in the output
  buffer, and the default launch is 1024 blocks. Block concentrations add up to 32
  segment products per kernel launch instead of one (concentration time -35 to -40%,
  results unchanged). The Gyroid example (32^3, box relaxation) runs in 4.5 s instead of
  5.05 s.
- CPU: the CrysFFT path of the RQM4, RK2 and discrete solvers no longer expands the
  propagator to the full grid every step, and their work arrays are kept per thread
  by the solver and released with it (1.2-1.6x faster with Im-3m at 64^3).
- Every CUDA kernel now loops over its index range, so results do not depend on the
  launch configuration.

### Added

- Space-group reduced basis with CrysFFT fast paths (Pmmm DCT, Recursive 3m, ObliqueZ) and
  physical bases (m3, pmmm, z-mirror) for 3D periodic real fields.
- Crystal systems with angle optimization (monoclinic, triclinic, 2D oblique/hexagonal).
- `TestSpaceGroupPhases` with 14 converged phase fixtures.
- Well-tempered metadynamics (`WTMD`) for L-FTS, `Smearing`, `SCFTResult`/`IterationInfo`,
  YAML/JSON parameter files (`load_config`, `SCFT.from_file`).
- Wheel installation records let pip manage package upgrades and removal.
- `TestStressNonPeriodic` (2026-10-20): the absorbing-wall tolerance of the continuous
  chain is 1e-5 instead of 1.5e-2 (the measured finite-difference mismatch is 3e-7 since
  the chain-end stress fix; it was 5e-3), and the 3D mixed reflecting/absorbing case also
  runs for continuous chains (tolerance 1e-4, measured 1.2e-5, was 4.8e-3).
- Tests (2026-10-20): `TestConfigFile`, `TestScftCallback`, `TestScftResult`,
  `TestSaveVtk` and `TestCompressors` cover the Python helpers (those that run a solver
  skip on builds without a CPU platform); `TestGridLimits` covers the grid-size limits and
  `TestSpaceGroupCell` the cell check of space groups.
- Tests (2026-10-19): `TestStressComplexPlatforms` (complex stress on every platform, 64^3
  grid), `TestStressNonPeriodicRK2`, `TestComplexMask` (array handling and object
  lifetimes of the bindings), `TestLftsRun` (L-FTS/CL-FTS run loops and checkpoint
  continuation); `TestConvergenceOrderLite` had no `__main__` block and never ran under
  ctest; `TestParams` regained the solver-option assertions.

## [0.9.1] - 2025-05-02

- Complex-valued fields: stress computation and `CudaPseudo` for complex type.
- Pure-Python library split out of the example scripts (`src/python/`).
- Fixes: wrong dH_ref/dχN, `np.complex64` → `np.complex128`.
- `SymmetricPolymerTheory`: eigenvalues and eigenvectors reordered.

## [0.9.0] - 2024-10-02

- `SymmetricPolymerTheory` applied throughout the examples; variable renamed `eigen` →
  `auxiliary`.
- Fix: wrong sign of dH/dW.
- SCFT fields can be saved in YAML; `reduce_gpu_memory_usage` added to all SCFT examples.

## [0.8.4] - 2024-08-03

- Grand canonical ensemble functions (`get_total_concentration_gce`, under development).
- Discrete chains: segment indices start from 1; `propagator_junction_start` removed.
- Fix: standard deviation of the stress in `TestPseudo`.

## [0.8.3.1] - 2024-06-22

- Propagator aggregation reworked (variable names follow the paper); fixes for complex side
  chains; Simpson-rule dependence removed from aggregation.

## [0.8.3] - 2024-05-25

- Free energy printed per chain (H → H/CV).
- Async operations in parallel propagator jobs; PSCF conversion tools; propagator code
  order redefined; `check_total_partition()` simplified.

## [0.8.2] - 2024-03-22

- Fix: dendrimer handling in `PropagatorAnalyzer::add_polymer`; key decomposition
  simplified.

## [0.8.1] - 2024-01-15

- Crank-Nicolson real-space solver up to 3D (CPU and CUDA).
- Branched-polymer architecture drawing; multi-GPU fix for the discrete chain model;
  OpenMP replaces batched cuFFT.

## [0.8.0] - 2023-07-11

- First tagged release of the C++/CUDA core with Python bindings: SCFT and L-FTS for
  arbitrary acyclic branched polymers, continuous and discrete chain models.
