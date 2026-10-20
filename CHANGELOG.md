# Changelog

Notable changes to PolymerFTS. The project follows a loose semantic versioning scheme: the
minor version changes when results or the Python interface change, the patch version for
fixes that leave converged results unchanged.

Commit hashes refer to the `main`/`devel` history on GitHub.

## [0.10.0] - Unreleased

The first release since v0.9.1 (May 2025). It bundles a year of development; the entries
below list the changes that alter simulation results or the user-facing interface, so that
old inputs and reference numbers can be re-checked. Pure refactoring and documentation
commits are omitted.

### Results-affecting fixes

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

### Interface changes

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
- Parameter checks (2026-10-19): SCFT rejects a singular chi N matrix (a single monomer
  type, or a zero pair in a three-monomer system) with a clear message, crystal systems
  are checked against the box dimension, `max_iter` must be positive; L-FTS/CL-FTS accept
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
- Build: C++20, CMake 3.17+, spglib and LAPACKE are required at configure time; the CPU
  backends share a `cpu-common` object library; `POLYMERFTS_WERROR` turns warnings into
  errors (on in CI); the LAPACKE header may be `mkl_lapacke.h` (conda-forge's MKL build of
  liblapacke ships no `lapacke.h`); `POLYMERFTS_CUDA_ARCHITECTURES` selects the compute capabilities;
  the installed `_core.so` carries an RPATH to the libraries it was linked against; the
  staged package in `build/python` drops modules that no longer exist in the sources. The generated build no longer depends on `LIBRARY_PATH` or
  `CPLUS_INCLUDE_PATH` of the configuring shell: libraries are linked by absolute path and
  the CUDA toolkit directories are always passed explicitly. Packaging with pip (`pyproject.toml`), conda (`conda-recipe/`) and
  Docker (`docker/`) was added (2026-01-09, fe3a54df).

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
- `TestSpaceGroupPhases` with 14 converged phase fixtures and
  `scripts/reconverge_phase_fixtures.py` to regenerate them.
- Well-tempered metadynamics (`WTMD`) for L-FTS, `Smearing`, `SCFTResult`/`IterationInfo`,
  YAML/JSON parameter files (`load_config`, `SCFT.from_file`).
- `scripts/install_versioned.sh`: wheel per version plus a per-version virtual environment,
  for keeping several releases side by side (documented in Installation.md).
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
