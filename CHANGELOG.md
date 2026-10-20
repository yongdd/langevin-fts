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
  boundaries); it now packs into its own scratch buffer.
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

### Interface changes

- The Python layer is organised for researchers: each simulation class reads its options
  straight from the `params` dictionary, and the package has fewer files (`compressor.py`
  merged into `lfts.py`; `config.py` and `io.py` into `utils.py`; `result.py` into
  `scft.py`). `LR`, `LRAM`, `load_config`, `save_vtk`, `SCFTResult` and `IterationInfo`
  stay importable from `polymerfts`. The `SystemSpec`/`*Options` layer tried during the
  2026-10 cleanup was removed again before release.
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
