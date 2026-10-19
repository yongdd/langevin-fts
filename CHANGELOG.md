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

### Interface changes

- New configuration layer: `parse_system()` returns an immutable `SystemSpec`
  (`GridSpec`, `ChainSpec`), and `PropagatorSolver.from_spec()` builds a solver from it.
  `SCFT`, `LFTS` and `CLFTS` are built on it (2026-10-19, 9e69f9e2).
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
  to the constructor or `from_spec`), the SCFT context manager, `LFTSResult`,
  `save_config`/`create_template_config` (2026-10-19).
- `SpaceGroup.physical_basis_name()` reports the active physical basis
  (`irreducible`, `pmmm`, `m3`, `z-mirror`).
- Build: C++20, CMake 3.17+, spglib and LAPACKE are required at configure time; the CPU
  backends share a `cpu-common` object library; `POLYMERFTS_WERROR` turns warnings into
  errors (on in CI). Packaging with pip (`pyproject.toml`), conda (`conda-recipe/`) and
  Docker (`docker/`) was added (2026-01-09, fe3a54df).

### Added

- Space-group reduced basis with CrysFFT fast paths (Pmmm DCT, Recursive 3m, ObliqueZ) and
  physical bases (m3, pmmm, z-mirror) for 3D periodic real fields.
- Crystal systems with angle optimization (monoclinic, triclinic, 2D oblique/hexagonal).
- `TestSpaceGroupPhases` with 14 converged phase fixtures and
  `scripts/reconverge_phase_fixtures.py` to regenerate them.
- Well-tempered metadynamics (`WTMD`) for L-FTS, `Smearing`, runtime validation options,
  `SCFTResult`/`IterationInfo`, YAML/JSON configuration loading.

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
