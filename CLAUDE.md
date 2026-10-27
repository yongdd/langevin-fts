# CLAUDE.md

Shared project guidance for coding agents. Keep this file in English and focused
on portable project requirements. Local accounts, machine paths, credentials,
cluster quotas and personal workload details belong outside the repository guides.

## Project Overview

`polymerfts` implements self-consistent field theory (SCFT), Langevin
field-theoretic simulations (LFTS) and complex Langevin simulations (CLFTS).
Python simulation loops call C++20 computational kernels through pybind11.
The computational backends are CPU-MKL, CPU-FFTW and CUDA.

Use the current source as the authority for supported capabilities and defaults:

- [CMakeLists.txt](CMakeLists.txt): native build requirements and backend options.
- [pyproject.toml](pyproject.toml): Python dependencies and wheel configuration.
- [Parameter Reference](docs/references/Parameters.md): public simulation options.
- [Developer Guide](docs/internals/DeveloperGuide.md): implementation architecture.

## Mandatory Workflow Rules

### Commits and Existing Work

- Run `git commit` only after an explicit user instruction to commit. A request
  to edit, fix or update code does not authorize a commit.
- Before committing, run the checks relevant to the change and review the diff
  in a careful second pass. Preserve unrelated changes in the shared workspace.
- Leave `devel/` alone: read its contents only when the user explicitly asks,
  do not audit, modify, stage or commit them, and do not link to it from
  documentation.
- Reserve `scripts/` for tools used by library users: preparing simulations,
  converting inputs/results and analyzing output. Keep personal installation,
  benchmarking and fixture-maintenance tools outside the checkout. Shared CI
  utilities belong under `.github/`; regression helpers belong under `tests/`.

### Preserve Numerical Test Conditions

Existing tests contain calibrated physical and numerical parameters. Unless the
user explicitly requests a change, do not alter their tolerances, field strengths,
grid sizes, box dimensions, polymer parameters or other numerical conditions.
Never weaken a check to make a test pass. Investigate and report failures.

### Shared Computing Resources

- On a Slurm cluster, submit computations expected to exceed one minute as jobs.
  Run simulations, convergence studies, benchmarks and parameter sweeps on
  allocated compute nodes.
- Submit each parameter configuration as a separate job when running a sweep.
- Follow the site's resource policy, explicitly request memory and let the
  scheduler select GPU devices. Preserve its `CUDA_VISIBLE_DEVICES` setting.
- Keep site-specific scheduler launchers outside the checkout; do not commit
  personal paths, partition choices or local module setup in project scripts.
- Keep permitted login-node builds short, use at most four workers and lower
  their priority with `nice -n 10`. Use compute jobs for longer builds and tests.
- Check job status at most once per minute. Keep searches inside the project;
  avoid recursive walks of shared filesystem roots or entire home directories.

## Build and Environment

Use [Installation](docs/getting-started/Installation.md) for setup details.
The native build needs C++20 with `std::source_location`, Python 3.11+, pybind11,
spglib and LAPACKE. Enable at least one computational backend. MKL and FFTW are
disabled by default; CUDA defaults to `AUTO`.

CUDA builds require a compatible host compiler, CUDA toolkit 12.2+ and CMake
3.25.2+. CPU-only CMake builds support CMake 3.18+. A CUDA runtime package does
not supply the `nvcc` compiler. Select the toolkit through the local module system,
`PATH`, `CUDACXX` or `CMAKE_CUDA_COMPILER`; use a compatible host compiler.

Development and integrated CPU regression builds enable both MKL and FFTW
(GPL distribution requirements are described in Installation). The user
installation guide provides separate MKL and FFTW setup paths:

```bash
conda activate polymerfts
conda install --override-channels -c conda-forge --freeze-installed fftw=3.3.11 mkl-devel=2026.1.0
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DPOLYMERFTS_USE_FFTW=ON \
    -DPOLYMERFTS_USE_MKL=ON \
    -DPOLYMERFTS_USE_CUDA=AUTO \
    -DPOLYMERFTS_WERROR=ON
nice -n 10 cmake --build build --parallel 4
```

- The base environment includes LAPACKE but installs neither MKL nor FFTW.
  Install the libraries separately before enabling their CPU backends.
  See Installation for setup details.
- The environment uses conda-forge (Miniforge) on Linux and Windows. On Windows,
  build with MSVC from an x64 developer prompt in the activated environment and
  select Ninja with `-G Ninja` (wheels: `-Ccmake.args=-GNinja`): the Visual Studio
  generator cannot use conda's `nvcc`. conda-forge's `cuda-nvcc`,
  `cuda-cudart-dev` and `libcufft-dev` supply the CUDA toolkit there. See
  Installation.
- Use `-DPOLYMERFTS_USE_CUDA=ON` when CUDA must be built: configuration then fails
  if the toolchain is unsuitable. `AUTO` can fall back to CPU; inspect the output.
- Set `POLYMERFTS_CUDA_ARCHITECTURES` for the intended toolkit and target devices.
  The defaults are defined in CMake; avoid copying a fixed architecture list here.
- Rebuild after C++/CUDA changes. Python changes also need the staging target:
  `cmake --build build --target polymerfts_stage --parallel 4`.
- Tests import the staged package in `build/python/`, not an installed copy.
  Install Python packages only through a wheel built with `python -m pip wheel`
  and installed with `python -m pip install`; remove them with `pip uninstall`.
  Do not use direct CMake installation for Python packages. For an uninstalled
  example, put `build/python` on `PYTHONPATH` using the appropriate absolute path.
- Use a separate build directory for a fresh configuration rather than deleting
  an existing build tree.
- Use an optimized build with `POLYMERFTS_WERROR=ON` to check host C++ warnings.
  A syntax-only check misses some optimization-dependent warnings. This option
  does not cover nvcc-compiled `.cu` files.

Where supported by the shell, `ulimit -s unlimited` and `OMP_STACKSIZE=1G` provide
the stack space used by the examples. Set OpenMP options before importing the
package. `OMP_NUM_THREADS` controls propagator-level workers; the CUDA backend
uses up to four streams. `OMP_MAX_ACTIVE_LEVELS=0` serializes that work.

## Testing and Numerical Validation

Discover the tests for the current build instead of relying on fixed test counts
or hardware-specific runtimes:

```bash
ctest --test-dir build -N
ctest --test-dir build -L basic --output-on-failure
ctest --test-dir build -L numerical --output-on-failure
ctest --test-dir build --output-on-failure
```

Run long suites through the scheduler. Use `-R` to select tests relevant to the
change. The available tests depend on the compiled backends. A CUDA build needs
an allocated, visible GPU to validate its CUDA tests; without one, every test
with a CUDA part fails after running its CPU parts. Report failures and skips accurately.

`LFTS` and `CLFTS` runs write `polymer_<id>.png` into `recording.dir`; `SCFT` writes it
only into `polymer_figure_dir` when given. Run temporary checks and examples from a
scratch directory so generated files do not enter the source tree. Some native tests read inputs from `tests/`;
use CTest to preserve their configured working directory.

For changes to numerical algorithms or propagator kernels:

1. Verify integrated material conservation with the correct normalization for
   the model and boundary conditions. In the usual incompressible bulk case,
   the spatial mean of the summed concentrations is one.
2. Call `check_total_partition()` to check forward/backward propagator consistency.
3. Use random test fields with standard deviation around five when appropriate;
   weak fields can conceal numerical errors. Preserve existing calibrated fields.
4. Check analytical limits, contour/time-step refinement and backend agreement
   as appropriate to the change. Separate correctness tests from performance runs.
5. Finish relevant solver changes with `examples/scft/Gyroid.py` using the
   continuous chain model. Check convergence rather than a fixed execution time.

For FTS changes, also check uninterrupted versus restarted runs, full RNG and
previous-noise restoration, pending measurements and extension state. Statistical
validation needs error estimates and effective sample size. Same-backend restart
checks can require bitwise equality; comparisons across backends need suitable
numerical or statistical tolerances.

See [Numerical Validation](docs/internals/NumericalValidation.md) for regression
coverage and sampling studies. Relevant tests include `TestParameterNames`,
`TestFtsReliability`, `TestCheckpointValidation`, `TestExtensions`,
`TestPropagatorAnalytic`, `TestConvergenceOrderLite` and `TestSamplingStatistics`.
Measure performance separately from correctness tests.

## Architecture and Extension Points

### Native Computation

- `src/common/PlatformSelector.cpp` and `AbstractFactory.h` select and construct
  backends. CPU implementations live in `src/platforms/cpu/`; CUDA implementations
  live in `src/platforms/cuda/`.
- The computation order is implemented once in `src/common/Computation*.cpp`.
  Grid operations go through `ComputationBackend`, implemented by
  `CpuComputationBackend` and `CudaComputationBackend`.
- `ComputationCommon`, `ComputationContinuous` and `ComputationDiscrete` handle
  shared and chain-specific work. The `ComputationReduceMemory*` classes add
  checkpoint-based storage. Full and reduced-memory modes must agree bit for bit
  on the same backend.
- Keep each backend operation a physically meaningful step on whole fields.
  `BackendArray` and `CheckpointArray` are opaque to the common computation layer.
- Propagator aggregation reuses equivalent calculations for branched and mixed
  systems. Preserve that optimization rather than adding manual duplicate work.
- New computational features need CPU and CUDA implementations unless explicitly
  platform-specific. `src/pybind11/polymerfts_core.cpp` exposes the native API.

### Python Simulation Loops

Researchers should be able to follow and modify the physical sequence directly.
Keep each simulation class readable from top to bottom:

- Read options directly from the `params` dictionary in the constructor, with
  defaults next to their use. Avoid introducing parameter dataclass/spec layers.
- Keep one topic per file. Duplication among `scft.py`, `lfts.py` and `clfts.py`
  is acceptable when it keeps the loops self-contained.
- Put shared parameter validation and legacy-name normalization in `validation.py`;
  keep common mechanical helpers in `utils.py`.
- `scft.py` owns SCFT iteration and optional cell relaxation; `lfts.py` owns
  saddle-constrained Langevin sampling and its field compressors; `clfts.py`
  evolves complex auxiliary fields.
- `integrators.py` implements built-in LM and Euler steps. Use the existing
  `integrator=` interface for additional methods and named `callbacks=` for
  observables. Preserve accepted-step ordering and checkpoint contracts.
- `propagator_solver.py` wraps the native solver API; `polymer_field_theory.py`
  handles monomer/auxiliary-field transformations. `smearing.py`, `wtmd.py`,
  `electrostatics.py` and `phase_equilibrium.py` contain their respective physical
  operations.

Use [Extensions](docs/internals/Extensions.md) and
[CustomExtensions.py](examples/lfts/CustomExtensions.py) for callback and integrator
contracts. The additive-noise example is called predictor-corrector; it uses one
explicit Heun correction with the same noise in both stages.

## Numerical Methods and Parameter Conventions

[NumericalMethods.h](src/common/NumericalMethods.h) is the common method registry.
`PropagatorSolver.numerical_methods()` exposes its names and capabilities to Python.
Add a descriptor and the CPU/CUDA backend construction cases when adding a method;
keep its algorithm in the solver classes.

| Method | Chain model | Key properties |
|--------|-------------|----------------|
| `rqm4` | Continuous | Fourth-order contour integration using Richardson extrapolation |
| `rk2` | Continuous | Second-order Rasmussen-Kalosakas splitting; RK does not mean Runge-Kutta here |
| `cn-adi2` | Continuous | Second-order finite differences; real fields and orthogonal cells; mixed boundary types supported; stress unavailable |
| `discrete` | Discrete | Bond convolution for the Chapman-Kolmogorov equation, not continuum MDE integration |

Pseudo-spectral solvers require all directions to be periodic or all directions
to be non-periodic. Reflecting and absorbing directions can coexist. Mixing
periodic and non-periodic directions requires `cn-adi2`. Consult the registry and
[Parameter Reference](docs/references/Parameters.md) for further restrictions.

Cell-centered grids are the default. Hexagonal-axis space-group calculations use
a hybrid grid convention; preserve their symmetry mapping and reciprocal-space
alias handling. See [Space Group Theory](docs/theory/SpaceGroup.md).

Current input names include `polymers`, `relax_cell`, `stress_scale`,
`blocks[].monomer_fractions`, `space_group.hall_number` and the three `recording`
intervals: `checkpoint_interval`, `sample_interval`, `statistics_save_interval`.
`numerical_method` selects the continuous-chain propagator method; discrete
chains always select their dedicated solver. `langevin.integrator` selects the
stochastic integrator. `langevin.max_step` is the final global step number,
including restored steps. WTMD retains its separate `recording_period` option.

Constructors copy and normalize legacy input names. Supplying both names of the
same option raises `ValidationError`. Preserve old-checkpoint compatibility and
verify stored settings hashes before translating names for comparison.

Lengths use the reference statistical segment length times the square root of
the reference polymerization index. `segment_lengths` stores the linear relative
segment lengths, which the solver squares internally. `ds` is the contour step;
field potentials use the per-reference-chain convention. See
[Numerical Methods](docs/theory/NumericalMethods.md) and
[Polymer Field Theory](docs/theory/PolymerFieldTheory.md) for equations and references.

## Memory, FFT and CUDA Rules

- Use owning RAII buffers and smart pointers. Device memory uses
  `CudaDeviceBuffer`, pinned host memory uses `CudaPinnedBuffer`; host arrays use
  standard containers. Raw pointers lend access rather than transferring ownership,
  except where an existing backend allocation interface requires it.
- Preserve allocation and resource counters used by safety tests. CUDA destruction
  and release paths must not throw; use the existing release helpers.
- Share solver mechanics through `CpuSolverPseudoBase` and `CudaSolverPseudoBase`.
  Put chain-specific evolution and stress factors in the derived solvers.
- FFT calls can run concurrently across propagators. Keep execution scratch
  buffers thread-local or otherwise isolated per concurrent operation. Shared
  planning buffers are not execution scratch buffers.
- Disable internal FFT threading; parallelism belongs at the propagator level.
  FFTW execution on thread-local arrays uses its new-array execute functions.
- Preserve FFT inputs that later computation still needs. FFTW complex-to-real
  and cuFFT Z2D execution can overwrite their input even out of place. Copy such
  inputs into execution-local scratch first; do not assume every transform has
  the same preservation behavior.
- Keep CUDA allocations, frees and synchronous default-stream copies out of
  propagator loops. Reuse buffers and enqueue transfers on the operation's stream.
- Avoid adding OpenMP regions inside the per-iteration CUDA host path; work there
  stays serial or moves to GPU kernels.
- CUDA element kernels use grid-stride loops and `grid_blocks` for launch sizing.
  Line solvers distribute lines across blocks and loop when needed.
- Wrap runtime/library errors with `CUDA_CHECK`/`CUFFT_CHECK` and check every kernel
  launch with `CUDA_CHECK_LAUNCH()`. Asynchronous device errors can appear at a later
  synchronization; use the CUDA debugging tools when diagnosing them.

Use `compute_propagators()` and the public `PropagatorSolver` accessors. Do not use
the internal `advance_propagator_single_segment` API.

See [FFT Implementation](docs/internals/FFTImplementation.md) and the Developer
Guide for detailed transform dispatch, ownership and computation hierarchies.

## Documentation

- Use [Parameter Reference](docs/references/Parameters.md) for complete input formats
  and compatibility rules; avoid copying a second parameter reference into this file.
- Examples live in `examples/scft/`, `examples/lfts/` and `examples/clfts/`.
  [Tutorials](tutorials/README.md) provides a learning sequence.
- Explain physical conventions and derivations under `docs/theory/`. Keep measured
  performance results with their problem size, backend, hardware and precision.
- Keep personal paths, accounts, tokens, internal hostnames and local cluster
  configuration out of these two agent guides.

For GitHub math, use protected inline syntax (`` $`...`$ ``) and unindented fenced
blocks with the `math` language. Put display blocks outside list indentation.
Within table math use `\vert` instead of a literal vertical bar. Write `\lt` for a
`<` directly before a letter; GitHub can take it as an HTML tag and break the
formula. Escape literal dollar signs. Check documentation edits with:

```bash
python .github/scripts/check_latex.py AGENTS.md CLAUDE.md
```

The checker validates TeX syntax only when the optional KaTeX CLI is available.
For broader documentation changes, run it without file arguments.
