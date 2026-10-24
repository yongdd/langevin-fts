# Numerical validation and reproducibility

Keep the existing calibrated test parameters and tolerances. A passing solver comparison is evidence for that case, not a general proof of physical validity. Use independent reference calculations and refinement alongside regression fixtures.

| Check | Existing or added test | Scope |
|---|---|---|
| Reaction and diffusion against an analytic solution | `TestPropagatorAnalytic` | Constant potential with a nonzero Fourier-mode initial condition; RQM4/RK2 on available backends |
| Contour refinement | `TestConvergenceOrderLite` | RQM4 and RK2 asymptotic orders on a nonuniform field |
| Forward/backward consistency | `TestPartitionFunctionCheck`, propagator tests | Every contour pairing yields the same partition function |
| Material conservation | Propagator tests, `TestSpaceGroupPlatformEquivalence` | Volume-averaged total concentration; expand reduced fields or use orbit weights |
| Analytical-limit structure factors | `TestStructureFunctionRPA` | Disordered Gaussian-fluctuation regime, L-FTS/CL-FTS with and without smearing; RPA formulas in [Section 6.4 of the L-FTS and CL-FTS theory](../theory/LangevinFieldTheoreticSimulation.md#64-rpa-reference) |
| Stress derivatives | `TestStress1D`, `TestStressSpaceGroup`, `TestStressNonPeriodic*`, `TestStressAliasTies` | Lengths, angles, constrained cell deformations and boundary conditions |
| CPU/GPU agreement | `TestSpaceGroupPlatformEquivalence`, `TestStressComplexPlatforms` | Same input fields, not independently translated/rotated converged phases |
| Restart equivalence | `TestLftsRun`, `TestLftsDetails`, `TestFtsReliability` | RNG, previous noise, fields, WTMD and pending measurements across recording boundaries |
| Failure handling and compatibility | `TestFtsReliability`, `TestCheckpointValidation` | Initial/terminal failure, unchanged RNG/extension/WTMD state on corrupt restarts, atomic-write failure, earlier WTMD checkpoints loaded into a reused simulation |
| Extension contracts | `TestExtensions` | Harmonic stochastic moments, read-only observers and sampling intervals, multistage forces, extension-state restart and method registry on available backends |
| Sampling error estimates | `TestSamplingStatistics` | Independent Gaussian draws and an AR(1) process with known asymptotic variance |

The Python sampling/restart tests do not replace long statistical checks. Existing RPA tolerances remain unchanged. Its continuum reference applies in its specified weak-fluctuation regime; finite contour/grid errors and finite fluctuation corrections must be distinguished from sampling uncertainty.

## Running regression validation

After building, `ctest --test-dir build -L numerical --output-on-failure` runs the selected numerical regressions. Basic and full test selections remain available. Missing CPU backends can skip CPU-specific tests. The GitHub-hosted workflow builds without CUDA; its class-name consistency check does not execute GPU numerics.

Run the selected tests and the unmodified continuous-chain Gyroid example on
allocated compute resources. Set up the environment and GPU allocation using
your site's scheduler, and set `PYTHONPATH` to the staged build package when
validating uninstalled changes. For example, inside the allocation:

```bash
ctest --test-dir /path/to/polymerfts/build -L numerical --output-on-failure \
    --output-junit /path/to/report/numerical.xml
cd /path/to/report
python /path/to/polymerfts/examples/scft/Gyroid.py > gyroid.log 2>&1
```

For CPU/GPU agreement, the build must include both backends and the job must
have a visible GPU. Preserve JUnit results, the CMake cache, environment details,
GPU information and the Gyroid convergence result in a distinct report directory
per job. A CPU-only job can run the same CTest selection, but does not certify
GPU agreement. Submit long checks and parameter sweeps through the scheduler,
one configuration per job. Keep cluster-specific launchers, paths and module
settings outside the repository.

To run the full suite, omit `-L numerical` from the CTest command after building
the required targets.

## Statistical refinement

The test harness `tests/validation/validate_fts_sampling.py` runs one configuration and writes post-equilibration Hamiltonian estimates, Monte Carlo standard errors, effective sample sizes, block structure factors and continuum RPA references. CL-FTS also reports imaginary observables. Scaled drift maxima are summarized by quantiles, including the maximum. The full histories and run status remain in the adjacent `_data` directory. Existing output directories are rejected to prevent mixing independent runs.

```bash
python tests/validation/validate_fts_sampling.py --model lfts --dt 0.5 \
    --seed 7 --steps 20000 --burn-in 4000 --block 1000 \
    --output /path/to/report/lfts-dt05-seed7.json
```

Run this command inside an allocated compute job with resources appropriate to
the selected backend. For GPU sampling, add `--platform cuda` and request a GPU
through the scheduler. The script draws chain diagrams inside its output
directory. It never changes the calibrated tests.

Compare separate jobs along these axes:

- Time step: three decreasing values. Keep comparable post-equilibration Langevin time; increase step counts and burn-in when reducing `dt`.
- Grid: increase `--nx` at fixed physical box and model; then refine `--ds` separately.
- Independent realizations: repeat each comparison with multiple `--seed` values. Account for both within-run autocorrelation and between-run variation.
- Saddle accuracy: tighten `--saddle-tolerance`; report `saddle_failures` and `noise_resamples`. The default retry policy consumes one new noise per accepted proposal attempt, even when its saddle solve is repeated. Explicit resampling studies assess the former policy and cannot establish that it is unbiased merely because failure is rare.
- CL-FTS: vary `--alpha` toward zero, inspect imaginary observables and drift tails, and cross-check in regimes where an independent reference is applicable. Bounded trajectories alone do not establish correct sampling.

The ESS estimator uses an initial positive sequence of paired autocorrelations and conservatively caps ESS at the number of samples. Structure-factor uncertainty is estimated from saved block averages, so at least four full post-burn-in blocks are required. These are approximate error estimates, not automatic equilibrium or complex-Langevin correctness certificates. Persistent shifts under refinement require investigation even when one comparison falls within its error bars.

An initial CPU validation on 2026-10-22 used 20 separate jobs: L-FTS time steps 0.5/0.25/0.125, CL-FTS time steps 1/0.5/0.25, two independent seeds per time step, 32/64 grid points, contour steps 0.02/0.01, and CL stabilization 0/0.01/0.05. The box length was 4, chi N was 6 and nbar was one million; post-equilibration Langevin time was held fixed within each model's time-step comparison. All completed. The four-mode mean ratio S_AA/RPA ranged from 0.9755 to 1.0168; every individual mode was within 8 percent of RPA. No saddle failures occurred, so this pilot does not assess noise-selection bias under actual failures. The raw Hamiltonian differed between grids; agreement of these low-wavevector structure factors does not establish a continuum limit for every observable or validate strong-fluctuation CL-FTS.

## Provenance and continuation

Simulation files contain `simulation_metadata_json`; FTS also writes `run_status.json` after success or an exception. Metadata identifies the loaded native extension's embedded build version/commit/modification flag, compiler and toolkit flags, binary and Python-source hashes, runtime numerical-library versions, selected backend/method/precision, CPU/GPU information, threading environment, input hash and seed. Initial fields have a separate hash. A resumed run records its parent checkpoint and preserves the originating seed.

The embedded Git information is captured at CMake configuration. Reconfigure after source changes to refresh it. Hashes identify the actual installed artifacts even when a build is dirty or the simulation runs outside a checkout. Preserve the corresponding source/wheel and environment/build files; recording versions alone does not recreate them. `run_status.json` records the final execution outcome; periodic checkpoints describe an in-progress run at their saved accepted step.

Versioned checkpoints compare effective model and sampling settings before mutating the RNG. Output location, final step and backend are not part of physical compatibility. Same-backend deterministic regression checks auxiliary fields, concentrations and pending observables exactly; cross-backend continuations can differ by floating-point rounding. Unversioned files require explicit legacy opt-in and cannot recover measurements that were never stored.

## Performance measurements

Keep accuracy tests and performance measurements separate. Propagator timings
and fixed-iteration SCFT timings measure different work; neither alone measures
time to convergence. Record provenance and input hashes with timing results.

Report problem geometry, architecture, contour discretization, field input, hardware, selected backend/method, precision, compilation options, threads/streams, reduced basis and memory mode. State warm-up, repetitions, timing boundaries and variability. Compare total SCFT time at matched residual/accuracy, and sampling efficiency at matched bias and Monte Carlo error. Cross-project comparisons also match physical units, ensemble, bead/bond conventions, free-energy offsets and Fourier normalization.
