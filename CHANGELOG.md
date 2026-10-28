# Changelog

Notable changes to PolymerFTS. The project follows a loose semantic versioning scheme: the
minor version changes when results or the Python interface change, the patch version for
fixes that leave converged results unchanged.

Commit hashes refer to the `main`/`devel` history on GitHub.

## [Unreleased]

### Installation and platforms

- The GitHub repository is renamed from `langevin-fts` to `polymerfts`, the name of the
  package and its import (2026-10-27); GitHub redirects the old URLs.
- The Conda environment and CI use conda-forge (Miniforge) instead of Anaconda's
  `defaults` channel, on Linux and Windows (2026-10-24). OpenBLAS is selected with
  `libblas=*=*openblas` and supplies LAPACKE through `openblas`; spglib comes from
  conda-forge with its C library; the MKL backend installs `mkl-devel`.
- The supplied environment uses Python 3.15 and pins every package to an exact version
  (2026-10-26), so a new environment gets the libraries tested on Linux and Windows:
  among them NumPy 2.5.4, SciPy 1.18.1, OpenBLAS 0.3.34, spglib 2.8.0, CMake 4.4.4 and
  pybind11 3.1.0. The installation commands and CI pin MKL 2026.1.0 and FFTW 3.3.11.
  Python 3.15 runs in UTF-8 mode, so output redirected to a file on Windows keeps the
  Greek symbols of the console output. The conda recipe builds for Python 3.11-3.15 and
  no longer pins NumPy or the pybind11 ABI, and the CI lint job uses Python 3.15.
  The environment no longer installs `git`: on Windows, conda-forge's git brings MSYS
  tools such as `link.exe` that shadow the MSVC linker when conda's paths come first,
  and a fresh CUDA configuration then failed with a misleading message about nvcc.
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

- CL-FTS records the partition ratio |Q_s|/<|e^(-u_s)|> of every step for each canonical
  single-site species, discrete chains of one segment and point ions
  (`partition_ratio_step_history`, columns named by `partition_ratio_species_json`; 2026-10-29).
  It is 1 for real fields and approaches 0 near a zero of Q_s, the pole of -n_s ln Q_s. The end
  of a run prints its smallest value and the fractions of steps below 0.1 and 0.01,
  `scripts/analysis/drift_tail.py` summarizes the records, and checkpoints keep the pending
  steps (NaN for the steps of a checkpoint without the record).
- Reproductions of published results as examples and validation records (2026-10-28):
  `examples/charged/scft/GouyChapmanPlates.py` (Netz 2001, Poisson-Boltzmann between plates),
  `examples/charged/clfts/CounterionsBetweenPlates.py` (Netz's loop expansion at weak coupling),
  `SmearedSaltSelfEnergy.py` (Born self-energy and screening; Wang 2010, Agrawal and Wang 2022),
  `McCartyPolyampholyte.py` (McCarty et al. 2019), `LeeCoacervation.py` (Lee, Popov and
  Fredrickson 2008), `examples/block_copolymer/clfts/LequieuDiblockMelt.py` (Lequieu 2024) and
  `examples/blend/clfts/HomopolymerSolutionBook.py` (Fredrickson and Delaney 2023).
  docs/internals/NumericalValidation.md lists the papers and codes (PSCF, BioFTS, polycomp)
  that polymerfts reproduces, and examples/README.md gives the compared numbers, including
  exact Monte Carlo checks of the CL-FTS of point ions and the results that are not reproduced
  (the phase diagram of Lee et al., the dense branch of Danielsen et al.).
- Documentation: a closed box is the boundary to a medium of vanishing dielectric constant, so
  ions are repelled by images of the same sign (it read "insulating walls without image
  charges"); the checked range of charged CL-FTS (counterions between plates up to Ξ ≈ 0.6,
  the O(δτ) bias of the ψ step); the recorded pressure dilates the box at a fixed number of grid
  points, which differs from the pressure at fixed grid spacing when the smearing is not
  resolved; comparisons with BioFTS (with the corrections of its issue 1, which removes the
  time-step dependence the structure-function comparison reported) and polycomp.
- Documentation: a system of point ions alone takes a neutral solvent of one segment as the
  reference chain, with `zeta_n: 0` so that it has no field (2026-10-29). With ζN > 0 the
  field of the solvent made CL-FTS diverge below about 0.01 solvent molecules per grid cell;
  `CounterionsBetweenPlates.py` and `SmearedSaltSelfEnergy.py` use ζN = 0, which leaves ψ
  and the ions bit for bit unchanged.
- Examples of charged systems (2026-10-28): `examples/charged/scft/ChargedDiblockLamella.py` relaxes
  the period of a charged diblock lamella with point counterions over a range of added salt,
  `examples/charged/scft/PolyelectrolyteBrush.py` solves a charged brush grafted to a charged wall of a
  closed box with the real-space `cn-adi2` method, and `examples/charged/clfts/SaltReservoir.py` runs a
  charged diblock melt in contact with a salt reservoir (grand-canonical point ions) and compares
  the uptake of ions with the Donnan equilibrium. `examples/block_copolymer/scft/crystal_systems/`
  relaxes cylinders in hexagonal, oblique and triclinic cells and a lamella in a monoclinic cell, and
  scans the free energy against the angle of fixed cells.
- Electrostatics in closed boxes and fixed charges (2026-10-28): `electrostatics` accepts
  boxes with walls on every side, where ψ lives in the cosine series (no normal derivative at
  the walls), in CL-FTS and SCFT, and `electrostatics.fixed_charge`, a fixed charge density on
  the grid that adds <ψ c_f> to H, c_f to the charge density and its mean to the
  electroneutrality and the counterions. With walls the charge structure function is not
  recorded, the structure functions lack the ψ completion, and the SCFT stress uses the cosine
  modes of ψ and of the ions. SCFT takes a fixed charge in the axis order of `nx` with
  `axis_ordering`, like the initial fields. `TestConfinedElectrostatics` checks the
  Poisson-Boltzmann limit between charged layers against an independent solution, the forces,
  restarts, the stress and pressure with fixed charges in a periodic box, the SCFT stress in a
  closed box against dF/dL and charged chains grafted to a wall. `SCFT`, `LFTS` and `CLFTS` reject a `mask` parameter, which they
  ignored before: masks remain a `PropagatorSolver` option.
- Gibbs-ensemble CL-FTS (`GibbsEnsemble`, 2026-10-28): two CL-FTS boxes exchange chains,
  canonical point ions and volume at fixed totals until the exchange potentials
  βμ_p - α_p βΠR₀³/C (βμ_s for ions) agree; with charges, no screening and only canonical ions
  only neutral combinations move, while charged grand-canonical ions give each box its Donnan
  potential and every species moves. Every species needs a positive amount in both boxes.
  Each cycle continues the boxes from their fields files (fields, ψ and the previous noises)
  with the uniform modes moved to the homogeneous saddle point of the new composition; `run()`
  starts from the input boxes and `continue_run()` continues an interrupted run from
  `gibbs_history.json`. CLFTS gains `draw_architectures` and `run(psi_noise_prev=...)`.
  `TestGibbsEnsemble` checks, in the mean-field limit, the Flory-Huggins binodal of a blend,
  the coexistence of charged chains with canonical ions (neutral combinations) and with
  grand-canonical salt (Donnan potentials) against independent solutions, the input checks and
  bitwise restarts; `examples/blend/clfts/GibbsEnsemble.py` runs a blend.
- Mean-field electrostatics in SCFT (2026-10-28): SCFT accepts `electrostatics` (with
  `coupling`), point ions and grand-canonical ions, and iterates the electrostatic potential ψ
  with the fields. The residual of ψ is the Newton step of the linearized Poisson equation with
  the charge response of the homogeneous state, which keeps the iteration stable for large E.
  The free energy adds -(1/2E)<ψ(k² + κ²)ψ> and the ions, the homogeneous reference the
  electrostatic energy of the homogeneous state (Donnan potential or screened net charge), and
  the stress the Coulomb kernel and the ion smearing. `SCFT.compute_stress(w, phi)` gives the
  lattice derivatives at fixed fields, `run(initial_psi=...)` an initial ψ, and the results
  `psi` and `ion_chemical_potentials`; canonical ions enter `grand_potential` with
  -N c_s βμ_s. The coupling must be positive, and `coexistence_quantities()` rejects results
  with electrostatics. `TestScftElectrostatics` checks the neutral reduction, the
  strong-screening limit (χ_eff), the stress against dF/dL, the homogeneous states, the
  equivalence of canonical and grand-canonical ions, a fixed charge with `axis_ordering` and
  the platforms.
- CL-FTS grand-canonical species (2026-10-28): a polymer with `chemical_potential` in place of
  `volume_fraction`, or a point ion with `chemical_potential` in place of `concentration`, has a
  fixed chemical potential βμ (in the convention of the recorded chemical potentials) instead
  of a fixed amount. Its term of H is -e^(βμ) Q per reference chain, its concentration
  e^(βμ) times the unnormalized density of the chains or ions, and the volume fractions of the
  canonical polymers sum to at most 1. A charged grand-canonical species makes the uniform
  mode of ψ evolve as the Donnan potential, so its input need not be electroneutral (a
  polyampholyte without net charge does not). The
  osmotic pressure includes their ideal gas e^(βμ) Q and conformations, and
  `number_density_history` records the numbers of chains and ions of all species.
  `TestClftsGrandCanonical` checks the agreement of the ensembles on the same fields, the
  forces, the Donnan and Flory-Huggins mean-field limit with LM and etd, the pressure against
  -dH/dV at fixed chemical potentials and restarts.
- CL-FTS point ions (`electrostatics.ions`, 2026-10-28): ions with a valence, a Gaussian smearing
  length and a concentration per segment of the reference density, without volume and without
  χ. An ion feels z_s (Γ_s * ψ)/N like a segment, H gains -N c_s ln Q_s and the charge density
  the smeared ion charge. The counterions that neutralize the polymers are added to one ion
  species (`counterion`, automatic when unambiguous). The ions add their chemical potentials
  (after the polymers), their ideal gas and smearing dilation to the osmotic pressure, and their
  charge to the charge structure function. `electrostatics.n_segments` gives the number of
  segments of the reference chain, which continuous chains need for point ions and now also
  for `bjerrum_length`. `TestClftsPointIons` checks the forces, the linear response, the
  pressure against -dH/dV, restarts and the Debye-Hückel statistics of ψ.
- Fourier transforms of whole fields, `PropagatorSolver.fourier` (2026-10-27): the same
  interface on CPU (FFTW, MKL) and CUDA (cuFFT) for `forward`/`backward` transforms and for
  Fourier multipliers, `apply([(M, fields), ...])` = backward(Σ M(k) · forward(fields)) with
  matrix or diagonal multipliers stored once by `set_multiplier` (in GPU memory with
  CUDA). `k_squared()` gives |k|² in the coefficient layout: the `rfftn` half spectrum for
  real fields, the full spectrum for complex fields, the cosine series along walls. For real
  fields a multiplier acts as a real operator, made Hermitian where the half spectrum holds
  both k and -k, so every backend gives the same result.
  `AbstractFactory::create_fourier_transform()`; `TestFourierTransform`; explained step
  by step in `examples/tools/FourierTransform.py`.
- CL-FTS integrator `"etd"` (`langevin.integrator`, 2026-10-26): Leimkuhler-Matthews steps
  whose linear part is integrated exactly for each wavevector. Before the run `CLFTS`
  measures the Jacobian of the forces at the homogeneous state with complex-step
  derivatives, one computation of the concentrations per field for all wavevectors, so it
  holds the response of any chain architecture, the smearing and, with electrostatics, ψ,
  which this integrator advances together with the auxiliary fields. For a linear force
  the step is the exact flow and its stationary covariance is exact for any time step; the
  noise is averaged over two steps as in LM. Eigen-directions that are unstable at the
  homogeneous state (near the order-disorder transition and in ordered phases) take LM
  steps and limit the time step there. In the disordered diblock of
  `TestStructureFunctionRPA`, LM diverges at δτN = 4, while `etd` stays within 1 % of the
  RPA up to δτN = 64; the test runs it at δτN = 16. Integrators declare the hook with
  `linear_response` and `set_linear_response()`; each simulation keeps its own copy of such
  an integrator, and L-FTS rejects them. `TestClftsEtd`; `tests/validation/validate_etd_order.py`
  measures the order of the time-step error.
- CL-FTS chemical potentials and osmotic pressure (2026-10-26): every sample records
  βμ_p = ln(φ_p/α_p) - ln Q_p of all species (`chemical_potential_history`), and with
  `recording.pressure` the osmotic pressure ∂ln Z/∂V (`pressure_history`): the chain
  stress, the box dependence of the smearing functions, the field terms of H, the
  normalizations of the Gaussian field integrals and the terms of ψ. Checkpoints keep the
  pending samples; one without them gives NaN for those samples.
  `TestClftsThermodynamics` checks the pressure against -dH/dV at fixed grid values and both
  operators against Flory-Huggins at the homogeneous saddle point.
- Screened electrostatics (`electrostatics.screening_length`, 2026-10-26): a background
  electrolyte that is not simulated explicitly turns the Coulomb kernel into E/(k² + κ²),
  with the screening length 1/κ, the Debye length of the background alone (explicit ions
  screen in addition). The uniform mode of ψ then fluctuates and the explicit species may
  carry a net charge; unscreened runs keep their checkpoint settings.
- CL-FTS records the largest drift over the grid of every Langevin step, for each auxiliary
  field (`drift_max_step_history`) and for ψ (`psi_drift_max_step_history`), in the `dH`
  files (2026-10-26). Checkpoints keep the steps since the last statistics file, and the
  end of a run prints a summary of the distribution. The tail of this distribution decides
  whether complex Langevin averages converge correctly (Nagata, Nishimura & Shimasaki,
  PRD 94, 114515 (2016)); `drift_max_history` holds only the sampled steps.
  `scripts/analysis/drift_tail.py` reads the records of a run and prints the tail and its
  log-log slopes. `TestDriftStepHistory` checks the records, restarts and checkpoints.
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
  Example: `examples/charged/clfts/ChargedDiblock.py`.

### Interface changes

- `recording.pressure` records the virial form of the CL-FTS osmotic pressure
  (`CLFTS.pressure_virial`; 2026-10-29). It adds to the dilation operator, which `CLFTS.pressure`
  still returns, the Schwinger-Dyson terms (1/2)[mean(Omega dH/dOmega) - n_g/(C V/R0^3)] of every
  auxiliary field and the same, mode by mode, for psi. Their exact average is zero; they remove
  the Gaussian fluctuation of the field terms and normalizations, of order
  sqrt(n_g/2)/(C V/R0^3) per field and sample. The averages are unchanged and the standard
  deviation per sample is smaller (0.41 to 0.078 in the diblock of `TestClftsThermodynamics`,
  10.3 to 0.013 in the homopolymer solution of `HomopolymerSolutionBook.py` at L = 1.6 R_g with
  16³ grid points), so `GibbsEnsemble` and the examples that average the pressure need fewer
  samples.
- The examples are grouped by physical system and then by simulation method (2026-10-28):
  `examples/block_copolymer/`, `examples/blend/` and `examples/charged/`, each with `scft/`,
  `lfts/` and `clfts/` as available, and `examples/tools/` for the Fourier transform and the
  extension examples. `examples/README.md` lists every example in a table of systems and
  methods. The scripts are unchanged apart from the paths in their comments.
- The C++ factory functions return `std::unique_ptr` (2026-10-26):
  `PlatformSelector::create_factory_real`/`create_factory_complex` and every `create_*`
  function of `AbstractFactory`. C++ callers no longer delete the objects. An object
  created from others still borrows raw pointers to them, so they must outlive it; local
  `unique_ptr`s declared in creation order are destroyed in the right order. The C++
  tests that use the factory hold the objects in `unique_ptr`s; 19 of them deleted the box
  before the computation, which was harmless only because none used `cn-adi2`, whose
  computation releases its hold on the box when it is destroyed. The Python interface is
  unchanged.
- Constructing `SCFT`, `LFTS` or `CLFTS` no longer writes `polymer_<id>.png` into the
  current directory, where it could overwrite an existing file of that name (2026-10-26).
  `LFTS` and `CLFTS` draw the chain architectures into `recording.dir` when a run starts;
  `SCFT` draws them only into the new `polymer_figure_dir`, when given.
  `draw_polymer_architectures()` takes a `directory`, and its colors beyond the sixth
  monomer type come from a fixed local stream instead of the global NumPy random state.
  The scripts of the PSCF conversion draw no figures and no longer change the working
  directory.
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
- `check_total_partition()` no longer prints the error of each polymer, evaluates every
  polymer instead of stopping at the first failure and takes an optional `tolerance`
  (2026-10-26). The errors come from the new `PropagatorSolver.partition_errors()`
  (native `compute_partition_errors()`); the simulation loops print them.
- Checks move to the layer that owns the state (2026-10-26):
  - `PropagatorComputation` returns only results of the current fields and box. Before
    `compute_propagators()`, after a failed computation and after
    `update_laplacian_operator()` its getters raise "Propagators not computed"; the
    concentration getters also need `compute_concentrations()`, `get_stress()` needs
    `compute_stress()`. They returned Q = 0, arrays never written, or the results of the
    previous box. `PropagatorSolver` relies on these checks instead of its own flag.
  - The native factory rejects mixed periodic and non-periodic boundaries for the
    pseudo-spectral methods (`supports_mixed_boundaries`), not only `PropagatorSolver`;
    an unknown method lists the valid ones, and `PropagatorSolver` resolves the method
    with the new `resolve_numerical_method()` of the native registry.
  - While a `cn-adi2` computation uses a box, `set_lattice_parameters()` rejects
    non-orthogonal angles before changing the box. `PropagatorSolver` checked this in
    Python, and its constructor used 1e-10 degrees instead of the 1e-10 radians of the
    box. `PropagatorSolver` creates the box in its constructor and uses its orthogonality.
  - `SCFT` checks the cell against the space group with the new `SpaceGroup.check_cell()`,
    the check the computation box applies, instead of a Python copy with other angle
    tolerances (`ValidationError` as before; monoclinic and triclinic cells are now also
    checked at construction). The rule against box relaxation in rhombohedral-axes
    settings stays in Python.
  - The binding of `create_anderson_mixing()` takes a 64-bit size and rejects too many
    variables with the native message (`RuntimeError`); `SCFT` and `LFTS` no longer
    repeat the check. The native binding also checks the sizes of `w` and `q_init`, so
    `PropagatorSolver.compute_propagators()` raises `RuntimeError` instead of `ValueError`
    for a wrong size.
- `SpaceGroup` prints nothing while it is constructed or a physical basis is enabled;
  `display_info()` prints the summary, which `SCFT` calls, and `PropagatorSolver` reports
  the active basis with its size (2026-10-26).
- The CPU z-mirror crystallographic FFT no longer reads the undocumented
  `FTS_FFTW_EXHAUSTIVE`, `FTS_FFTW_PATIENT`, `FTS_FFTW_ESTIMATE`, `FTS_FFTW_WISDOM_ONLY`,
  `FTS_FFTW_WISDOM_FILE` and `FTS_CRYSFFT_HEX_DCT_FFT` (2026-10-26). It plans with
  `FFTW_MEASURE` and the native DCT, the previous default, and the FFT-based DCT path
  that only the last variable enabled is removed, as are the unused `print_array`
  helpers of Anderson mixing.

### Fixed

- The propagator statistics printed `reduction = -nan %` when the chains need no propagator
  steps, as with discrete chains of one segment only; they print 0 % (2026-10-29).
- The drift summary at the end of a CL-FTS run labelled an auxiliary field that is left out at
  a zero eigenvalue as imaginary; it labels it dropped (2026-10-29).
- `CudaCommon.h` includes `<cuda_runtime.h>` itself (2026-10-27). Host-compiled sources
  reached the CUDA runtime API only through Thrust, whose CUDA 12.9 headers no longer
  include it outside nvcc, so the Python module failed to compile with CUDA 12.9.
- Script tools on Windows (2026-10-26):
  - Saving a script on the server from the Simulation Script Builder failed on Windows,
    which has none of the directory descriptors (`os.O_DIRECTORY`, `dir_fd`) that keep a
    POSIX save inside its base directory. Windows creates the folders one at a time,
    refuses symbolic links and junctions by their resolved paths, and writes the script
    with LF line endings. A folder that another process replaces by a link during a save
    is not caught there, unlike on POSIX.
  - The editor's server subfolder and the PSCF converter's output paths accepted paths
    such as `/tmp/x`, which Windows does not count as absolute but roots on the current
    drive, so they were written outside the target directory. Both now reject any path
    with a root or a drive, and the editor rejects `:` in folders and filenames.
  - The Windows CI job runs `tests/scripts`; the symbolic-link tests are skipped where
    links cannot be created (Windows without Developer Mode).
- Native state and input checks (2026-10-26):
  - `Molecules.add_solvent()` after a propagator computation was created made the
    computation read past its solvent arrays; `add_solvent()` and `add_polymer()` now
    raise once a computation exists.
  - The CUDA launch configuration is checked against the limits of the current device
    when it is set (`CudaCommon::check_launch`), not only by `display_info()`. A value of
    `LFTS_GPU_NUM_BLOCKS` or `LFTS_GPU_NUM_THREADS` that is not an integer names the
    variable instead of failing in `stoi`.
  - `PropagatorSolver.get_dv()` and `get_grid_points()` ignored the lattice angles and
    later box changes; they use the volume and the lattice vectors of the box (new
    `get_lattice_vectors()`), and the grid points of hexagonal-axes space groups follow
    their node-centered in-plane convention.
  - `PropagatorSolver.n_grid` gave the full grid before the first computation even with
    a space group, so fields sized from it were rejected; it gives the reduced basis.
- SCFT and L-FTS raise `ValidationError` when a partition function with real fields is
  not positive and finite at an accepted state: the final SCFT fields, or an L-FTS saddle
  point whose Hamiltonian is not finite (2026-10-26). A discrete chain grafted next to an
  absorbing wall in a strong field (TODO.md, item 29) leaves its propagator below the
  round-off level of the spectral transforms, and Q came out negative (about -3e-19
  against +4.67e-28) on every platform; SCFT reported no error and L-FTS only a
  non-finite Hamiltonian. Intermediate iterates are not checked, since `rqm4` and
  `cn-adi2` can pass through Q <= 0 during saddle iterations. The check cannot detect a
  positive Q made of round-off noise; `check_total_partition()` reports that
  inconsistency. `TestPartitionGuard` covers the checks.
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
- An m3 physical basis enabled on a grid with nz/2 < 8, such as Im-3m 8³ or 12³, made
  `PropagatorSolver` fail with "recursive 3m CrysFFT is unavailable" on every platform
  (2026-10-26). The CrysFFT selector applied the nz/2 >= 8 size rule, meant for choosing
  a basis, to a basis already enabled. The rule now only chooses
  (`SpaceGroup.m3_crysfft_feasible()`, which SCFT uses), and the recursive 3m engines,
  exact for nz/2 >= 2, run an enabled m3 basis on any grid it maps.
  `TestCrysFFTM3SmallGrid` compares these runs with full-grid ones on every platform.
- CUDA runs boxes with reflecting walls on odd grid sizes, like the CPU backends
  (2026-10-26). It refused every odd non-periodic size, although only the DST-2 of
  absorbing axes needs an even length, which its error message now names.
  `TestCudaOddReflecting` compares CUDA with the CPU in 1D, 2D and 3D boxes, and
  `TestCudaFFTMixedBC` checks odd lengths against direct DCT and DST sums.
- MKL ObliqueZ passed `int` literals where `DftiSetValue` reads `MKL_LONG` (undefined
  behaviour on LP64). The FFTW recursive 3m engine planned without checking its
  allocations and executed its plans on `std::vector` storage, whose alignment need not
  match that of the planning arrays (2026-10-26).

### Performance

- The transforms of every Langevin step in the Python layer use `PropagatorSolver.fourier`,
  the platform FFT, on the GPU with CUDA, instead of NumPy/SciPy on the host (2026-10-27):
  the `etd` step (its matrices are Fourier multipliers, applied in one call), smearing
  (except in boxes that mix periodic and wall axes), the transforms of ψ, the LR compressor
  of L-FTS and WTMD; at each sample, the structure functions of L-FTS and CL-FTS (all
  densities in one call, all fields in another; boxes with walls keep numpy.fft, whose
  coefficients they record) and the dilation of the smearing for the osmotic pressure.
  Results change at round-off: over 8 steps of the charged diblock of
  `TestClftsEtd` the Hamiltonians and ψ agree with the previous code to 8e-16 (LM and
  `etd`, CPU and CUDA), an L-FTS run with LRAM, smearing and WTMD to 7e-10, within its
  saddle tolerance, and an SCFT run with smearing and cell relaxation to 3e-11, within its
  tolerance. Lamella with Gaussian smearing on an A10, ms per step: CL-FTS 32³ LM 20.8 → 17.1,
  `etd` 23.7 → 17.3; 64³ LM 172 → 99, `etd` 231 → 99; L-FTS (LRAM) 32³ 136 → 94, 64³
  808 → 291. `set_linear_response()` of an integrator now also receives `fourier`;
  `Smearing`, `LR` and `WTMD` take the transform as an optional argument.
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
- The CPU CrysFFT engines (Pmmm, recursive 3m, ObliqueZ) share their k-space tables
  between the OpenMP threads instead of building a copy per thread (2026-10-26). Each
  table is built once, under a lock, and a thread keeps the pointers it has looked up
  (`src/platforms/cpu/CrysFFTSharedTables.h`). For a 128³ recursive 3m run with four
  contour coefficients and eight threads this saves about 0.9 GB and seven of eight table
  builds after every cell change. The FFTW ObliqueZ and Pmmm engines free their planning
  arrays after planning. The tables are the same, so results are unchanged.
- Code shared between backends, results unchanged (2026-10-26): MklFFT and the MKL Pmmm
  engine share one DCT-II/III implementation (`MklDctLines.h`); the CPU and CUDA
  recursive 3m engines share the fold, twiddle and multiplier helpers
  (`src/common/Recursive3mHelpers.h`), and the CPU and CUDA ObliqueZ engines the
  reciprocal metric (`src/common/ReciprocalMetric.h`); both solvers choose their CrysFFT
  engine through `plan_crysfft()` in `CrysFFTSelector.h`.

### Tests

- `tests/scripts/TestDriftTail.py` runs the one-pole model of Aarts, Seiler, Sexty and
  Stamatescu (2017) with complex Langevin dynamics (2026-10-29): at β = 1.6, where the walkers
  reach the zero of the weight, `drift_tail.py` finds the power-law tail and Re<z²> is wrong by
  more than 20 standard errors although Im<z²> and the Schwinger-Dyson identities of z and z²
  vanish within their errors; at β = 4.8 no step exceeds ten times the median drift and Re<z²>
  agrees. `TestClftsThermodynamics` checks the virial pressure at the homogeneous saddle point
  and in CL runs, `TestDriftStepHistory` and `TestClftsPointIons` the partition ratios.

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
- Computation state (2026-10-26): `TestComputationSafety` checks the getters before,
  between and after the computations and after a box change, the molecules after a
  computation exists, the mixed-boundary rule of the factory, the orthogonal box of a
  `cn-adi2` computation, the grid geometry with angles and `n_grid` with a space group;
  `TestComputationState` expects the box itself to reject non-orthogonal angles for
  `cn-adi2`. `TestPartitionGuard` checks that the solver reports a non-positive Q
  without raising or printing, and that SCFT and L-FTS reject it at accepted states.
  `TestStressLinear3D` no longer reads concentrations of the previous box in its
  finite-difference steps: the values were never used, and the getter now rejects them.

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
