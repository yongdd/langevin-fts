# Parameter Reference

This document provides a complete reference for all simulation parameters.

## Table of Contents

1. [Simulation Setup](#simulation-setup)
2. [Chain Definition](#chain-definition)
3. [Numerical Methods](#numerical-methods)
4. [Platform](#platform)
5. [SCFT Parameters](#scft-parameters)
6. [L-FTS Parameters](#l-fts-parameters)
7. [CL-FTS Parameters](#cl-fts-parameters) (with the [Gibbs ensemble](#gibbs-ensemble))
8. [Advanced Options](#advanced-options)
9. [Example Configurations](#example-configurations)

---

## Simulation Setup

### Grid Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `nx` | list[int] | Grid points per dimension | `[32, 32, 32]` |
| `lx` | list[float] | Box lengths (in units of $`b N_{ref}^{1/2}`$) | `[4.0, 4.0, 4.0]` |

### Boundary Conditions

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `bc` | list[str] | Boundary conditions, low and high side of each direction (6 values for 3D) | `["periodic"] * (2 * dim)` |

**Boundary condition types:**
- `"periodic"`: Cyclic boundary (uses FFT)
- `"reflecting"`: Zero flux / Neumann (uses DCT)
- `"absorbing"`: Zero value / Dirichlet (uses DST)

Note: FFT/DCT/DST apply to pseudo-spectral methods (RQM4, RK2). CN-ADI2 implements these boundary conditions using ghost cells.

**Restrictions:**
- Pseudo-spectral methods (RQM4, RK2, and the discrete-chain solver) require every direction to be periodic or every direction to be non-periodic; each non-periodic direction may be reflecting or absorbing independently of the others (e.g. reflecting in x and z, absorbing in y). Mixing periodic with non-periodic directions raises an error on all platforms.
- Mixing periodic and non-periodic directions (e.g., periodic in x/y, reflecting in z) requires `"numerical_method": "cn-adi2"`, which is available for **continuous chains only**. Discrete chains cannot mix periodic and non-periodic directions.
- Stress computation (and therefore `relax_cell`) supports periodic, reflecting, and absorbing boundary conditions (pseudo-spectral methods, real fields).
- Reflecting or absorbing walls need an orthogonal cell (all angles 90°), and a `space_group` needs periodic boundaries in every direction; both combinations are rejected.
- Impenetrable regions (a `mask`) are available through `PropagatorSolver(mask=...)` only; `SCFT`, `LFTS` and `CLFTS` reject a `mask` parameter.

**Example (reflecting in z, periodic in x/y — requires `"numerical_method": "cn-adi2"`):**
```python
"bc": ["periodic", "periodic",      # x: low, high
       "periodic", "periodic",      # y: low, high
       "reflecting", "reflecting"]  # z: low, high
```

---

## Chain Definition

### Basic Chain Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `chain_model` | str | `"continuous"` or `"discrete"` | `"continuous"` |
| `ds` | float | Contour step size (typically $`1/N_{ref}`$) | `0.01` |

### Monomer Properties

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `segment_lengths` | dict | Relative statistical segment lengths $`b/b_{ref}`$ (linear ratio; the code squares this internally) | `{"A": 1.0, "B": 1.0}` |
| `chi_n` | dict | Flory-Huggins parameters × $`N_{ref}`$; pairs that are omitted do not interact (χN = 0). A singular χN matrix (e.g. a zero pair in a three-monomer system, or chemically identical monomers) is accepted. Incompressible SCFT rejects a single monomer type | `{"A,B": 20.0}` |
| `zeta_n` | float | (Optional) Compressibility parameter $`\zeta N \geq 0`$ of the compressible model. Omitted: incompressible ($`\sum_i \phi_i = 1`$) in SCFT and L-FTS; CL-FTS needs a finite value and defaults to `100.0` | `100.0` |

### Polymer Definition

| Parameter | Type | Description |
|-----------|------|-------------|
| `polymers` | list[dict] | List of polymer species |

**Each polymer dictionary:**

| Key | Type | Description |
|-----|------|-------------|
| `volume_fraction` | float | Volume fraction in system |
| `chemical_potential` | float | (CL-FTS) In place of `volume_fraction`: a grand-canonical species with the chemical potential $`\beta\mu_p`$ per chain, in the convention $`\beta\mu_p = \ln(\phi_p/\alpha_p) - \ln Q_p`$ of the recorded chemical potentials; the number of chains per reference chain is $`e^{\beta\mu_p} Q_p`$. Not for grafted chains or random copolymers |
| `blocks` | list | Block specifications |
| `initial_conditions` | dict | (Optional) Custom propagator initial conditions per node, e.g. `{0: "G"}` maps node 0 to the initial condition labeled `"G"` in `q_init` (used for grafted chains). Labels may contain letters, digits and underscores only. Only SCFT takes `q_init`: L-FTS and CL-FTS reject grafted chains |

**Block format:** Each block is a dictionary with:
- `"type"`: Monomer type (string)
- `"length"`: Contour length (float)
- `"v"`: Start vertex (optional for linear chains)
- `"u"`: End vertex (optional for linear chains)
- `"monomer_fractions"`: (Optional) Random copolymer, see below

```python
"polymers": [{
    "volume_fraction": 1.0,
    "blocks": [
        {"type": "A", "length": 0.5},  # A block, 50% of chain
        {"type": "B", "length": 0.5}   # B block, 50% of chain
    ]
}]
```

The volume fractions sum to 1. With grand-canonical polymers (`chemical_potential`) those of the canonical polymers sum to at most 1, and the grand-canonical chains fill the rest on average ([model](../theory/PolymerFieldTheory.md#6-grand-canonical-species)).

**Random copolymer:** a polymer with a single block that has a `"monomer_fractions"` dictionary, e.g. `{"type": "R", "length": 1.0, "monomer_fractions": {"A": 0.5, "B": 0.5}}`, is a random copolymer of the listed monomer types. `"type"` is a new name that is not in `segment_lengths`, the fractions must sum to 1, and the chain uses the field $`\sum_i f_i w_i`$ and the segment length $`(\sum_i f_i b_i^2)^{1/2}`$; its concentration is split among the monomer types by the fractions (see `examples/lfts/MixtureBlockRandom.py`).

### Polymer Architectures

**Linear AB diblock:**
```python
"blocks": [
    {"type": "A", "length": 0.5},
    {"type": "B", "length": 0.5}
]
#   0 --A-- 1 --B-- 2
```

**Star polymer (3-arm AB):**
```python
"blocks": [
    {"type": "A", "length": 0.5, "v": 0, "u": 1},
    {"type": "A", "length": 0.5, "v": 0, "u": 2},
    {"type": "A", "length": 0.5, "v": 0, "u": 3},
    {"type": "B", "length": 0.5, "v": 1, "u": 4},
    {"type": "B", "length": 0.5, "v": 2, "u": 5},
    {"type": "B", "length": 0.5, "v": 3, "u": 6}
]
#      4        5        6
#      |B       |B       |B
#      1        2        3
#       \   A   |   A   /
#        \      |      /
#         ------0------
```

**ABC triblock:**
```python
"blocks": [
    {"type": "A", "length": 0.33},
    {"type": "B", "length": 0.34},
    {"type": "C", "length": 0.33}
]
#   0 --A-- 1 --B-- 2 --C-- 3
```

---

## Numerical Methods

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `numerical_method` | str | Algorithm for propagator computation | `"rqm4"` |

**Available methods:**

| Value | Order | Type | Description |
|-------|-------|------|-------------|
| `"rqm4"` | 4th | Pseudo-spectral | Richardson extrapolation (default) |
| `"rk2"` | 2nd | Pseudo-spectral | Rasmussen-Kalosakas operator splitting |
| `"cn-adi2"` | 2nd | Real-space | Crank-Nicolson ADI (continuous chains only) |

**Restrictions:**
- `numerical_method` applies to the **continuous** chain model only. For `"chain_model": "discrete"`, the parameter is ignored (a note is printed) and the dedicated discrete-chain pseudo-spectral solver is always used.
- `"cn-adi2"` is the only method that supports mixing periodic and non-periodic boundary conditions; the pseudo-spectral methods require every direction to be periodic or every direction to be non-periodic (see [Boundary Conditions](#boundary-conditions)).

See [NumericalMethods.md](../theory/NumericalMethods.md) for benchmarks.

---

## Platform

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `platform` | str | Computational backend | Auto-selected |

**Available platforms:**
- `"auto"`: automatic selection (the default; see below)
- `"cuda"`: NVIDIA GPU (cuFFT)
- `"cpu-mkl"`: CPU (Intel MKL library)
- `"cpu-fftw"`: CPU (FFTW library, GPL license)

**Auto-selection logic** (same for `PropagatorSolver`, `SCFT`, `LFTS` and `CLFTS`):
- 1D simulations → the first CPU backend of the build
- 2D/3D simulations → `"cuda"` if available, else CPU
- Inputs that the CUDA pseudo-spectral solvers do not run stay on the CPU: an odd grid size, or more than 6142 points, along an absorbing direction in 2D/3D, and complex fields of continuous chains with non-periodic boundaries. With `"platform": "cuda"` these inputs raise an error

---

## SCFT Parameters

`SCFT(params)` reads the shared parameters above plus the keys below. These keys and `space_group` are read by `SCFT` only; `LFTS` and `CLFTS` ignore them (a `space_group` given to them has no effect).

### Optimizer

| Parameter | Type | Description |
|-----------|------|-------------|
| `optimizer` | dict | Anderson mixing configuration (the only optimizer) |

| Key | Type | Description | Typical value |
|-----|------|-------------|---------------|
| `name` | str | Must be `"am"` | `"am"` |
| `max_hist` | int | Maximum history length | `20` |
| `start_error` | float | Error threshold to start Anderson mixing (simple mixing before) | `1e-1` |
| `mix_min` | float | Minimum mixing parameter | `0.1` |
| `mix_init` | float | Initial mixing parameter | `0.1` |

All keys are **required**: omitting one, a name other than `"am"`, a `max_hist` that is not an integer >= 0, or a non-finite or negative `start_error`, `mix_min` or `mix_init` raises `ValidationError`, as do a `tolerance` or `stress_scale` that is not finite and positive. The values above are typical starting points; reduce `mix_min`/`mix_init` to `0.01` if the iteration diverges.

```python
"optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-1, "mix_min": 0.1, "mix_init": 0.1}
```

### Convergence and Box Relaxation

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `max_iter` | int | Maximum number of SCFT iterations | `2000` |
| `tolerance` | float | Stop when the self-consistency error (plus the norm of the stress along the relaxed lattice parameters when the box relaxes) is below this value | `1e-8` |
| `relax_cell` | bool | Relax the box lengths (and, for monoclinic/triclinic/oblique systems, the angles) together with the fields | Required |
| `stress_scale` | float | Scaling of the stress entries in the optimizer vector | `1` |
| `stress_interval` | int or `"adaptive"` | Compute the stress every N iterations; `"adaptive"` uses 1, 2 or 4 as the error drops below 1e-2 and 1e-4 | `1` |

Box relaxation needs a pseudo-spectral method with real fields (see [Box Optimization](#box-optimization) for the restrictions). Each iteration changes every box length by at most 10 % and every angle by at most 5°.

### Crystal System

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `crystal_system` | str | Which lattice parameters may vary during box relaxation | `"Orthorhombic"` (see below) |

| Value | Free lengths | Free angles |
|-------|--------------|-------------|
| `"Orthorhombic"` | a, b, c independently | none (all 90°) |
| `"Tetragonal"` | a = b, c | none |
| `"Cubic"` | a = b = c | none |
| `"Hexagonal"` | a = b, c (3D only) | none (γ = 120°) |
| `"Monoclinic"` | a, b, c (3D only) | β |
| `"Triclinic"` | a, b, c (3D only) | α, β, γ |

Without `crystal_system`, a 2D cell with `angles` is classified from γ: 120° → hexagonal (a = b), other non-90° → oblique (a, b and γ free). Relaxation never changes the angles that are not free, and its first step overwrites the tied lengths with the first one: with `relax_cell` and without a space group, a given `crystal_system` (and a 2D cell classified as hexagonal) needs `lx` and `angles` that already satisfy the table, for example `"angles": [90, 90, 120]` with a = b for `"Hexagonal"`; otherwise SCFT raises `ValidationError`. A fixed cell (`"relax_cell": false`) is not checked. Tied lengths relax along the sum of their stresses, which is also what the convergence test measures; when the structure lacks the symmetry of the tie and the stresses of the tied lengths do not vanish separately, SCFT prints a warning that names a crystal system with free lengths (Orthorhombic, Triclinic for Hexagonal, Oblique2D for Hexagonal2D). With a `space_group` the crystal system comes from the group and the angles never vary. The table lives in `CRYSTAL_SYSTEMS` at the top of `scft.py`. Valid values depend on the dimension: 3D accepts Orthorhombic, Tetragonal, Cubic, Hexagonal, Monoclinic and Triclinic; 2D accepts Orthorhombic, Hexagonal2D and Oblique2D; 1D only Orthorhombic.

### Axis Ordering

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `axis_ordering` | list[str] | Order in which `nx`, `lx`, `bc` and `angles` are given, e.g. `["b", "a", "c"]`; the code converts to the standard `[a, b, c]`. The arrays of `initial_fields` (and `q_init`) are given in the same order as `nx` and transposed too. Results (fields, concentrations, saved files with their `nx` and `lx`) are in the standard order; to pass them back as initial fields of a run with `axis_ordering`, transpose them to its order | `["a", "b", "c"]` |

### Output

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `polymer_figure_dir` | str | Directory into which the constructor draws the chain architectures (`polymer_<id>.png`); without it no figures are written | none |

`space_group` and `smearing` are described under [Advanced Options](#advanced-options).

### Electrostatics (Mean Field)

SCFT accepts `electrostatics` as described under [Electrostatics](#electrostatics) and solves the Poisson equation for the electrostatic potential $`\psi`$ together with the fields ([theory](../theory/SelfConsistentFieldTheory.md#25-charged-systems)). It needs `coupling` ($`E`$; SCFT has no $`\bar N`$ for `bjerrum_length`), takes no `psi_dt_scaling`, one smearing length for all monomer types and no `space_group`; box relaxation needs a periodic box. Point ions may be canonical or grand canonical; grand-canonical polymers are not available. `run(..., initial_psi=...)` takes an initial $`\psi`$, by default the uniform potential of the homogeneous state. The results hold `psi` and `ion_chemical_potentials` (also in `save_results`).

---

## L-FTS Parameters

### Langevin Dynamics

| Parameter | Type | Description |
|-----------|------|-------------|
| `langevin` | dict | Langevin dynamics configuration |

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `dt` | float | Langevin time step $`\Delta\tau \cdot N_{ref}`$ | Required |
| `nbar` | float | Invariant polymerization index $`\bar{N}`$ | Required |
| `max_step` | int | Final Langevin step number, including steps restored from a checkpoint | Required |
| `integrator` | str | Built-in Langevin method: `"lm"`, `"euler"` or, in CL-FTS with a periodic box, `"etd"` (Leimkuhler-Matthews steps with the exact linear response of the homogeneous state, [CL-FTS theory](../theory/ComplexLangevinSimulation.md#21-exponential-time-differencing)); an `integrator=` constructor argument takes precedence | `"lm"` |

### Saddle Point

| Parameter | Type | Description |
|-----------|------|-------------|
| `saddle` | dict | Saddle point iteration configuration |

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `max_iter` | int | Maximum iterations for saddle point | Required |
| `tolerance` | float | Convergence tolerance | Required |
| `failure_policy` | str | `"retry"`: continue the saddle solve with the same proposed real fields and noise; `"abort"`: stop on the first failure; `"resample"`: explicitly opt into selecting fresh noise | `"retry"` |
| `max_retries` | int | Additional saddle solves after the first failed attempt | `5` |

The initial saddle must converge before drawing Langevin noise. By default a failed step keeps its proposed real fields and noise and continues from the last imaginary-field iterate with a reset compressor history. After exhausting the retries, `SaddlePointError` is raised. The failed step contributes no measurements. `run()` and `continue_run()` return a status dictionary on success; `sim.run_status` and `recording.dir/run_status.json` identify completion or failure and the last accepted step. Other numerical errors propagate and are recorded as `failed`.

The optional `"resample"` policy reproduces the earlier noise-selection behavior. Each redraw emits a warning and increments `noise_resamples`; convergence-dependent selection can alter the sampled distribution. Initial-saddle retries never draw noise, regardless of the policy.

### Recording

| Parameter | Type | Description |
|-----------|------|-------------|
| `recording` | dict | Data recording configuration |

| Key | Type | Description |
|-----|------|-------------|
| `dir` | str | Output directory name; a run also draws the chain architectures here (`polymer_<id>.png`) |
| `checkpoint_interval` | int | Interval for saving complete restart checkpoints |
| `sample_interval` | int | Collect H, dH/dχN, drift and structure-function samples every N Langevin steps |
| `statistics_save_interval` | int | Save the collected histories and averaged structure functions, then reset the accumulators, every N Langevin steps |
| `pressure` | bool | (CL-FTS, optional) Also record the osmotic pressure at every sample; needs a periodic box and free chains. Default `False` |

All intervals are positive integer counts of Langevin steps. Structure functions are normalized by the actual sample count, including when `statistics_save_interval` is not a multiple of `sample_interval`. Measurement files include `sample_count` and a JSON string `simulation_metadata_json`. Hamiltonian files also record `drift_max_history`: the maximum absolute scaled force used for each sampled transition, including dynamical stabilization in CL-FTS. CL-FTS also records `drift_max_step_history`, the maximum of every step for each auxiliary field (one column per field), and with electrostatics `psi_drift_max_step_history`; checkpoints keep the steps since the last statistics file. See [ComplexLangevinSimulation.md](../theory/ComplexLangevinSimulation.md#3-correct-convergence) for their use. At every sample CL-FTS records the chemical potentials βμ_p of all species (`chemical_potential_history_real`/`_imag`, one row per sample), with `pressure` the osmotic pressure βΠR₀³/C (`pressure_history_real`/`_imag`) and, with grand-canonical species, the numbers of chains and ions per reference chain of all species (`number_density_history_real`/`_imag`, the columns of the chemical potentials) ([CL-FTS theory, Section 7](../theory/ComplexLangevinSimulation.md#7-chemical-potential-and-osmotic-pressure)). Custom callbacks use their own `interval` independently of these built-in measurements.

Field checkpoints use format version 2 and are replaced atomically. They store the full PCG64 state, previous noise, auxiliary fields, concentrations, pending measurement sums/histories, WTMD state, and custom integrator/callback settings and states. Version-1 files continue with default LM and no callbacks. `continue_run(path)` validates the effective physical and numerical settings before restoring the RNG. The output directory, checkpoint interval, maximum final step, verbosity and backend may change; sampling periods, model, integrator settings, named callbacks and WTMD bias settings must match. On a different backend only numerical/statistical equivalence is expected. Old unversioned files require `continue_run(path, allow_legacy=True)` and emit a warning because compatibility and pending measurements cannot be verified.

See [Numerical Validation](../internals/NumericalValidation.md) for regression coverage, provenance and statistical validation jobs. Both `LFTS` and `CLFTS` accept keyword arguments `integrator=` and `callbacks=`; see [Extensions](../internals/Extensions.md) for their contracts and a runnable example.

### Compressor

| Parameter | Type | Description |
|-----------|------|-------------|
| `compressor` | dict | Field compressor configuration |

| Key | Type | Description | Typical value |
|-----|------|-------------|---------------|
| `name` | str | Compressor type: `"am"`, `"lr"`, or `"lram"` | Required |
| `max_hist` | int | Anderson mixing history length | `20` |
| `start_error` | float | Error threshold to start AM | `5e-1` |
| `mix_min` | float | Minimum mixing parameter | `0.01` |
| `mix_init` | float | Initial mixing parameter | `0.01` |

For `"am"` and `"lram"`, the keys `max_hist`, `start_error`, `mix_min`, and `mix_init` are **required** (no built-in defaults; omitting one raises `ValidationError`). `"lr"` and `"lram"` require exactly one imaginary auxiliary field and periodic axes or reflecting walls (their Jacobian is measured in the Fourier basis of the box: the FFT along periodic axes, the cosine series along reflecting walls); otherwise, as with absorbing walls, `LFTS` raises `ValidationError`.

### Well-Tempered Metadynamics (Optional)

| Parameter | Type | Description |
|-----------|------|-------------|
| `wtmd` | dict | WTMD configuration (optional) |

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `ell` | int | $`\ell`$-norm for order parameter | `4` |
| `kc` | float | Cutoff wavenumber | `6.02` |
| `delta_t` | float | Well-tempering factor $`\Delta T/T`$ | `5.0` |
| `sigma_psi` | float | Gaussian width $`\sigma_\Psi`$ | `0.16` |
| `psi_min` | float | Minimum $`\Psi`$ | `0.0` |
| `psi_max` | float | Maximum $`\Psi`$ | `10.0` |
| `dpsi` | float | Bin width $`d\Psi`$; a new run requires it to divide `psi_max - psi_min` into a whole number of bins | `1e-3` |
| `update_freq` | int | Statistics update frequency | `1000` |
| `recording_period` | int | Data recording frequency | `100000` |

WTMD requires a 3D grid (`nx` with three entries) and a real auxiliary (exchange) field, that is a negative value among the `Eigenvalues` the constructor prints (of the projected interaction matrix, see [Polymer Field Theory](../theory/PolymerFieldTheory.md#2-incompressible-model); for two monomer types, a positive `chi_n`).

### Other L-FTS Options

| Parameter | Type | Description |
|-----------|------|-------------|
| `verbose_level` | int | Output verbosity: 0 (no line per Langevin step), 1 (the Langevin step number and one saddle-point summary line per Langevin step), 2 (the step number and one line per saddle-point iteration). Other values raise `ValidationError` |
| `smearing` | dict | (Optional) Finite-range interactions, see [Smearing](#smearing) |

`LFTS(params, random_seed=None)`: pass an integer `random_seed` for reproducible noise.

---

## CL-FTS Parameters

`CLFTS(params, random_seed=None)` (complex Langevin) reads the shared parameters, `langevin` and `recording` exactly as L-FTS does (no `compressor`, `saddle` or `wtmd`: all fields fluctuate; a `wtmd` block raises `ValidationError`), and `verbose_level` (0, 1 or 2; 0 prints nothing per Langevin step, 1 and 2 both print one status line per step), plus:

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `zeta_n` | float | Compressibility parameter; complex Langevin needs a finite value | `100.0` |
| `dynamic_stabilization` | float | Dynamical stabilization constant αds (Willis & Matsen, *J. Chem. Phys.* **2024**, 161, 244903): adds $`i\,\alpha_{ds}\,\mathrm{Im}[W_-]`$ to the exchange-field force to suppress runaway solutions. `alpha_ds` is accepted as an alias. Willis & Matsen used 0.001 at $`\bar{N} = 10^6`$–$`10^7`$ and 0.01–0.02 at $`\bar{N} = 10^5`$; 0.1 already biased their results | `0.0` |
| `smearing` | dict | (Optional) Finite-range interactions, see [Smearing](#smearing); recommended for CL-FTS stability. CL-FTS also accepts one Gaussian length per monomer type | none |
| `electrostatics` | dict | (Optional) Smeared Coulomb interactions, see [Electrostatics](#electrostatics) | none |

### Electrostatics

CL-FTS samples the electrostatic potential; SCFT takes its saddle point ([Electrostatics (Mean Field)](#electrostatics-mean-field)). Charged monomer types interact through the smeared charge density $`c = \sum_i z_i\,(\Gamma_i * \phi_i)`$ with the Coulomb kernel $`E/k^2`$, or the screened kernel $`E/(k^2 + \kappa^2)`$ with `screening_length` ([model](../theory/PolymerFieldTheory.md#5-electrostatic-interactions)). CL-FTS decouples the interaction with a fluctuating electrostatic potential $`\psi`$ ([sampling](../theory/ComplexLangevinSimulation.md#6-charged-systems)). Small ions are either molecules of one segment in `polymers`, which take volume like the other species, or point ions (`ions`), which have no volume and no $`\chi`$ ([model](../theory/PolymerFieldTheory.md#55-point-ions)).

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `charges` | dict | Valence per segment of each charged monomer type, `{"A": -0.2, "Na": 1.0}`; other types are neutral | required |
| `bjerrum_length` | float | (CL-FTS) Bjerrum length $`l_B`$ in units of $`R_0`$; the coupling is $`E = 4\pi (l_B/R_0) N^2 \bar{N}^{1/2}`$ with the number of segments $`N`$ of the reference chain (`n_segments`) | — |
| `coupling` | float | The coupling $`E = 4\pi l_B \rho_0 N R_0^2`$ directly, with the physical $`N`$. Give exactly one of `bjerrum_length` and `coupling` | — |
| `n_segments` | float | Number of segments $`N`$ of the reference chain. Discrete chains have $`N = 1/\Delta s`$ (an integer) and need not give it; continuous chains, whose $`\Delta s`$ is a contour step, need it for `bjerrum_length` and `ions` | $`1/\Delta s`$ (discrete) |
| `ions` | dict | (Optional) Point ions, `{"Na": {"valence": 1, "concentration": 0.02, "smearing_length": 0.1}, ...}`: valence, concentration of added salt in ions per segment of the reference density $`\rho_0`$, and Gaussian smearing length in units of $`R_0`$ (positive). In place of `concentration`, `chemical_potential` $`\beta\mu_s`$ makes the species grand canonical, in the convention $`\beta\mu_s = \ln(N c_s) - \ln Q_s`$. Names must differ from the monomer types | none |
| `counterion` | str or None | The canonical point ion that neutralizes the canonical polymers and the given canonical ions: its concentration grows by the net charge divided by its valence. Default: the only canonical ion species whose valence has the opposite sign of the net charge (an error if there are several). `None` adds no counterions | automatic |
| `psi_dt_scaling` | float | (CL-FTS) Mobility of $`\psi`$ relative to the auxiliary fields | `1.0` |
| `fixed_charge` | array | (Optional) Fixed charge density $`c_f(\mathbf{r})`$ in charges per segment of the reference density, one value per grid point (as `nx`), for example charged surfaces or particles. It adds to the charge density as given (not smeared) and to the electroneutrality through its mean | none |
| `screening_length` | float | (Optional) Screening length $`1/\kappa`$ in units of $`R_0`$ of a background electrolyte that is not simulated explicitly: the kernel becomes $`E/(k^2 + \kappa^2)`$. It is the Debye length of the background alone; explicit ions screen in addition ([model](../theory/PolymerFieldTheory.md#54-screened-coulomb-interaction)). Omit it for unscreened charges | none |

**Requirements:**
- Without `screening_length` the system must be electroneutral: $`\sum_i z_i \phi_i + \sum_s z_s c_s = 0`$ with the overall segment fraction $`\phi_i`$ of each type and the point-ion concentrations $`c_s`$, counterions included (to $`10^{-10}`$), plus the mean of `fixed_charge`. With counterions in `polymers`, their fraction follows from the polymer charge; point ions take it automatically (`counterion`). With `screening_length` the background electrolyte screens a net charge. A charged grand-canonical species (polymer or point ion) also lifts the condition: the uniform mode of $`\psi`$ then evolves as the Donnan potential that makes the system neutral ([model](../theory/PolymerFieldTheory.md#6-grand-canonical-species)).
- `smearing` must be given; with per-type Gaussian lengths $`a_i`$ (Born radii) each type has its own $`\Gamma_i(k)`$. Charged types need a positive Gaussian length, since $`a_i = 0`$ leaves point charges.
- A periodic box, or a closed box with walls on every side (`bc` reflecting or absorbing; not both kinds of axes). With walls the transforms are cosine series, so $`\psi`$ has no normal derivative at the walls; the charge structure function, the osmotic pressure and the `etd` integrator need a periodic box. No random copolymers.
- `zeta_n` is finite, as always in CL-FTS.

```python
params["smearing"] = {"type": "gaussian", "a_int": {"A": 0.1, "B": 0.1, "C": 0.15}}
params["electrostatics"] = {"charges": {"A": 0.2, "C": -1.0}, "bjerrum_length": 0.01}

# The same polymers in a salt of point ions: Cl- also neutralizes the charge of A
params["electrostatics"] = {"charges": {"A": 0.2}, "bjerrum_length": 0.01,
                            "ions": {"Na": {"valence": 1, "concentration": 0.02, "smearing_length": 0.1},
                                     "Cl": {"valence": -1, "concentration": 0.02, "smearing_length": 0.1}}}
```

Grand-canonical salt in contact with a reservoir, for example a polyelectrolyte gel, takes chemical potentials instead of concentrations; the Donnan potential $`\psi_0`$ adjusts the numbers of ions:

```python
mu = np.log(N * 0.02)   # c = 0.02 per segment in the reservoir (psi = 0)
params["electrostatics"] = {"charges": {"A": 0.2}, "bjerrum_length": 0.01,
                            "ions": {"Na": {"valence": 1, "chemical_potential": mu, "smearing_length": 0.1},
                                     "Cl": {"valence": -1, "chemical_potential": mu, "smearing_length": 0.1}}}
```

The point ions add one chemical potential each to `chemical_potential_history`, after the polymers.

### Gibbs Ensemble

`GibbsEnsemble(params, random_seed=None)` runs two CL-FTS boxes that exchange chains, point ions and volume toward coexisting phases ([theory](../theory/ComplexLangevinSimulation.md#8-gibbs-ensemble)). `params` is a CL-FTS input with the section `gibbs_ensemble`; the volume fractions of `polymers` and the top-level `lx` are not used, and `langevin.max_step` and the recording intervals other than `sample_interval` are set by the cycles. The boxes write into `recording.dir`/`box0` and `box1`, and `gibbs_history.json` holds every cycle. `run(initial_fields=None, initial_psi=None)` (one dict for both boxes or a list of two) starts from the boxes of the input and replaces an earlier `gibbs_history.json`; `continue_run()` continues it. Both return `summary()`, the averages of the second half of the cycles.

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `boxes` | list | Two dicts with `lx` (the lengths at the starting composition), `volume_fractions` (one per polymer, summing to 1) and optionally `ion_concentrations` of the canonical point ions (by default those of `electrostatics` with the counterions of the box) | required |
| `cycles` | int | Number of exchanges | required |
| `cycle_steps` | int | Langevin steps of each box per exchange, a multiple of `recording.sample_interval` | required |
| `exchange_rate` | float | Rate $`r`$ of the exchange | `0.2` |
| `skip_steps` | int | Steps at the start of each cycle left out of the averages (smaller than `cycle_steps`) | `0` |

The polymers must be canonical free chains and the boxes periodic, and every polymer and canonical point ion needs a positive amount in both boxes (leave ions that are absent out of `electrostatics.ions`). Without screening, charged boxes with only canonical point ions must be neutral; charged grand-canonical point ions keep their chemical potential in both boxes and set the Donnan potential of each box, which then need not be neutral by itself. `random_seed` is a non-negative integer.

The checkpoints store $`\psi`$ and its random-number stream, and a continuation must use the same electrostatic settings. A run continued by hand with `run(..., start_langevin_step=...)` takes $`\psi`$ through `initial_psi`, for example `psi_real + 1j * psi_imag` of a fields file, and the previous noise of its Leimkuhler-Matthews step through `psi_noise_prev` (zero if omitted); without `initial_psi` $`\psi`$ restarts at zero with a warning. Such a run does not restore the random-number streams; `continue_run()` continues a checkpoint exactly. [ChargedDiblock.py](../../examples/clfts/ChargedDiblock.py) is a complete script.

---

## Advanced Options

### Memory Management

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `reduce_memory` | bool | Enable checkpoint-based memory saving | `False` |
| `aggregate_propagator_computation` | bool | Let the propagator optimizer merge equivalent propagators of branched/mixed systems (turn off only when per-block concentrations are needed from the C++ API) | `True` |

When `True`:
- Stores only checkpoints every ~$`2\sqrt{N_{tot}}`$ contour steps, i.e. ~$`\sqrt{N_{tot}}/2`$ checkpoints per propagator (where $`N_{tot}`$ is the total number of contour steps)
- Reduces the memory that grows with the grid and the number of contour steps: for `examples/scft/Gyroid.py` (AB diblock, $`ds = 0.01`$) the GPU memory of the process on an A10 falls from 332 to 254 MiB on its 32³ grid and from 1064 to 420 MiB on a 64³ grid, of which about 236 MiB is the CUDA context. The CUDA checkpoints are kept in pinned host memory, which raises the host memory at 64³ from 298 to 494 MiB. On cpu-mkl the peak host memory at 64³ falls from 783 to 448 MiB
- Increases computation time by 2-4x

`reduce_memory=True` works with every chain model and with space-group symmetry, including **discrete chains combined with a space group**, on both CPU and CUDA (see [SpaceGroup.md](../theory/SpaceGroup.md)).

### Space Group Symmetry (Beta)

| Parameter | Type | Description |
|-----------|------|-------------|
| `space_group` | dict | Space group constraints (SCFT only) |

```python
"space_group": {
    "symbol": "Im-3m",   # Hermann-Mauguin symbol
    "hall_number": 529        # Hall number (optional)
}
```

The box must match the crystal system of the group: cubic groups need equal `lx` and 90° angles, tetragonal a = b with 90° angles, orthorhombic 90° angles, hexagonal and trigonal groups on hexagonal axes a = b with `"angles": [90, 90, 120]`, and rhombohedral-axes settings a = b = c with α = β = γ (box relaxation is not supported for them). `SCFT` raises `ValidationError` otherwise. A space group needs periodic boundaries in every direction, and a `mask` used with it must have the symmetry of the group (a mask without it is rejected); the reduced-basis integrals weight each orbit by its size times the mask.

**Platform support:**
- **Discrete chains + space group** is supported on both CPU and CUDA platforms, in both standard and `reduce_memory=True` modes.

See [SpaceGroup.md](../theory/SpaceGroup.md) for available space groups.

### Smearing

| Parameter | Type | Description |
|-----------|------|-------------|
| `smearing` | dict | (Optional, SCFT/L-FTS/CL-FTS) Finite-range interactions: the fields are convolved with a smearing function before the propagators and the concentrations before the forces, $`w_\Gamma(k) = w(k)\,\Gamma(k)`$ ([theory](../theory/PolymerFieldTheory.md#4-smeared-interactions)) |

| Type | Keys | $`\Gamma(k)`$ |
|------|------|--------------|
| `{"type": "gaussian", "a_int": a}` | smearing length $`a`$ in units of $`R_0`$ (universality needs $`a \lesssim 0.02 R_0`$) | $`\exp(-a^2 k^2 / 2)`$ |
| `{"type": "gaussian", "a_int": {"A": a_A, ...}}` | CL-FTS only: one length for every monomer type; type $`i`$ is smeared with its own $`\Gamma_i`$, and the structure function of a pair $`i \le j`$ is divided by $`\Gamma_i`$. No random copolymers | $`\exp(-a_i^2 k^2 / 2)`$ |
| `{"type": "sigmoidal", "k_int": k_c, "dk_int": dk}` | cutoff wavenumber $`k_c \gtrsim 20 / R_0`$, transition width $`dk`$ (default 5.0) | $`C_0 [1 - \tanh((k - k_c)/dk)]`$ |

$`k`$ is the physical wavenumber of the cell, including the cell angles. Along an axis with reflecting or absorbing walls the field is smeared in the cosine (DCT-II) basis, i.e. together with its even image across the wall (across an absorbing wall the propagators are odd, so fields and concentrations are even); the two sides of an axis may have different walls. In SCFT box relaxation the kernel follows the box, and the stress includes the derivative of $`\Gamma(k)`$. Not available together with `space_group`. References: Delaney & Fredrickson, *J. Phys. Chem. B* **2016**, 120, 7615; Matsen et al., *J. Chem. Phys.* **2026**, 164, 014905.

### Box Optimization

`relax_cell`, `stress_scale`, `stress_interval` and `crystal_system` are listed under [SCFT Parameters](#scft-parameters).

**Restrictions:** Stress computation supports periodic, reflecting, and absorbing boundary conditions for pseudo-spectral methods with real fields (complex fields with non-periodic boundaries are rejected), but is not implemented for the real-space `"cn-adi2"` method. Therefore `relax_cell=True` requires a pseudo-spectral method. With non-periodic boundaries only box lengths can be optimized (angles stay at 90°); discrete-chain stress is exact against d(-lnQ)/dL, continuous-chain stress carries the contour-quadrature error (see docs/theory/StressTensor.md).

### Lattice Angles (Non-orthogonal)

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `angles` | list[float] | Unit cell angles [α, β, γ] in degrees | `[90, 90, 90]` |

Following crystallographic convention:
- α: angle between **b** and **c**
- β: angle between **a** and **c**
- γ: angle between **a** and **b**

```python
"angles": [90.0, 90.0, 120.0]  # Hexagonal: alpha=90, beta=90, gamma=120
```

Non-orthogonal cells need periodic boundaries in every direction (walls are rejected), and the angles must give a cell that is not nearly flat: the ratio of the cell volume to $`abc`$, $`\sqrt{1 - \cos^2\alpha - \cos^2\beta - \cos^2\gamma + 2\cos\alpha\cos\beta\cos\gamma}`$ ($`\sin\gamma`$ in 2D), must be at least 0.1. With one oblique angle this allows 5.74° to 174.26°. A flatter cell raises an error, also when box relaxation reaches it, because the Fourier modes of a flat cell are expensive to set up and its volume loses precision. Describe such a lattice with its reduced (Niggli) cell, spanned by its shortest vectors, whose angles lie between 60° and 120° and whose volume ratio is at least $`1/\sqrt{2}`$; with a rhombohedral space group, use the hexagonal-axes setting of the group.

### Parameter Files (YAML/JSON)

The parameter dictionary can be read from a file:

```python
from polymerfts import SCFT, LFTS, load_config
calc = SCFT.from_file("gyroid.yaml", max_iter=500)     # keyword arguments override file values
sim  = LFTS.from_file("lamella.yaml", random_seed=12)  # CLFTS.from_file works the same way
params = load_config("gyroid.yaml")                    # just the dictionary
```

The format follows the extension (`.yaml`, `.yml` or `.json`). A string value that uses numpy, such as `lx: ["4.0*np.sqrt(3)/2", 2.0]`, is evaluated as an arithmetic expression: numbers, `+ - * / **`, parentheses, `np.pi`, `np.e` and the one-argument functions `sqrt, cbrt, exp, log, log10, sin, cos, tan, arcsin, arccos, arctan, deg2rad, rad2deg, abs`. Every intermediate value must be a finite real number. Nothing else is executed; any other string containing `np.` is kept as text with a warning. A missing file raises `FileNotFoundError`; an unsupported extension, invalid YAML/JSON, an empty file or content that is not a dictionary raises `ConfigError`.

---

## Example Configurations

### Compatibility with Earlier Parameter Names

SCFT, LFTS and CLFTS accept the following legacy input names. Constructors copy
and normalize the input; `sim.params` and newly saved parameters use the names
documented above. The caller's dictionary is unchanged. Specify one name per
option: supplying both a current name and its alias raises `ValidationError`,
even when their values are equal.

| Earlier name | Current name |
|--------------|--------------|
| `distinct_polymers` | `polymers` |
| `box_is_altering` | `relax_cell` |
| `scale_stress` | `stress_scale` |
| `alpha_ds` | `dynamic_stabilization` |
| `blocks[].fraction` | `blocks[].monomer_fractions` |
| `space_group.number` | `space_group.hall_number` |
| `recording.recording_period` | `recording.checkpoint_interval` |
| `recording.sf_computing_period` | `recording.sample_interval` |
| `recording.sf_recording_period` | `recording.statistics_save_interval` |

Version-1 and version-2 checkpoints written with earlier names can be resumed
using current parameter names. The stored settings hash is verified before
translating aliases for the compatibility comparison. WTMD's separate
`wtmd.recording_period` option keeps its existing name and meaning.

### SCFT Lamellar Phase

```python
params = {
    "nx": [1, 1, 64],
    "lx": [1.0, 1.0, 4.0],
    "chain_model": "continuous",
    "ds": 0.01,
    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 20.0},
    "polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": 0.5},
            {"type": "B", "length": 0.5}
        ]
    }],
    "numerical_method": "rqm4",
    "relax_cell": False,
    "optimizer": {
        "name": "am",
        "max_hist": 20,
        "start_error": 1e-1,
        "mix_min": 0.1,
        "mix_init": 0.1
    }
}
```

### SCFT Gyroid Phase with Box Optimization

```python
params = {
    "nx": [32, 32, 32],
    "lx": [3.65, 3.65, 3.65],
    "chain_model": "continuous",
    "ds": 0.01,
    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 18.0},
    "polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": 0.375},
            {"type": "B", "length": 0.625}
        ]
    }],
    "numerical_method": "rqm4",
    "relax_cell": True,
    "stress_scale": 0.5,
    "space_group": {
        "symbol": "Ia-3d",
        "hall_number": 530
    },
    "optimizer": {
        "name": "am",
        "max_hist": 20,
        "start_error": 1e-1,
        "mix_min": 0.1,
        "mix_init": 0.1
    }
}
```

### Non-Periodic Boundaries (Confined Film)

All numerical methods support boundary conditions that are non-periodic in every direction (reflecting or absorbing, chosen per direction): RQM4 and RK2 use DCT/DST transforms, while CN-ADI2 uses ghost cells. **Mixing** periodic and non-periodic directions, as in the confined-film example below, is supported only by `"cn-adi2"` (continuous chains only) — the pseudo-spectral methods raise an error for mixed boundary conditions. On CUDA, the pseudo-spectral methods need an even grid size of at most 6142 points along absorbing directions (in 1D also at most 4096 points along reflecting ones), and continuous chains with complex fields need periodic boundaries; `"platform": "auto"` runs such inputs on the CPU. `PropagatorSolver.fourier`, with which `"cn-adi2"` runs apply smearing and the `lr`/`lram` compressors in a box with walls on every axis, uses the cosine series along every wall, absorbing ones included (a box that mixes periodic axes and walls transforms them with NumPy/SciPy instead); on CUDA its 1D transform takes at most 4096 points (`"auto"` runs 1D inputs on the CPU), and in 2D and 3D it has no size limit.

```python
params = {
    "nx": [1, 1, 64],
    "lx": [1.0, 1.0, 4.0],
    "bc": ["periodic", "periodic",
           "periodic", "periodic",
           "reflecting", "reflecting"],  # Confined in z
    "chain_model": "continuous",
    "ds": 0.01,
    "numerical_method": "cn-adi2",  # required for mixed BCs (rqm4/rk2 raise an error)
    # ... other parameters
}
```

---

## Units and Conventions

The unit of length is $`b N_{ref}^{1/2}`$ for both chain models, where:
- $`b`$ is a reference statistical segment length
- $`N_{ref}`$ is a reference polymerization index

Fields are defined as **per reference chain** potential. To obtain **per reference segment** potential, multiply each field by `ds`.

This convention follows [*Macromolecules* **2013**, 46, 8037].
