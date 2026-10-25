# Parameter Reference

This document provides a complete reference for all simulation parameters.

## Table of Contents

1. [Simulation Setup](#simulation-setup)
2. [Chain Definition](#chain-definition)
3. [Numerical Methods](#numerical-methods)
4. [Platform](#platform)
5. [SCFT Parameters](#scft-parameters)
6. [L-FTS Parameters](#l-fts-parameters)
7. [CL-FTS Parameters](#cl-fts-parameters)
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
| `blocks` | list | Block specifications |
| `initial_conditions` | dict | (Optional) Custom propagator initial conditions per node, e.g. `{0: "G"}` maps node 0 to the initial condition labeled `"G"` in `q_init` (used for grafted chains). Labels may contain letters, digits and underscores only |

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

All keys are **required** (omitting one raises `KeyError`). The values above are typical starting points; reduce `mix_min`/`mix_init` to `0.01` if the iteration diverges.

```python
"optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-1, "mix_min": 0.1, "mix_init": 0.1}
```

### Convergence and Box Relaxation

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `max_iter` | int | Maximum number of SCFT iterations | `2000` |
| `tolerance` | float | Stop when the self-consistency error (plus the stress norm when the box relaxes) is below this value | `1e-8` |
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

Without `crystal_system`, a 2D cell with `angles` is classified from γ: 120° → hexagonal (a = b), other non-90° → oblique (a, b and γ free). With a `space_group` the crystal system comes from the group and the angles never vary. The table lives in `CRYSTAL_SYSTEMS` at the top of `scft.py`. Valid values depend on the dimension: 3D accepts Orthorhombic, Tetragonal, Cubic, Hexagonal, Monoclinic and Triclinic; 2D accepts Orthorhombic, Hexagonal2D and Oblique2D; 1D only Orthorhombic.

### Axis Ordering

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `axis_ordering` | list[str] | Order in which `nx`, `lx`, `bc` and `angles` are given, e.g. `["b", "a", "c"]`; the code converts to the standard `[a, b, c]`. The arrays of `initial_fields` (and `q_init`) are given in the same order as `nx` and transposed too. Results (fields, concentrations, saved files with their `nx` and `lx`) are in the standard order; to pass them back as initial fields of a run with `axis_ordering`, transpose them to its order | `["a", "b", "c"]` |

`space_group` and `smearing` are described under [Advanced Options](#advanced-options).

---

## L-FTS Parameters

### Langevin Dynamics

| Parameter | Type | Description |
|-----------|------|-------------|
| `langevin` | dict | Langevin dynamics configuration |

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `dt` | float | Langevin time step $`\Delta\tau \cdot N_{ref}`$ | Required |
| `nbar` | float | Invariant polymerization index $`\bar{n}`$ | Required |
| `max_step` | int | Final Langevin step number, including steps restored from a checkpoint | Required |
| `integrator` | str | Built-in Langevin method: `"lm"` or `"euler"`; an `integrator=` constructor argument takes precedence | `"lm"` |

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
| `dir` | str | Output directory name |
| `checkpoint_interval` | int | Interval for saving complete restart checkpoints |
| `sample_interval` | int | Collect H, dH/dχN, drift and structure-function samples every N Langevin steps |
| `statistics_save_interval` | int | Save the collected histories and averaged structure functions, then reset the accumulators, every N Langevin steps |

All intervals are positive integer counts of Langevin steps. Structure functions are normalized by the actual sample count, including when `statistics_save_interval` is not a multiple of `sample_interval`. Measurement files include `sample_count` and a JSON string `simulation_metadata_json`. Hamiltonian files also record `drift_max_history`: the maximum absolute scaled force used for each sampled transition, including dynamical stabilization in CL-FTS. Custom callbacks use their own `interval` independently of these built-in measurements.

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

For `"am"` and `"lram"`, the keys `max_hist`, `start_error`, `mix_min`, and `mix_init` are **required** (no built-in defaults; omitting one raises `KeyError`). `"lr"` and `"lram"` require exactly one imaginary auxiliary field; otherwise `LFTS` raises `ValidationError`.

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
| `dpsi` | float | Bin width $`d\Psi`$ | `1e-3` |
| `update_freq` | int | Statistics update frequency | `1000` |
| `recording_period` | int | Data recording frequency | `100000` |

WTMD requires a 3D grid (`nx` with three entries).

### Other L-FTS Options

| Parameter | Type | Description |
|-----------|------|-------------|
| `verbose_level` | int | Output verbosity: 0 (only the Langevin step number), 1 (plus one saddle-point summary line per Langevin step), 2 (one line per saddle-point iteration). Other values raise `ValidationError` |
| `smearing` | dict | (Optional) Finite-range interactions, see [Smearing](#smearing) |

`LFTS(params, random_seed=None)`: pass an integer `random_seed` for reproducible noise.

---

## CL-FTS Parameters

`CLFTS(params, random_seed=None)` (complex Langevin) reads the shared parameters, `langevin` and `recording` exactly as L-FTS does (no `compressor`, `saddle` or `wtmd`: all fields fluctuate), and `verbose_level` (0, 1 or 2; 0 prints nothing per Langevin step, 1 and 2 both print one status line per step), plus:

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `zeta_n` | float | Compressibility parameter; complex Langevin needs a finite value | `100.0` |
| `dynamic_stabilization` | float | Dynamical stabilization constant αds (Willis & Matsen, *J. Chem. Phys.* **2024**, 161, 244903): adds $`i\,\alpha_{ds}\,\mathrm{Im}[W_-]`$ to the exchange-field force to suppress runaway solutions. `alpha_ds` is accepted as an alias. Willis & Matsen used 0.001 at $`\bar{N} = 10^6`$–$`10^7`$ and 0.01–0.02 at $`\bar{N} = 10^5`$; 0.1 already biased their results | `0.0` |
| `smearing` | dict | (Optional) Finite-range interactions, see [Smearing](#smearing); recommended for CL-FTS stability. CL-FTS also accepts one Gaussian length per monomer type | none |
| `electrostatics` | dict | (Optional) Smeared Coulomb interactions, see [Electrostatics](#electrostatics) | none |

### Electrostatics

Charged monomer types interact through the smeared charge density $`c = \sum_i z_i\,(\Gamma_i * \phi_i)`$ with the Coulomb kernel $`E/k^2`$. CL-FTS decouples the interaction with a fluctuating electrostatic potential $`\psi`$ ([theory, Section 5.6](../theory/LangevinFieldTheoreticSimulation.md#56-charged-systems)). Small ions are molecules of one segment and take volume like the other species.

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `charges` | dict | Valence per segment of each charged monomer type, `{"A": -0.2, "Na": 1.0}`; other types are neutral | required |
| `bjerrum_length` | float | Bjerrum length $`l_B`$ in units of $`R_0`$; the coupling is $`E = 4\pi (l_B/R_0) N^2 \bar{N}^{1/2}`$ with $`N = 1/\Delta s`$ (an integer). Discrete chains only: $`\Delta s`$ of continuous chains is a contour step, not the number of segments | — |
| `coupling` | float | The coupling $`E = 4\pi l_B \rho_0 N R_0^2`$ directly, with the physical $`N`$; required for continuous chains. Give exactly one of `bjerrum_length` and `coupling` | — |
| `psi_dt_scaling` | float | Mobility of $`\psi`$ relative to the auxiliary fields | `1.0` |

**Requirements:**
- The system must be electroneutral: $`\sum_i z_i \phi_i = 0`$ with the overall segment fraction $`\phi_i`$ of each type (to $`10^{-10}`$). The counterion fraction therefore follows from the polymer charge.
- `smearing` must be given; with per-type Gaussian lengths $`a_i`$ (Born radii) each type has its own $`\Gamma_i(k)`$. Charged types need a positive Gaussian length, since $`a_i = 0`$ leaves point charges.
- Periodic boxes only; no random copolymers.
- `zeta_n` is finite, as always in CL-FTS.

```python
params["smearing"] = {"type": "gaussian", "a_int": {"A": 0.1, "B": 0.1, "C": 0.15}}
params["electrostatics"] = {"charges": {"A": 0.2, "C": -1.0}, "bjerrum_length": 0.01}
```

The checkpoints store $`\psi`$ and its random-number stream, and a continuation must use the same electrostatic settings. A run continued by hand with `run(..., start_langevin_step=...)` takes $`\psi`$ through `initial_psi`, for example `psi_real + 1j * psi_imag` of a fields file; without it $`\psi`$ restarts at zero with a warning. Such a run does not restore the random-number streams; `continue_run()` continues a checkpoint exactly. [ChargedDiblock.py](../../examples/clfts/ChargedDiblock.py) is a complete script.

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
| `smearing` | dict | (Optional, SCFT/L-FTS/CL-FTS) Finite-range interactions: the fields are convolved with a smearing function before the propagators and the concentrations before the forces, $`w_\Gamma(k) = w(k)\,\Gamma(k)`$ |

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

Non-orthogonal cells need periodic boundaries in every direction (walls are rejected), and the angles must form a cell ($`1 - \cos^2\alpha - \cos^2\beta - \cos^2\gamma + 2\cos\alpha\cos\beta\cos\gamma > 0`$).

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

All numerical methods support boundary conditions that are non-periodic in every direction (reflecting or absorbing, chosen per direction): RQM4 and RK2 use DCT/DST transforms, while CN-ADI2 uses ghost cells. **Mixing** periodic and non-periodic directions, as in the confined-film example below, is supported only by `"cn-adi2"` (continuous chains only) — the pseudo-spectral methods raise an error for mixed boundary conditions.

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
