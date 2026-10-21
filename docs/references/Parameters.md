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
| `bc` | list[str] | Boundary conditions (6 values for 3D) | `["periodic"]*6` |

**Boundary condition types:**
- `"periodic"`: Cyclic boundary (uses FFT)
- `"reflecting"`: Zero flux / Neumann (uses DCT)
- `"absorbing"`: Zero value / Dirichlet (uses DST)

Note: FFT/DCT/DST apply to pseudo-spectral methods (RQM4, RK2). CN-ADI2 implements these boundary conditions using ghost cells.

**Restrictions:**
- Pseudo-spectral methods (RQM4, RK2, and the discrete-chain solver) require **uniform** boundary conditions: all-periodic, all-reflecting, or all-absorbing. Mixing periodic with non-periodic directions raises an error on all platforms.
- Mixed boundary conditions (e.g., periodic in x/y, reflecting in z) require `"numerical_method": "cn-adi2"`, which is available for **continuous chains only**. Discrete chains cannot use mixed boundary conditions at all.
- Stress computation (and therefore `box_is_altering`) supports periodic, reflecting, and absorbing boundary conditions (pseudo-spectral methods, real fields).

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
| `chi_n` | dict | Flory-Huggins parameters × $`N_{ref}`$; pairs that are omitted do not interact (χN = 0). SCFT needs an invertible χN matrix (its field residual uses χ⁻¹): a single monomer type, or a zero pair in a three-monomer system, is rejected; L-FTS and CL-FTS accept zero pairs | `{"A,B": 20.0}` |
| `zeta_n` | float | (Optional) Compressibility parameter $`\zeta N`$ of the compressible model. Omitted: incompressible ($`\sum_i \phi_i = 1`$) in SCFT and L-FTS; CL-FTS needs a finite value and defaults to `100.0` | `100.0` |

### Polymer Definition

| Parameter | Type | Description |
|-----------|------|-------------|
| `distinct_polymers` | list[dict] | List of polymer species |

**Each polymer dictionary:**

| Key | Type | Description |
|-----|------|-------------|
| `volume_fraction` | float | Volume fraction in system |
| `blocks` | list | Block specifications |
| `initial_conditions` | dict | (Optional) Custom propagator initial conditions per node, e.g. `{0: "G"}` maps node 0 to the initial condition labeled `"G"` in `q_init` (used for grafted chains) |

**Block format:** Each block is a dictionary with:
- `"type"`: Monomer type (string)
- `"length"`: Contour length (float)
- `"v"`: Start vertex (optional for linear chains)
- `"u"`: End vertex (optional for linear chains)

```python
"distinct_polymers": [{
    "volume_fraction": 1.0,
    "blocks": [
        {"type": "A", "length": 0.5},  # A block, 50% of chain
        {"type": "B", "length": 0.5}   # B block, 50% of chain
    ]
}]
```

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
- `"cn-adi2"` is the only method that supports mixing periodic and non-periodic boundary conditions; the pseudo-spectral methods require uniform boundary conditions (see [Boundary Conditions](#boundary-conditions)).

See [NumericalMethods.md](../theory/NumericalMethods.md) for benchmarks.

---

## Platform

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `platform` | str | Computational backend | Auto-selected |

**Available platforms:**
- `"cuda"`: NVIDIA GPU (cuFFT)
- `"cpu-mkl"`: CPU (Intel MKL library)
- `"cpu-fftw"`: CPU (FFTW library, GPL license)

**Auto-selection logic** (same for `PropagatorSolver`, `SCFT`, `LFTS` and `CLFTS`):
- 1D simulations → the first CPU backend of the build
- 2D/3D simulations → `"cuda"` if available, else CPU

---

## SCFT Parameters

`SCFT(params)` reads the shared parameters above plus the keys below.

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
| `box_is_altering` | bool | Relax the box lengths (and, for monoclinic/triclinic/oblique systems, the angles) together with the fields | Required |
| `scale_stress` | float | Scaling of the stress entries in the optimizer vector | `1` |
| `stress_interval` | int or `"adaptive"` | Compute the stress every N iterations; `"adaptive"` uses 1, 2 or 4 as the error drops below 1e-2 and 1e-4 | `1` |
| `fts_consistent_energy` | bool | Pin the field means to the saddle point so that the SCFT free energy matches the field-theoretic Hamiltonian of L-FTS/CL-FTS (disordered phase: $`F = -\chi N (f - 1/2)^2`$ instead of 0) | `False` |

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
| `axis_ordering` | list[str] | Order in which `nx`, `lx` and `angles` are given, e.g. `["b", "a", "c"]`; the code converts to the standard `[a, b, c]` | `["a", "b", "c"]` |

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
| `max_step` | int | Maximum Langevin steps | Required |

### Saddle Point

| Parameter | Type | Description |
|-----------|------|-------------|
| `saddle` | dict | Saddle point iteration configuration |

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `max_iter` | int | Maximum iterations for saddle point | Required |
| `tolerance` | float | Convergence tolerance | Required |

### Recording

| Parameter | Type | Description |
|-----------|------|-------------|
| `recording` | dict | Data recording configuration |

| Key | Type | Description |
|-----|------|-------------|
| `dir` | str | Output directory name |
| `recording_period` | int | Period for saving fields |
| `sf_computing_period` | int | Period for computing structure function |
| `sf_recording_period` | int | Period for saving structure function |

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

For `"am"` and `"lram"`, the keys `max_hist`, `start_error`, `mix_min`, and `mix_init` are **required** (no built-in defaults; omitting one raises `KeyError`).

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

### Other L-FTS Options

| Parameter | Type | Description |
|-----------|------|-------------|
| `verbose_level` | int | Output verbosity: 0 (only the Langevin step number), 1 (plus one saddle-point summary line per Langevin step), 2 (one line per saddle-point iteration). Other values raise `ValidationError` |
| `smearing` | dict | (Optional) Finite-range interactions, see [Smearing](#smearing) |

`LFTS(params, random_seed=None)`: pass an integer `random_seed` for reproducible noise.

---

## CL-FTS Parameters

`CLFTS(params, random_seed=None)` (complex Langevin) reads the shared parameters, `langevin`, `recording` and `verbose_level` exactly as L-FTS does (no `compressor` or `saddle`: all fields fluctuate), plus:

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `zeta_n` | float | Compressibility parameter; complex Langevin needs a finite value | `100.0` |
| `dynamic_stabilization` | float | Dynamical stabilization constant αds (Willis & Matsen, *J. Chem. Phys.* **2024**, 161, 244903): adds $`i\,\alpha_{ds}\,\mathrm{Im}[W_-]`$ to the exchange-field force to suppress runaway solutions. `alpha_ds` is accepted as an alias. Typical: 0.01–0.1 for $`\bar{N} \ge 10^5`$ | `0.0` |
| `smearing` | dict | (Optional) Finite-range interactions, see [Smearing](#smearing); recommended for CL-FTS stability | none |

---

## Advanced Options

### Memory Management

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `reduce_memory` | bool | Enable checkpoint-based memory saving | `False` |
| `aggregate_propagator_computation` | bool | Let the propagator optimizer merge equivalent propagators of branched/mixed systems (turn off only when per-block concentrations are needed from the C++ API) | `True` |

When `True`:
- Stores only checkpoints every ~$`2\sqrt{N_{tot}}`$ contour steps, i.e. ~$`\sqrt{N_{tot}}/2`$ checkpoints per propagator (where $`N_{tot}`$ is the total number of contour steps)
- Reduces memory by ~90%
- Increases computation time by 2-4x

`reduce_memory=True` works with every chain model and with space-group symmetry, including **discrete chains combined with a space group**, on both CPU and CUDA (see [SpaceGroup.md](../theory/SpaceGroup.md)).

### Space Group Symmetry (Beta)

| Parameter | Type | Description |
|-----------|------|-------------|
| `space_group` | dict | Space group constraints |

```python
"space_group": {
    "symbol": "Im-3m",   # Hermann-Mauguin symbol
    "number": 529        # Hall number (optional)
}
```

The box must match the crystal system of the group: cubic groups need equal `lx` and 90° angles, tetragonal a = b with 90° angles, orthorhombic 90° angles, hexagonal and trigonal groups on hexagonal axes a = b with `"angles": [90, 90, 120]`, and rhombohedral-axes settings a = b = c with α = β = γ (box relaxation is not supported for them). `SCFT` raises `ValidationError` otherwise.

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
| `{"type": "sigmoidal", "k_int": k_c, "dk_int": dk}` | cutoff wavenumber $`k_c \gtrsim 20 / R_0`$, transition width $`dk`$ (default 5.0) | $`C_0 [1 - \tanh((k - k_c)/dk)]`$ |

Not available together with `space_group`. References: Delaney & Fredrickson, *J. Phys. Chem. B* **2016**, 120, 7615; Matsen et al., *J. Chem. Phys.* **2026**, 164, 014905.

### Box Optimization

`box_is_altering`, `scale_stress`, `stress_interval` and `crystal_system` are listed under [SCFT Parameters](#scft-parameters).

**Restrictions:** Stress computation supports periodic, reflecting, and absorbing boundary conditions for pseudo-spectral methods with real fields (complex fields with non-periodic boundaries are rejected), but is not implemented for the real-space `"cn-adi2"` method. Therefore `box_is_altering=True` requires a pseudo-spectral method. With non-periodic boundaries only box lengths can be optimized (angles stay at 90°); discrete-chain stress is exact against d(-lnQ)/dL, continuous-chain stress carries the contour-quadrature error (see docs/theory/StressTensor.md).

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

### SCFT Lamellar Phase

```python
params = {
    "nx": [1, 1, 64],
    "lx": [1.0, 1.0, 4.0],
    "chain_model": "continuous",
    "ds": 0.01,
    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 20.0},
    "distinct_polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": 0.5},
            {"type": "B", "length": 0.5}
        ]
    }],
    "numerical_method": "rqm4",
    "box_is_altering": False,
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
    "distinct_polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": 0.375},
            {"type": "B", "length": 0.625}
        ]
    }],
    "numerical_method": "rqm4",
    "box_is_altering": True,
    "scale_stress": 0.5,
    "space_group": {
        "symbol": "Ia-3d",
        "number": 530
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

All numerical methods support **uniform** non-periodic boundary conditions (all-reflecting or all-absorbing): RQM4 and RK2 use DCT/DST transforms, while CN-ADI2 uses ghost cells. **Mixing** periodic and non-periodic directions, as in the confined-film example below, is supported only by `"cn-adi2"` (continuous chains only) — the pseudo-spectral methods raise an error for mixed boundary conditions.

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
