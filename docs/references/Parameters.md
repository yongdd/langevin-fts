# Parameter Reference

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document provides a complete reference for all simulation parameters.

## Table of Contents

1. [Simulation Setup](#simulation-setup)
2. [Chain Definition](#chain-definition)
3. [Numerical Methods](#numerical-methods)
4. [Platform](#platform)
5. [Optimizer (SCFT)](#optimizer-scft)
6. [Advanced Options](#advanced-options)
7. [Example Configurations](#example-configurations)

---

## Simulation Setup

### Grid Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `nx` | list[int] | Grid points per dimension | `[32, 32, 32]` |
| `lx` | list[float] | Box lengths (in units of $b N_{ref}^{1/2}$) | `[4.0, 4.0, 4.0]` |

### Boundary Conditions

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `bc` | list[str] | Boundary conditions (6 values for 3D) | `["periodic"]*6` |

**Boundary condition types:**
- `"periodic"`: Cyclic boundary (uses FFT)
- `"reflecting"`: Zero flux / Neumann (uses DCT)
- `"absorbing"`: Zero value / Dirichlet (uses DST)

Note: FFT/DCT/DST apply to pseudo-spectral methods (RQM4, RK2). CN-ADI2 implements these boundary conditions using ghost cells.

**Example (reflecting in z, periodic in x/y):**
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
| `ds` | float | Contour step size (typically $1/N_{ref}$) | `0.01` |

### Monomer Properties

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `segment_lengths` | dict | Relative statistical segment lengths $(b/b_{ref})^2$ | `{"A": 1.0, "B": 1.0}` |
| `chi_n` | dict | Flory-Huggins parameters × $N_{ref}$ | `{"A,B": 20.0}` |

### Polymer Definition

| Parameter | Type | Description |
|-----------|------|-------------|
| `distinct_polymers` | list[dict] | List of polymer species |

**Each polymer dictionary:**

| Key | Type | Description |
|-----|------|-------------|
| `volume_fraction` | float | Volume fraction in system |
| `blocks` | list | Block specifications |
| `grafting` | dict | (Optional) Grafting conditions |

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
| `"cn-adi2"` | 2nd | Real-space | Crank-Nicolson ADI |

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

**Auto-selection logic:**
- 1D simulations → CPU
- 2D/3D simulations → `"cuda"` if available, else CPU

---

## Optimizer (SCFT)

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `optimizer` | dict | Optimizer configuration | See below |

### Anderson Mixing (`"name": "am"`)

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `name` | str | Optimizer name | `"am"` |
| `max_hist` | int | Maximum history length | `20` |
| `start_error` | float | Error threshold to start Anderson mixing | `1e-1` |
| `mix_min` | float | Minimum mixing parameter | `0.1` |
| `mix_init` | float | Initial mixing parameter | `0.1` |

**Example:**
```python
"optimizer": {
    "name": "am",
    "max_hist": 20,
    "start_error": 1e-1,
    "mix_min": 0.1,
    "mix_init": 0.1
}
```

### ADAM Optimizer (`"name": "adam"`)

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `name` | str | `"adam"` | |
| `lr` | float | Learning rate | `0.01` |
| `b1` | float | Momentum parameter (β₁) | `0.9` |
| `b2` | float | Second moment parameter (β₂) | `0.999` |
| `gamma` | float | Learning rate decay factor | `1.0` |

### SCFT Advanced Options

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `stress_interval` | int or str | Stress computation interval. Integer for fixed interval, `"adaptive"` for automatic adjustment | `3` |

---

## L-FTS Parameters

### Langevin Dynamics

| Parameter | Type | Description |
|-----------|------|-------------|
| `langevin` | dict | Langevin dynamics configuration |

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `dt` | float | Langevin time step $\Delta\tau \cdot N_{ref}$ | Required |
| `nbar` | float | Invariant polymerization index $\bar{n}$ | Required |
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

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `name` | str | Compressor type: `"am"`, `"lr"`, or `"lram"` | Required |
| `max_hist` | int | Anderson mixing history length | `20` |
| `start_error` | float | Error threshold to start AM | `5e-1` |
| `mix_min` | float | Minimum mixing parameter | `0.01` |
| `mix_init` | float | Initial mixing parameter | `0.01` |

### Well-Tempered Metadynamics (Optional)

| Parameter | Type | Description |
|-----------|------|-------------|
| `wtmd` | dict | WTMD configuration (optional) |

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `ell` | int | $\ell$-norm for order parameter | `4` |
| `kc` | float | Cutoff wavenumber | `6.02` |
| `delta_t` | float | Well-tempering factor $\Delta T/T$ | `5.0` |
| `sigma_psi` | float | Gaussian width $\sigma_\Psi$ | `0.16` |
| `psi_min` | float | Minimum $\Psi$ | `0.0` |
| `psi_max` | float | Maximum $\Psi$ | `10.0` |
| `dpsi` | float | Bin width $d\Psi$ | `1e-3` |
| `update_freq` | int | Statistics update frequency | `1000` |
| `recording_period` | int | Data recording frequency | `100000` |

### Other L-FTS Options

| Parameter | Type | Description |
|-----------|------|-------------|
| `verbose_level` | int | Output verbosity (1: per step, 2: per iteration) |

---

## Advanced Options

### Memory Management

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `reduce_memory` | bool | Enable checkpoint-based memory saving | `False` |

When `True`:
- Stores only checkpoints (~$2\sqrt{N_{tot}}$ per propagator, where $N_{tot}$ is the total number of contour steps)
- Reduces memory by ~90%
- Increases computation time by 3-4x

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

See [SpaceGroup.md](../theory/SpaceGroup.md) for available space groups.

### Box Optimization

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `box_is_altering` | bool | Enable box size optimization | `False` |
| `scale_stress` | float | Stress scaling factor for box updates | `1.0` |

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
    }
}
```

### Non-Periodic Boundaries (Confined Film)

All numerical methods support non-periodic boundary conditions: RQM4 and RK2 use DCT/DST transforms, while CN-ADI2 uses ghost cells.

```python
params = {
    "nx": [1, 1, 64],
    "lx": [1.0, 1.0, 4.0],
    "bc": ["periodic", "periodic",
           "periodic", "periodic",
           "reflecting", "reflecting"],  # Confined in z
    "chain_model": "continuous",
    "ds": 0.01,
    "numerical_method": "cn-adi2",  # or "rqm4", "rk2"
    # ... other parameters
}
```

---

## Units and Conventions

The unit of length is $b N_{ref}^{1/2}$ for both chain models, where:
- $b$ is a reference statistical segment length
- $N_{ref}$ is a reference polymerization index

Fields are defined as **per reference chain** potential. To obtain **per reference segment** potential, multiply each field by `ds`.

This convention follows [*Macromolecules* **2013**, 46, 8037].
