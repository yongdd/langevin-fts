# Parameter Reference

> **Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

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

**Block format:** `[monomer_type, contour_length, start_vertex, end_vertex]`

```python
"distinct_polymers": [{
    "volume_fraction": 1.0,
    "blocks": [
        ["A", 0.5, 0, 1],  # A block, 50% of chain, vertex 0→1
        ["B", 0.5, 1, 2]   # B block, 50% of chain, vertex 1→2
    ]
}]
```

### Polymer Architectures

**Linear AB diblock:**
```python
"blocks": [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]]
#   0 --A-- 1 --B-- 2
```

**Star polymer (3-arm):**
```python
"blocks": [
    ["A", 0.33, 0, 1],
    ["B", 0.33, 0, 2],
    ["C", 0.34, 0, 3]
]
#        1
#        |A
#   2-B--0--C-3
```

**ABC triblock:**
```python
"blocks": [
    ["A", 0.33, 0, 1],
    ["B", 0.34, 1, 2],
    ["C", 0.33, 2, 3]
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
- `"cpu-fftw"`: CPU (FFTW library)

**Auto-selection logic:**
- 1D simulations → `"cpu-fftw"`
- 2D/3D simulations → `"cuda"` if available, else `"cpu-fftw"`

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
| `start_error` | float | Error threshold to start Anderson mixing | `1e-2` |
| `mix_min` | float | Minimum mixing parameter | `0.1` |
| `mix_init` | float | Initial mixing parameter | `0.1` |

**Example:**
```python
"optimizer": {
    "name": "am",
    "max_hist": 20,
    "start_error": 1e-2,
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
        "blocks": [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]]
    }],
    "numerical_method": "rqm4",
    "optimizer": {
        "name": "am",
        "max_hist": 20,
        "start_error": 1e-2,
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
        "blocks": [["A", 0.375, 0, 1], ["B", 0.625, 1, 2]]
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
