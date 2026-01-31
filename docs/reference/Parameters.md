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
| `lx` | list[float] | Box lengths (in units of $bN^{1/2}$) | `[4.0, 4.0, 4.0]` |

### Boundary Conditions

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `bc` | list[str] | Boundary conditions (6 values for 3D) | `["periodic"]*6` |

**Boundary condition types:**
- `"periodic"`: Cyclic boundary (uses FFT)
- `"reflecting"`: Zero flux / Neumann (uses DCT)
- `"absorbing"`: Zero value / Dirichlet (uses DST)

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
| `ds` | float | Contour step size (typically 1/N) | `0.01` |

### Monomer Properties

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `segment_lengths` | dict | Relative statistical segment lengths $(a/a_{ref})^2$ | `{"A": 1.0, "B": 1.0}` |
| `chi_n` | dict | Flory-Huggins parameters × N | `{"A,B": 20.0}` |

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
| `alpha` | float | Learning rate | `0.01` |
| `beta1` | float | Momentum parameter | `0.9` |
| `beta2` | float | Second moment parameter | `0.999` |

---

## Advanced Options

### Memory Management

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `reduce_memory` | bool | Enable checkpoint-based memory saving | `False` |

When `True`:
- Stores only checkpoints (~$2\sqrt{N}$ per propagator)
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

For non-orthogonal crystal systems:

```python
"lx_angles": [90.0, 90.0, 120.0]  # alpha, beta, gamma in degrees
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

### L-FTS with Memory Saving

```python
params = {
    "nx": [64, 64, 64],
    "lx": [4.0, 4.0, 4.0],
    "chain_model": "discrete",
    "ds": 0.002,  # 1/500
    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 12.0},
    "distinct_polymers": [{
        "volume_fraction": 1.0,
        "blocks": [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]]
    }],
    "numerical_method": "rk2",
    "platform": "cuda",
    "reduce_memory": True
}
```

### Non-Periodic Boundaries (Confined Film)

```python
params = {
    "nx": [1, 1, 64],
    "lx": [1.0, 1.0, 4.0],
    "bc": ["periodic", "periodic",
           "periodic", "periodic",
           "reflecting", "reflecting"],  # Confined in z
    "chain_model": "continuous",
    "ds": 0.01,
    "numerical_method": "cn-adi2",  # Real-space method
    # ... other parameters
}
```

---

## Units and Conventions

The unit of length is $bN^{1/2}$ for both chain models, where:
- $b$ is a reference statistical segment length
- $N$ is a reference polymerization index

Fields are defined as **per reference chain** potential. To obtain **per reference segment** potential, multiply each field by `ds`.

This convention follows [*Macromolecules* **2013**, 46, 8037].
