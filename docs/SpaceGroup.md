# Space Group Symmetry

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the `SpaceGroup` class, which applies crystallographic symmetry constraints to reduce computational cost in polymer field theory simulations.

**Note**: This is a **beta feature**. Validate results against full simulations without symmetry constraints.

## Overview

Many polymer phases (BCC, gyroid, HCP, etc.) have crystallographic symmetry. By exploiting this symmetry, fields can be represented using only **irreducible mesh points** rather than the full simulation grid.

**Benefits:**
- Reduced memory: Store only irreducible points (up to ~192x reduction for FCC, ~96x for BCC/Gyroid)
- Faster convergence: Field updates operate on smaller arrays
- Enforced symmetry: Results guaranteed to have correct space group symmetry

## Common Space Groups for Polymer Phases

| Phase | Symbol | No. | Hall | Ops | Crystal System | Grid Constraint |
|-------|--------|-----|------|-----|----------------|-----------------|
| BCC | Im-3m | 229 | 529 | 96 | Cubic | nx = ny = nz |
| FCC | Fm-3m | 225 | 523 | 192 | Cubic | nx = ny = nz |
| A15 | Pm-3n | 223 | 520 | 48 | Cubic | nx = ny = nz |
| Gyroid | Ia-3d | 230 | 530 | 96 | Cubic | nx = ny = nz |
| Diamond | Fd-3m | 227 | 525 | 192 | Cubic | nx = ny = nz |
| HCP | P6_3/mmc | 194 | 488 | 24 | Hexagonal | nx = ny |
| PL | P6_3/mmc | 194 | 488 | 24 | Hexagonal | nx = ny |
| C14 | P6_3/mmc | 194 | 488 | 24 | Hexagonal | nx = ny |
| Sigma | P4_2/mnm | 136 | 419 | 16 | Tetragonal | nx = ny |

## Mathematical Foundation

### Symmetry Operations

A space group is a set of symmetry operations that leave the crystal structure invariant. Each symmetry operation is a pair $(\mathbf{R}, \mathbf{t})$ where:
- $\mathbf{R}$: 3×3 rotation/reflection matrix (orthogonal, $\mathbf{R}^T \mathbf{R} = \mathbf{I}$)
- $\mathbf{t}$: translation vector in fractional coordinates

Applied to a point $\mathbf{p}$ in fractional coordinates:

$$\mathbf{p}' = \mathbf{R} \cdot \mathbf{p} + \mathbf{t}$$

where each component is wrapped to the range $[0, 1)$ using periodic boundary conditions.

**Example: Im-3m (BCC) symmetry operations**

The BCC space group has 96 symmetry operations (48 point group operations × 2 for body centering). Some examples:

| Operation | $\mathbf{R}$ | $\mathbf{t}$ | Description |
|-----------|--------------|--------------|-------------|
| Identity | $\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$ | $(0, 0, 0)$ | No change |
| Inversion | $\begin{pmatrix} -1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{pmatrix}$ | $(0, 0, 0)$ | Point reflection |
| Body-center | $\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$ | $(\frac{1}{2}, \frac{1}{2}, \frac{1}{2})$ | Translation to body center |
| 4-fold rotation | $\begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}$ | $(0, 0, 0)$ | 90° rotation about z-axis |

### Orbit of a Point

The **orbit** of a grid point $\mathbf{p}$ is the set of all symmetrically equivalent points:

$$\text{Orbit}(\mathbf{p}) = \{ \mathbf{R}_i \cdot \mathbf{p} + \mathbf{t}_i \mid i = 1, \ldots, n_{\text{ops}} \}$$

(each component wrapped to $[0, 1)$ via periodic boundary conditions)

**Example: Orbit calculation for BCC on 4×4×4 grid**

Consider the point $\mathbf{p} = (1, 0, 0)$ in grid indices, which is $\mathbf{p}_{\text{frac}} = (0.25, 0, 0)$ in fractional coordinates.

Applying some symmetry operations:
- Identity: $(0.25, 0, 0)$ → grid point $(1, 0, 0)$
- Body-center translation: $(0.25 + 0.5, 0 + 0.5, 0 + 0.5) = (0.75, 0.5, 0.5)$ → grid point $(3, 2, 2)$
- 4-fold rotation about z: $(0, 0.25, 0)$ → grid point $(0, 1, 0)$
- Inversion: $(-0.25, 0, 0)$ → wrapped to $(0.75, 0, 0)$ → grid point $(3, 0, 0)$

All these points belong to the same orbit and share one irreducible mesh index.

### Irreducible Mesh

The irreducible mesh is the minimal set of grid points from which the full field can be reconstructed using symmetry operations.

**Algorithm:**
1. Mark all grid points as unvisited
2. For the first unvisited point $\mathbf{p}$:
   - Add $\mathbf{p}$ to irreducible mesh with index $k$
   - Compute $\text{Orbit}(\mathbf{p})$
   - Mark all orbit points with index $k$
3. Repeat until all points are visited

**Example: Gyroid (Ia-3d) on 64×64×64 grid**
```
Full mesh:        64 × 64 × 64 = 262,144 points
Symmetry ops:     96 (48 point group × 2 for body centering)
Irreducible mesh: 2,761 points
Reduction factor: 262,144 / 2,761 ≈ 95x
```

The reduction factor approaches the number of symmetry operations when most orbits have maximum size. Points on symmetry elements (axes, planes, centers) have smaller orbits.

### Field Transformations

#### To Reduced Basis

Extract field values at irreducible mesh points:

$$w_{\text{reduced}}[k] = w_{\text{full}}[\mathbf{p}_k]$$

where $\mathbf{p}_k$ is the $k$-th irreducible mesh point.

#### From Reduced Basis

Reconstruct full field by copying values to all orbit members:

$$w_{\text{full}}[\mathbf{p}] = w_{\text{reduced}}[\text{index}(\mathbf{p})]$$

where $\text{index}(\mathbf{p})$ maps grid point $\mathbf{p}$ to its irreducible mesh index.

#### Symmetrization

Average field values over each orbit to enforce perfect symmetry:

$$w_{\text{sym}}[\mathbf{p}] = \frac{1}{|\text{Orbit}(\mathbf{p})|} \sum_{\mathbf{q} \in \text{Orbit}(\mathbf{p})} w[\mathbf{q}]$$

This is useful for initializing SCFT iterations from non-symmetric initial guesses.

## Usage

### In SCFT Parameters

```python
params = {
    "nx": [64, 64, 64],
    "lx": [4.0, 4.0, 4.0],
    # ... other parameters ...

    "space_group": {
        "symbol": "Ia-3d",      # ITA symbol
        "number": 530           # Hall number (optional if unique)
    }
}

scft = SCFT(params)
scft.run()
```

### Direct Usage

```python
from polymerfts.space_group import SpaceGroup
import numpy as np

# Create space group object
nx = [64, 64, 64]
sg = SpaceGroup(nx, "Ia-3d", hall_number=530)

# Output:
# Using Hall number: 530 for symbol 'Ia-3d'
# International space group number: 230
# Crystal system: Cubic
# The number of symmetry operations: 96
# Original mesh size: 262144
# Irreducible mesh size: 2761

# Convert full field to reduced basis
w_full = np.random.randn(2, 64*64*64)  # 2 fields
w_reduced = sg.to_reduced_basis(w_full)
print(w_reduced.shape)  # (2, 2761)

# Reconstruct full field from reduced
w_reconstructed = sg.from_reduced_basis(w_reduced)
print(w_reconstructed.shape)  # (2, 262144)

# Symmetrize a field (average over orbits)
w_symmetrized = sg.symmetrize(w_full)
```

## Crystal Systems and Grid Constraints

| Crystal System | Lattice Parameters | Grid Constraint |
|----------------|-------------------|-----------------|
| Cubic | a | nx = ny = nz |
| Tetragonal | a, c | nx = ny |
| Hexagonal | a, c | nx = ny |
| Trigonal | a, c | nx = ny |
| Orthorhombic | a, b, c | none |
| Monoclinic | a, b, c, β | none |
| Triclinic | a, b, c, α, β, γ | none |

### Grid Divisibility Requirements

Some space groups require grid dimensions to be divisible by specific numbers:

| Space Group | Required Divisor | Reason |
|-------------|-----------------|--------|
| Ia-3d (Gyroid) | 4 | 1/4, 3/4 translations |
| Fd-3m (Diamond) | 4 | 1/4, 3/4 translations |
| Im-3m (BCC) | 2 | 1/2 translations |
| P6_3/mmc (HCP) | 6 | 1/3, 2/3, 1/2 positions |

## Class Methods

| Method | Description |
|--------|-------------|
| `to_reduced_basis(fields)` | Convert full fields to irreducible representation |
| `from_reduced_basis(reduced)` | Reconstruct full fields from reduced |
| `symmetrize(fields)` | Average fields over orbits for perfect symmetry |

## Attributes

| Attribute | Description |
|-----------|-------------|
| `hall_number` | Hall number (1-530) |
| `spacegroup_number` | International space group number (1-230) |
| `spacegroup_symbol` | ITA short symbol |
| `crystal_system` | Crystal system name |
| `lattice_parameters` | Free lattice parameters for this system |
| `symmetry_operations` | List of (rotation, translation) pairs |
| `irreducible_mesh` | List of irreducible point coordinates |
| `indices` | Map from full grid to irreducible point index |

## Hexagonal Systems

Hexagonal crystal systems (P6_3/mmc, etc.) require special attention:

- Lattice: a = b, c independent
- Angles: α = β = 90°, γ = 120°
- Grid: nx = ny (first two dimensions must be equal)
- All dimensions must be divisible by 6

**Example:**
```python
nx = [48, 48, 96]  # a = b, c different, all divisible by 6
sg = SpaceGroup(nx, "P6_3/mmc", hall_number=488)
```

## Worked Example: BCC on 4×4×4 Grid

This example demonstrates the irreducible mesh calculation for Im-3m (BCC) on a small grid.

### Setup

- Grid: 4×4×4 = 64 points
- Space group: Im-3m (Hall 529), 96 symmetry operations
- Fractional coordinate of grid point $(i, j, k)$: $(i/4, j/4, k/4)$

### Orbit Calculation

**Point $(0, 0, 0)$** - Corner (high symmetry)
- Fractional: $(0, 0, 0)$
- After all 96 operations: only maps to itself and body center
- Orbit size: 2

**Point $(0, 0, 1)$** - Edge
- Fractional: $(0, 0, 0.25)$
- Maps to 6 points via point group, doubled by body centering
- Orbit size: 12

**Point $(0, 1, 1)$** - Face
- Fractional: $(0, 0.25, 0.25)$
- Orbit size: 24

**Point $(1, 1, 1)$** - Body diagonal
- Fractional: $(0.25, 0.25, 0.25)$
- Orbit size: 8

### Irreducible Mesh

| Index | Representative | Orbit Size | Description |
|-------|---------------|------------|-------------|
| 0 | $(0,0,0)$ | 2 | Corner + body center |
| 1 | $(0,0,1)$ | 12 | Edge |
| 2 | $(0,0,2)$ | 6 | Face center |
| 3 | $(0,1,1)$ | 24 | Face |
| 4 | $(0,1,2)$ | 12 | General |
| 5 | $(1,1,1)$ | 8 | Body diagonal |

**Verification:**
- Total points: $2 + 12 + 6 + 24 + 12 + 8 = 64$ ✓
- Irreducible mesh size: 6 points
- Reduction factor: $64 / 6 \approx 10.7 \times$

Note: The reduction factor (10.7x) is less than the number of symmetry operations (96) because high-symmetry points have small orbits.

## Limitations

- Only **periodic boundary conditions** supported
- Grid must be compatible with space group symmetry
- Beta feature - validate results carefully
- Requires `spglib` library

## References

- `space_group.py`: SpaceGroup implementation
- `scft.py`: Integration with SCFT simulations
- International Tables for Crystallography, Vol. A (2016)
- spglib documentation: https://spglib.github.io/spglib/
