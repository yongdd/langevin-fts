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

When CrysFFT is available (periodic 3D simulations meeting the grid and cell
requirements listed in [Physical Bases for CrysFFT](#physical-bases-for-crysfft)),
the solver automatically switches to a **physical grid** (1/8 or 1/2 size) to
avoid gather/scatter mapping during diffusion steps. See
[FFTImplementation.md](../internals/FFTImplementation.md).

## Common Space Groups for Polymer Phases

| Phase | Symbol | No. | Hall | Ops | Crystal System | Grid Constraint |
|-------|--------|-----|------|-----|----------------|-----------------|
| A15 | Pm-3n | 223 | 520 | 48 | Cubic | nx = ny = nz |
| BCC | Im-3m | 229 | 529 | 96 | Cubic | nx = ny = nz |
| FCC | Fm-3m | 225 | 523 | 192 | Cubic | nx = ny = nz |
| SC | Pm-3m | 221 | 517 | 48 | Cubic | nx = ny = nz |
| SD (Diamond) | Fd-3m | 227 | 526 | 192 | Cubic | nx = ny = nz |
| DG (Gyroid) | Ia-3d | 230 | 530 | 96 | Cubic | nx = ny = nz |
| DP | Im-3m | 229 | 529 | 96 | Cubic | nx = ny = nz |
| DD | Pn-3m | 224 | 522 | 48 | Cubic | nx = ny = nz |
| SG | I4_132 | 214 | 510 | 48 | Cubic | nx = ny = nz |
| SP | Pm-3m | 221 | 517 | 48 | Cubic | nx = ny = nz |
| Sigma | P4_2/mnm | 136 | 419 | 16 | Tetragonal | nx = ny |
| HCP | P6_3/mmc | 194 | 488 | 24 | Hexagonal | nx = ny |
| PL | P6/mmm | 191 | 485 | 24 | Hexagonal | nx = ny |
| Fddd | Fddd | 70 | 336 | 32 | Orthorhombic | none |

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

**Cell-centered grid convention**: The implementation uses a cell-centered grid, so grid point $(i, j, k)$ on an $N_x \times N_y \times N_z$ mesh corresponds to the fractional coordinate

$$\mathbf{p}_{\text{frac}} = \left( \frac{i + 0.5}{N_x}, \frac{j + 0.5}{N_y}, \frac{k + 0.5}{N_z} \right)$$

**Example: Orbit calculation for BCC on 4×4×4 grid**

Consider the point $\mathbf{p} = (1, 0, 0)$ in grid indices, which is $\mathbf{p}_{\text{frac}} = (0.375, 0.125, 0.125)$ in cell-centered fractional coordinates.

Applying some symmetry operations:
- Identity: $(0.375, 0.125, 0.125)$ → grid point $(1, 0, 0)$
- Body-center translation: $(0.375 + 0.5, 0.125 + 0.5, 0.125 + 0.5) = (0.875, 0.625, 0.625)$ → grid point $(3, 2, 2)$
- 4-fold rotation about z: $(-0.125, 0.375, 0.125)$ → wrapped to $(0.875, 0.375, 0.125)$ → grid point $(3, 1, 0)$
- Inversion: $(-0.375, -0.125, -0.125)$ → wrapped to $(0.625, 0.875, 0.875)$ → grid point $(2, 3, 3)$

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
Irreducible mesh: 2,736 points
Reduction factor: 262,144 / 2,736 ≈ 96x
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
    },
}

scft = SCFT(params)
scft.run()
```

### Direct Usage

```python
from polymerfts import SpaceGroup
import numpy as np

# Create space group object
nx = [64, 64, 64]
sg = SpaceGroup(nx, "Ia-3d", hall_number=530)

# Output:
# ---------- Space Group ----------
# Hall number: 530
# International space group number: 230
# Symbol: Ia-3d
# Crystal system: Cubic
# Number of symmetry operations: 96
# Original mesh size: 262144
# Reduced basis size (irreducible): 2736
# Pmmm physical basis size (if enabled): n/a
# M3 physical basis size (if enabled): 32768

# Convert full field to reduced basis
w_full = np.random.randn(2, 64*64*64)  # 2 fields
w_reduced = sg.to_reduced_basis(w_full)
print(w_reduced.shape)  # (2, 2736)

# Reconstruct full field from reduced
w_reconstructed = sg.from_reduced_basis(w_reduced)
print(w_reconstructed.shape)  # (2, 262144)

# Symmetrize a field (average over orbits)
w_symmetrized = sg.symmetrize(w_full)
```

### Physical Bases for CrysFFT

When CrysFFT is available (periodic 3D only), the solver may switch the reduced
basis from the irreducible mesh to a **physical basis** to avoid gather/scatter
mapping during diffusion. This selection is automatic:

- **Orthogonal boxes** (α = β = γ = 90°, even grid): the 3m physical basis
  (1/8 grid) is preferred when the space group provides the required 3m
  translations and $(n_z / 2)$ is divisible by 8; otherwise the solver falls
  back to the Pmmm physical basis (1/8 grid, requires mirror planes along
  x, y, z).
- **Non-orthogonal cells with an orthogonal z-axis** (α = β = 90°, γ
  arbitrary — e.g. hexagonal or monoclinic-γ cells): the z-mirror (ObliqueZ)
  physical basis (1/2 grid) is used when the space group has a z-mirror
  operation and $n_z$ is even.
- If no physical basis applies, the irreducible basis is used.

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
| P6_3/mmc (HCP) | 2 | 1/2 translations |

The required divisor is computed from the denominators of the translation
components of the symmetry operations (e.g. a 1/4 translation requires the
grid to be divisible by 4).

## Class Methods

| Method | Description |
|--------|-------------|
| `to_reduced_basis(fields)` | Convert full fields to irreducible representation |
| `from_reduced_basis(reduced)` | Reconstruct full fields from reduced |
| `symmetrize(fields)` | Average fields over orbits for perfect symmetry |

## Accessor Methods

Metadata is exposed through getter methods (not attributes):

| Method | Description |
|--------|-------------|
| `get_hall_number()` | Hall number (1-530) |
| `get_spacegroup_number()` | International space group number (1-230) |
| `get_spacegroup_symbol()` | ITA short symbol |
| `get_crystal_system()` | Crystal system name |
| `get_n_symmetry_ops()` | Number of symmetry operations |
| `get_n_reduced_basis()` | Number of irreducible mesh points |
| `get_reduced_basis_indices()` | Flat grid indices of the irreducible points |
| `get_full_to_reduced_map()` | Map from full grid to irreducible point index |
| `get_orbit_counts()` | Orbit size for each irreducible point |
| `get_nx()` | Grid dimensions |
| `get_total_grid()` | Total number of grid points |

## Hexagonal Systems

Hexagonal crystal systems (P6_3/mmc, etc.) require special attention:

- Lattice: a = b, c independent
- Angles: α = β = 90°, γ = 120°
- Grid: nx = ny (first two dimensions must be equal)
- All dimensions must be divisible by 2 (for P6_3/mmc, from its 1/2 translations)

**Example:**
```python
nx = [48, 48, 96]  # a = b, c different, all divisible by 2
sg = SpaceGroup(nx, "P6_3/mmc", hall_number=488)
```

## Worked Example: BCC on 4×4×4 Grid

This example demonstrates the irreducible mesh calculation for Im-3m (BCC) on a small grid.

### Setup

- Grid: 4×4×4 = 64 points
- Space group: Im-3m (Hall 529), 96 symmetry operations
- Fractional coordinate of grid point $(i, j, k)$ (cell-centered): $\left( \frac{i+0.5}{4}, \frac{j+0.5}{4}, \frac{k+0.5}{4} \right)$

### Orbit Calculation

Because the grid is cell-centered, no grid point sits exactly on a
high-symmetry position such as the corner $(0,0,0)$ or the body center, so
orbits are larger than on a node-centered grid.

**Point $(0, 0, 0)$** - Body diagonal
- Fractional: $(0.125, 0.125, 0.125)$
- All three coordinates equal: 8 sign combinations $(\pm 0.125, \pm 0.125, \pm 0.125)$, doubled by body centering
- Orbit size: 16

**Point $(0, 0, 1)$** - General position on a mirror plane
- Fractional: $(0.125, 0.125, 0.375)$
- Two coordinates equal: 3 placements of the distinct coordinate × 8 sign combinations, doubled by body centering
- Orbit size: 48

### Irreducible Mesh

| Index | Representative | Orbit Size | Description |
|-------|---------------|------------|-------------|
| 0 | $(0,0,0)$ | 16 | Body diagonal $(0.125, 0.125, 0.125)$ |
| 1 | $(0,0,1)$ | 48 | $(0.125, 0.125, 0.375)$ |

**Verification:**
- Total points: $16 + 48 = 64$ ✓
- Irreducible mesh size: 2 points
- Reduction factor: $64 / 2 = 32 \times$

Note: The reduction factor (32x) is less than the number of symmetry operations (96) because points on the body diagonal have smaller orbits. On larger grids, most points sit in general positions and the reduction factor approaches 96 (e.g. ≈96x for Ia-3d on a 64³ grid).

## Limitations

- Only **periodic boundary conditions** supported
- Grid must be compatible with space group symmetry
- Beta feature - validate results carefully
- Requires `spglib` library
- **Initial-field symmetry and stress**: input fields are projected onto the
  *enabled* symmetry basis, which may be a physical-basis **subgroup** (m3,
  Pmmm mirrors, or z-mirror) of the full space group. A field that is
  subgroup-symmetric but not fully symmetric still yields correct partition
  functions and concentrations, but the space-group **stress** path assumes
  full-group symmetry of the propagators, so early box-relaxation steps can
  be distorted. In practice this is self-correcting: the SCFT iteration
  converges to the fully symmetric saddle point, and the stress becomes
  exact as the fields converge. Analytic initial guesses built with the
  corner convention `round(x*nx)` are index-shifted relative to the
  cell-centered `(i+0.5)/N` orbit map and are therefore never exactly
  symmetric — this is harmless for the converged answer, but supply fully
  symmetric fields if the early box-size trajectory matters.
- **Discrete chains**: space group symmetry is supported on both **CPU and CUDA**
  (standard mode)
- **`reduce_memory=True` + discrete chains + space group** is not supported on any
  platform (throws on both CPU and CUDA)
- **Hexagonal/trigonal artifact**: Cell-centered grids are mathematically incompatible with hexagonal rotation matrices. Rotations with even row sums (e.g., $[1, -1, 0]$) map cell-centered positions $(i+0.5)/N$ to cell boundaries $(integer)/N$, causing inconsistent orbit assignments and X-shaped density artifacts. Cubic/orthorhombic space groups are unaffected (rotation matrix row sums are always odd). The `star` git branch implements a Fourier star basis that eliminates this artifact by working in reciprocal space where wavevector rotations are exact integer operations.

## References

- `src/common/SpaceGroup.h` / `src/common/SpaceGroup.cpp`: SpaceGroup implementation
- `src/pybind11/polymerfts_core.cpp`: Python bindings
- `src/python/scft.py`: Integration with SCFT simulations
- International Tables for Crystallography, Vol. A (2016)
- spglib documentation: https://spglib.github.io/spglib/
