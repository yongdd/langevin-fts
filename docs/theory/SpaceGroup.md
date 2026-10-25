# Space Group Symmetry

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
| PL | P6_3/mmc | 194 | 488 | 24 | Hexagonal | nx = ny |
| Fddd | Fddd | 70 | 336 | 32 | Orthorhombic | none |

## Mathematical Foundation

### Symmetry Operations

A space group is a set of symmetry operations that leave the crystal structure invariant. Each symmetry operation is a pair $`(\mathbf{R}, \mathbf{t})`$ where:
- $`\mathbf{R}`$: 3×3 integer rotation/reflection matrix acting on fractional coordinates; it preserves the cell metric, $`\mathbf{R}^T \mathbf{g} \mathbf{R} = \mathbf{g}`$ ($`\mathbf{g}`$: metric tensor of the cell), and is orthogonal only on orthogonal cells (the hexagonal $`\mathbf{R}_6`$ below is not)
- $`\mathbf{t}`$: translation vector in fractional coordinates

Applied to a point $`\mathbf{p}`$ in fractional coordinates:

```math
\mathbf{p}' = \mathbf{R} \cdot \mathbf{p} + \mathbf{t}
```

where each component is wrapped to the range $`[0, 1)`$ using periodic boundary conditions.

**Example: Im-3m (BCC) symmetry operations**

The BCC space group has 96 symmetry operations (48 point group operations × 2 for body centering). Some examples:

| Operation | $`\mathbf{R}`$ | $`\mathbf{t}`$ | Description |
|-----------|--------------|--------------|-------------|
| Identity | $`\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}`$ | $`(0, 0, 0)`$ | No change |
| Inversion | $`\begin{pmatrix} -1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{pmatrix}`$ | $`(0, 0, 0)`$ | Point reflection |
| Body-center | $`\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}`$ | $`(\frac{1}{2}, \frac{1}{2}, \frac{1}{2})`$ | Translation to body center |
| 4-fold rotation | $`\begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}`$ | $`(0, 0, 0)`$ | 90° rotation about z-axis |

### Orbit of a Point

The **orbit** of a grid point $`\mathbf{p}`$ is the set of all symmetrically equivalent points:

```math
\text{Orbit}(\mathbf{p}) = \{ \mathbf{R}_i \cdot \mathbf{p} + \mathbf{t}_i \mid i = 1, \ldots, n_{\text{ops}} \}
```

(each component wrapped to $`[0, 1)`$ via periodic boundary conditions)

**Cell-centered grid convention**: For cubic, tetragonal, orthorhombic,
monoclinic, and triclinic settings the implementation uses a cell-centered
grid, so grid point $`(i, j, k)`$ on an $`N_x \times N_y \times N_z`$ mesh
corresponds to the fractional coordinate

```math
\mathbf{p}_{\text{frac}} = \left( \frac{i + 0.5}{N_x}, \frac{j + 0.5}{N_y}, \frac{k + 0.5}{N_z} \right)
```

Hexagonal-axes settings of hexagonal/trigonal groups instead use a **hybrid
convention** — node-centered in-plane coordinates
$`\left(\frac{i}{N_x}, \frac{j}{N_y}\right)`$ with cell-centered $`z`$ —
because in-plane hexagonal
rotations are exact on the grid only in node-centered coordinates (see
[How Hexagonal/Trigonal Groups Are
Handled](#how-hexagonaltrigonal-groups-are-handled)).

**Example: Orbit calculation for BCC on 4×4×4 grid**

Consider the point $`\mathbf{p} = (1, 0, 0)`$ in grid indices, which is $`\mathbf{p}_{\text{frac}} = (0.375, 0.125, 0.125)`$ in cell-centered fractional coordinates.

Applying some symmetry operations:
- Identity: $`(0.375, 0.125, 0.125)`$ → grid point $`(1, 0, 0)`$
- Body-center translation: $`(0.375 + 0.5, 0.125 + 0.5, 0.125 + 0.5) = (0.875, 0.625, 0.625)`$ → grid point $`(3, 2, 2)`$
- 4-fold rotation about z: $`(-0.125, 0.375, 0.125)`$ → wrapped to $`(0.875, 0.375, 0.125)`$ → grid point $`(3, 1, 0)`$
- Inversion: $`(-0.375, -0.125, -0.125)`$ → wrapped to $`(0.625, 0.875, 0.875)`$ → grid point $`(2, 3, 3)`$

All these points belong to the same orbit and share one irreducible mesh index.

### Irreducible Mesh

The irreducible mesh is the minimal set of grid points from which the full field can be reconstructed using symmetry operations.

**Algorithm:**
1. Mark all grid points as unvisited
2. For the first unvisited point $`\mathbf{p}`$:
   - Add $`\mathbf{p}`$ to irreducible mesh with index $`k`$
   - Compute $`\text{Orbit}(\mathbf{p})`$
   - Mark all orbit points with index $`k`$
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

```math
w_{\text{reduced}}[k] = w_{\text{full}}[\mathbf{p}_k]
```

where $`\mathbf{p}_k`$ is the $`k`$-th irreducible mesh point.

#### From Reduced Basis

Reconstruct full field by copying values to all orbit members:

```math
w_{\text{full}}[\mathbf{p}] = w_{\text{reduced}}[\text{index}(\mathbf{p})]
```

where $`\text{index}(\mathbf{p})`$ maps grid point $`\mathbf{p}`$ to its irreducible mesh index.

#### Symmetrization

Average field values over each orbit to enforce perfect symmetry:

```math
w_{\text{sym}}[\mathbf{p}] = \frac{1}{|\text{Orbit}(\mathbf{p})|} \sum_{\mathbf{q} \in \text{Orbit}(\mathbf{p})} w[\mathbf{q}]
```

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
        "hall_number": 530           # Hall number (optional if unique)
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
sg.display_info()

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

When CrysFFT is available (periodic 3D only), the reduced basis can be switched
from the irreducible mesh to a **physical basis** to avoid gather/scatter
mapping during diffusion. SCFT makes this choice automatically; with
`PropagatorSolver`, enable a basis on the `SpaceGroup` (`enable_*_physical_basis()`)
before passing it in. SCFT chooses:

- **Orthogonal boxes** (α = β = γ = 90°, even grid): the 3m physical basis
  (1/8 grid) is preferred when the space group provides the required 3m
  translations, the translation-parity fold permutation is valid (each
  generator translation must be an integral number of grid cells with a
  bijective parity map — e.g. for Fddd's quarter-cell d-glides at most one
  of $`n_x/4`$, $`n_y/4`$, $`n_z/4`$ may be odd), and $`n_z/2 \ge 8`$; otherwise the
  Pmmm physical basis is used (1/8 grid, requires mirror
  planes along x, y, z).
- **Non-orthogonal cells with an orthogonal z-axis** (α = β = 90°, γ
  arbitrary — e.g. hexagonal cells): the z-mirror (ObliqueZ)
  physical basis (1/2 grid) is used when the space group has a z-mirror
  $`(x, y, -z + t_z)`$ with integral x and y translations and $`t_z = 0`$ or
  $`\tfrac{1}{2}`$, and $`n_z`$ is even ($`n_z`$ must be a multiple of 4 when
  $`t_z = \tfrac{1}{2}`$, as for P6_3/mmc).
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
| P6_3/mmc (HCP) | 2 (nz only) | 1/2 translation along z (6_3 screw) |

The required divisor is computed per axis from the denominators of the
translation components of the symmetry operations along that axis (e.g. a 1/4
translation requires the grid to be divisible by 4).

## Class Methods

| Method | Description |
|--------|-------------|
| `to_reduced_basis(fields)` | Convert full fields to irreducible representation |
| `from_reduced_basis(reduced)` | Reconstruct full fields from reduced |
| `symmetrize(fields)` | Average fields over the classes of the active basis: the full space-group orbits on the irreducible basis, only the physical-basis classes once a physical basis is enabled |
| `symmetrize_reduced_basis(reduced)` | Average a reduced-basis field in place over the full space-group orbits (a physical basis can hold several representatives of one orbit) |
| `reduced_to_irreducible(reduced)` | Map a reduced-basis (physical sub-grid) field to the irreducible-orbit representation |
| `irreducible_to_reduced(irreducible)` | Map an irreducible-orbit field back to the reduced (physical) basis |

The conversions accept one field (1-D) or several (`(n_fields, n_in)`) and keep that layout: a 1-D input gives a 1-D output, a 2-D input an `(n_fields, n_out)` array; `symmetrize` returns the shape of its input and `symmetrize_reduced_basis` works in place (C-contiguous float64 array).

## Accessor Methods

Metadata is exposed through getter methods (not attributes):

| Method | Description |
|--------|-------------|
| `get_hall_number()` | Hall number (1-530) |
| `get_spacegroup_number()` | International space group number (1-230) |
| `get_spacegroup_symbol()` | ITA short symbol |
| `get_crystal_system()` | Crystal system name |
| `check_cell(lx, angles)` | Raise unless the cell (lengths, angles in degrees) belongs to the crystal system of the group; the computation box applies the same check |
| `display_info()` | Print the group, its crystal system and the basis sizes (construction prints nothing) |
| `uses_hexagonal_axes()` | True for hexagonal/trigonal groups in a hexagonal-axes setting (a = b, γ = 120°); False for rhombohedral-axes R settings (a = b = c, α = β = γ) |
| `get_n_symmetry_ops()` | Number of symmetry operations |
| `get_n_reduced_basis()` | Number of reduced-basis points (irreducible mesh points, or the physical sub-grid size when a physical basis is enabled) |
| `get_n_reduced_basis_full()` | Number of irreducible-orbit points of the full space group (equals `get_n_reduced_basis()` unless a physical basis is active) |
| `physical_basis_name()` | Active physical basis: `'irreducible'`, `'pmmm'`, `'m3'` or `'z-mirror'` |
| `m3_crysfft_feasible()` | True when the recursive 3m CrysFFT (required by the m3 basis) is feasible on this grid |
| `enable_pmmm_physical_basis()` / `using_pmmm_physical_basis()` | Switch the reduced basis to the Pmmm physical sub-grid (1/8 grid) / query whether it is active |
| `enable_m3_physical_basis()` / `using_m3_physical_basis()` | Switch to the M3 physical sub-grid (1/8 even-index grid) / query whether it is active |
| `enable_z_mirror_physical_basis()` / `using_z_mirror_physical_basis()` | Switch to the z-mirror physical basis (half grid along z) / query whether it is active |
| `get_reduced_basis_indices()` | Flat grid indices of the reduced-basis points (the irreducible points, or the physical sub-grid when a physical basis is enabled) |
| `get_full_to_reduced_map()` | Map from full grid index to reduced-basis index |
| `get_orbit_counts()` | Number of grid points mapped to each reduced-basis point (the orbit size on the irreducible basis) |
| `get_nx()` | Grid dimensions |
| `get_total_grid()` | Total number of grid points |

## Hexagonal Systems

Hexagonal crystal systems (P6_3/mmc, etc.) require special attention:

- Lattice: a = b, c independent
- Angles: α = β = 90°, γ = 120° (pass `"angles": [90, 90, 120]` in params)
- Grid: nx = ny (first two dimensions must be equal)
- nz must be even (for P6_3/mmc, from the 1/2 translation of its 6_3 screw); nx = ny may be odd

**Example:**
```python
nx = [48, 48, 96]  # a = b, c different, nz even
sg = SpaceGroup(nx, "P6_3/mmc", hall_number=488)
```

### How Hexagonal/Trigonal Groups Are Handled

Hexagonal-axes settings are treated differently from the other crystal
systems in two places, both for the same reason: the discretization must
respect the point group *exactly*, not just to rounding accuracy. Without
them the perforation layers of PL/HCP phases show X-shaped artifacts and the
free energy is biased by $`\sim 10^{-3}`$; with them, space-group on/off
results agree to machine precision (e.g., PL on 48×48×72:
$`\max|\Delta\phi| \approx 5 \times 10^{-15}`$).

#### 1. Hybrid orbit convention (real space)

In-plane hexagonal rotations contain matrix rows whose entries sum to zero,
e.g. the 6-fold rotation

```math
\mathbf{R}_{6} = \begin{pmatrix} 1 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}
```

Applied to a cell-centered coordinate, the row $`[1, -1, 0]`$ gives
$`\frac{(i+0.5) - (j+0.5)}{N} = \frac{i-j}{N}`$ — the $`0.5`$ offset is
annihilated, so a cell **center** is mapped onto a cell **boundary**, which
is not a grid point. No rounding rule can fix this: the cell-centered grid
is simply not invariant under the in-plane hexagonal rotations, and forcing
nearest-point assignment produces wrong orbits at every grid point (for
P6_3/mmc on 24×24×16, every point has images on cell boundaries, and after
rounding the images of no point form a set closed under the group).

The fix is a **hybrid convention** used only for hexagonal-axes settings:

```math
\mathbf{p}_{\text{frac}} = \left( \frac{i}{N_x}, \frac{j}{N_y}, \frac{k + 0.5}{N_z} \right)
```

i.e. node-centered in-plane, cell-centered along $`z`$. This works because
every symmetry operation of every hexagonal-axes Hall setting is
block-diagonal: a $`2 \times 2`$ in-plane block acting on $`(x, y)`$ and $`\pm 1`$
acting on $`z`$. The in-plane block sees node-centered coordinates (invariant
under the hexagonal rotations), while $`\pm z`$ (with its screw/glide
translations, which are integral in grid cells) remains exact on the
cell-centered $`z`$ axis. Every orbit map is then evaluated with **exact
integer modular arithmetic** — no floating-point rounding is involved at
all. Cubic, tetragonal, orthorhombic, monoclinic, and triclinic settings
keep the pure cell-centered convention, which is exact for them.

#### 2. Group-invariant alias representative (Fourier space)

Exact real-space orbits are not sufficient: the discretized diffusion
operator itself must commute with the point group. The pseudo-spectral
Boltzmann factor $`\exp(-b^2 |\mathbf{k}|^2 \Delta s / 6)`$ is tabulated per
discrete frequency $`\mathbf{m}`$, but on the discrete grid $`\mathbf{m}`$ is
only defined modulo $`N`$ (aliasing). On an **oblique** cell the squared
magnitude

```math
|\mathbf{k}|^2 = \mathbf{m}^{\top} \mathbf{G}^{*} \mathbf{m}, \qquad \mathbf{G}^{*} = (2\pi)^2 \, \mathbf{g}^{-1}
```

depends on which alias representative is chosen, because the reciprocal
metric $`\mathbf{G}^{*}`$ has off-diagonal terms. The conventional
principal-zone representative $`m_d \in [-N_d/2, N_d/2)`$ is **not invariant**
under hexagonal rotations: a rotation can carry a principal-zone frequency
outside the zone, and re-aliasing it back changes $`|\mathbf{k}|^2`$. The
tabulated operator then breaks the point-group symmetry, so even a
perfectly symmetrized field loses its symmetry after one propagator step.

The fix is to choose, within each alias class $`\{\mathbf{m} + N \mathbf{n}\}`$,
the representative that **minimizes the metric form**
$`\mathbf{m}^{\top} \mathbf{G}^{*} \mathbf{m}`$ (the first-Brillouin-zone
representative). This choice is group-invariant by construction — rotations
permute alias classes and preserve the minimal value — so the discretized
diffusion and stress operators recover exact point-group symmetry. The
minimal-form search runs only when the cell is actually oblique, detected by
a *relative* tolerance on the off-diagonal metric elements
($`|G^{*}_{ij}| > 10^{-12} \cdot \max_d G^{*}_{dd}`$); orthogonal cells use the
principal-zone representative unchanged (both conventions coincide there).
The same representative is used in the k-space Boltzmann tables, the stress
tables, and the CrysFFT ObliqueZ generators (`src/common/MinimalFormAlias.h`).
The search is exact rather than limited to one period around the centered
index, which misses the minimum when the grid spacing differs strongly between
axes (e.g. 4×32 points on a 120-degree cell with equal lengths).

**Initial-guess note**: analytic initial fields for hexagonal phases should
be built with the hybrid convention (in-plane $`i/N`$, cell-centered $`z`$) to
be exactly symmetric; fields built with other conventions are projected
onto the symmetric basis, which is harmless for the converged answer.

## Worked Example: BCC on 4×4×4 Grid

This example demonstrates the irreducible mesh calculation for Im-3m (BCC) on a small grid.

### Setup

- Grid: 4×4×4 = 64 points
- Space group: Im-3m (Hall 529), 96 symmetry operations
- Fractional coordinate of grid point $`(i, j, k)`$ (cell-centered): $`\left( \frac{i+0.5}{4}, \frac{j+0.5}{4}, \frac{k+0.5}{4} \right)`$

### Orbit Calculation

Because the grid is cell-centered, no grid point sits exactly on a
high-symmetry position such as the corner $`(0,0,0)`$ or the body center, so
orbits are larger than on a node-centered grid.

**Point $`(0, 0, 0)`$** - Body diagonal
- Fractional: $`(0.125, 0.125, 0.125)`$
- All three coordinates equal: 8 sign combinations $`(\pm 0.125, \pm 0.125, \pm 0.125)`$, doubled by body centering
- Orbit size: 16

**Point $`(0, 0, 1)`$** - General position on a mirror plane
- Fractional: $`(0.125, 0.125, 0.375)`$
- Two coordinates equal: 3 placements of the distinct coordinate × 8 sign combinations, doubled by body centering
- Orbit size: 48

### Irreducible Mesh

| Index | Representative | Orbit Size | Description |
|-------|---------------|------------|-------------|
| 0 | $`(0,0,0)`$ | 16 | Body diagonal $`(0.125, 0.125, 0.125)`$ |
| 1 | $`(0,0,1)`$ | 48 | $`(0.125, 0.125, 0.375)`$ |

**Verification:**
- Total points: $`16 + 48 = 64`$ ✓
- Irreducible mesh size: 2 points
- Reduction factor: $`64 / 2 = 32 \times`$

Note: The reduction factor (32x) is less than the number of symmetry operations (96) because points on the body diagonal have smaller orbits. On larger grids, most points sit in general positions and the reduction factor approaches 96 (e.g. ≈96x for Ia-3d on a 64³ grid).

## Limitations

- Only **periodic boundary conditions** supported
- A **mask** (impenetrable regions) must have the symmetry of the group; a mask without
  it is rejected (the reduced basis keeps one value per orbit)
- Grid must be compatible with space group symmetry: along each axis, a translation
  part p/q of a symmetry operation requires that axis to be a multiple of q (checked
  per axis, so a 6_3 screw constrains only nz)
- The box must match the crystal system of the group (cubic: a = b = c and 90° angles;
  tetragonal: a = b and 90° angles; orthorhombic: 90° angles; hexagonal/trigonal on
  hexagonal axes: a = b and γ = 120°;
  rhombohedral axes: a = b = c and α = β = γ, without box relaxation). `SCFT` raises a
  `ValidationError` otherwise.
- Beta feature - validate results carefully
- Requires `spglib` library
- **Initial-field symmetry**: input fields are projected onto the **full**
  space group (orbit average over all operations) before the first iteration,
  and the optimizer works in the irreducible basis, so propagators,
  concentrations and the space-group stress are consistent with the full
  symmetry from the first step. Analytic initial guesses built with the
  corner convention `round(x*nx)` are index-shifted relative to the
  cell-centered `(i+0.5)/N` orbit map; the projection removes their
  asymmetric part, which changes only the starting point, not the converged
  answer.
- **Discrete chains**: space group symmetry is supported on both **CPU and CUDA**,
  in both standard and `reduce_memory=True` modes
- **Hexagonal/trigonal groups (hexagonal-axes settings)** are fully
  supported via a hybrid orbit convention and a group-invariant k-space
  alias representative; space-group on/off results agree to machine
  precision. See [How Hexagonal/Trigonal Groups Are
  Handled](#how-hexagonaltrigonal-groups-are-handled).
- **Rhombohedral-axes R settings** contain axis-permuting operations that are
  exact on cell-centered grids only when $`n_x = n_y = n_z`$; incompatible
  grids are rejected with an explicit error instead of being silently
  projected onto wrong orbits.

## References

- `src/common/SpaceGroup.h` / `src/common/SpaceGroup.cpp`: SpaceGroup implementation
- `src/pybind11/polymerfts_core.cpp`: Python bindings
- `src/python/scft.py`: Integration with SCFT simulations
- International Tables for Crystallography, Vol. A (2016)
- spglib documentation: https://spglib.github.io/spglib/
