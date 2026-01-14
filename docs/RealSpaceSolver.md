# Real-Space Solver Documentation

This document describes the real-space finite difference solver for continuous chain propagators, including its numerical methods, boundary condition support, and usage.

For detailed benchmark comparisons, see [NumericalMethodsPerformance.md](NumericalMethodsPerformance.md) and [RealSpaceConvergence.md](RealSpaceConvergence.md).

## Overview

The real-space solver uses the **CN-ADI** (Crank-Nicolson Alternating Direction Implicit) method to solve the modified diffusion equation:

$$\frac{\partial q}{\partial s} = \frac{b^2}{6} \nabla^2 q - w(\mathbf{r}) q$$

Unlike the pseudo-spectral method which requires periodic boundary conditions, the real-space method supports:

| Boundary Condition | Description | Mathematical Form |
|-------------------|-------------|-------------------|
| **Periodic** | Cyclic boundary | $q(0) = q(L)$ |
| **Reflecting** | Neumann (zero flux) | $\partial q / \partial n = 0$ |
| **Absorbing** | Dirichlet (zero value) | $q = 0$ at boundary |

## Numerical Method

### Crank-Nicolson Scheme

The Crank-Nicolson method is a semi-implicit scheme that is unconditionally stable:

$$(I - \frac{\Delta s}{2} L) q^{n+1} = (I + \frac{\Delta s}{2} L) q^n$$

where $L$ is the discrete Laplacian operator combined with the potential term.

### ADI Splitting (3D)

For 3D problems, the operator is split into three sequential 1D solves:

1. **X-direction sweep**: Solve tridiagonal system for $q^*$
2. **Y-direction sweep**: Solve tridiagonal system for $q^{**}$
3. **Z-direction sweep**: Solve tridiagonal system for $q^{n+1}$

Each direction requires solving a tridiagonal (or cyclic tridiagonal for periodic BC) system using:
- **Thomas algorithm**: For non-periodic boundaries
- **Sherman-Morrison formula**: For periodic boundaries

### Symmetric Splitting with Potential

The propagator advancement includes the potential term using symmetric splitting:

$$q^{n+1} = e^{-w \Delta s/2} \cdot \text{Diffusion}(\Delta s) \cdot e^{-w \Delta s/2} \cdot q^n$$

## Available Methods

| Method | Order | Description |
|--------|-------|-------------|
| **cn-adi2** | 2nd | Standard Crank-Nicolson ADI |
| **cn-adi4-lr** | 4th | CN-ADI with Local Richardson extrapolation |
| **cn-adi4-gr** | 4th | CN-ADI with Global Richardson extrapolation |

### Richardson Extrapolation for 4th-Order Accuracy

The 4th-order methods use Richardson extrapolation combining one full step with two half-steps:

$$q_{\text{out}} = \frac{4 \cdot q_{\text{half}} - q_{\text{full}}}{3}$$

This cancels the $O(\Delta s^2)$ error term, yielding $O(\Delta s^4)$ accuracy.

### Runtime Selection

```python
from polymerfts import PropagatorSolver

# CN-ADI2 (default - faster)
solver = PropagatorSolver(..., numerical_method="cn-adi2")

# CN-ADI4 with local Richardson (good balance)
solver = PropagatorSolver(..., numerical_method="cn-adi4-lr")

# CN-ADI4 with global Richardson (highest accuracy in 1D)
solver = PropagatorSolver(..., numerical_method="cn-adi4-gr")
```

## Usage

### Python Interface

```python
params = {
    "nx": [32, 32, 32],
    "lx": [4.0, 4.0, 4.0],
    "chain_model": "continuous",
    "ds": 0.01,

    # Boundary conditions: "periodic", "reflecting", or "absorbing"
    "bc": ["reflecting", "reflecting",    # x-direction (low, high)
           "reflecting", "reflecting",    # y-direction (low, high)
           "absorbing", "absorbing"],     # z-direction (low, high)

    # Select real-space method
    "numerical_method": "cn-adi2",

    # ... other parameters
}
```

### Mixed Boundary Conditions

Different boundary conditions can be specified for each direction:

```python
# Confined film: reflecting in z, periodic in x and y
"bc": ["periodic", "periodic",      # x: periodic
       "periodic", "periodic",      # y: periodic
       "reflecting", "reflecting"]  # z: reflecting (confined)

# Semi-infinite slab: absorbing on one side
"bc": ["periodic", "periodic",
       "periodic", "periodic",
       "reflecting", "absorbing"]   # z: reflecting bottom, absorbing top
```

### Grafted Brush Example

For polymer brushes grafted to a surface:

```python
from polymerfts import PropagatorSolver
import numpy as np

# 1D grafted brush with absorbing boundaries
solver = PropagatorSolver(
    nx=[128],
    lx=[4.0],
    ds=0.01,
    bond_lengths={"A": 1.0},
    bc=["absorbing", "absorbing"],
    chain_model="continuous",
    numerical_method="cn-adi2",
    platform="cpu-mkl"
)

# Add polymer with grafting point at node 0
solver.add_polymer(
    volume_fraction=1.0,
    blocks=[["A", 1.0, 0, 1]],
    grafting_points={0: "G"}
)

# Create delta-function initial condition at grafting point
dx = 4.0 / 128
x = (np.arange(128) + 0.5) * dx
x0 = 0.5  # Grafting point position

# Gaussian approximation to delta function
sigma = 0.1
q_init = np.exp(-(x - x0)**2 / (2 * sigma**2))
q_init = q_init / (np.sum(q_init) * dx)  # Normalize

# Compute propagators with zero field
w_field = np.zeros(128)
solver.compute_propagators({"A": w_field}, q_init={"G": q_init})

# Get results
q_end = solver.get_propagator(polymer=0, v=0, u=1, step=100)
Q = solver.get_partition_function(polymer=0)
```

## Grafted Brush Validation

### Absorbing Boundaries with Sharp Initial Conditions

For absorbing boundaries, the pseudo-spectral method using Discrete Sine Transform (DST) achieves spectral accuracy (~10⁻¹⁵ error), while real-space CN-ADI achieves ~10⁻⁶ error.

**Key findings:**
1. **Pseudo-spectral (DST) achieves machine precision** for all Gaussian widths
2. **Real-space requires sufficient resolution**: $\sigma/\Delta x > 3$-$5$ for accurate results
3. **Both methods handle absorbing BCs correctly**

### Stability Warning

**CN-ADI4** can become unstable when initial conditions are close to absorbing boundaries:

| Distance from boundary | CN-ADI4 | CN-ADI2 |
|------------------------|---------|---------|
| $> 5\sigma$ | Stable | Stable |
| $2$-$3\sigma$ | **Unstable** | Stable |
| $< 2\sigma$ | **Diverges** | Stable |

**Recommendation**: Use CN-ADI2 (not CN-ADI4) when grafting points are near absorbing boundaries.

## Implementation Details

### Files

| File | Description |
|------|-------------|
| `src/platforms/cpu/CpuSolverCNADI.cpp` | CPU CN-ADI solver |
| `src/platforms/cuda/CudaSolverCNADI.cu` | CUDA CN-ADI solver |
| `src/common/FiniteDifference.cpp` | Tridiagonal coefficient generation |

### Tridiagonal Solvers

**Non-periodic (Thomas Algorithm)**:
- Forward elimination followed by back substitution
- CUDA: Uses shared memory for coefficient caching

**Periodic (Sherman-Morrison)**:
- Converts cyclic system to standard tridiagonal + correction
- Solves two systems and combines results

## Limitations

1. **Stress computation**: Not yet implemented for real-space method
2. **Discrete chains**: Only continuous chain model supported
3. **Non-orthogonal cells**: Only orthogonal unit cells supported
4. **Complex fields**: Only real-valued fields supported

## When to Use Real-Space vs Pseudo-Spectral

### Use Real-Space When:
- Non-periodic boundary conditions are required
- CN-ADI4 accuracy is needed with non-periodic boundaries

### Use Pseudo-Spectral When:
- Periodic boundary conditions are acceptable
- Stress calculations are needed
- Maximum performance is required

## References

1. **Crank-Nicolson Method**: J. Crank and P. Nicolson, *Proc. Cambridge Phil. Soc.*, **1947**, 43, 50-67.

2. **ADI Method**: D. W. Peaceman and H. H. Rachford, *J. Soc. Indust. Appl. Math.*, **1955**, 3, 28-41.

3. **Richardson Extrapolation**: L. F. Richardson, *Phil. Trans. R. Soc. A*, **1911**, 210, 307-357.
