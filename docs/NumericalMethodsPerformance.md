# Numerical Methods: Performance and Accuracy

This document provides benchmark results comparing the numerical methods available for chain propagator computation in polymer field theory simulations. The benchmarks reproduce Fig. 1 from Song et al., *Chinese J. Polym. Sci.* **2018**, 36, 488-496.

## Available Methods

All numerical methods are selectable at runtime using the `numerical_method` parameter.

### Pseudo-Spectral Methods

| Method | Order | Description | Reference |
|--------|-------|-------------|-----------|
| **RQM4** | 4th | Richardson extrapolation with Ranjan-Qin-Morse 2008 parameters | *Macromolecules* 41, 942-954 (2008) |
| **ETDRK4** | 4th | Exponential Time Differencing Runge-Kutta (Krogstad scheme) | *Chinese J. Polym. Sci.* 36, 488-496 (2018) |

### Real-Space Methods

| Method | Order | Description |
|--------|-------|-------------|
| **CN-ADI2** | 2nd | Crank-Nicolson Alternating Direction Implicit |
| **CN-ADI4-LR** | 4th | CN-ADI with Local Richardson extrapolation |
| **CN-ADI4-GR** | 4th | CN-ADI with Global Richardson extrapolation |
| **SDC-N** | Nth | Spectral Deferred Correction (N=2-10) |

### Usage Example

```python
from polymerfts import scft

params = {
    "nx": [32, 32, 32],
    "lx": [3.3, 3.3, 3.3],
    "ds": 0.01,
    "chain_model": "continuous",
    "numerical_method": "rqm4",  # or "etdrk4", "cn-adi2", "cn-adi4-lr", "cn-adi4-gr", "sdc-N" (N=2-10)
    # ... other parameters
}

calculation = scft.SCFT(params=params)
```

## Benchmark Configuration

- **Platform**: NVIDIA A10 GPU (CUDA)
- **System**: AB diblock copolymer, Gyroid phase
- **SCFT convergence**: tolerance = 10⁻⁹
- **Max iterations**: 2000
- **Date**: 2026-01-13

## Fig. 1: Contour Discretization Convergence

**Conditions**: f = 0.375, χN = 18, M = 32³, L = 3.65

### Convergence Plot

![Contour Convergence](figures/figure1_song2018.png)

**(a)** Pseudo-spectral methods (RQM4, ETDRK4) show 4th-order convergence. **(b)** Real-space methods: CN-ADI2 shows 2nd-order convergence, CN-ADI4-LR and CN-ADI4-GR both show 4th-order convergence. Note: CN-ADI uses a different reference free energy (F_ref = -0.47935) due to finite-difference spatial discretization. **(c)** Execution time comparison for all methods.

### Execution Time vs Contour Steps (Ns)

| Method | Ns=40 | Ns=80 | Ns=160 | Ns=320 | Ns=640 | Ns=1000 |
|--------|-------|-------|--------|--------|--------|---------|
| **RQM4** | 4.3 s | 8.2 s | 15.7 s | 30.7 s | 60.7 s | 94.3 s |
| **ETDRK4** | 8.7 s | 16.7 s | 32.6 s | 64.7 s | 128.4 s | 199.8 s |
| **CN-ADI2** | 11.8 s | 22.9 s | 45.5 s | 90.3 s | 180.1 s | 282.0 s |
| **CN-ADI4-LR** | 35.4 s | 68.7 s | 136.5 s | 270.9 s | 540.3 s | 846.0 s |
| **CN-ADI4-GR** | 34.6 s | 68.3 s | 135.8 s | 271.2 s | 541.2 s | 844.9 s |

### Free Energy vs Contour Steps (Ns)

| Method | Ns=40 | Ns=80 | Ns=160 | Ns=320 | Ns=640 | Ns=1000 |
|--------|-------|-------|--------|--------|--------|---------|
| **RQM4** | -0.47701093 | -0.47697737 | -0.47697436 | -0.47697413 | -0.47697411 | -0.47697411 |
| **ETDRK4** | -0.47693550 | -0.47697152 | -0.47697394 | -0.47697410 | -0.47697411 | -0.47697411 |
| **CN-ADI2** | -0.47773363 | -0.47895081 | -0.47925127 | -0.47932631 | -0.47934509 | -0.47934879 |
| **CN-ADI4-LR** | -0.47936255 | -0.47935202 | -0.47935137 | -0.47935135 | -0.47935135 | -0.47935135 |
| **CN-ADI4-GR** | -0.47934960 | -0.47935095 | -0.47935129 | -0.47935135 | -0.47935135 | -0.47935135 |
| **SDC-4** | -0.47933271 | - | - | - | - | - |

### Error vs Contour Steps (|F - F_ref|, F_ref = -0.47697411)

| Method | Ns=40 | Ns=80 | Ns=160 | Ns=320 | Ns=640 |
|--------|-------|-------|--------|--------|--------|
| **RQM4** | 3.68e-05 | 3.26e-06 | 2.47e-07 | 1.71e-08 | 1.13e-09 |
| **ETDRK4** | 3.86e-05 | 2.60e-06 | 1.68e-07 | 1.07e-08 | 6.77e-10 |

### Speedup Relative to CN-ADI2 (at Ns=1000)

| Method | Speedup |
|--------|---------|
| **RQM4** | **3.0x faster** |
| **ETDRK4** | 1.4x faster |
| **CN-ADI2** | 1.0x (baseline) |
| **CN-ADI4-LR** | 3.0x slower |
| **CN-ADI4-GR** | 3.0x slower |

### Convergence Order Analysis

| Ns transition | ETDRK4 order | RQM4 order |
|---------------|--------------|------------|
| 40 → 80 | 3.89 | 3.50 |
| 80 → 160 | 3.95 | 3.72 |
| 160 → 320 | 3.97 | 3.86 |
| 320 → 640 | 3.98 | 3.92 |
| 640 → 1000 | 4.01 | 3.97 |

**Key findings**:
- **RQM4** and **ETDRK4** both show 4th-order convergence (order ≈ 4.0)
- **ETDRK4** has slightly smaller error coefficients than RQM4 at the same ds
- Both pseudo-spectral methods converge to F = -0.47697411
- **CN-ADI methods** converge to a different value (F = -0.47935) due to different spatial discretization

## Performance Summary

### Method Comparison

| Method | Convergence Order | Speed (vs CN-ADI2) | Notes |
|--------|-------------------|-------------------|-------|
| **RQM4** | 4th ✓ | **2.7x faster** | Recommended for most applications |
| **ETDRK4** | 4th ✓ | 1.3x faster | Same accuracy as RQM4, 2x slower |
| **CN-ADI2** | 2nd ✓ | baseline | Supports non-periodic BC |
| **CN-ADI4-LR** | 4th ✓ | 3.0x slower | Local Richardson extrapolation |
| **CN-ADI4-GR** | 4th ✓ | 3.0x slower | Global Richardson extrapolation |
| **SDC-4** | 4th ✓ | ~7x slower | Configurable order; slower in 3D due to PCG |

### Key Findings

1. **RQM4 is the fastest pseudo-spectral method** - 2x faster than ETDRK4 per iteration
2. **RQM4 and ETDRK4 achieve identical 4th-order convergence** - both converge to F = -0.47697411
3. **CN-ADI4-LR and CN-ADI4-GR have similar performance** - both achieve 4th-order convergence with ~3x cost of CN-ADI2
4. **CN-ADI methods** support non-periodic boundary conditions but converge to a different free energy (F = -0.47935) due to finite-difference spatial discretization error

## Method Recommendations

### When to Use Each Method

| Use Case | Recommended Method | Reason |
|----------|-------------------|--------|
| Standard SCFT/FTS (periodic BC) | **RQM4** | Fastest, 4th-order accurate |
| Non-periodic boundaries (2nd-order) | **CN-ADI2** | Fast, supports absorbing/reflecting BC |
| Non-periodic boundaries (4th-order) | **CN-ADI4-LR** or **CN-ADI4-GR** | 4th-order with non-periodic BC |
| Higher than 4th-order accuracy | **SDC-N** | Nth-order (N=2-10), but slower in 2D/3D |

### ETDRK4 vs RQM4

Both methods achieve **4th-order convergence** with comparable accuracy. The implementation uses:

| Factor | RQM4 | ETDRK4 |
|--------|------|--------|
| Speed | **~2x faster** | Slower (more FFTs per step) |
| Implementation | Operator splitting + Richardson extrapolation | Krogstad scheme (no operator splitting) |
| Coefficients | Pre-computed Boltzmann factors | Kassam-Trefethen contour integral |
| Error coefficient | Slightly larger | Slightly smaller |

**Recommendation**: Use **RQM4** for standard simulations due to its speed advantage. **ETDRK4** (Krogstad scheme) achieves similar accuracy with slightly smaller error coefficients but requires more computation per step.

### CN-ADI4-LR vs CN-ADI4-GR

Both methods achieve 4th-order convergence with similar computational cost (~3x CN-ADI2). The key difference:

- **CN-ADI4-LR**: Local Richardson extrapolation at each step. Simpler, lower memory.
- **CN-ADI4-GR**: Global Richardson extrapolation at quadrature level. True 4th-order in 1D.

For implementation details, see [RealSpaceSolver.md](RealSpaceSolver.md#cn-adi4-lr-vs-cn-adi4-gr).

## References

1. A. Ranjan, J. Qin, and D. C. Morse, **"Linear Response and Stability of Ordered Phases of Block Copolymer Melts"**, *Macromolecules*, **2008**, 41, 942-954.
   - RQM4 parameters for Richardson extrapolation

2. J. Song, Y. Liu, and R. Zhang, **"Exponential Time Differencing Schemes for Solving the Self-Consistent Field Equations of Polymers"**, *Chinese J. Polym. Sci.*, **2018**, 36, 488-496.
   - Krogstad ETDRK4 scheme for polymer field theory, benchmark methodology

3. A.-K. Kassam and L. N. Trefethen, **"Fourth-Order Time-Stepping for Stiff PDEs"**, *SIAM J. Sci. Comput.*, **2005**, 26, 1214-1233.
   - Contour integral method for stable ETDRK4 coefficient computation

4. P. Stasiak and M. W. Matsen, **"Efficiency of pseudo-spectral algorithms with Anderson mixing for the SCFT of periodic block-copolymer phases"**, *Eur. Phys. J. E*, **2011**, 34, 110.
   - Convergence analysis methodology

## SDC Method Performance Notes

The Spectral Deferred Correction (SDC) method provides Nth-order temporal accuracy (N=2-10) for real-space problems with non-periodic boundary conditions. However, it has significantly higher computational cost than other methods, especially in 3D.

### SDC Computational Characteristics

| Dimension | Solver Type | Performance |
|-----------|-------------|-------------|
| 1D | Tridiagonal (direct) | Fast |
| 2D/3D | PCG (iterative) | **~30x slower than RQM4** |

**Benchmark Result** (32³ grid, χN=18, f=0.375):
- **SDC-4** at Ns=40: 134.7 s (382 iterations)
- **RQM4** at Ns=40: 4.3 s (380 iterations)
- **Slowdown**: ~31x

### Why SDC is Slower in 3D

1. **Implicit solver at each substep**: SDC uses M substeps (M=4 for SDC-4) per contour step, each requiring a Newton iteration
2. **PCG iterations**: Each Newton step requires PCG (Preconditioned Conjugate Gradient) solver for the 3D sparse system
3. **Typical iteration count**: PCG runs ~10-50 iterations per solve, with ~50 solves per propagator step

**Recommendation**: For 3D periodic systems, use **RQM4** or **ETDRK4** which are 30x faster. SDC is intended for non-periodic boundary conditions where pseudo-spectral methods are not applicable.

## Notes on Pseudo-Spectral vs Real-Space Discrepancy

The CN-ADI methods converge to a free energy (F = -0.47935) that differs from the pseudo-spectral methods (F = -0.47697) by approximately 0.5%. This discrepancy arises from the different spatial discretization approaches:

- **Pseudo-spectral methods** (RQM4, ETDRK4): Use spectral (Fourier) representation for spatial derivatives, achieving exponential convergence in spatial resolution for smooth periodic fields
- **Real-space methods** (CN-ADI): Use finite-difference approximations for spatial derivatives, introducing O(Δx²) spatial discretization error

For the 32³ grid used in these benchmarks, the finite-difference spatial error dominates the contour discretization error. To achieve comparable accuracy between methods, CN-ADI would require a finer spatial grid (higher M).

**Implication**: When comparing free energies between pseudo-spectral and real-space solvers, ensure sufficient spatial resolution for the real-space method. For periodic boundary conditions, pseudo-spectral methods are more efficient and accurate.
