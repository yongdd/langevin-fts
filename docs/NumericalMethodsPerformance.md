# Numerical Methods: Performance and Accuracy

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document provides benchmark results comparing the numerical methods available for chain propagator computation in polymer field theory simulations. The benchmarks reproduce Fig. 1 from Song et al., *Chinese J. Polym. Sci.* **2018**, 36, 488-496.

## Available Methods

All numerical methods are selectable at runtime using the `numerical_method` parameter.

### Pseudo-Spectral Methods

| Method | Order | Description | Reference |
|--------|-------|-------------|-----------|
| **RQM4** | 4th | Richardson extrapolation with Ranjan-Qin-Morse 2008 parameters | *Macromolecules* 41, 942-954 (2008) |
| **RK2** | 2nd | Rasmussen-Kalosakas operator splitting (no Richardson extrapolation) | *J. Polym. Sci. B* 40, 1777 (2002) |
| **ETDRK4** | 4th | Exponential Time Differencing Runge-Kutta (Krogstad scheme) | *Chinese J. Polym. Sci.* 36, 488-496 (2018) |

> **Note**: RK2 for continuous chains is mathematically equivalent to the **N-bond model** for discrete chains described in Park et al., *J. Chem. Phys.* 150, 234901 (2019).

### Real-Space Methods

| Method | Order | Description |
|--------|-------|-------------|
| **CN-ADI2** | 2nd | Crank-Nicolson Alternating Direction Implicit |
| **CN-ADI4-LR** | 4th | CN-ADI with Local Richardson extrapolation |

### Usage Example

```python
from polymerfts import scft

params = {
    "nx": [32, 32, 32],
    "lx": [3.3, 3.3, 3.3],
    "ds": 0.01,
    "chain_model": "continuous",
    "numerical_method": "rqm4",  # or "rk2", "etdrk4", "cn-adi2", "cn-adi4-lr"
    # ... other parameters
}

calculation = scft.SCFT(params=params)
```

## Benchmark Configuration

- **Platform**: NVIDIA A10 GPU (CUDA)
- **System**: AB diblock copolymer, Gyroid phase
- **SCFT convergence**: tolerance = 10⁻⁹
- **Max iterations**: 2000
- **Date**: 2026-01-15

## Fig. 1: Contour Discretization Convergence

**Conditions**: f = 0.375, χN = 18, M = 32³, L = 3.65

### Convergence Plot

![Contour Convergence](figures/figure1_song2018.png)

**(a)** Pseudo-spectral 4th-order methods (RQM4, ETDRK4) show 4th-order convergence, while RK2 shows 2nd-order convergence. **(b)** Real-space methods: CN-ADI2 shows 2nd-order convergence, CN-ADI4-LR shows 4th-order convergence. Note: Real-space methods use F_ref = -0.47935 due to finite-difference spatial discretization. **(c)** Execution time comparison for all methods.

### Execution Time vs Contour Steps (Ns)

| Method | Ns=40 | Ns=80 | Ns=160 | Ns=320 | Ns=640 | Ns=1000 |
|--------|-------|-------|--------|--------|--------|---------|
| **RQM4** | 4.3 s | 8.2 s | 15.7 s | 30.7 s | 60.7 s | 94.3 s |
| **RK2** | 3.1 s | 4.4 s | 7.3 s | 12.9 s | 23.4 s | 36.0 s |
| **ETDRK4** | 8.7 s | 16.7 s | 32.6 s | 64.7 s | 128.4 s | 199.8 s |
| **CN-ADI2** | 11.8 s | 22.9 s | 45.5 s | 90.3 s | 180.1 s | 282.0 s |
| **CN-ADI4-LR** | 35.4 s | 68.7 s | 136.5 s | 270.9 s | 540.3 s | 846.0 s |

### Free Energy vs Contour Steps (Ns)

| Method | Ns=40 | Ns=80 | Ns=160 | Ns=320 | Ns=640 | Ns=1000 |
|--------|-------|-------|--------|--------|--------|---------|
| **RQM4** | -0.477010930978363 | -0.476977370031310 | -0.476974360495475 | -0.476974130350877 | -0.476974114332803 | -0.476974113395157 |
| **RK2** | -0.475796762870488 | -0.476698716000712 | -0.476906701588300 | -0.476957357088977 | -0.476969930359763 | -0.476972400411139 |
| **ETDRK4** | -0.476935495624106 | -0.476971516036433 | -0.476973944733011 | -0.476974102472743 | -0.476974112524746 | -0.476974113087715 |
| **CN-ADI2** | -0.477733629652859 | -0.478950812741258 | -0.479251268938022 | -0.479326309667460 | -0.479345088848207 | -0.479348787593391 |
| **CN-ADI4-LR** | -0.479362551579412 | -0.479352016719520 | -0.479351370587765 | -0.479351351917473 | -0.479351354344542 | -0.479351354862727 |

### Error vs Contour Steps

**Pseudo-spectral methods** ($F_{\rm ref}$ = -0.476974113087715):

| Method | Ns=40 | Ns=80 | Ns=160 | Ns=320 | Ns=640 | Ns=1000 |
|--------|-------|-------|--------|--------|--------|---------|
| **RQM4** | 3.6818e-05 | 3.2569e-06 | 2.4741e-07 | 1.7263e-08 | 1.2451e-09 | 3.0744e-10 |
| **RK2** | 1.1774e-03 | 2.7540e-04 | 6.7411e-05 | 1.6756e-05 | 4.1827e-06 | 1.7127e-06 |
| **ETDRK4** | 3.8617e-05 | 2.5971e-06 | 1.6835e-07 | 1.0615e-08 | 5.6297e-10 | (ref) |

**Real-space methods** ($F_{\rm ref}$ = -0.479351354862727):

| Method | Ns=40 | Ns=80 | Ns=160 | Ns=320 | Ns=640 | Ns=1000 |
|--------|-------|-------|--------|--------|--------|---------|
| **CN-ADI2** | 1.6177e-03 | 4.0054e-04 | 1.0009e-04 | 2.5045e-05 | 6.2662e-06 | 2.5675e-06 |
| **CN-ADI4-LR** | 1.1197e-05 | 6.6162e-07 | 1.5492e-08 | 3.1779e-09 | 7.5084e-10 | (ref) |

### Speedup Relative to CN-ADI2 (at Ns=1000)

| Method | Speedup |
|--------|---------|
| **RK2** | **7.8x faster** |
| **RQM4** | **3.0x faster** |
| **ETDRK4** | 1.4x faster |
| **CN-ADI2** | 1.0x (baseline) |
| **CN-ADI4-LR** | 3.0x slower |

### Convergence Order Analysis

| Ns transition | ETDRK4 order | RQM4 order | RK2 order |
|---------------|--------------|------------|-----------|
| 40 → 80 | 3.89 | 3.50 | 2.10 |
| 80 → 160 | 3.95 | 3.72 | 2.06 |
| 160 → 320 | 3.97 | 3.86 | 2.13 |
| 320 → 640 | 3.98 | 3.92 | 2.61 |

**Key findings**:
- **RQM4** and **ETDRK4** both show 4th-order convergence (order ≈ 4.0)
- **RK2** shows 2nd-order convergence (order ≈ 2.2), with error ~1.7e-6 at Ns=1000
- **ETDRK4** has slightly smaller error coefficients than RQM4 at the same ds
- All pseudo-spectral methods converge to F = -0.47697411
- **CN-ADI methods** converge to a different value (F = -0.47935) due to finite-difference spatial discretization

## Performance Summary

### Method Comparison

| Method | Convergence Order | Speed (vs CN-ADI2) | Notes |
|--------|-------------------|-------------------|-------|
| **RK2** | 2nd ✓ | **7.8x faster** | Fastest, lower accuracy |
| **RQM4** | 4th ✓ | **3.0x faster** | Recommended for most applications |
| **ETDRK4** | 4th ✓ | 1.4x faster | Same accuracy as RQM4, 2x slower |
| **CN-ADI2** | 2nd ✓ | baseline | Supports non-periodic BC |
| **CN-ADI4-LR** | 4th ✓ | 3.0x slower | Local Richardson extrapolation |

### Key Findings

1. **RK2 is the fastest method** - 2.6x faster than RQM4 per iteration, but only 2nd-order accurate
2. **RQM4 is the fastest 4th-order pseudo-spectral method** - 2x faster than ETDRK4 per iteration
3. **RQM4 and ETDRK4 achieve identical 4th-order convergence** - both converge to F = -0.47697411
4. **RK2 achieves 2nd-order convergence** - converges to F = -0.47697240 (differs by ~1.7e-6 from 4th-order methods)
5. **CN-ADI4-LR achieves 4th-order convergence** with ~3x cost of CN-ADI2
6. **CN-ADI methods** support non-periodic boundary conditions but converge to a different free energy (F = -0.47935) due to finite-difference spatial discretization error

## Method Recommendations

### When to Use Each Method

| Use Case | Recommended Method | Reason |
|----------|-------------------|--------|
| Standard SCFT/FTS (periodic BC) | **RQM4** | Fastest 4th-order, recommended default |
| Fast iterations, prototyping | **RK2** | Fastest overall, lower accuracy |
| Non-periodic boundaries (2nd-order) | **CN-ADI2** | Fast, supports absorbing/reflecting BC |
| Non-periodic boundaries (4th-order) | **CN-ADI4-LR** | 4th-order with non-periodic BC |

### ETDRK4 vs RQM4

Both methods achieve **4th-order convergence** with comparable accuracy. The implementation uses:

| Factor | RQM4 | ETDRK4 |
|--------|------|--------|
| Speed | **~2x faster** | Slower (more FFTs per step) |
| Implementation | Operator splitting + Richardson extrapolation | Krogstad scheme (no operator splitting) |
| Coefficients | Pre-computed Boltzmann factors | Kassam-Trefethen contour integral |
| Error coefficient | Slightly larger | Slightly smaller |

**Recommendation**: Use **RQM4** for standard simulations due to its speed advantage. **ETDRK4** (Krogstad scheme) achieves similar accuracy with slightly smaller error coefficients but requires more computation per step.

## Notes on Pseudo-Spectral vs Real-Space Discrepancy

The CN-ADI methods converge to a free energy (F = -0.47935) that differs from the pseudo-spectral methods (F = -0.47697) by approximately 0.5%. This discrepancy arises from the different spatial discretization approaches:

- **Pseudo-spectral methods** (RQM4, ETDRK4): Use spectral (Fourier) representation for spatial derivatives, achieving exponential convergence in spatial resolution for smooth periodic fields
- **Real-space methods** (CN-ADI): Use finite-difference approximations for spatial derivatives, introducing O(Δx²) spatial discretization error

For the 32³ grid used in these benchmarks, the finite-difference spatial error dominates the contour discretization error. To achieve comparable accuracy between methods, CN-ADI would require a finer spatial grid (higher M).

**Implication**: When comparing free energies between pseudo-spectral and real-space solvers, ensure sufficient spatial resolution for the real-space method. For periodic boundary conditions, pseudo-spectral methods are more efficient and accurate.

## References

1. A. Ranjan, J. Qin, and D. C. Morse, **"Linear Response and Stability of Ordered Phases of Block Copolymer Melts"**, *Macromolecules*, **2008**, 41, 942-954.
   - RQM4 parameters for Richardson extrapolation

2. J. Song, Y. Liu, and R. Zhang, **"Exponential Time Differencing Schemes for Solving the Self-Consistent Field Equations of Polymers"**, *Chinese J. Polym. Sci.*, **2018**, 36, 488-496.
   - Krogstad ETDRK4 scheme for polymer field theory, benchmark methodology

3. A.-K. Kassam and L. N. Trefethen, **"Fourth-Order Time-Stepping for Stiff PDEs"**, *SIAM J. Sci. Comput.*, **2005**, 26, 1214-1233.
   - Contour integral method for stable ETDRK4 coefficient computation

4. P. Stasiak and M. W. Matsen, **"Efficiency of pseudo-spectral algorithms with Anderson mixing for the SCFT of periodic block-copolymer phases"**, *Eur. Phys. J. E*, **2011**, 34, 110.
   - Convergence analysis methodology
