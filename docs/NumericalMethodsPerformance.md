# Numerical Methods: Performance and Accuracy

This document provides benchmark results comparing the numerical methods available for chain propagator computation in polymer field theory simulations. The benchmarks reproduce Fig. 1 and Fig. 2 from Song et al., *Chinese J. Polym. Sci.* **2018**, 36, 488-496.

## Available Methods

All numerical methods are selectable at runtime using the `numerical_method` parameter.

### Pseudo-Spectral Methods

| Method | Order | Description | Reference |
|--------|-------|-------------|-----------|
| **RQM4** | 4th | Richardson extrapolation with Ranjan-Qin-Morse 2008 parameters | *Macromolecules* 41, 942-954 (2008) |
| **ETDRK4** | 4th | Exponential Time Differencing Runge-Kutta | *J. Comput. Phys.* 176, 430-455 (2002) |

### Real-Space Methods

| Method | Order | Description |
|--------|-------|-------------|
| **CN-ADI2** | 2nd | Crank-Nicolson Alternating Direction Implicit |
| **CN-ADI4** | 4th | CN-ADI with Richardson extrapolation |

### Usage Example

```python
from polymerfts import scft

params = {
    "nx": [32, 32, 32],
    "lx": [3.3, 3.3, 3.3],
    "ds": 0.01,
    "chain_model": "continuous",
    "numerical_method": "rqm4",  # or "etdrk4", "cn-adi2", "cn-adi4"
    # ... other parameters
}

calculation = scft.SCFT(params=params)
```

## Benchmark Configuration

- **Platform**: NVIDIA A10 GPU (CUDA)
- **System**: AB diblock copolymer, Gyroid phase
- **SCFT convergence**: tolerance = 10⁻⁹
- **Max iterations**: 2000
- **Date**: 2026-01-12

## Fig. 1: Contour Discretization Convergence

**Conditions**: f = 0.375, χN = 18, M = 32³, L = 3.65

### Convergence Plot

![Contour Convergence](figures/figure1_song2018.png)

### Execution Time vs Contour Steps (Ns)

| Method | Ns=100 | Ns=200 | Ns=400 | Ns=1000 | Ns=4000 |
|--------|--------|--------|--------|---------|---------|
| **RQM4** | 7.5 s | 14.4 s | 28.4 s | 70.3 s | 280.8 s |
| **ETDRK4** | 15.2 s | 29.8 s | 59.2 s | 146.6 s | 588.7 s |
| **CN-ADI2** | 19.1 s | 38.0 s | 76.2 s | 191.2 s | 766.0 s |
| **CN-ADI4** | 57.4 s | 113.6 s | 228.5 s | 570.5 s | 2290.2 s |

### Free Energy vs Contour Steps (Ns)

| Method | Ns=40 | Ns=80 | Ns=160 | Ns=320 | Ns=1000 | Ns=4000 |
|--------|-------|-------|--------|--------|---------|---------|
| **RQM4** | -0.4770 | -0.4770 | -0.4770 | -0.4770 | -0.4770 | -0.4770 |
| **ETDRK4** | -0.4769 | -0.4770 | -0.4770 | -0.4770 | -0.4770 | -0.4770 |
| **CN-ADI2** | -0.4777 | -0.4790 | -0.4793 | -0.4793 | -0.4793 | -0.4794 |
| **CN-ADI4** | -0.4794 | -0.4794 | -0.4794 | -0.4794 | -0.4794 | -0.4794 |

### Speedup Relative to CN-ADI2 (at Ns=1000)

| Method | Speedup |
|--------|---------|
| **RQM4** | **2.7x faster** |
| **ETDRK4** | 1.3x faster |
| **CN-ADI2** | 1.0x (baseline) |
| **CN-ADI4** | 3.0x slower |

### Convergence Order Analysis

**Key findings**:
- **RQM4** and **ETDRK4** both show 4th-order convergence, following the slope -4 reference line
- **CN-ADI4** shows 4th-order convergence
- **CN-ADI2** shows 2nd-order convergence (follows slope -2)
- Both pseudo-spectral methods converge to F = -0.4770

## Fig. 2: Spatial Resolution Convergence

**Conditions**: f = 0.32, Ns = 101, χN = 40 and 80

### Free Energy vs Grid Size (χN = 40, L = 3.85)

| Method | Nx=24 | Nx=32 | Nx=48 | Nx=64 | Nx=96 | Nx=128 |
|--------|-------|-------|-------|-------|-------|--------|
| **RQM4** | -3.351 | -3.337 | -3.334 | -3.334 | -3.334 | -3.334 |
| **ETDRK4** | -3.350 | -3.336 | -3.334 | -3.334 | -3.334 | - |
| **CN-ADI2** | -3.445 | -3.384 | -3.354 | -3.346 | -3.341 | -3.339 |
| **CN-ADI4** | -3.446 | -3.383 | -3.353 | -3.344 | -3.338 | -3.337 |

### Execution Time vs Grid Size (χN = 40)

| Method | Nx=32 | Nx=64 | Nx=96 | Nx=128 |
|--------|-------|-------|-------|--------|
| **RQM4** | 14.6 s | 63.7 s | 293.5 s | 653.8 s |
| **ETDRK4** | 29.4 s | 126.8 s | 576.5 s | - |
| **CN-ADI2** | 34.8 s | 96.1 s | 208.8 s | 2493.7 s |

### Speedup at High Resolution (Nx=96, χN=40)

| Method | Time | Speedup vs CN-ADI2 |
|--------|------|-------------------|
| **RQM4** | 294 s | **1.4x slower** |
| **CN-ADI2** | 209 s | 1.0x (baseline) |

At high spatial resolution, CN-ADI2 becomes competitive due to its O(N) FFT cost vs O(N log N) for pseudo-spectral methods.

## Performance Summary

### Method Comparison

| Method | Convergence Order | Speed (vs CN-ADI2) | Notes |
|--------|-------------------|-------------------|-------|
| **RQM4** | 4th ✓ | **2.7x faster** | Recommended for most applications |
| **ETDRK4** | 4th ✓ | 1.3x faster | Same accuracy as RQM4, 2x slower |
| **CN-ADI2** | 2nd ✓ | baseline | Supports non-periodic BC |
| **CN-ADI4** | 4th ✓ | 3.0x slower | High accuracy with non-periodic BC |

### Key Findings

1. **RQM4 is the fastest pseudo-spectral method** - 2x faster than ETDRK4 per iteration
2. **RQM4 and ETDRK4 achieve identical 4th-order convergence** - both converge to F = -0.4770
3. **Pseudo-spectral methods** (RQM4, ETDRK4) show exponential spatial convergence
4. **CN-ADI methods** support non-periodic boundary conditions

## Method Recommendations

### When to Use Each Method

| Use Case | Recommended Method | Reason |
|----------|-------------------|--------|
| Standard SCFT/FTS (periodic BC) | **RQM4** | Fastest, 4th-order accurate |
| Non-periodic boundaries | **CN-ADI2** | Supports absorbing/reflecting BC |
| High-precision confined systems | **CN-ADI4** | 4th-order with non-periodic BC |

### ETDRK4 vs RQM4

Both methods achieve **identical accuracy** (4th-order convergence). The choice depends on:

| Factor | RQM4 | ETDRK4 |
|--------|------|--------|
| Speed | **2x faster** | Slower |
| Implementation | Operator splitting | Exponential integrator |
| Coefficients | Pre-computed | Contour integral |

**Recommendation**: Use **RQM4** for standard simulations. ETDRK4 is available as an alternative but offers no advantage over RQM4 for polymer SCFT.

## References

1. A. Ranjan, J. Qin, and D. C. Morse, **"Linear Response and Stability of Ordered Phases of Block Copolymer Melts"**, *Macromolecules*, **2008**, 41, 942-954.
   - RQM4 parameters for Richardson extrapolation

2. S. M. Cox and P. C. Matthews, **"Exponential Time Differencing for Stiff Systems"**, *J. Comput. Phys.*, **2002**, 176, 430-455.
   - ETDRK4 algorithm

3. P. Stasiak and M. W. Matsen, **"Efficiency of pseudo-spectral algorithms with Anderson mixing for the SCFT of periodic block-copolymer phases"**, *Eur. Phys. J. E*, **2011**, 34, 110.
   - Convergence analysis methodology

4. J. Song, Y. Liu, and R. Zhang, **"Exponential Time Differencing Schemes for Solving the Self-Consistent Field Equations of Polymers"**, *Chinese J. Polym. Sci.*, **2018**, 36, 488-496.
   - ETDRK4 for polymer field theory, benchmark methodology
