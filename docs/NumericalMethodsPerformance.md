# Numerical Methods: Performance and Accuracy

This document provides benchmark results comparing the numerical methods available for chain propagator computation in polymer field theory simulations.

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

## Performance Benchmarks

### Benchmark Configuration

- **Platform**: NVIDIA A10 GPU (CUDA)
- **Phases tested**: Gyroid (Ia-3d, f=0.36) and Fddd (O^70, f=0.43)
- **Polymer**: AB diblock copolymer
- **SCFT iterations**: 100 per test
- **Date**: 2026-01-12

### Gyroid Phase Results

#### Time per Iteration vs Contour Steps (N)

*Grid size: 32³, χN = 15-25*

| Method | N=50 | N=100 | N=200 | N=400 |
|--------|------|-------|-------|-------|
| **RQM4** | 14.1 ms | 25.5 ms | 49.3 ms | 96.8 ms |
| **ETDRK4** | 27.0 ms | 52.0 ms | 102.2 ms | 202.7 ms |
| **CN-ADI2** | 38.5 ms | 75.3 ms | 148.7 ms | 295.5 ms |
| **CN-ADI4** | 111.5 ms | 221.3 ms | 440.7 ms | 879.6 ms |

**Key findings:**
- RQM4 is **2.0x faster** than ETDRK4
- RQM4 is **2.9x faster** than CN-ADI2
- RQM4 is **8.6x faster** than CN-ADI4
- Time scales linearly with N (doubling N doubles time)

#### Time per Iteration vs Grid Size

*N=100 (ds=0.01), χN = 15*

| Method | 24³ | 32³ | 48³ | 64³ |
|--------|-----|-----|-----|-----|
| **RQM4** | 17.2 ms | 25.6 ms | 57.4 ms | 96.9 ms |
| **ETDRK4** | 34.4 ms | 51.8 ms | 98.6 ms | 151.3 ms |
| **CN-ADI2** | 55.0 ms | 75.2 ms | 119.5 ms | 163.2 ms |
| **CN-ADI4** | 162.1 ms | 221.4 ms | 349.1 ms | 468.9 ms |

**Key findings:**
- Pseudo-spectral methods (RQM4, ETDRK4) scale as O(M log M)
- CN-ADI methods scale as O(M) but with higher constant overhead
- At large grids (64³), RQM4 maintains significant advantage

### Fddd Phase Results

#### Time per Iteration vs Contour Steps (N)

*Grid size: 84×48×24, χN = 12-16*

| Method | N=50 | N=100 | N=200 | N=400 |
|--------|------|-------|-------|-------|
| **RQM4** | 27.6 ms | 49.7 ms | 94.8 ms | 185.0 ms |
| **ETDRK4** | 43.2 ms | 80.1 ms | 155.3 ms | 305.3 ms |
| **CN-ADI2** | 55.0 ms | 103.3 ms | 201.9 ms | 399.2 ms |
| **CN-ADI4** | 155.7 ms | 300.8 ms | 596.8 ms | 1189.1 ms |

![Time per iteration vs contour steps](figures/ds_variation.png)

#### Time per Iteration vs Grid Size

*N=100 (ds=0.01), χN = 12*

| Method | 56×32×16 | 84×48×24 | 112×64×32 |
|--------|----------|----------|-----------|
| **RQM4** | 25.0 ms | 50.2 ms | 89.0 ms |
| **ETDRK4** | 49.1 ms | 80.5 ms | 140.3 ms |
| **CN-ADI2** | 69.0 ms | 103.5 ms | 148.8 ms |
| **CN-ADI4** | 202.9 ms | 301.0 ms | 428.4 ms |

![Time per iteration vs grid size](figures/nx_variation.png)

### Performance Summary

#### Speedup Relative to CN-ADI2 (at N=100)

| Method | Gyroid (32³) | Fddd (84×48×24) |
|--------|--------------|-----------------|
| **RQM4** | 2.9x faster | 2.1x faster |
| **ETDRK4** | 1.4x faster | 1.3x faster |
| **CN-ADI4** | 2.9x slower | 2.9x slower |

![Method comparison](figures/method_comparison.png)

#### χN Independence

Time per iteration is essentially **independent of χN** for all methods:
- Gyroid: <1% variation across χN = 15, 20, 25
- Fddd: <2% variation across χN = 12, 14, 16

## Accuracy Analysis

### Free Energy Convergence

The free energy (F) was measured after 100 SCFT iterations for each method and contour discretization.

#### Gyroid Phase (χN = 15)

| Method | N=50 | N=100 | N=200 | N=400 |
|--------|------|-------|-------|-------|
| **RQM4** | -0.116745 | -0.116746 | -0.116746 | -0.116746 |
| **CN-ADI4** | -0.117272 | -0.117272 | -0.117272 | -0.117272 |
| **CN-ADI2** | -0.117179 | -0.117250 | -0.117267 | -0.117271 |

#### Fddd Phase (χN = 14)

| Method | N=50 | N=100 | N=200 | N=400 |
|--------|------|-------|-------|-------|
| **RQM4** | -0.156641 | -0.156641 | -0.156641 | -0.156641 |
| **CN-ADI4** | -0.156854 | -0.156854 | -0.156854 | -0.156854 |
| **CN-ADI2** | -0.156310 | -0.156631 | -0.156799 | -0.156840 |

**Key findings:**
- **RQM4** and **CN-ADI4** show excellent convergence - free energy is independent of ds
- **CN-ADI2** shows 2nd-order convergence behavior (F improves as ds decreases)
- All 4th-order methods (RQM4, CN-ADI4) achieve consistent F values even at coarse discretization (N=50)

### Contour Discretization Order

| Method | Error Scaling | Convergence Order |
|--------|---------------|-------------------|
| **RQM4** | $O(\Delta s^4)$ | 4 |
| **ETDRK4** | $O(\Delta s^4)$ | 4 |
| **CN-ADI2** | $O(\Delta s^2)$ | 2 |
| **CN-ADI4** | $O(\Delta s^4)$ | 4 |

For the same accuracy, RQM4/ETDRK4 require fewer contour steps than CN-ADI2, making them more efficient for high-precision calculations.

## Method Recommendations

### When to Use Each Method

| Use Case | Recommended Method | Reason |
|----------|-------------------|--------|
| Standard SCFT/FTS (periodic BC) | **RQM4** | Fastest, 4th-order accurate |
| Alternative pseudo-spectral | **ETDRK4** | Similar accuracy, ~2x slower |
| Non-periodic boundaries | **CN-ADI2** | Supports absorbing/reflecting BC |
| High-precision confined systems | **CN-ADI4** | 4th-order with non-periodic BC |

### Performance Summary

For periodic boundary conditions (standard SCFT):
1. **RQM4** is the best choice for most applications
2. **ETDRK4** is a valid alternative with similar accuracy
3. **CN-ADI** methods are not recommended for periodic BC due to lower efficiency

For non-periodic boundary conditions:
1. **CN-ADI2** for moderate accuracy requirements
2. **CN-ADI4** when higher accuracy is needed

## References

1. A. Ranjan, J. Qin, and D. C. Morse, **"Linear Response and Stability of Ordered Phases of Block Copolymer Melts"**, *Macromolecules*, **2008**, 41, 942-954.
   - RQM4 parameters for Richardson extrapolation

2. S. M. Cox and P. C. Matthews, **"Exponential Time Differencing for Stiff Systems"**, *J. Comput. Phys.*, **2002**, 176, 430-455.
   - ETDRK4 algorithm

3. P. Stasiak and M. W. Matsen, **"Efficiency of pseudo-spectral algorithms with Anderson mixing for the SCFT of periodic block-copolymer phases"**, *Eur. Phys. J. E*, **2011**, 34, 110.
   - Convergence analysis methodology

4. J. Song, Y. Liu, and R. Zhang, **"Exponential Time Differencing Schemes for Solving the Self-Consistent Field Equations of Polymers"**, *Chinese J. Polym. Sci.*, **2018**, 36, 488-496.
   - ETDRK4 for polymer field theory, performance benchmarks
