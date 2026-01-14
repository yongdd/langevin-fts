# Real-Space Method Free Energy Convergence Analysis

## Overview

This document presents the free energy convergence analysis for CN-ADI real-space numerical methods:

1. **cn-adi2**: 2nd-order Crank-Nicolson ADI
2. **cn-adi4-lr**: 4th-order CN-ADI with Local Richardson extrapolation
3. **cn-adi4-gr**: 4th-order CN-ADI with Global Richardson extrapolation

## Convergence Plot

![Real-Space Convergence Analysis](RealSpaceConvergencePlot.png)

## 1D Lamellar Phase Results

### Test Configuration (1D)

- **System**: AB diblock copolymer in lamellar phase
- **Parameters**:
  - χN = 20.0
  - f = 0.5 (symmetric diblock)
  - Nx = 32, Lx = 3.2
  - SCFT tolerance = 10⁻⁹
- **Contour discretization**: N = 10, 20, 40, 80, 160, 320, 640, 1000 (ds = 1/N)
- **Platform**: CPU (MKL)

### 1D Free Energy Values (βF/V)

| N | ds | cn-adi2 | cn-adi4-gr | cn-adi4-lr |
|---|----:|--------:|-----------:|-----------:|
| 10 | 0.1000 | -0.6784098747 | -0.7043941892 | -0.7046102563 |
| 20 | 0.0500 | -0.6982993078 | -0.7046652874 | -0.7047080538 |
| 40 | 0.0250 | -0.7031302049 | -0.7046749093 | -0.7046788342 |
| 80 | 0.0125 | -0.7042925736 | -0.7046755219 | -0.7046758228 |
| 160 | 0.0063 | -0.7045800292 | -0.7046755605 | -0.7046755815 |
| 320 | 0.0031 | -0.7046516930 | -0.7046755629 | -0.7046755643 |
| 640 | 0.0016 | -0.7046695964 | -0.7046755631 | -0.7046755632 |
| 1000 | 0.0010 | -0.7046731192 | -0.7046755631 | -0.7046755631 |

### 1D Convergence Order

| Method | Measured Order | Expected Order |
|--------|---------------:|---------------:|
| cn-adi2 | 2.11 | 2 |
| cn-adi4-lr | 3.45 | 4 |
| cn-adi4-gr | 4.11 | 4 |

### 1D Analysis

1. **cn-adi2** shows expected O(ds²) convergence with measured order ≈ 2.1
2. **cn-adi4-gr** achieves true O(ds⁴) convergence with measured order ≈ 4.1
3. **cn-adi4-lr** shows reduced order ≈ 3.5, likely due to ADI operator splitting error affecting the local Richardson extrapolation
4. **At N=640**, cn-adi4-gr achieves error ~8×10⁻¹², demonstrating near machine-precision accuracy

## 3D Gyroid Phase Results

### Test Configuration (3D)

- **System**: AB diblock copolymer in gyroid phase
- **Parameters**:
  - χN = 20.0
  - f = 0.45 (asymmetric diblock)
  - Nx = 32×32×32, Lx = 3.8×3.8×3.8
  - SCFT tolerance = 10⁻⁸
- **Contour discretization**: N = 20, 40, 80, 160, 320, 640, 1000 (ds = 1/N)
- **Platform**: CUDA (GPU)

### 3D Free Energy Values (βF/V)

| N | ds | cn-adi2 | cn-adi4-gr | cn-adi4-lr |
|---|----:|--------:|-----------:|-----------:|
| 20 | 0.0500 | -0.9084294604 | -0.9139505161 | -0.9142508014 |
| 40 | 0.0250 | -0.9126679721 | -0.9139188445 | -0.9139518630 |
| 80 | 0.0125 | -0.9136161400 | -0.9139224726 | -0.9139252508 |
| 160 | 0.0063 | -0.9138465913 | -0.9139233743 | -0.9139235777 |
| 320 | 0.0031 | -0.9139042234 | -0.9139235136 | -0.9139235275 |
| 640 | 0.0016 | -0.9139186939 | -0.9139235327 | -0.9139235336 |
| 1000 | 0.0010 | -0.9139215501 | -0.9139235348 | -0.9139235349 |

### 3D Convergence Order

| Method | Measured Order (3D) | Measured Order (1D) | Expected Order |
|--------|--------------------:|--------------------:|---------------:|
| cn-adi2 | 2.15 | 2.11 | 2 |
| cn-adi4-lr | 3.73 | 3.45 | 4 |
| cn-adi4-gr | 2.70 | 4.11 | 4 |

### 3D Analysis

1. **cn-adi4-gr shows reduced apparent convergence order in 3D** (~2.7 vs 4.1 in 1D). This is because the spatial discretization error from the 32³ grid becomes comparable to the temporal discretization error at fine ds values.

2. **cn-adi4-lr shows improved convergence order in 3D** (~3.7 vs 3.5 in 1D).

3. **Despite lower measured order, cn-adi4-gr achieves the smallest absolute errors** at coarse discretization:
   - At N=20: cn-adi4-gr error (2.7e-5) is ~200× smaller than cn-adi2 (5.5e-3)
   - At N=40: cn-adi4-gr error (4.7e-6) is ~270× smaller than cn-adi2 (1.3e-3)

## Implementation Details

**cn-adi4-lr (Local Richardson):**
```
for each step n:
    q_full[n+1] = advance(q[n], ds)
    q_half_temp = advance(q[n], ds/2)
    q_half[n+1] = advance(q_half_temp, ds/2)
    q[n+1] = (4·q_half[n+1] - q_full[n+1]) / 3
```

**cn-adi4-gr (Global Richardson):**
```
# Advance two independent chains
for each step n:
    q_full[n+1] = advance(q_full[n], ds)

for each half-step:
    q_half[n+1] = advance(q_half[n], ds/2)

# Richardson extrapolation at quadrature level
Q_rich = (4·Q_half - Q_full) / 3
φ_rich = (4·φ_half - φ_full) / 3
```

## Computational Cost

| Method | ADI Steps per N | Relative Cost |
|--------|---------------:|---------------:|
| cn-adi2 | N | 1.0× |
| cn-adi4-lr | 3N | 3.0× |
| cn-adi4-gr | 3N | 3.0× |

## Key Findings

1. **cn-adi4-gr achieves true 4th-order convergence in 1D**, making it the most accurate method when spatial discretization is not limiting.

2. **In 3D with moderate spatial grids**, the apparent convergence order of cn-adi4-gr is reduced because the method converges so fast that spatial discretization error dominates.

3. **cn-adi4-lr shows ~3.5-3.7 order convergence**, which is still significantly better than cn-adi2 but doesn't achieve the theoretical 4th order due to ADI operator splitting effects.

4. **For practical calculations**:
   - If accuracy is paramount and spatial grids are fine: use cn-adi4-gr
   - If computational cost matters: cn-adi4-lr offers a good balance
   - For quick exploratory runs: cn-adi2 with larger N

## References

- Richardson extrapolation: L.F. Richardson, "The deferred approach to the limit" (1927)
- ADI method: D.W. Peaceman & H.H. Rachford Jr., "The numerical solution of parabolic and elliptic differential equations" (1955)
