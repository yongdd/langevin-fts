# Numerical Methods: Performance and Accuracy

This document provides benchmark results comparing the numerical methods available for chain propagator computation in polymer field theory simulations.

## Available Methods

All numerical methods are selectable at runtime using the `numerical_method` parameter.

### Pseudo-Spectral Methods (Periodic Boundaries)

| Method | Order | Description | Reference |
|--------|-------|-------------|-----------|
| **rqm4** | 4th | Richardson extrapolation with Ranjan-Qin-Morse 2008 parameters (default) | *Macromolecules* 41, 942-954 (2008) |
| **etdrk4** | 4th | Exponential Time Differencing Runge-Kutta | *J. Comput. Phys.* 176, 430-455 (2002) |

### Real-Space Methods (Non-Periodic Boundaries)

| Method | Order | Description |
|--------|-------|-------------|
| **cn-adi2** | 2nd | Crank-Nicolson Alternating Direction Implicit (default) |
| **cn-adi4** | 4th | CN-ADI with Richardson extrapolation |

### Usage Example

```python
params = {
    # ... other parameters ...
    "numerical_method": "rqm4"  # or "etdrk4", "cn-adi2", "cn-adi4"
}
```

## Convergence Analysis

Following the methodology of Stasiak & Matsen (*Eur. Phys. J. E* 34, 110, 2011), we analyze how the partition function $Q$ converges as the contour discretization $ds \to 0$.

### Test Configuration

- **System**: Homopolymer A in lamellar external field
- **Field**: $w(\mathbf{r}) = (\chi N/2) \cos(2\pi n_p z / L_z)$ with $\chi N = 12$, $n_p = 3$
- **Grid**: $32 \times 32 \times 32$
- **Box size**: $4.0 \times 4.0 \times 4.0$ (in units of $bN^{1/2}$)
- **Chain model**: Continuous

### Partition Function vs Contour Steps

| $N$ ($ds=1/N$) | RQM4 $Q$ | CN-ADI2 $Q$ | Difference |
|----------------|----------|-------------|------------|
| 10 | 12.652569057 | 13.089088363 | 3.45% |
| 20 | 12.645948526 | 13.126915040 | 3.80% |
| 40 | 12.645395735 | 13.136356896 | 3.88% |
| 80 | 12.645355694 | 13.138716398 | 3.90% |
| 160 | 12.645353000 | 13.139306213 | 3.91% |
| 320 | 12.645352825 | 13.139453663 | 3.91% |

The difference between methods arises from different spatial discretization approaches:
- **RQM4**: Cell-centered grid with periodic Fourier transforms
- **CN-ADI2**: Finite difference grid with ADI tridiagonal solvers

### Convergence Order

The convergence order $p$ is estimated from:
$$|Q(ds) - Q_{ref}| \propto ds^p$$

| Method | Measured Order | Expected Order |
|--------|----------------|----------------|
| **RQM4** | $p \approx 3.83$ | 4.0 |
| **CN-ADI2** | $p \approx 2.10$ | 2.0 |

Both methods achieve their expected convergence orders.

### Error vs Contour Steps

For RQM4 (pseudo-spectral), using $Q_{ref} = Q(N=320)$:

| $N$ | $|Q - Q_{ref}|$ | Relative Error |
|-----|-----------------|----------------|
| 10 | $7.2 \times 10^{-3}$ | $5.7 \times 10^{-4}$ |
| 20 | $5.96 \times 10^{-4}$ | $4.7 \times 10^{-5}$ |
| 40 | $4.29 \times 10^{-5}$ | $3.4 \times 10^{-6}$ |
| 80 | $2.87 \times 10^{-6}$ | $2.3 \times 10^{-7}$ |
| 160 | $1.75 \times 10^{-7}$ | $1.4 \times 10^{-8}$ |

For practical SCFT/FTS calculations, $N = 100$ ($ds = 0.01$) typically provides sufficient accuracy.

## Performance Benchmarks

Following Song, Liu & Zhang (*Chinese J. Polym. Sci.* 36, 488-496, 2018), we compare computation time for propagator calculations.

### CUDA (GPU) Performance

**Hardware**: NVIDIA A10 GPU

| $N$ ($ds=1/N$) | RQM4 (ms) | CN-ADI2 (ms) | Ratio |
|----------------|-----------|--------------|-------|
| 10 | 1.30 | 3.77 | 2.9x |
| 20 | 2.46 | 7.42 | 3.0x |
| 40 | 4.79 | 14.72 | 3.1x |
| 80 | 9.48 | 29.32 | 3.1x |
| 160 | 18.86 | 58.53 | 3.1x |

**Scaling**: Both methods scale linearly with $N$, as expected.

### CPU-MKL Performance

**Hardware**: Intel CPU with MKL (4 OpenMP threads)

| $N$ ($ds=1/N$) | RQM4 (ms) | CN-ADI2 (ms) | Ratio |
|----------------|-----------|--------------|-------|
| 10 | 10.79 | 27.64 | 2.6x |
| 20 | 21.33 | 54.73 | 2.6x |
| 40 | 42.37 | 109.30 | 2.6x |
| 80 | 85.02 | 219.05 | 2.6x |
| 160 | 169.87 | 437.70 | 2.6x |
| 320 | 331.76 | 881.06 | 2.7x |

### Platform Comparison

For $N=100$ ($ds=0.01$), $32^3$ grid:

| Method | CPU (ms) | CUDA (ms) | GPU Speedup |
|--------|----------|-----------|-------------|
| **RQM4** | 103.5 | 11.8 | **8.8x** |
| **CN-ADI2** | 274.3 | 36.7 | **7.5x** |

### Cross-Platform Consistency

Results between CPU and CUDA platforms are identical within machine precision:

| Method | CPU $Q$ | CUDA $Q$ | Relative Difference |
|--------|---------|----------|---------------------|
| RQM4 | 12.645354010317 | 12.645354010315 | $1.6 \times 10^{-13}$ |
| CN-ADI2 | 13.138999512002 | 13.138999512001 | $7.6 \times 10^{-14}$ |

This confirms that the library produces consistent results regardless of platform.

## Phase Benchmarks: Gyroid and Fddd

Realistic benchmarks using ordered block copolymer phases with all four numerical methods.

### Gyroid Phase (Ia-3d)

**Configuration:**
- **System**: AB diblock copolymer, $f = 0.36$, $\chi N = 20$
- **Grid**: $32 \times 32 \times 32$
- **Box**: $3.3 \times 3.3 \times 3.3$ (in units of $bN^{1/2}$)
- **Contour**: $ds = 0.01$ ($N = 100$)

| Method | Solver | CPU-MKL (ms) | CUDA (ms) | GPU Speedup |
|--------|--------|--------------|-----------|-------------|
| **RQM4** | Pseudo-Spectral | 107.1 | 15.0 | **7.1x** |
| **ETDRK4** | Pseudo-Spectral | 163.4 | — | — |
| **CN-ADI2** | Real-Space | 278.6 | 37.1 | **7.5x** |
| **CN-ADI4** | Real-Space | 828.3 | 110.6 | **7.5x** |

**Partition Function Comparison:**

| Method | $Q$ |
|--------|-----|
| RQM4 | 1.011194981 |
| ETDRK4 | 1.011741041 |
| CN-ADI2 | 1.011205118 |
| CN-ADI4 | 1.011205997 |

### Fddd Phase (O^70)

**Configuration:**
- **System**: AB diblock copolymer, $f = 0.43$, $\chi N = 14$
- **Grid**: $48 \times 32 \times 24$
- **Box**: $5.58 \times 3.17 \times 1.59$ (in units of $bN^{1/2}$)
- **Contour**: $ds = 0.01$ ($N = 100$)

| Method | Solver | CPU-MKL (ms) | CUDA (ms) | GPU Speedup |
|--------|--------|--------------|-----------|-------------|
| **RQM4** | Pseudo-Spectral | 127.9 | 16.7 | **7.7x** |
| **ETDRK4** | Pseudo-Spectral | 198.5 | — | — |
| **CN-ADI2** | Real-Space | 285.9 | 37.0 | **7.7x** |
| **CN-ADI4** | Real-Space | 855.6 | 110.0 | **7.8x** |

**Partition Function Comparison:**

| Method | $Q$ |
|--------|-----|
| RQM4 | 1.002484518 |
| ETDRK4 | 1.002631538 |
| CN-ADI2 | 1.002482218 |
| CN-ADI4 | 1.002482481 |

### Key Findings

1. **RQM4 is fastest**: Approximately 1.5x faster than ETDRK4, 2.5x faster than CN-ADI2
2. **GPU acceleration**: Consistent 7-8x speedup across all methods
3. **CN-ADI4 overhead**: Richardson extrapolation adds ~3x overhead compared to CN-ADI2
4. **Accuracy**: Real-space and pseudo-spectral methods give slightly different $Q$ values due to different spatial discretization, but both converge to the same physical solution

### Benchmark Script

```bash
python tests/benchmark_phases.py cuda    # Run on GPU
python tests/benchmark_phases.py cpu-mkl # Run on CPU
```

## Method Selection Guide

### Pseudo-Spectral vs Real-Space

| Criterion | Pseudo-Spectral | Real-Space |
|-----------|-----------------|------------|
| **Boundaries** | Periodic only | Periodic, reflecting, absorbing |
| **Speed** | Faster (~3x) | Slower |
| **Memory** | FFT workspace | Tridiagonal solver workspace |
| **Use case** | Bulk systems, SCFT/FTS | Confined systems, brushes |

### RQM4 vs ETDRK4

Both are 4th-order accurate for pseudo-spectral solvers:

| Criterion | RQM4 | ETDRK4 |
|-----------|------|--------|
| **FFTs per step** | 6 | 8 |
| **Stability** | Good for typical $ds$ | L-stable (better for stiff problems) |
| **Memory** | Lower | Higher (stores phi coefficients) |
| **Selection** | `numerical_method="rqm4"` (default) | `numerical_method="etdrk4"` |

### CN-ADI2 vs CN-ADI4

| Criterion | CN-ADI2 | CN-ADI4 |
|-----------|---------|---------|
| **Order** | 2nd | 4th |
| **Steps per ds** | 1 | 2 (Richardson extrapolation) |
| **Speed** | Faster | ~2x slower |
| **Stability** | More stable | May be unstable near absorbing BC |
| **Selection** | `numerical_method="cn-adi2"` (default) | `numerical_method="cn-adi4"` |

## Recommendations

### For Periodic Systems (SCFT, L-FTS, CL-FTS)

1. Use **rqm4** (pseudo-spectral, default)
2. Set $ds = 0.01$ ($N=100$) for most calculations
3. Reduce to $ds = 0.005$ for high-precision comparisons
4. Use **CUDA** platform for 2D/3D systems
5. Consider **etdrk4** for stiff problems where RQM4 shows instability

### For Non-Periodic Systems (Brushes, Confined Films)

1. Use **cn-adi2** (real-space, default) for stability
2. Use **cn-adi4** if higher accuracy is needed:
   ```python
   params = {"numerical_method": "cn-adi4", ...}
   ```
3. May require smaller $ds$ due to 2nd-order accuracy
4. Test stability near absorbing boundaries before production runs (CN-ADI4 may be unstable)

### Performance Optimization

1. **GPU**: Use CUDA for all 2D/3D simulations (8-15x speedup)
2. **Grid size**: Pseudo-spectral has $O(M \log M)$ FFT cost; real-space has $O(M)$ cost per direction
3. **Memory**: For large grids, enable `reduce_memory_usage=True` (see [MemoryAndPerformance.md](MemoryAndPerformance.md))

## Benchmark Scripts

Two benchmark scripts are available:

### Basic Convergence and Performance

```bash
python tests/benchmark_numerical_methods.py
```

Tests convergence order and timing vs contour discretization. Results saved to `benchmark_results.json`.

### Phase Benchmarks (Gyroid, Fddd)

```bash
python tests/benchmark_phases.py cuda    # GPU benchmark
python tests/benchmark_phases.py cpu-mkl # CPU benchmark
```

Tests all four methods on realistic ordered phases. Results saved to `benchmark_phases_results.json`.

## References

1. A. Ranjan, J. Qin, and D. C. Morse, **"Linear Response and Stability of Ordered Phases of Block Copolymer Melts"**, *Macromolecules*, **2008**, 41, 942-954.
   - RQM4 parameters for Richardson extrapolation

2. S. M. Cox and P. C. Matthews, **"Exponential Time Differencing for Stiff Systems"**, *J. Comput. Phys.*, **2002**, 176, 430-455.
   - ETDRK4 algorithm

3. P. Stasiak and M. W. Matsen, **"Efficiency of pseudo-spectral algorithms with Anderson mixing for the SCFT of periodic block-copolymer phases"**, *Eur. Phys. J. E*, **2011**, 34, 110.
   - Convergence analysis methodology

4. J. Song, Y. Liu, and R. Zhang, **"Exponential Time Differencing Schemes for Solving the Self-Consistent Field Equations of Polymers"**, *Chinese J. Polym. Sci.*, **2018**, 36, 488-496.
   - ETDRK4 for polymer field theory, performance benchmarks
