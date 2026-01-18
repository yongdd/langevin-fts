# Memory Usage and Performance Guide

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the memory management strategies and performance characteristics of the polymer field theory simulation library for both CUDA (GPU) and CPU-MKL platforms.

For numerical method benchmarks (RQM4 vs CN-ADI2, convergence analysis), see [NumericalMethodsPerformance.md](NumericalMethodsPerformance.md).

## Overview

The library provides two memory modes controlled by the `reduce_memory` parameter:

| Mode | Parameter | Description |
|------|-----------|-------------|
| **Standard** | `reduce_memory=False` | Stores all propagators for maximum speed |
| **Memory-Saving** | `reduce_memory=True` | Stores only checkpoints, recomputes as needed |

```python
params = {
    "platform": "cuda",  # or "cpu-mkl"
    "reduce_memory": True,  # Enable memory-saving mode
    # ... other parameters
}
```

## Memory Scaling

Memory usage depends on:
- $M$: Total grid points (e.g., $64^3 = 262,144$)
- $N$: Total segments across all propagators
- $C$: Number of checkpoints (approximately $2\sqrt{N}$ for each propagator)

| Mode | Memory Complexity | Computation Complexity |
|------|-------------------|------------------------|
| Standard | $O(N \times M)$ | $O(N \times M)$ |
| Memory-Saving | $O(C \times M) \approx O(\sqrt{N} \times M)$ | $O(N \times \sqrt{N} \times M)$ |

## CUDA (GPU) Platform

### Standard Mode

All propagators are stored in **GPU device memory**:

```
GPU Device Memory:
├── d_propagator[key][n]           # All segment propagators
├── d_propagator_half_steps[key][n] # Junction half-bonds
├── d_phi_block[key]               # Block concentrations
└── Solver workspace               # FFT, Boltzmann factors
```

**Memory usage**: Scales linearly with chain length and polymer complexity.

### Memory-Saving Mode (Checkpointing)

Checkpoints stored in **pinned host memory**, minimal GPU workspace:

```
GPU Device Memory (Fixed ~10-20 arrays):
├── d_q_one[2]           # Current/next propagator (ping-pong)
├── d_q_block_v/u[2]     # Concentration computation
├── d_phi                # Temporary concentration
├── d_q_pair[2]          # Stress computation
└── Solver workspace     # FFT, Boltzmann factors

Pinned Host Memory (Checkpoints):
├── propagator_at_check_point[(key, n)]      # Segment checkpoints at 2√N intervals
├── propagator_half_steps_at_check_point     # Junction checkpoints
├── q_recal[0..2√N]                          # Recomputation workspace
├── phi_block[key]                           # Block concentrations
└── q_pair[2]                                # Ping-pong buffers
```

**Memory usage**: GPU memory is nearly constant; checkpoints stored in host RAM.

## CPU-MKL Platform

### Standard Mode

All propagators stored in **system RAM**:

```
System RAM:
├── propagator[key][n]              # All segment propagators
├── propagator_half_steps[key][n]   # Junction half-bonds
├── phi_block[key]                  # Block concentrations
└── Solver workspace                # FFT plans, Boltzmann factors
```

### Memory-Saving Mode (Checkpointing)

Same checkpointing strategy as CUDA, but all in system RAM:

```
System RAM (Reduced):
├── propagator_at_check_point[(key, n)]      # Segment checkpoints at 2√N intervals
├── propagator_half_steps_at_check_point     # Junction checkpoints
├── q_recal[0..2√N]                          # Recomputation workspace
├── phi_block[key]                           # Block concentrations
└── q_pair[2]                                # Ping-pong buffers
```

## Benchmark Results

### Test Configuration

- **Polymer**: AB Diblock Copolymer (f = 0.5, χN = 12)
- **Chain model**: Discrete
- **N = 500** contour steps (ds = 1/500)
- **Checkpoint interval**: $\lceil 2\sqrt{500} \rceil = 45$
- **Hardware**: NVIDIA A10 GPU (23 GB), Intel MKL CPU (8 threads)

### CUDA (GPU) Results

| Mode | Grid | GPU Memory | Host Memory | Time/iter | Total (5 iter) |
|------|------|------------|-------------|-----------|----------------|
| **Standard** | $64^3$ | 2.0 GB | 181 MB | **0.146 s** | 3.4 s |
| **Standard** | $80^3$ | 3.9 GB | 58 MB | **0.359 s** | 6.2 s |
| **Standard** | $96^3$ | 6.8 GB | 151 MB | **0.645 s** | 10.6 s |
| **Reduce** | $96^3$ | 0.5 GB | 918 MB | **2.44 s** | 15.1 s |
| **Reduce** | $128^3$ | 1.3 GB | 2.2 GB | **5.40 s** | 34.5 s |
| **Reduce** | $160^3$ | 2.5 GB | 4.2 GB | **12.0 s** | 71.4 s |
| **Reduce** | $200^3$ | 4.9 GB | 8.3 GB | **25.5 s** | 145.8 s |

**Key observations**:
- GPU memory reduced by **~92%** with memory-saving mode
- Checkpoints stored in pinned host RAM for async transfers
- Computation **3.8x slower** due to recomputation from checkpoints
- Standard mode for $200^3$ would require ~60 GB GPU memory (infeasible)

### CPU-MKL Results

| Mode | Grid | CPU Memory | Time/iter | Total (5 iter) |
|------|------|------------|-----------|----------------|
| **Standard** | $64^3$ | 2.1 GB | **1.66 s** | 8.8 s |
| **Standard** | $80^3$ | 4.2 GB | **4.02 s** | 20.5 s |
| **Standard** | $96^3$ | 7.3 GB | **7.26 s** | 36.8 s |
| **Standard** | $112^3$ | 11.5 GB | **11.8 s** | 59.9 s |
| **Reduce** | $96^3$ | 0.5 GB | **27.3 s** | 82.5 s |
| **Reduce** | $128^3$ | 1.3 GB | **65.8 s** | 198.7 s |
| **Reduce** | $160^3$ | 2.5 GB | **147.7 s** | 444.5 s |
| **Reduce** | $200^3$ | 4.9 GB | **378.3 s** | 1137 s |

**Key observations**:
- Memory reduced by **~93%** with memory-saving mode
- Computation **3.8x slower** due to recomputation
- Measured memory closely matches theoretical predictions

### Performance Comparison

| Grid | CUDA Std | CUDA Red | CPU Std | CPU Red | CUDA Speedup |
|------|----------|----------|---------|---------|--------------|
| $64^3$ | 0.15 s | - | 1.66 s | - | **11.4x** |
| $80^3$ | 0.36 s | - | 4.02 s | - | **11.2x** |
| $96^3$ | 0.65 s | 2.44 s | 7.26 s | 27.3 s | **11.2x** |
| $128^3$ | - | 5.40 s | - | 65.8 s | **12.2x** |
| $160^3$ | - | 12.0 s | - | 147.7 s | **12.3x** |
| $200^3$ | - | 25.5 s | - | 378.3 s | **14.8x** |

### Memory Savings Analysis

At $96^3$ grid (common grid sizes for comparison):

| Platform | Standard | Reduce Memory | Savings | Time Overhead |
|----------|----------|---------------|---------|---------------|
| **CUDA** | 6.8 GB GPU | 0.5 GB GPU + 0.9 GB host | **93%** | 3.8x |
| **CPU-MKL** | 7.3 GB | 0.5 GB | **93%** | 3.8x |

## Memory Estimation

### Estimating Standard Mode Memory

For a discrete chain system:

$$\text{Memory (bytes)} \approx N_{\text{propagators}} \times (N+1) \times M_{\text{grid}} \times 8$$

**Example**: AB diblock (2 propagators), N=500, $64^3$ grid

$$\text{Memory} \approx 2 \times 501 \times 262144 \times 8 \approx 2.0 \text{ GB}$$

### Estimating Memory-Saving Mode

$$\text{Memory} \approx (N_{\text{checkpoints}} + 2\sqrt{N}) \times M_{\text{grid}} \times 8$$

where $N_{\text{checkpoints}} \approx N / (2\sqrt{N}) = \sqrt{N}/2$ per propagator.

### Theoretical Memory Requirements (N=500)

| Grid | Standard (Diblock) | Reduce Memory | Savings |
|------|-------------------|---------------|---------|
| $64^3$ | 2.0 GB | 0.14 GB | 93% |
| $96^3$ | 6.8 GB | 0.55 GB | 92% |
| $128^3$ | 16.1 GB | 1.3 GB | 92% |
| $160^3$ | 31.4 GB | 2.5 GB | 92% |
| $200^3$ | **61.5 GB** | **4.9 GB** | **92%** |

**Key insight**: For N=500, memory savings are consistently ~92% because:
$$\text{Savings} \approx 1 - \frac{O(\sqrt{N})}{O(N)} = 1 - \frac{1}{\sqrt{N}} \approx 1 - \frac{1}{22.4} \approx 95.5\%$$

### Target Configuration: N=500, M=200³

| Metric | Standard Mode | Reduce Memory Mode |
|--------|---------------|-------------------|
| **Memory (Diblock)** | 61.5 GB | 4.9 GB |
| **Memory Savings** | - | **92%** |
| **CUDA Time/iter** | N/A (OOM) | **25.5 s** |
| **CPU Time/iter** | N/A (OOM) | **378.3 s** |
| **CUDA vs CPU** | - | **14.8x faster** |

## Choosing the Right Mode

### Use Standard Mode (`reduce_memory=False`) when:
- Memory is not a constraint
- Maximum performance is required
- Running small to medium grids ($\leq 96^3$ for N=500)
- Short polymer chains (N < 200)

### Use Memory-Saving Mode (`reduce_memory=True`) when:
- **CUDA**: GPU memory is limited but host RAM is available
- **CPU**: System RAM is constrained
- Large grids ($> 96^3$)
- Long polymer chains (N > 200)
- Complex architectures with many blocks
- Running multiple simulations concurrently

### Quick Reference

| N | Max Grid (Standard, 24GB GPU) | Recommended Mode |
|---|-------------------------------|------------------|
| 100 | ~$160^3$ | Standard |
| 200 | ~$128^3$ | Standard or Reduce |
| 500 | ~$96^3$ | **Reduce Memory** |
| 1000 | ~$64^3$ | **Reduce Memory** |

## Technical Implementation Details

### Checkpointing Algorithm

The checkpointing algorithm stores propagators only at strategic positions and recomputes intermediate values on-the-fly. For a propagator $q(\mathbf{r}, s)$ with $N$ segments:

1. **During propagator computation**:
   - Compute propagator step by step: $q(\mathbf{r}, s+\Delta s) = \Gamma[q(\mathbf{r}, s)]$
   - Store at checkpoint positions: segment 1, $N$, junctions, and every $2\sqrt{N}$ steps
   - Discard intermediate values

2. **During concentration calculation** (block-based algorithm):
   - Divide the computation into blocks of size $k = \lceil 2\sqrt{N}\rceil$
   - For each block:
     - Load the left propagator from the nearest checkpoint
     - Recompute forward to required positions within the block
     - Advance the right propagator step by step
     - Accumulate concentration contributions: $\phi(\mathbf{r}) \propto q(\mathbf{r}, s) \cdot q^\dagger(\mathbf{r}, N-s)$
   - This reduces redundant recomputation by processing contiguous ranges

### Block-Based Computation

The concentration calculation requires products of forward and backward propagators:
$$\phi(\mathbf{r}) = \sum_{n=0}^{N} c_n \cdot q(\mathbf{r}, n) \cdot q^\dagger(\mathbf{r}, N-n)$$

Instead of recomputing all $q$ values at once, the block-based algorithm:
1. Divides $n$ into blocks of size $2\sqrt{N}$
2. For each block, loads the nearest checkpoint for $q$
3. Recomputes only the positions needed for that block
4. Advances $q^\dagger$ continuously using ping-pong buffers

This reduces the workspace from $O(N)$ to $O(\sqrt{N})$ arrays while minimizing redundant computation.

### CUDA-Specific Optimizations

- **Pinned memory** (`cudaMallocHost`): Enables async host-device transfers
- **Dual streams**: Overlap kernel execution with memory transfers
- **Minimal device allocation**: Fixed workspace regardless of chain length

### CPU-MKL-Specific Details

- **Block-based computation**: Processes concentration in blocks of $2\sqrt{N}$ for efficient checkpoint reuse
- **$2\sqrt{N}$ checkpoints**: Stores checkpoints at optimal intervals
- **Single-threaded recomputation**: Minimizes memory during checkpoint reconstruction
- **MKL FFT**: Intel-optimized FFT for spectral operations

## Troubleshooting

### Out of GPU Memory
```python
# Enable memory-saving mode
params["reduce_memory"] = True
```

### Slow Performance with Memory-Saving
- Consider if standard mode fits in memory
- Check if polymer has high symmetry (star, dendritic) for better performance
- Reduce grid size if possible

### Verifying Memory Mode is Active

In debug builds, look for these messages:
```
# Standard mode
"--------- Discrete Chain Solver, GPU Version ---------"

# Memory-saving mode
"--------- Discrete Chain Solver, GPU Memory Saving Version (Checkpointing) ---------"
```

## References

1. J. He and Q. Wang, **"PSCF+: An Extended and Improved Open-Source Software Package for Polymer Self-Consistent Field Calculations"**, *Journal of Chemical Theory and Computation*, **2025**, 21, 9879-9889.
   - [SavMem.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/SavMem.pdf): Detailed mathematical derivation of optimal checkpoint placement for memory-saving algorithm.

2. D. Yong, Y. Kim, S. Jo, D. Y. Ryu, and J. U. Kim, **"Order-to-Disorder Transition of Cylinder-Forming Block Copolymer Films Confined within Neutral Interfaces"**, *Macromolecules*, **2021**, 54, 11304-11317.
   - CUDA implementation with memory optimization techniques.
