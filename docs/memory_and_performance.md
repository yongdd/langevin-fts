# Memory Usage and Performance Guide

This document describes the memory management strategies and performance characteristics of the polymer field theory simulation library for both CUDA (GPU) and CPU-MKL platforms.

## Overview

The library provides two memory modes controlled by the `reduce_memory_usage` parameter:

| Mode | Parameter | Description |
|------|-----------|-------------|
| **Standard** | `reduce_memory_usage=False` | Stores all propagators for maximum speed |
| **Memory-Saving** | `reduce_memory_usage=True` | Stores only checkpoints, recomputes as needed |

```python
params = {
    "platform": "cuda",  # or "cpu-mkl"
    "reduce_memory_usage": True,  # Enable memory-saving mode
    # ... other parameters
}
```

## Memory Scaling

Memory usage depends on:
- $M$: Total grid points (e.g., $64^3 = 262,144$)
- $N$: Total segments across all propagators
- $C$: Number of checkpoints (approximately $\sqrt{N}$ for each propagator)

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
├── propagator_at_check_point[(key, n)]      # Segment checkpoints
├── propagator_half_steps_at_check_point     # Junction checkpoints
├── q_recal[0..max_segment]                  # Recomputation workspace
├── phi_block[key]                           # Block concentrations
└── q_pair[2]                                # Ping-pong buffers
```

**Memory usage**: GPU memory is nearly constant; checkpoints stored in host RAM.

### CUDA Benchmark Results

**Test Configuration**: Bottle Brush polymer (20 side chains, 42 blocks)

| Grid | Std GPU | Red GPU | GPU Saved | Std RAM | Red RAM | Slowdown |
|------|---------|---------|-----------|---------|---------|----------|
| $32^3$ | ~144 MB | 12 MB   | 132 MB    | ~2 MB   | ~84 MB  | 2.2x     |
| $48^3$ | 572 MB  | 48 MB   | 524 MB    | ~0 MB   | 380 MB  | 3.4x     |
| $64^3$ | 1182 MB | 120 MB  | 1062 MB   | ~0 MB   | 794 MB  | 3.3x     |

**Key observations**:
- GPU memory reduced by **~90%**
- Checkpoints moved to host RAM (pinned memory for async transfers)
- Computation **2-4x slower** due to recomputation

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
├── propagator_at_check_point[(key, n)]      # Segment checkpoints only
├── propagator_half_steps_at_check_point     # Junction checkpoints
├── q_recal[0..max_segment]                  # Recomputation workspace
├── phi_block[key]                           # Block concentrations
└── q_pair[2]                                # Ping-pong buffers
```

### CPU-MKL Benchmark Results

**Test Configuration**: Various polymer architectures, $32^3$ grid

| Architecture | Blocks | Std Time | Red Time | Slowdown |
|-------------|--------|----------|----------|----------|
| Diblock (AB) | 2 | 1.51s | 4.71s | 3.1x |
| Triblock (ABA) | 3 | 1.37s | 3.09s | 2.3x |
| 3-arm Star | 6 | 0.98s | 1.74s | 1.8x |
| 9-arm Star | 18 | 0.48s | 0.73s | 1.5x |
| Bottle Brush (10 SC) | 22 | 3.43s | 8.65s | 2.5x |
| Bottle Brush (20 SC) | 42 | 5.09s | 10.95s | 2.2x |

**Larger grid (Bottle Brush 20 SC)**:

| Grid | Std RAM | Red RAM | RAM Saved | Slowdown |
|------|---------|---------|-----------|----------|
| $48^3$ | 106 MB  | 75 MB   | 31 MB     | 1.9x     |

**Key observations**:
- Memory savings visible for larger grids
- Slowdown varies by architecture (1.5x - 3.1x)
- Star polymers benefit from propagator aggregation optimization

## Performance by Polymer Architecture

The slowdown in memory-saving mode varies with polymer architecture:

| Architecture Type | Typical Slowdown | Notes |
|-------------------|------------------|-------|
| Linear chains | 3-4x | No propagator reuse |
| Star polymers | 1.5-2x | Propagator aggregation reduces recomputation |
| Bottle brush | 2-3x | Mixed linear and branched characteristics |
| Dendritic | 1.5-2.5x | High propagator reuse at junctions |

**Why star polymers are faster**: The propagator computation optimizer detects equivalent propagator calculations in symmetric architectures and computes them only once. In memory-saving mode, this means fewer checkpoints and less recomputation.

## Choosing the Right Mode

### Use Standard Mode (`reduce_memory_usage=False`) when:
- Memory is not a constraint
- Maximum performance is required
- Running small to medium grids ($\leq 48^3$)
- Short polymer chains

### Use Memory-Saving Mode (`reduce_memory_usage=True`) when:
- **CUDA**: GPU memory is limited but host RAM is available
- **CPU**: System RAM is constrained
- Large grids ($64^3$+)
- Long polymer chains (high $N$)
- Complex architectures with many blocks
- Running multiple simulations concurrently

## Memory Estimation

### Estimating Standard Mode Memory

For a discrete chain system:

$$\text{Memory (bytes)} \approx N_{\text{segments}} \times M_{\text{grid}} \times 8 \times 2$$

where the factor of 2 accounts for forward and backward propagators.

**Example**: Bottle brush (42 blocks, ~90 segments each), $64^3$ grid

$$\text{Memory} \approx 42 \times 90 \times 262144 \times 8 \times 2 \approx 1.6 \text{ GB}$$

### Estimating Memory-Saving Mode

$$\text{GPU Memory (CUDA)} \approx 20 \times M_{\text{grid}} \times 8 \approx \text{constant}$$

$$\text{Host Memory} \approx N_{\text{checkpoints}} \times M_{\text{grid}} \times 8$$

where $N_{\text{checkpoints}} \approx 2\text{-}5$ per propagator (endpoints + junctions).

## Technical Implementation Details

### Checkpointing Algorithm

The checkpointing algorithm stores propagators only at strategic positions and recomputes intermediate values on-the-fly. For a propagator $q(\mathbf{r}, s)$ with $N$ segments:

1. **During propagator computation**:
   - Compute propagator step by step: $q(\mathbf{r}, s+\Delta s) = \Gamma[q(\mathbf{r}, s)]$
   - Store only at checkpoint positions (segment 1, $N$, and junctions)
   - Discard intermediate values

2. **During concentration calculation**:
   - Load propagator from nearest checkpoint
   - Recompute forward to required position
   - Compute concentration contribution: $\phi(\mathbf{r}) \propto q(\mathbf{r}, s) \cdot q^\dagger(\mathbf{r}, N-s)$
   - Repeat for each segment

### CUDA-Specific Optimizations

- **Pinned memory** (`cudaMallocHost`): Enables async host-device transfers
- **Dual streams**: Overlap kernel execution with memory transfers
- **Minimal device allocation**: Fixed workspace regardless of chain length

### CPU-MKL-Specific Details

- **Single-threaded recomputation**: Minimizes memory during checkpoint reconstruction
- **Same algorithm**: Identical checkpointing strategy as CUDA version
- **MKL FFT**: Intel-optimized FFT for spectral operations

## Troubleshooting

### Out of GPU Memory
```python
# Enable memory-saving mode
params["reduce_memory_usage"] = True
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

1. G. H. Fredrickson, **"Equilibrium Theory of Inhomogeneous Polymers"**, Oxford University Press, 2006.
   - [SavMem.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/SavMem.pdf): Detailed mathematical derivation of optimal checkpoint placement for memory-saving algorithm.

2. D. Yong and J. U. Kim, **"Accelerating Langevin Field-Theoretic Simulation of Polymers with Deep Learning"**, *Journal of Chemical Theory and Computation*, **2025**, 21, 3676-3689.
   - Propagator computation optimization for branched polymers.

3. S. M. Park, Y. O. Lee, and G. H. Fredrickson, **"Parallel Algorithm for Numerical Self-Consistent Field Theory Simulations of Block Copolymer Structure"**, *Journal of Chemical Physics*, **2019**, 150, 234901.
   - Discrete chain model (N-1 bond model) implementation.

4. D. Yong, Y. Kim, S. Jo, D. Y. Ryu, and J. U. Kim, **"Order-to-Disorder Transition of Cylinder-Forming Block Copolymer Films Confined within Neutral Interfaces"**, *Macromolecules*, **2021**, 54, 11304-11317.
   - CUDA implementation with memory optimization techniques.
