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
├── propagator_at_check_point[(key, n)]      # Segment checkpoints at √N intervals
├── propagator_half_steps_at_check_point     # Junction checkpoints
├── q_recal[0..√N]                           # Recomputation workspace (O(√N) size)
├── phi_block[key]                           # Block concentrations
└── q_pair[2]                                # Ping-pong buffers
```

**Memory usage**: GPU memory is nearly constant; checkpoints stored in host RAM.

### CUDA Benchmark Results

#### Bottle Brush Polymer

**Test Configuration**: Bottle Brush polymer (20 side chains, 42 blocks), discrete chain model, ds=0.01, total N=698, checkpoint_interval=27

| Grid | Std GPU | Red GPU | GPU Saved | Slowdown |
|------|---------|---------|-----------|----------|
| $32^3$ | 392 MB  | 250 MB  | 142 MB    | 3.1x     |
| $48^3$ | 856 MB  | 292 MB  | 564 MB    | 3.4x     |
| $64^3$ | 1490 MB | 362 MB  | 1128 MB   | 3.7x     |

#### AB Diblock Copolymer

**Test Configuration**: AB diblock copolymer (f=0.5), discrete chain model, varying N

**N = 40 segments** (checkpoint_interval = 7):

| Grid | Std GPU | Red GPU | GPU Saved | Slowdown |
|------|---------|---------|-----------|----------|
| $48^3$ | 206 MB  | 50 MB   | 156 MB (76%) | 2.4x  |
| $64^3$ | 434 MB  | 120 MB  | 314 MB (72%) | 2.3x  |

**N = 80 segments** (checkpoint_interval = 9):

| Grid | Std GPU | Red GPU | GPU Saved | Slowdown |
|------|---------|---------|-----------|----------|
| $32^3$ | 72 MB   | 14 MB   | 58 MB (81%)  | 2.7x  |
| $48^3$ | 286 MB  | 50 MB   | 236 MB (83%) | 2.6x  |
| $64^3$ | 592 MB  | 124 MB  | 468 MB (79%) | 2.8x  |

**N = 160 segments** (checkpoint_interval = 13):

| Grid | Std GPU | Red GPU | GPU Saved | Slowdown |
|------|---------|---------|-----------|----------|
| $32^3$ | 112 MB  | 14 MB   | 98 MB (88%)  | 3.0x  |
| $48^3$ | 446 MB  | 50 MB   | 396 MB (89%) | 3.2x  |
| $64^3$ | 912 MB  | 122 MB  | 790 MB (87%) | 3.5x  |

**Key observations**:
- GPU memory reduced by **~75-90%** depending on chain length and grid size
- Checkpoints stored in pinned host RAM for async transfers
- Computation **2.3-3.5x slower** due to recomputation from checkpoints
- Slowdown increases with chain length N (more recomputation needed)

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
├── propagator_at_check_point[(key, n)]      # Segment checkpoints at √N intervals
├── propagator_half_steps_at_check_point     # Junction checkpoints
├── q_recal[0..√N]                           # Recomputation workspace (O(√N) size)
├── phi_block[key]                           # Block concentrations
└── q_pair[2]                                # Ping-pong buffers
```

### CPU-MKL Benchmark Results

**Test Configuration**: Various polymer architectures, $32^3$ grid, single thread

**Continuous Chain Model**:

| Architecture | Std Time | Red Time | Slowdown |
|-------------|----------|----------|----------|
| Diblock (N=40) | 0.087s | 0.155s | 1.79x |
| Diblock (N=100) | 0.214s | 0.414s | 1.93x |
| Triblock (ABA) | 0.065s | 0.148s | 2.28x |
| 9-arm Star | 0.213s | 0.412s | 1.93x |
| Bottle Brush (10 SC) | 0.364s | 0.663s | 1.82x |
| Bottle Brush (20 SC) | 0.414s | 0.686s | 1.66x |

**Discrete Chain Model**:

| Architecture | Std Time | Red Time | Slowdown |
|-------------|----------|----------|----------|
| Diblock (N=40) | 0.028s | 0.049s | 1.72x |
| Diblock (N=100) | 0.071s | 0.130s | 1.84x |
| Triblock (ABA) | 0.023s | 0.047s | 2.10x |
| 9-arm Star | 0.072s | 0.131s | 1.82x |
| Bottle Brush (10 SC) | 0.156s | 0.231s | 1.48x |
| Bottle Brush (20 SC) | 0.204s | 0.271s | 1.33x |

**Key observations**:
- Slowdown varies by architecture (1.33x - 2.3x)
- Bottle brush polymers show lower slowdown (1.33-1.82x) due to block-based computation
- Discrete chains are generally faster than continuous chains

### CPU-MKL Memory Usage

#### Bottle Brush Polymer

**Test Configuration**: Bottle Brush polymer (20 side chains, 42 blocks), discrete chain model, ds=0.01, total N=698, checkpoint_interval=27

| Grid | Standard | Reduced | Saved | Slowdown |
|------|----------|---------|-------|----------|
| $32^3$ | 272 MB | 223 MB | 49 MB (18%) | 1.33x |
| $48^3$ | 634 MB | 468 MB | 165 MB (26%) | 1.41x |
| $64^3$ | 1336 MB | 946 MB | 390 MB (29%) | 1.39x |

#### AB Diblock Copolymer

**Test Configuration**: AB diblock copolymer (f=0.5), discrete chain model, varying N

**N = 40 segments** (checkpoint_interval = 7):

| Grid | Standard | Reduced | Saved | Slowdown |
|------|----------|---------|-------|----------|
| $32^3$ | 168 MB | 154 MB | 14 MB (8%)   | 1.60x |
| $48^3$ | 286 MB | 239 MB | 47 MB (16%)  | 1.51x |
| $64^3$ | 518 MB | 404 MB | 114 MB (22%) | 1.63x |

**N = 80 segments** (checkpoint_interval = 9):

| Grid | Standard | Reduced | Saved | Slowdown |
|------|----------|---------|-------|----------|
| $32^3$ | 187 MB | 156 MB | 30 MB (16%)  | 1.79x |
| $48^3$ | 354 MB | 252 MB | 102 MB (29%) | 1.85x |
| $64^3$ | 678 MB | 434 MB | 244 MB (36%) | 1.82x |

**N = 160 segments** (checkpoint_interval = 13):

| Grid | Standard | Reduced | Saved | Slowdown |
|------|----------|---------|-------|----------|
| $32^3$ | 229 MB | 160 MB | 69 MB (30%)  | 1.80x |
| $48^3$ | 490 MB | 260 MB | 230 MB (47%) | 1.77x |
| $64^3$ | 998 MB | 451 MB | 548 MB (55%) | 1.79x |

**Key observations**:
- Memory reduced by **~8-55%** depending on chain length and grid size
- Memory savings increase with chain length N (more propagators to checkpoint)
- Memory savings increase with grid size (larger arrays benefit more from checkpointing)
- Checkpoints stored at $\sqrt{N_{\text{total}}}$ intervals reduce propagator storage
- Computation **1.5-1.9x slower** due to recomputation

## Performance by Polymer Architecture

The slowdown in memory-saving mode varies with polymer architecture:

### CPU-MKL Slowdown

| Architecture Type | Continuous | Discrete | Notes |
|-------------------|------------|----------|-------|
| Linear chains (diblock) | 1.8-1.9x | 1.6-1.8x | Block-based recomputation |
| Triblock | 2.3x | 2.1x | Multiple blocks increase recomputation |
| Star polymers | 1.9x | 1.8x | Symmetric architecture |
| Bottle brush | 1.7-1.8x | 1.3-1.4x | Many junctions with shared checkpoints |

### CUDA Slowdown

| Architecture Type | Discrete | Notes |
|-------------------|----------|-------|
| Linear chains (diblock, N=40-160) | 2.3-3.5x | Increases with chain length |
| Bottle brush (N=698) | 3.1-3.7x | GPU-CPU transfer overhead |

**Why bottle brush polymers show lower slowdown on CPU**: Bottle brush polymers have many junction points that share checkpoints. The block-based computation algorithm efficiently reuses these checkpoints, minimizing redundant recomputation.

**CUDA vs CPU slowdown**: CUDA shows higher slowdown (2.3-3.7x) compared to CPU (1.3-1.9x) because of the additional GPU-host memory transfer overhead for checkpoints stored in pinned host memory. CUDA slowdown increases with chain length due to more checkpoint transfers.

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

## Manual Checkpoint Management

In memory-saving mode, you can manually add checkpoints at specific propagator positions using the `add_checkpoint()` method. This is useful when you need to access propagator values at specific contour positions using `get_chain_propagator()`.

### API Usage

```python
from polymerfts.propagator_solver import PropagatorSolver

# Create solver with memory-saving mode
solver = PropagatorSolver(
    nx=[64, 64, 64], lx=[4.0, 4.0, 4.0],
    ds=0.01, bond_lengths={'A': 1.0, 'B': 1.0},
    bc=['periodic']*6,
    chain_model='continuous', method='pseudospectral',
    platform='cpu-mkl', reduce_memory_usage=True
)

solver.add_polymer(1.0, [['A', 0.5, 0, 1], ['B', 0.5, 1, 2]])

# Add a checkpoint at step 30 for the first block's propagator
# This must be called BEFORE compute_propagators()
success = solver.add_checkpoint(polymer=0, v=0, u=1, n=30)
# Returns True if checkpoint was added, False if it already exists

# Now compute propagators - checkpoint at n=30 will be stored
solver.compute_propagators({'A': w_A, 'B': w_B})

# Access the propagator at checkpoint position
q_30 = solver.get_propagator(polymer=0, v=0, u=1, step=30)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `polymer` | Polymer index (0-based) |
| `v` | Starting vertex of the propagator direction |
| `u` | Ending vertex of the propagator direction |
| `n` | Contour step index (0 to n_segment for continuous, 1 to n_segment for discrete) |

### Notes

- **Call before `compute_propagators()`**: Checkpoints must be added before computing propagators
- **Standard mode**: Returns `False` in standard mode (checkpoints not needed)
- **Duplicate checkpoints**: Returns `False` if checkpoint already exists at that position
- **Memory cost**: Each checkpoint uses $M \times 8$ bytes (one grid array)

## Technical Implementation Details

### Checkpointing Algorithm

The checkpointing algorithm stores propagators only at strategic positions and recomputes intermediate values on-the-fly. For a propagator $q(\mathbf{r}, s)$ with $N$ segments:

1. **During propagator computation**:
   - Compute propagator step by step: $q(\mathbf{r}, s+\Delta s) = \Gamma[q(\mathbf{r}, s)]$
   - Store at checkpoint positions: segment 1, $N$, junctions, and every $\sqrt{N}$ steps
   - Discard intermediate values

2. **During concentration calculation** (block-based algorithm):
   - Divide the computation into blocks of size $k = \lceil\sqrt{N}\rceil$
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
1. Divides $n$ into blocks of size $\sqrt{N}$
2. For each block, loads the nearest checkpoint for $q$
3. Recomputes only the positions needed for that block
4. Advances $q^\dagger$ continuously using ping-pong buffers

This reduces the workspace from $O(N)$ to $O(\sqrt{N})$ arrays while minimizing redundant computation.

### CUDA-Specific Optimizations

- **Pinned memory** (`cudaMallocHost`): Enables async host-device transfers
- **Dual streams**: Overlap kernel execution with memory transfers
- **Minimal device allocation**: Fixed workspace regardless of chain length

### CPU-MKL-Specific Details

- **Block-based computation**: Processes concentration in blocks of $\sqrt{N}$ for efficient checkpoint reuse
- **$\sqrt{N}$ checkpoints**: Stores checkpoints at every $\sqrt{N}$ steps in addition to junctions
- **Single-threaded recomputation**: Minimizes memory during checkpoint reconstruction
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

1. J. He and Q. Wang, **"PSCF+: An Extended and Improved Open-Source Software Package for Polymer Self-Consistent Field Calculations"**, *Journal of Chemical Theory and Computation*, **2025**, 21, 9879-9889.
   - [SavMem.pdf](https://github.com/qwcsu/pscfplus/blob/master/doc/notes/SavMem.pdf): Detailed mathematical derivation of optimal checkpoint placement for memory-saving algorithm.

2. D. Yong, Y. Kim, S. Jo, D. Y. Ryu, and J. U. Kim, **"Order-to-Disorder Transition of Cylinder-Forming Block Copolymer Films Confined within Neutral Interfaces"**, *Macromolecules*, **2021**, 54, 11304-11317.
   - CUDA implementation with memory optimization techniques.
