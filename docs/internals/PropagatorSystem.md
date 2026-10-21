# Propagator System

This document describes the propagator identification system and parallel scheduling algorithm used in the polymer field theory simulation library.

## Table of Contents

1. [Overview](#1-overview)
2. [Propagator Identification](#2-propagator-identification)
3. [Initial Conditions](#3-initial-conditions)
4. [Scheduling Algorithm](#4-scheduling-algorithm)
5. [Data Structures](#5-data-structures)
6. [Execution](#6-execution)
7. [Usage Examples](#7-usage-examples)
8. [Performance Considerations](#8-performance-considerations)
9. [References](#9-references)

---

## 1. Overview

For branched polymers and polymer mixtures, multiple propagators need to be computed. Some propagators depend on others (e.g., a junction propagator depends on its branch propagators), while independent propagators can run in parallel.

The propagator system:
1. **Identifies** propagators using a string-based key format
2. **Optimizes** by detecting and reusing equivalent computations
3. **Schedules** parallel execution across multiple streams/threads

**Measured speedup** (A10 GPU, ds = 0.01, time of one propagator and concentration computation): computing independent propagators on 4 CUDA streams instead of 1 gives 1.9x for the bottle-brush of `examples/scft/BottleBrushLamella3D.py` (50 side chains, discrete chains) on a 32³ grid but 1.1x on 64³, and 1.0–1.1x for the 9-arm star of `examples/scft/Star9ArmsGyroid.py` and for a comb with 9 side chains on either grid. Merging equivalent propagators (item 2, `aggregate_propagator_computation`) saves more: with one stream it gives 3.6x for the bottle-brush and 2.0x for the comb, and nothing for the star, whose identical arms are computed once either way.

---

## 2. Propagator Identification

### 2.1 Chain Model Differences

| Aspect | Continuous Chain | Discrete Chain |
|--------|------------------|----------------|
| Block length constraint | Arbitrary (non-integer multiples of ds allowed) | Must be integer multiple of ds |
| Numbers in aggregated keys | `length_index` (0 for sliced) | `n_segment` directly |
| Minimum sliced segments | 0 | 1 |

### 2.2 ContourLengthMapping

The `ContourLengthMapping` class manages the relationship between contour lengths and their indices (primarily for continuous chains).

| Property | Description |
|----------|-------------|
| `length_index` | Index for unique contour lengths (1-based; 0 reserved for sliced) |
| `ds_index` | Index for unique ds values (0-based) |
| `n_segment` | Number of segments = round(contour_length / ds) |

**Example Mapping** (blocks of 0.3, 0.35, 0.5 with ds=0.1):

| contour_length | length_index | n_segment | local_ds | ds_index |
|----------------|--------------|-----------|----------|----------|
| 0.0 (sliced) | 0 | 0 | - | - |
| 0.3 | 1 | 3 | 0.1 | 1 |
| 0.35 | 2 | 4 | 0.0875 | 0 |
| 0.5 | 3 | 5 | 0.1 | 1 |

### 2.3 Code Format: DKN

The code format is used during propagator code generation and encodes the polymer topology.

**Structure:**
```
[Dependencies]MonomerTypeN
```

Where N is the block's `length_index` for both chain models (the keys are generated from the `ContourLengthMapping`).

**Examples:**

| Code | Description |
|------|-------------|
| `A3` | A-type propagator with N=3 (free end) |
| `B2` | B-type propagator with N=2 (free end) |
| `(A3)B2` | B-type propagator depending on A3 |
| `(A3B2)C4` | C-type propagator at junction of A3 and B2 |
| `(A3B2:2)C4` | C-type with two identical B2 branches |

**Dependency Notation:**
- **Parentheses `()`**: Junction point merging multiple branches
- **Colon `:`**: Repetition count (e.g., `B2:3` = three identical B2 branches)
- **Curly braces `{}`**: Custom initial condition (e.g., `{wall}A3`)

### 2.4 Key Format: DK+M

The key format is derived from the code format and includes the ds_index for computation lookup.

**Structure:**
```
[Dependencies]MonomerType+ds_index
```

**Conversion Examples:**

| Code (DKN) | Key (DK+M) | Notes |
|------------|------------|-------|
| `A3` | `A+0` | ds_index=0 for N=3 |
| `(A3)B2` | `(A3)B+0` | Only outer level has +M |
| `(A3B2)C4` | `(A3B2)C+0` | Inner deps keep N format |

### 2.5 Aggregated Key Format

When identical propagator branches are detected, they are merged to reduce redundancy.

**Structure:**
```
[dep_code1,dep_code2,...]MonomerType+ds_index
```

> **Note**: Inside brackets, dependencies use the DK format (without the original length_index N) plus the sliced length_index (always 0, continuous chains) or the sliced n_segment (discrete chains). The outer key uses DK+M format where M is the ds_index.

**Examples:**

**Continuous chain aggregation:**
```
Input propagators:
  (D4)B+0: n_segment_right=4
  (E4)B+0: n_segment_right=4

After slicing (n_segment=0, length_index=0):
  (D4)B+0: n_segment_right=0
  (E4)B+0: n_segment_right=0

Aggregated key: [(D4)B0,(E4)B0]B+0
                      ^      ^
                  length_index=0 (for continuous chains)
```

**Discrete chain aggregation:**
```
Input propagators:
  (D4)B+0: n_segment_right=4
  (E4)B+0: n_segment_right=4

After slicing (minimum_n_segment=1):
  (D4)B+0: n_segment_right=1
  (E4)B+0: n_segment_right=1

Aggregated key: [(D4)B1,(E4)B1]B+0
                      ^      ^
                  n_segment=1 (for discrete chains)
```

---

## 3. Initial Conditions

The dependency part **D** of the propagator code determines the initial condition. The formulas below are for continuous chains, which start at $`q(\mathbf{r}, 0)`$; the discrete case follows at the end of this section.

### Free End (Empty D)

For a propagator starting from a free chain end (e.g., `A+0`):

```math
q(\mathbf{r}, 0) = 1
```

### Junction Point (D with Parentheses)

For a propagator at a junction (e.g., `(A3B2)C+0`):

```math
q(\mathbf{r}, 0) = \prod_{i} \left[ q_i(\mathbf{r}, N_i) \right]^{n_i}
```

where $`n_i`$ is the repetition count (from `:n` notation, default 1).

### Aggregated Computation (D with Square Brackets)

For aggregated propagators (e.g., `[(A3)B0,(C3)B0]B+0`; the aggregated propagators all continue the same block, so they share its monomer type):

```math
q(\mathbf{r}, 0) = \sum_{i} n_i \cdot q_i(\mathbf{r}, N_i)
```

### Custom Initial Condition (D with Curly Braces)

For user-specified initial conditions (e.g., `{wall}A+0`):

```math
q(\mathbf{r}, 0) = q_{\text{init}}[\text{name}](\mathbf{r})
```

### Discrete Chains

Discrete chains start at $`q(\mathbf{r}, 1)`$: at a free end or grafting point it is the value above times $`e^{-w(\mathbf{r})\Delta s}`$; at a junction the product is taken over the half-bond states $`q_i(\mathbf{r}, N_i + 1/2)`$ of the branches and followed by a half-bond step and $`e^{-w(\mathbf{r})\Delta s}`$; an aggregated sum is formed from junction values (then a half-bond step and $`e^{-w(\mathbf{r})\Delta s}`$) or from sliced segments (then a full bond step) (`ComputationDiscrete::compute_propagators`).

---

## 4. Scheduling Algorithm

### Step 1: Group by Height

Propagators are grouped by their **dependency height**:

| Height | Description | Example |
|--------|-------------|---------|
| 0 | No dependencies (free-end) | `A+0`, `B+0` |
| 1 | Depends only on height-0 | `(A3)C+0` |
| 2 | Depends on height-1 | `((A3)C4)D+0` |

```
Height 0: A+0, B+0, C+0           (can all run in parallel)
Height 1: (A3)D+0, (B3)E+0        (can run after their dependencies)
Height 2: ((A3)D4(B3)E4)F+0       (runs after D and E complete)
```

### Step 2: Resolve Dependencies

For each propagator, compute when it can start:

```
resolved_time[key] = max over dependencies of (start_time[dep] + max(n_segment needed from dep, 1))
```

A propagator can thus start before a dependency ends, when it needs only the first segments of that dependency (e.g. a free-end propagator shared by blocks of different lengths).

### Step 3: Greedy Stream Assignment

Process propagators level by level:
1. Sort by resolved time (earlier ready = scheduled first)
2. Assign to the stream with earliest available time
3. Update stream's available time

```
Stream 0: |--A+0--|--D+0--|----F+0----|
Stream 1: |--B+0--|--E+0--|
Stream 2: |--C+0--|
```

### Step 4: Time Slicing

Divide execution into time intervals where the set of active jobs is constant:

```
Time 0-10:  A+0 (stream 0), B+0 (stream 1), C+0 (stream 2)
Time 10-15: A+0 continues, B+0 continues
Time 15-20: D+0 (stream 0), E+0 (stream 1)
Time 20-30: F+0 (stream 0)
```

---

## 5. Data Structures

### Input: ComputationEdge

```cpp
struct ComputationEdge {
    int max_n_segment;  // Number of contour steps
    std::string monomer_type;
    std::vector<std::tuple<std::string, int, int>> deps;  // (key, n_segment, n_repeated)
    int height;         // Height of the propagator in the dependency tree (used by the scheduler)
    std::set<int> junction_ends;  // Indices where additional half-bond-step computation
                                  // is required (discrete chains)
};
```

### Output: Schedule

```cpp
// schedule[t] = jobs active during time interval t
// Each job: (propagator_key, segment_from, segment_to)
std::vector<std::vector<std::tuple<std::string, int, int>>> schedule;
```

### Internal Variables

| Variable | Type | Description |
|----------|------|-------------|
| `job_assignment` | `map<string, tuple<int,int,int>>` | Maps key → (stream, start, end) |
| `dependency_resolved_time` | `map<string, int>` | When each propagator can start |
| `sorted_jobs` | `vector<tuple<string, int>>` | Jobs sorted by start time |
| `time_points` | `vector<int>` | Discrete times where schedule changes |

---

## 6. Execution

The schedule is executed once for both platforms, in `ComputationContinuous::compute_propagators`
and `ComputationDiscrete::compute_propagators` (`src/common`). The propagators of one group run
concurrently on OpenMP threads; thread `i` passes stream index `i` to the backend, which runs the
work on that OpenMP thread (CPU) or queues it on CUDA stream `i` (GPU). After each group the
backend synchronizes, so the next group sees every propagator it depends on.

```cpp
for (const auto& parallel_job : sc->get_schedule())
{
    ParallelErrors errors;
    #pragma omp parallel for num_threads(n_streams)
    for (size_t job = 0; job < parallel_job.size(); job++)
    {
        errors.run([&]
        {
            const int STREAM = omp_get_thread_num();
            // initial condition, then backend->advance_propagator(STREAM, q[n], q[n+1], ...)
        });
    }
    backend->synchronize();   // CPU: nothing to do; CUDA: cudaDeviceSynchronize
    errors.rethrow();         // an exception inside the loop reaches the caller
}
```

Number of streams:

- CPU: `OMP_NUM_THREADS` when it is set, otherwise 4.
- CUDA: `min(OMP_NUM_THREADS, MAX_STREAMS)` when `OMP_NUM_THREADS` is set, otherwise `MAX_STREAMS` (= 4, defined in `CudaCommon.h`).
- Reduce-memory mode: one stream; the propagators are computed one at a time in schedule order.

---

## 7. Usage Examples

### Basic Usage

```cpp
#include "Scheduler.h"

// Get propagators from optimizer
auto& propagators = optimizer.get_computation_propagators();

// Create scheduler with 4 streams
int n_streams = 4;
Scheduler scheduler(propagators, n_streams);

// Get the schedule
auto& schedule = scheduler.get_schedule();

// Execute each time interval
for (size_t t = 0; t < schedule.size(); t++) {
    for (const auto& [key, seg_from, seg_to] : schedule[t]) {
        // Compute propagator 'key' from segment seg_from to seg_to
    }
    // Synchronize streams at interval boundary
}

// Display schedule for debugging
scheduler.display(propagators);
```

### Example Output

For a star polymer with 3 arms A, B and C of contour length 1.0 meeting at a junction (continuous chains, ds = 0.1, no aggregation, 4 streams):

```
=== Propagator Schedule ===
C+0:
    n_segment=10, start=0, end=10
B+0:
    n_segment=10, start=0, end=10
A+0:
    n_segment=10, start=0, end=10
(B1C1)A+0:
    n_segment=10, start=10, end=20
(A1C1)B+0:
    n_segment=10, start=10, end=20
(A1B1)C+0:
    n_segment=10, start=10, end=20

=== Time Slices ===
Time 0-10:
    C+0: segments 0-10
    B+0: segments 0-10
    A+0: segments 0-10
Time 10-20:
    (A1C1)B+0: segments 0-10
    (A1B1)C+0: segments 0-10
    (B1C1)A+0: segments 0-10
```

---

## 8. Performance Considerations

### When Scheduling Helps

- **Small grids with several independent propagators**: on an A10 at 32³, two streams make an AB diblock iteration 1.5x faster and four streams make the bottle-brush computation 1.9x faster
- **Polymer mixtures**: Multiple polymer species computed simultaneously

### When Scheduling Has Less Impact

- **Single homopolymer**: Only 1 propagator per height level (no parallelism)
- **GPU with large grids**: Single propagator already saturates the GPU (on 64³, 1.0-1.1x for the bottle-brush, star and comb above; 6-9% for an AB diblock on 64³-128³)

### Key Comparison

Keys are compared using `ComparePropagatorKey`:
1. First by height (lower height first)
2. Then in reverse (descending) lexicographic order (`str1 > str2`) — deterministic, but descending

---

## 9. References

- `PropagatorCode.h/cpp`: Code generation and parsing
- `PropagatorComputationOptimizer.h/cpp`: Optimization using keys
- `PropagatorAggregator.h/cpp`: Aggregated key generation
- `Scheduler.h/cpp`: Parallel computation scheduling
- `ContourLengthMapping.h/cpp`: Index mapping utilities
- D. Yong and J. U. Kim, "Dynamic Programming for Chain Propagator Computation of Branched Block Copolymers in Polymer Field Theory Simulations," *J. Chem. Theory Comput.* **2025**, 21, 3676.
