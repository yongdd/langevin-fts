# Parallel Propagator Scheduling

> **⚠️ Warning:** This document was generated with assistance from a large language model (LLM). While it is based on the referenced literature and the codebase, it may contain errors, misinterpretations, or inaccuracies. Please verify the equations and descriptions against the original references before relying on this document for research or implementation.

This document describes the `Scheduler` class, which creates an execution schedule for computing chain propagators in parallel across multiple GPU streams or CPU threads.

## Overview

For branched polymers and polymer mixtures, multiple propagators need to be computed. Some propagators depend on others (e.g., a junction propagator depends on its branch propagators), while independent propagators can run in parallel.

The Scheduler solves this scheduling problem by:
1. Analyzing propagator dependencies
2. Assigning propagators to streams
3. Creating a time-sliced execution plan

**Typical speedup**: 1.5-3x for branched polymers compared to single-stream execution.

## Scheduling Algorithm

The algorithm consists of four steps:

### Step 1: Group by Height

Propagators are grouped by their **dependency height**:

| Height | Description | Example |
|--------|-------------|---------|
| 0 | No dependencies (free-end) | `A+0`, `B+0` |
| 1 | Depends only on height-0 | `(A)C+0` |
| 2 | Depends on height-1 | `(AC)D+0` |

```
Height 0: A+0, B+0, C+0     (can all run in parallel)
Height 1: (A)D+0, (B)E+0    (can run after their dependencies)
Height 2: (DE)F+0           (runs after D and E complete)
```

### Step 2: Resolve Dependencies

For each propagator, compute when it can start (when all dependencies are resolved):

```
resolved_time[key] = max(end_time of all dependencies)
```

### Step 3: Greedy Stream Assignment

Process propagators level by level. For each propagator:
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

## Data Structures

### Input: ComputationEdge

The scheduler receives propagator information from `PropagatorComputationOptimizer`:

```cpp
struct ComputationEdge {
    std::string monomer_type;
    int max_n_segment;  // Number of contour steps
    std::vector<std::tuple<std::string, int, int>> deps;  // (key, n_segment, n_repeated)
};
```

### Output: Schedule

The schedule is a vector of time intervals, where each interval contains active jobs:

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

## Class Diagram

```
Scheduler
├── Private Members:
│   ├── job_assignment           - stream/time assignment per propagator
│   ├── dependency_resolved_time - when dependencies are resolved
│   ├── sorted_jobs              - propagators sorted by start time
│   ├── time_points              - discrete schedule change points
│   └── schedule                 - final execution schedule
│
├── Private Methods:
│   ├── group_by_height()        - group propagators by dependency depth
│   └── find_earliest_stream()   - find stream with minimum available time
│
└── Public Methods:
    ├── Scheduler(propagators, n_streams)  - constructor
    ├── get_schedule()                     - return computed schedule
    └── display(propagators)               - print schedule for debugging
```

## Usage Example

```cpp
#include "Scheduler.h"

// Get propagators from optimizer
auto& propagators = optimizer.get_computation_propagators();

// Create scheduler with 4 streams (typical for GPU)
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

## Example Output

For a star polymer with 3 arms (A, B, C) meeting at a junction:

```
=== Propagator Schedule ===
A+0:
    n_segment=10, start=0, end=10
B+0:
    n_segment=10, start=0, end=10
C+0:
    n_segment=10, start=0, end=10
(ABC)D+0:
    n_segment=5, start=10, end=15

=== Time Slices ===
Time 0-10:
    A+0: segments 0-10
    B+0: segments 0-10
    C+0: segments 0-10
Time 10-15:
    (ABC)D+0: segments 0-5
```

## Stream Count

Both CPU and GPU use multiple streams/threads to execute independent propagators in parallel according to the schedule.

| Platform | Default n_streams | Configuration |
|----------|------------------|---------------|
| GPU (CUDA) | 4 | Hardcoded in `CudaComputationContinuous` |
| CPU (FFTW) | 4 | `OMP_NUM_THREADS` environment variable |

### How Parallelization Works

```cpp
// CPU: OpenMP threads execute different propagators in parallel
#pragma omp parallel for num_threads(n_streams)
for (size_t job = 0; job < parallel_job->size(); job++) {
    // Each thread computes a different propagator
}

// GPU: CUDA streams execute different propagators concurrently
for (size_t job = 0; job < parallel_job->size(); job++) {
    cudaStreamSynchronize(streams[job % n_streams]);
    // Launch kernel on assigned stream
}
```

## Performance Considerations

### When Scheduling Helps

- **Block copolymers**: AB diblock has 2 propagators per height level (2x speedup)
- **Branched polymers**: Star, comb, dendritic have many independent branches (higher speedup)
- **Polymer mixtures**: Multiple polymer species computed simultaneously

### When Scheduling Has Less Impact

- **Single homopolymer**: Only 1 propagator per height level (no parallelism)
- **GPU with large grids**: Single propagator already saturates the GPU

## References

- `Scheduler.h/cpp`: Scheduler implementation
- `PropagatorComputationOptimizer.h/cpp`: Builds propagator dependency graph
- `CudaComputationContinuous.cu`: Executes schedule on GPU (continuous chains)
- `CudaComputationDiscrete.cu`: Executes schedule on GPU (discrete chains)
- `CpuComputationContinuous.cpp`: Executes schedule on CPU (continuous chains)
- `CpuComputationDiscrete.cpp`: Executes schedule on CPU (discrete chains)
- Chain propagator optimization: *J. Chem. Theory Comput.* **2025**, 21, 3676
