# Non-Monotonic Free Energy Convergence (Resolved)

This document describes a historical issue where SCFT free energy convergence became non-monotonic for certain values of the contour discretization parameter Ns. **This issue has been fixed** by using consistent local_ds values in both concentration normalization and Boltzmann factor computation.

## Historical Problem Description

When solving SCFT for block copolymers, the free energy F should converge smoothly as the contour discretization Ns increases (i.e., as ds = 1/Ns decreases). However, for specific values of Ns, the free energy exhibited large deviations from the expected convergence trend.

### Previously Affected Cases

The problem occurred when **f × Ns is a half-integer** (e.g., 4.5, 7.5, 10.5, ...), where f is the volume fraction of one block.

For f = 0.375 (A-block fraction), the problematic Ns values included:
- Ns = 12 (f×Ns = 4.5)
- Ns = 20 (f×Ns = 7.5)
- Ns = 28 (f×Ns = 10.5)
- Ns = 36 (f×Ns = 13.5)

## Root Cause

The issue stemmed from two separate bugs related to how contour discretization handles segment counts for each block.

### Segment Count Calculation

For a block with contour length L and global step size ds = 1/Ns:
```
n_segments = round(L / ds) = round(L × Ns)
```

For a diblock with fractions f and (1-f):
- A-block segments: n_A = round(f × Ns)
- B-block segments: n_B = round((1-f) × Ns)

### The Rounding Problem

When f × Ns is a half-integer:
- Both f × Ns and (1-f) × Ns are half-integers
- Standard rounding (round-half-away-from-zero) rounds both UP
- **Total segments = n_A + n_B > Ns**

Example for f = 0.375, Ns = 20:
- f × Ns = 7.5 → rounds to 8
- (1-f) × Ns = 12.5 → rounds to 13
- Total = 21 ≠ 20

### Why the Deviation Was Large (Before Fix)

Two bugs caused the mismatch:

**Bug 1: Concentration Normalization**

The concentration (phi) computation used **global ds** for normalization:
```cpp
norm = (molecules->get_ds() * volume_fraction / alpha) / Q;
```

While the propagator computation used **local_ds** (= contour_length / n_segment) for Boltzmann factors.

**Bug 2: Boltzmann Factor Reset**

The RQM4 solver's `update_laplacian_operator()` was called twice:
1. First in the solver constructor (correctly setting up local_ds values)
2. Again in the computation class constructor (resetting everything to global_ds)

The second call to `Pseudo::update()` reset `ds_values[1]` to global_ds and recomputed `boltz_bond` with the wrong value, undoing the local_ds setup.

## The Fix

The fix addressed both bugs:

### Fix 1: Concentration Normalization

Use local_ds (= alpha/n_segment_total) instead of global_ds (= 1/Ns):

```cpp
// Use local_ds = alpha/n_segment_total instead of global ds = 1/Ns
Polymer& pc = this->molecules->get_polymer(p);
T norm = (pc.get_volume_fraction()/pc.get_n_segment_total()*n_repeated)/this->single_polymer_partitions[p];
```

### Fix 2: Override update_laplacian_operator() in RQM4 Solvers

Override `update_laplacian_operator()` in `CpuSolverPseudoRQM4` and `CudaSolverPseudoRQM4` to re-register local_ds values after the base class resets them:

```cpp
void CpuSolverPseudoRQM4<T>::update_laplacian_operator()
{
    // Call base class implementation (updates Pseudo with global_ds)
    CpuSolverPseudoBase<T>::update_laplacian_operator();

    // Re-register local_ds values for each block
    const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
    int n_unique_ds = mapping.get_n_unique_ds();

    for (int ds_idx = 1; ds_idx <= n_unique_ds; ++ds_idx)
    {
        double local_ds = mapping.get_ds_from_index(ds_idx);
        this->pseudo->add_ds_value(ds_idx, local_ds);
    }

    // Finalize Pseudo to compute boltz_bond with correct local_ds
    this->pseudo->finalize_ds_values();
}
```

## Verification Results

Comprehensive tests were run for Ns = 10 to 40 with f = 0.375 and χN = 18. The key validation is that **when two Ns values produce the same segment counts, they must produce identical free energies**.

### Test Results Table

| Ns | f×Ns | n_A | n_B | Total | Free Energy F | Note |
|---:|-----:|----:|----:|------:|--------------:|------|
| 10 | 3.75 | 4 | 6 | 10 | -0.43678920 |  |
| 11 | 4.12 | 4 | 7 | 11 | -0.43402335 |  |
| 12 | 4.50 | 5 | 8 | 13 | -0.43331494 | half-integer |
| 13 | 4.88 | 5 | 8 | 13 | -0.43331494 |  |
| 14 | 5.25 | 5 | 9 | 14 | -0.43342758 |  |
| 15 | 5.62 | 6 | 9 | 15 | -0.43471746 |  |
| 16 | 6.00 | 6 | 10 | 16 | -0.43232649 |  |
| 17 | 6.38 | 6 | 11 | 17 | -0.43330480 |  |
| 18 | 6.75 | 7 | 11 | 18 | -0.43272504 |  |
| 19 | 7.12 | 7 | 12 | 19 | -0.43209884 |  |
| 20 | 7.50 | 8 | 13 | 21 | -0.43196052 | half-integer |
| 21 | 7.88 | 8 | 13 | 21 | -0.43196052 |  |
| 22 | 8.25 | 8 | 14 | 22 | -0.43213546 |  |
| 23 | 8.62 | 9 | 14 | 23 | -0.43264286 |  |
| 24 | 9.00 | 9 | 15 | 24 | -0.43171256 |  |
| 25 | 9.38 | 9 | 16 | 25 | -0.43225654 |  |
| 26 | 9.75 | 10 | 16 | 26 | -0.43194998 |  |
| 27 | 10.12 | 10 | 17 | 27 | -0.43168942 |  |
| 28 | 10.50 | 11 | 18 | 29 | -0.43164887 | half-integer |
| 29 | 10.88 | 11 | 18 | 29 | -0.43164887 |  |
| 30 | 11.25 | 11 | 19 | 30 | -0.43176552 |  |
| 31 | 11.62 | 12 | 19 | 31 | -0.43203699 |  |
| 32 | 12.00 | 12 | 20 | 32 | -0.43154914 |  |
| 33 | 12.38 | 12 | 21 | 33 | -0.43188572 |  |
| 34 | 12.75 | 13 | 21 | 34 | -0.43170010 |  |
| 35 | 13.12 | 13 | 22 | 35 | -0.43155713 |  |
| 36 | 13.50 | 14 | 23 | 37 | -0.43154173 | half-integer |
| 37 | 13.88 | 14 | 23 | 37 | -0.43154173 |  |
| 38 | 14.25 | 14 | 24 | 38 | -0.43162234 |  |
| 39 | 14.62 | 15 | 24 | 39 | -0.43179148 |  |
| 40 | 15.00 | 15 | 25 | 40 | -0.43149061 |  |

### Same-Segment Pair Verification

All cases where consecutive Ns values have the same segment counts now produce **identical** free energies:

| Ns pair | Total segments | |F(Ns) - F(Ns+1)| |
|---------|----------------|-------------------|
| 12, 13  | 13 | 0.00e+00 |
| 20, 21  | 21 | 0.00e+00 |
| 28, 29  | 29 | 0.00e+00 |
| 36, 37  | 37 | 0.00e+00 |

**Result: All 4 same-segment pairs have |ΔF| < 1e-10 (machine precision).**

## Technical Details

### Files Modified for Concentration Normalization Fix

- `CpuComputationContinuous.cpp`
- `CpuComputationDiscrete.cpp`
- `CpuComputationReduceMemoryContinuous.cpp`
- `CpuComputationReduceMemoryDiscrete.cpp`
- `CudaComputationContinuous.cu`
- `CudaComputationDiscrete.cu`
- `CudaComputationReduceMemoryContinuous.cu`
- `CudaComputationReduceMemoryDiscrete.cu`

### Files Modified for Boltzmann Factor Fix

- `CpuSolverPseudoRQM4.h` - Added `update_laplacian_operator()` declaration
- `CpuSolverPseudoRQM4.cpp` - Added `update_laplacian_operator()` implementation
- `CudaSolverPseudoRQM4.cu` - Updated `update_laplacian_operator()` implementation

### Why This Works

When total_segments ≠ Ns:
1. Propagators are computed with local_ds = alpha/total_segments
2. Boltzmann factors (boltz_bond) use the same local_ds
3. Concentration is normalized with local_ds = alpha/total_segments
4. All computations are now consistent, eliminating the mismatch

The segment count calculation in `Polymer.cpp` remains unchanged (round-to-nearest).

## References

- Contour discretization: Stasiak & Matsen, *Eur. Phys. J. E* **2011**, 34, 110
- Benchmark methodology: Song et al., *Chinese J. Polym. Sci.* **2018**, 36, 488
