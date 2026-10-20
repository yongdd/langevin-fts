/**
 * @file GridLimits.h
 * @brief Grid sizes the code supports with 32-bit grid indices.
 *
 * Grid arrays are indexed with int. A CUDA grid-stride loop
 * (`i += blockDim.x * gridDim.x`) stays within int as long as an array has at
 * most INT_MAX - MAX_GRID_THREADS elements and no launch uses more than
 * MAX_GRID_THREADS threads (grid_blocks() in CudaCommon.h and the checks in
 * CudaCommon enforce the latter). The largest grid arrays hold two fields
 * (propagator pairs, batched half steps), so a grid may have at most half
 * that many points; ComputationBox rejects larger grids.
 */

#ifndef GRID_LIMITS_H_
#define GRID_LIMITS_H_

#include <climits>

/// Threads in the grid of any grid-stride CUDA kernel launch
constexpr int MAX_GRID_THREADS = 1 << 20;

/// Elements of the largest array a grid-stride loop may run over
constexpr long long MAX_GRID_ARRAY = static_cast<long long>(INT_MAX) - MAX_GRID_THREADS;

/// Points of the largest supported grid (the largest arrays hold two fields)
constexpr long long MAX_GRID_POINTS = MAX_GRID_ARRAY / 2;

#endif
