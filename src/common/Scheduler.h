/**
 * @file Scheduler.h
 * @brief Schedules parallel propagator computations for multi-stream execution.
 *
 * This header provides the Scheduler class which creates an execution schedule
 * for computing multiple chain propagators in parallel. On GPU, this enables
 * concurrent kernel execution using multiple CUDA streams, significantly
 * improving performance for branched polymers and polymer mixtures.
 *
 * **Scheduling Problem:**
 *
 * For branched polymers, propagators have dependencies:
 * - A junction propagator cannot be computed until all its branch propagators are ready
 * - Independent propagators (from different branches) can run in parallel
 *
 * The scheduler creates an optimal assignment of propagators to streams,
 * respecting dependencies while maximizing parallelism.
 *
 * **Schedule Structure:**
 *
 * The schedule is organized by time intervals:
 * ```
 * Time 0-10: Stream 0 computes A+0, Stream 1 computes B+0
 * Time 10-20: Stream 0 computes (A)C+0, Stream 1 continues B+0
 * Time 20-30: Stream 0 computes (BC)D+0
 * ```
 *
 * @see PropagatorComputationOptimizer for propagator dependency analysis
 * @see CudaComputationContinuous for multi-stream execution
 */

#ifndef SCHEDULER_H_
#define SCHEDULER_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"

/**
 * @class Scheduler
 * @brief Creates parallel execution schedule for propagator computations.
 *
 * Given a set of propagators with dependencies, the Scheduler assigns each
 * propagator to a stream and determines the execution order. The goal is
 * to minimize total execution time by running independent propagators
 * concurrently.
 *
 * **Algorithm:**
 *
 * 1. Group propagators by dependency height (height 0 = no dependencies)
 * 2. For each propagator, compute when dependencies are resolved
 * 3. Greedily assign to stream with earliest available time
 * 4. Slice execution into time intervals for interleaved progress
 *
 * **Multi-Stream Benefits:**
 *
 * On GPU, using multiple streams enables:
 * - Concurrent kernel execution
 * - Overlap of memory transfers with computation
 * - Better GPU utilization for small propagators
 *
 * Typical speedup: 1.5-3x for branched polymers vs single-stream.
 */
class Scheduler
{
private:
    /**
     * @brief Job assignment for each propagator.
     *
     * Maps propagator key to (stream_id, start_time, end_time).
     */
    std::map<std::string, std::tuple<int, int, int>, ComparePropagatorKey> job_assignment;

    /**
     * @brief Time when each propagator's dependencies are resolved.
     */
    std::map<std::string, int> dependency_resolved_time;

    /**
     * @brief Propagators sorted by start time: (key, start_time).
     */
    std::vector<std::tuple<std::string, int>> sorted_jobs;

    /**
     * @brief Discrete time points where schedule changes (job starts or ends).
     */
    std::vector<int> time_points;

    /**
     * @brief The computed schedule.
     *
     * schedule[t] contains jobs active during time interval t.
     * Each job is (propagator_key, segment_from, segment_to).
     */
    std::vector<std::vector<std::tuple<std::string, int, int>>> schedule;

    /**
     * @brief Group propagators by dependency height.
     *
     * @param propagators Map of propagator keys to ComputationEdge
     * @return Vector of propagator groups, index = height level
     */
    std::vector<std::vector<std::string>> group_by_height(
        const std::map<std::string, ComputationEdge, ComparePropagatorKey>& propagators);

    /**
     * @brief Find stream with earliest available time.
     *
     * @param stream_available_time Vector of when each stream becomes free
     * @return Index of the stream with minimum available time
     */
    int find_earliest_stream(const std::vector<int>& stream_available_time);

public:
    /**
     * @brief Construct scheduler and compute the execution schedule.
     *
     * @param propagators Map of propagators with dependencies
     * @param n_streams   Number of parallel streams available
     *
     * @note On CPU, typically use n_streams=1 (OpenMP parallelizes within each propagator).
     *       On GPU, use n_streams=4 for good performance.
     */
    Scheduler(std::map<std::string, ComputationEdge, ComparePropagatorKey> propagators, int n_streams);

    /**
     * @brief Destructor.
     */
    ~Scheduler() {};

    /**
     * @brief Get the computed schedule.
     *
     * @return Reference to schedule vector. Each element is a time interval
     *         containing list of (key, segment_from, segment_to) tuples.
     */
    std::vector<std::vector<std::tuple<std::string, int, int>>>& get_schedule();

    /**
     * @brief Display schedule in human-readable format.
     *
     * @param propagators Propagator information for display
     */
    void display(std::map<std::string, ComputationEdge, ComparePropagatorKey> propagators);
};
#endif
