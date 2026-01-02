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
 * Time 0-10: Stream 0 computes A10, Stream 1 computes B10
 * Time 10-20: Stream 0 computes C10(A10), Stream 1 continues B10
 * Time 20-30: Stream 0 computes D20(B10,C10)
 * ```
 *
 * @see PropagatorComputationOptimizer for propagator dependency analysis
 * @see CudaComputationContinuous for multi-stream execution
 *
 * @example
 * @code
 * // Create scheduler for 4 CUDA streams
 * int N_STREAM = 4;
 * auto& propagators = optimizer.get_computation_propagators();
 * Scheduler scheduler(propagators, N_STREAM);
 *
 * // Get the schedule
 * auto& schedule = scheduler.get_schedule();
 *
 * // Each time interval contains jobs: (propagator_key, stream_id, contour_steps)
 * for (size_t t = 0; t < schedule.size(); t++) {
 *     for (auto& [key, stream, steps] : schedule[t]) {
 *         std::cout << "Time " << t << ": Stream " << stream
 *                   << " computes " << key << " for " << steps << " steps\n";
 *     }
 * }
 *
 * // Display human-readable schedule
 * scheduler.display(propagators);
 * @endcode
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
 * 1. Build dependency hierarchy (propagators grouped by height in tree)
 * 2. For each time point, assign ready propagators to available streams
 * 3. Track when each propagator finishes to resolve dependencies
 * 4. Output: list of (propagator, stream, duration) for each time interval
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
     * @brief Stream assignment for each propagator.
     *
     * Maps propagator key to (stream_number, start_time, finish_time).
     */
    std::map<std::string, std::tuple<int, int, int>, ComparePropagatorKey> stream_start_finish;

    /**
     * @brief Time when each propagator's dependencies are resolved.
     *
     * A propagator can start computing at resolved_time[key].
     */
    std::map<std::string, int> resolved_time;

    /**
     * @brief Propagators sorted by start time.
     *
     * List of (propagator_key, start_time) for scheduling order.
     */
    std::vector<std::tuple<std::string, int>> sorted_propagator_with_start_time;

    /**
     * @brief Discrete time points where schedule changes.
     *
     * Times when jobs start or finish.
     */
    std::vector<int> time_stamp;

    /**
     * @brief The computed schedule.
     *
     * schedule[t] contains jobs active during time interval t.
     * Each job is (propagator_key, stream_id, n_contour_steps).
     */
    std::vector<std::vector<std::tuple<std::string, int, int>>> schedule;

    /**
     * @brief Group propagators by dependency height.
     *
     * @param computation_propagators Map of propagator keys to ComputationEdge
     * @return Vector of propagator groups, index = height level
     */
    std::vector<std::vector<std::string>> make_propagator_hierarchies(
        std::map<std::string, ComputationEdge, ComparePropagatorKey> computation_propagators);

public:
    /**
     * @brief Construct scheduler and compute the execution schedule.
     *
     * @param computation_propagators Map of propagators with dependencies
     * @param N_STREAM               Number of parallel streams available
     *
     * @note On CPU, typically use N_STREAM=1 (OpenMP parallelizes within each propagator).
     *       On GPU, use N_STREAM=4 for good performance.
     */
    Scheduler(std::map<std::string, ComputationEdge, ComparePropagatorKey> computation_propagators, const int N_STREAM);

    /**
     * @brief Destructor.
     */
    ~Scheduler() {};

    /**
     * @brief Get the computed schedule.
     *
     * @return Reference to schedule vector. Each element is a time interval
     *         containing list of (key, stream_id, n_steps) tuples.
     */
    std::vector<std::vector<std::tuple<std::string, int, int>>>& get_schedule();

    /**
     * @brief Display schedule in human-readable format.
     *
     * Prints the schedule showing which propagators run on which streams
     * at each time point.
     *
     * @param computation_propagators Propagator information for display
     */
    void display(std::map<std::string, ComputationEdge, ComparePropagatorKey> computation_propagators);
};
#endif
