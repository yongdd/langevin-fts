/**
 * @file PropagatorAggregator.h
 * @brief Algorithms for aggregating equivalent propagator computations.
 *
 * This class provides static methods for identifying and merging equivalent
 * propagator computations in polymer chain models. Aggregation reduces
 * computational cost by computing shared sub-structures once.
 *
 * **Continuous Chain Aggregation:**
 *
 * For continuous Gaussian chains, branches with identical segment counts
 * can be merged into a single aggregated computation. The aggregated
 * propagator is computed once and reused for all equivalent branches.
 *
 * **Discrete Chain Aggregation:**
 *
 * For discrete bead-spring chains, aggregation follows a hierarchical
 * merging strategy. The minimum segment count is 1 (not 0) due to the
 * discrete nature of the chain.
 *
 * @see PropagatorComputationOptimizer for the optimization orchestrator
 * @see PropagatorCode for key generation
 */

#ifndef PROPAGATOR_AGGREGATOR_H_
#define PROPAGATOR_AGGREGATOR_H_

#include <string>
#include <map>
#include <vector>
#include <set>

// Forward declaration
struct ComputationBlock;
struct ComparePropagatorKey;

/**
 * @class PropagatorAggregator
 * @brief Static methods for propagator aggregation algorithms.
 *
 * Provides chain-model-specific aggregation strategies that identify
 * and merge equivalent propagator computations.
 */
class PropagatorAggregator
{
public:
    /**
     * @brief Aggregate propagators for continuous chain model.
     *
     * Merges branches with the same segment count into aggregated computations.
     * Creates new keys with bracket notation (e.g., "[A10:2,B10]C").
     *
     * **Algorithm:**
     *
     * For each unique segment count:
     * 1. Collect all propagators with that count
     * 2. Create aggregated key combining all branches
     * 3. Set individual branch segment counts to 0
     *
     * @param set_I Input set of computation blocks
     * @return Modified set with aggregated blocks
     */
    static std::map<std::string, ComputationBlock> aggregate_continuous_chain(
        std::map<std::string, ComputationBlock> set_I);

    /**
     * @brief Aggregate propagators for discrete chain model.
     *
     * Similar to continuous chain aggregation but accounts for discrete
     * chain specifics where minimum segment count is 1 (not 0).
     *
     * **Algorithm:**
     *
     * Iteratively merges the two largest segment counts until only one
     * aggregated propagator remains, creating a hierarchical structure.
     *
     * @param set_I Input set of computation blocks
     * @return Modified set with aggregated blocks
     */
    static std::map<std::string, ComputationBlock> aggregate_discrete_chain(
        std::map<std::string, ComputationBlock> set_I);
};

#endif
