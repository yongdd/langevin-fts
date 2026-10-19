/**
 * @file CpuComputationBase.h
 * @brief Common base class for CPU propagator computation classes.
 *
 * This header provides CpuComputationBase, a common base class that
 * consolidates shared functionality between CpuComputationContinuous
 * and CpuComputationDiscrete, including:
 *
 * - Common member variable declarations
 * - Shared concentration query methods
 * - Partition function accessors
 * - Laplacian operator updates
 *
 * **Design Decision:**
 *
 * The `single_partition_segment` has different tuple structures between
 * Continuous (4 elements) and Discrete (5 elements), so it remains in
 * derived classes. Similarly, constructors, destructors, and methods
 * with significantly different logic stay in derived classes.
 *
 * @see CpuComputationContinuous for continuous chain implementation
 * @see CpuComputationDiscrete for discrete chain implementation
 */

#ifndef CPU_COMPUTATION_BASE_H_
#define CPU_COMPUTATION_BASE_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "CpuSolver.h"
#include "Scheduler.h"
#include "CpuComputationCommon.h"

/**
 * @class CpuComputationBase
 * @brief Common base class for CPU propagator computation.
 *
 * Consolidates shared code between continuous and discrete chain
 * computation classes, including concentration queries and
 * partition function accessors.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Shared Functionality:**
 *
 * - Laplacian operator updates
 * - Total concentration queries (by monomer type, by polymer)
 * - Block concentration retrieval
 * - Solvent partition and concentration accessors
 * - Grand canonical ensemble concentration
 *
 * **Virtual Methods:**
 *
 * Derived classes must implement:
 * - compute_propagators(): Different propagator algorithms
 * - compute_concentrations(): Different integration formulas
 * - compute_stress(): Different stress computation logic
 * - get_chain_propagator(): Slightly different range checks
 * - check_total_partition(): Different validation logic
 */
template <typename T>
class CpuComputationBase : public CpuComputationCommon<T>
{
protected:

    /**
     * @brief Storage for computed propagators.
     *
     * Key: dependency code + monomer type (e.g., "v0u1_A")
     * Value: 2D array [contour_step][grid_point]
     */
    std::map<std::string, T **> propagator;

    /**
     * @brief Size of each propagator (number of contour steps).
     *
     * Used for proper deallocation.
     */
    std::map<std::string, int> propagator_size;

public:
    /**
     * @brief Construct CPU computation base.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Propagator optimizer with dependency info
     */
    CpuComputationBase(ComputationBox<T>* cb, Molecules *molecules,
                       PropagatorComputationOptimizer* propagator_computation_optimizer);

    /// Virtual destructor.
    virtual ~CpuComputationBase() {}
};

#endif
