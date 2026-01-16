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
class CpuComputationBase : public PropagatorComputation<T>
{
protected:
    CpuSolver<T> *propagator_solver;  ///< Solver for diffusion equation
    Scheduler *sc;                     ///< Propagator computation scheduler
    int n_streams;                     ///< Number of parallel computation streams

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

    #ifndef NDEBUG
    /**
     * @brief Debug: track which propagator steps are computed.
     */
    std::map<std::string, bool *> propagator_finished;
    #endif

    /**
     * @brief Block concentration fields.
     *
     * Key: (polymer_id, key_left, key_right) with key_left <= key_right
     * Value: Concentration array (size n_grid)
     */
    std::map<std::tuple<int, std::string, std::string>, T *> phi_block;

    /**
     * @brief Solvent concentration fields.
     *
     * One array per solvent species.
     */
    std::vector<T *> phi_solvent;

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

    /**
     * @brief Virtual destructor.
     */
    virtual ~CpuComputationBase() {}

    /**
     * @brief Update solver for new box dimensions.
     */
    void update_laplacian_operator() override;

    /**
     * @brief Get total partition function for a polymer.
     *
     * @param polymer Polymer index
     * @return Q = V^{-1} integral q_forward * q_backward dr
     */
    T get_total_partition(int polymer) override;

    /**
     * @brief Get total concentration of a monomer type.
     *
     * @param monomer_type Monomer type (e.g., "A")
     * @param phi          Output concentration array
     */
    void get_total_concentration(std::string monomer_type, T *phi) override;

    /**
     * @brief Get concentration of monomer type from a specific polymer.
     *
     * @param polymer      Polymer index
     * @param monomer_type Monomer type
     * @param phi          Output concentration array
     */
    void get_total_concentration(int polymer, std::string monomer_type, T *phi) override;

    /**
     * @brief Get concentration with fugacity weighting.
     *
     * @param fugacity     Chemical activity
     * @param polymer      Polymer index
     * @param monomer_type Monomer type
     * @param phi          Output concentration array
     */
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, T *phi) override;

    /**
     * @brief Get block concentration for a polymer.
     *
     * @param polymer Polymer index
     * @param phi     Output array (size n_grid * n_blocks)
     */
    void get_block_concentration(int polymer, T *phi) override;

    /**
     * @brief Get solvent partition function.
     *
     * @param s Solvent index
     * @return Solvent partition function
     */
    T get_solvent_partition(int s) override;

    /**
     * @brief Get solvent concentration field.
     *
     * @param s   Solvent index
     * @param phi Output concentration array
     */
    void get_solvent_concentration(int s, T *phi) override;

    /**
     * @brief Enable or disable cell-averaged bond function.
     *
     * @param enabled True to enable cell-averaging, false for standard bond function
     */
    void set_cell_averaged_bond(bool enabled) override;
};

#endif
