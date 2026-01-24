/**
 * @file CpuComputationContinuous.h
 * @brief CPU propagator computation for continuous chain model.
 *
 * This header provides CpuComputationContinuous, which implements the full
 * propagator computation pipeline for continuous Gaussian chains on CPU.
 * It orchestrates propagator solving, concentration calculation, and
 * stress computation.
 *
 * **Computation Pipeline:**
 *
 * 1. **compute_propagators()**: Solve modified diffusion equation for all chains
 * 2. **compute_concentrations()**: Calculate segment densities from propagators
 * 3. **compute_stress()**: Calculate stress tensor for box relaxation
 *
 * **Propagator Management:**
 *
 * Propagators are stored with keys based on their dependency codes from the
 * PropagatorComputationOptimizer. This enables efficient reuse of shared
 * propagator segments across different polymer chains.
 *
 * **Concentration Calculation:**
 *
 * For each block, concentration is computed as:
 *
 *     φ(r) = ∫₀^N q_forward(r,s) · q_backward(r,N-s) ds
 *
 * using Simpson's rule for the contour integration.
 *
 * @see PropagatorComputation for the abstract interface
 * @see CpuComputationDiscrete for discrete chain version
 * @see CpuComputationReduceMemoryContinuous for memory-saving version
 */

#ifndef CPU_PSEUDO_CONTINUOUS_H_
#define CPU_PSEUDO_CONTINUOUS_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "CpuComputationBase.h"
#include "CpuSolver.h"
#include "Scheduler.h"
#include "FFT.h"  // For FFTBackend enum

/**
 * @class CpuComputationContinuous
 * @brief CPU propagator computation for continuous Gaussian chains.
 *
 * Manages the complete propagator computation workflow including:
 * - Propagator storage and scheduling
 * - Forward/backward propagator computation
 * - Concentration field calculation
 * - Partition function and stress computation
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Memory Layout:**
 *
 * Propagators are stored as 2D arrays:
 *     propagator[key][s][grid_point]
 *
 * where key encodes the dependency structure and monomer type.
 *
 * **Optimizations:**
 *
 * - Propagator aggregation: Identical propagators computed once
 * - Scheduled computation: Ordered to maximize parallelism
 * - Simpson's rule: O(N) contour integration with O(ds²) accuracy
 *
 * @example
 * @code
 * CpuComputationContinuous<double> comp(cb, molecules, optimizer, "pseudospectral");
 *
 * // Compute everything for given fields
 * comp.compute_statistics(w_fields);
 *
 * // Get results
 * double Q = comp.get_total_partition(0);
 * comp.get_total_concentration("A", phi_A);
 * @endcode
 */
template <typename T>
class CpuComputationContinuous : public CpuComputationBase<T>
{
private:
    std::string method;                ///< Solver method ("pseudospectral" or "realspace")

    /**
     * @brief Segment pairs for partition function calculation.
     *
     * Each tuple: (polymer_id, q_forward_ptr, q_backward_ptr, n_repeated)
     * Stores one segment per chain for computing Q = integral q_f * q_b dr
     */
    std::vector<std::tuple<int, T *, T *, int>> single_partition_segment;

    /**
     * @brief Calculate concentration for one polymer block.
     *
     * Integrates product of forward and backward propagators:
     *     φ(r) = ∫ q_1(r,s) · q_2(r,N-s) ds
     *
     * @param phi      Output concentration array
     * @param q_1      Forward propagator array [step][point]
     * @param q_2      Backward propagator array [step][point]
     * @param N_LEFT   Starting contour index
     * @param N_RIGHT  Ending contour index
     */
    void calculate_phi_one_block(T *phi, T **q_1, T **q_2, const int N_LEFT, const int N_RIGHT);

public:
    /**
     * @brief Construct CPU computation for continuous chains.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Propagator optimizer with dependency info
     * @param method                          Solver method ("pseudospectral" or "realspace")
     * @param numerical_method                Numerical algorithm:
     *                                        - For pseudospectral: "rqm4" or "etdrk4"
     *                                        - For realspace: "cn-adi2" or "cn-adi4-lr"
     * @param backend                         FFT backend to use (FFTW, default: FFTW)
     */
    CpuComputationContinuous(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer, std::string method, std::string numerical_method = "", FFTBackend backend = FFTBackend::FFTW);

    /**
     * @brief Destructor. Frees all propagator and concentration arrays.
     */
    ~CpuComputationContinuous();

    /**
     * @brief Compute all propagators from potential fields.
     *
     * @param w_block Map of potential fields by monomer type
     * @param q_init  Optional initial conditions for propagators
     */
    void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /**
     * @brief Advance propagator by a single contour step.
     *
     * @param q_init Input propagator
     * @param q_out  Output propagator
     * @param p      Polymer index
     * @param v      Starting vertex of the block
     * @param u      Ending vertex of the block
     */
    void advance_propagator_single_segment(T* q_init, T *q_out, int p, int v, int u) override;

    /**
     * @brief Calculate all concentration fields from propagators.
     */
    void compute_concentrations() override;

    /**
     * @brief Compute propagators, concentrations, and partition functions.
     *
     * Convenience function that calls compute_propagators() and
     * compute_concentrations() in sequence.
     *
     * @param w_block Potential fields
     * @param q_init  Optional initial conditions
     */
    void compute_statistics(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /**
     * @brief Compute stress tensor components.
     */
    void compute_stress() override;

    /**
     * @brief Get propagator values at a specific point.
     *
     * @param q_out   Output array (size n_grid)
     * @param polymer Polymer index
     * @param v       Source vertex
     * @param u       Target vertex
     * @param n       Contour step index
     */
    void get_chain_propagator(T *q_out, int polymer, int v, int u, int n) override;

    /**
     * @brief Validate partition function calculation (for testing).
     *
     * @return True if partition function is consistent across segments
     */
    bool check_total_partition() override;
};
#endif
