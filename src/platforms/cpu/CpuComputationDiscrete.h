/**
 * @file CpuComputationDiscrete.h
 * @brief CPU propagator computation for discrete chain model.
 *
 * This header provides CpuComputationDiscrete, which implements the full
 * propagator computation pipeline for discrete chain models on CPU.
 *
 * **Discrete vs Continuous:**
 *
 * The discrete chain model treats polymers as a sequence of discrete segments
 * connected by bonds, rather than a continuous path. This is the natural
 * model for short chains and matches lattice-based theories.
 *
 * **Propagator Recurrence:**
 *
 *     q(r, n+1) = exp(-w(r)) · ∫ G(r-r') q(r', n) dr'
 *
 * where G(r) is the bond distribution function (Gaussian).
 *
 * **Half-Bond Steps:**
 *
 * For accurate concentration calculation and proper treatment of chain ends,
 * the discrete model uses half-bond propagators at junctions:
 *
 *     q(r, n+1/2) = B^(1/2) · q(r, n)
 *
 * where B^(1/2) is diffusion by half a bond length.
 *
 * @see PropagatorComputation for the abstract interface
 * @see CpuComputationContinuous for continuous chain version
 */

#ifndef CPU_PSEUDO_DISCRETE_H_
#define CPU_PSEUDO_DISCRETE_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "CpuComputationBase.h"
#include "CpuSolverPseudoDiscrete.h"
#include "Scheduler.h"

/**
 * @class CpuComputationDiscrete
 * @brief CPU propagator computation for discrete chain model.
 *
 * Implements the discrete chain propagator computation with proper
 * handling of half-bond steps at chain ends and junctions.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Memory Management:**
 *
 * - propagator: Full segment propagators q(r, n)
 * - propagator_half_steps: Half-bond propagators q(r, n+1/2)
 *
 * **Concentration Formula:**
 *
 * For discrete chains, concentration is a sum (not integral):
 *
 *     φ(r) = Σ_n exp(-w(r)) · q_forward(r,n) · q_backward(r,N-n)
 *
 * @example
 * @code
 * CpuComputationDiscrete<double> comp(cb, molecules, optimizer);
 *
 * // Compute all statistics
 * comp.compute_statistics(w_fields);
 *
 * // Get partition function
 * double Q = comp.get_total_partition(0);
 * @endcode
 */
template <typename T>
class CpuComputationDiscrete : public CpuComputationBase<T>
{
private:
    /**
     * @brief Half-bond step propagators q(r, n+1/2).
     *
     * Stores propagators after half-bond diffusion at junctions.
     * Key: dependency code + monomer type
     * Value: 2D array [junction_index][grid_point]
     */
    std::map<std::string, T **> propagator_half_steps;

    #ifndef NDEBUG
    std::map<std::string, std::map<int, bool>> propagator_half_steps_finished;  ///< Debug: half-steps
    int time_complexity;                                       ///< Debug: operation count
    #endif

    /**
     * @brief Segment pairs for partition function.
     *
     * Tuple: (polymer_id, q_forward, q_backward, monomer_type, n_repeated)
     * Includes monomer_type for proper Boltzmann factor weighting.
     */
    std::vector<std::tuple<int, T *, T *, std::string, int>> single_partition_segment;

    /**
     * @brief Calculate concentration for one block.
     *
     * Uses discrete sum instead of integral:
     *     φ(r) = Σ exp(-w) · q_1 · q_2
     *
     * @param phi      Output concentration
     * @param q_1      Forward propagators
     * @param q_2      Backward propagators
     * @param exp_dw   Boltzmann factor exp(-w)
     * @param N_LEFT   Starting segment
     * @param N_RIGHT  Ending segment
     */
    void calculate_phi_one_block(T *phi, T **q_1, T **q_2, const T *exp_dw, const int N_LEFT, const int N_RIGHT);

public:
    /**
     * @brief Construct CPU computation for discrete chains.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Propagator optimizer
     */
    CpuComputationDiscrete(ComputationBox<T>* cb, Molecules *molecules, PropagatorComputationOptimizer* propagator_computation_optimizer);

    /**
     * @brief Destructor. Frees propagators and concentrations.
     */
    ~CpuComputationDiscrete();

    /**
     * @brief Compute all propagators.
     *
     * @param w_block Potential fields by monomer type
     * @param q_init  Optional initial conditions
     */
    void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /**
     * @brief Advance propagator by one segment.
     *
     * @param q_init Input propagator
     * @param q_out  Output propagator
     * @param p      Polymer index
     * @param v      Starting vertex of the block
     * @param u      Ending vertex of the block
     */
    void advance_propagator_single_segment(T* q_init, T *q_out, int p, int v, int u) override;

    /**
     * @brief Calculate concentration fields from propagators.
     */
    void compute_concentrations() override;

    /**
     * @brief Compute all statistics from fields.
     *
     * @param w_block Potential fields
     * @param q_init  Optional initial conditions
     */
    void compute_statistics(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /**
     * @brief Compute stress tensor.
     */
    void compute_stress() override;

    /**
     * @brief Get propagator at specific point.
     *
     * @param q_out   Output array
     * @param polymer Polymer index
     * @param v       Source vertex
     * @param u       Target vertex
     * @param n       Segment index
     */
    void get_chain_propagator(T *q_out, int polymer, int v, int u, int n) override;

    /**
     * @brief Validate partition function (testing).
     */
    bool check_total_partition() override;
};
#endif    