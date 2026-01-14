/**
 * @file CpuComputationGlobalRichardson.h
 * @brief CPU propagator computation with Global Richardson extrapolation.
 *
 * This class implements Global Richardson extrapolation at the quadrature level,
 * maintaining two independent propagator chains that are combined via Richardson
 * extrapolation to compute physical quantities (Q, φ).
 *
 * **Global Richardson at Quadrature Level:**
 *
 * Unlike per-step Richardson (cn-adi4-lr) which applies extrapolation at every step,
 * this method:
 * 1. Maintains TWO independent propagator chains:
 *    - Full-step chain: q_full[0..N] advanced with step size ds
 *    - Half-step chain: q_half[0..2N] advanced with step size ds/2
 * 2. Computes Richardson-extrapolated propagators:
 *    q_rich[n] = (4·q_half[2n] - q_full[n]) / 3
 * 3. Uses q_rich for both Q and φ computation
 *
 * **Memory Layout:**
 *
 * For each propagator key:
 * - propagator_full[key][n] for n = 0..N (N+1 values)
 * - propagator_half[key][n] for n = 0..N (N+1 values, stores only even positions)
 * - propagator_richardson[key][n] for n = 0..N (N+1 values)
 *
 * **Computational Cost:**
 *
 * - Full-step chain: N steps
 * - Half-step chain: 2N steps
 * - Total: 3N ADI steps per propagator (vs 3N for cn-adi4-lr per-step)
 *
 * **Accuracy:**
 *
 * - Q and φ: O(ds⁴) via Richardson extrapolation
 *
 * @see CpuSolverGlobalRichardsonBase for the base solver
 * @see CpuComputationContinuous for per-step methods
 */

#ifndef CPU_COMPUTATION_GLOBAL_RICHARDSON_H_
#define CPU_COMPUTATION_GLOBAL_RICHARDSON_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputation.h"
#include "PropagatorComputationOptimizer.h"
#include "CpuSolverGlobalRichardsonBase.h"
#include "Scheduler.h"

/**
 * @class CpuComputationGlobalRichardson
 * @brief CPU computation with Global Richardson at quadrature level.
 *
 * Manages two independent propagator chains and applies Richardson
 * extrapolation only when computing the partition function Q.
 */
class CpuComputationGlobalRichardson : public PropagatorComputation<double>
{
private:
    CpuSolverGlobalRichardsonBase* solver;  ///< Base CN-ADI2 solver
    Scheduler* sc;                           ///< Computation scheduler
    int n_streams;                           ///< Number of parallel threads

    /// @name Full-step propagators (N+1 steps)
    /// @{
    std::map<std::string, double**> propagator_full;
    std::map<std::string, int> propagator_full_size;
    /// @}

    /// @name Half-step propagators (N+1 steps, only even positions stored)
    /// Only stores q_half at even positions (0, 2, 4, ..., 2N) which corresponds
    /// to full-step positions. Intermediate odd positions (1, 3, 5, ...) are
    /// computed on-the-fly but not stored.
    /// @{
    std::map<std::string, double**> propagator_half;
    std::map<std::string, int> propagator_half_size;
    /// @}

    /// @name Richardson-extrapolated propagators (N+1 steps, same size as full)
    /// q_rich[n] = (4·q_half[2n] - q_full[n]) / 3
    /// @{
    std::map<std::string, double**> propagator_richardson;
    /// @}

    /// @name Concentration fields
    /// @{
    std::map<std::tuple<int, std::string, std::string>, double*> phi_block;
    std::vector<double*> phi_solvent;
    /// @}

    /**
     * @brief Segment pairs for partition function calculation.
     * Tuple: (polymer_idx, q_full_left, q_full_right, q_half_left, q_half_right, n_aggregated)
     * Q is computed as: Q_rich = (4*Q_half - Q_full) / 3
     */
    std::vector<std::tuple<int, double*, double*, double*, double*, int>> partition_segment_info;

    #ifndef NDEBUG
    std::map<std::string, bool*> propagator_finished;  ///< Debug: tracks completed segments
    #endif

    /**
     * @brief Calculate concentration with Richardson extrapolation of phi.
     *
     * Computes φ_full and φ_half separately using full/half-step propagators,
     * each normalized by their respective Q values:
     *   φ_full = (norm / Q_full) * integral_full
     *   φ_half = (norm / Q_half) * integral_half
     * Then applies Richardson extrapolation: φ_rich = (4*φ_half - φ_full) / 3
     *
     * @param phi Output concentration array
     * @param q_1_full, q_2_full Full-step propagators
     * @param q_1_half, q_2_half Half-step propagators
     * @param N_LEFT, N_RIGHT Segment counts
     * @param Q_full Partition function from full-step chain
     * @param Q_half Partition function from half-step chain
     * @param norm Base normalization factor (ds * vf / alpha * n_repeated)
     */
    void calculate_phi_one_block(
        double* phi,
        double** q_1_full, double** q_2_full,
        double** q_1_half, double** q_2_half,
        const int N_LEFT,
        const int N_RIGHT,
        const double Q_full,
        const double Q_half,
        const double norm
    );

public:
    /**
     * @brief Construct Global Richardson computation.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Propagator optimizer
     */
    CpuComputationGlobalRichardson(
        ComputationBox<double>* cb,
        Molecules* molecules,
        PropagatorComputationOptimizer* propagator_computation_optimizer);

    ~CpuComputationGlobalRichardson();

    /**
     * @brief Update solver operators for new box dimensions.
     */
    void update_laplacian_operator() override;

    /**
     * @brief Compute all propagators from potential fields.
     *
     * Advances both full-step and half-step chains independently.
     */
    void compute_propagators(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {}) override;

    /**
     * @brief Not supported for Global Richardson.
     */
    void advance_propagator_single_segment(double*, double*, std::string) override
    {
        throw_with_line_number("advance_propagator_single_segment not supported for Global Richardson.");
    }

    /**
     * @brief Compute concentrations using Richardson-extrapolated propagators.
     */
    void compute_concentrations() override;

    /**
     * @brief Compute propagators and concentrations.
     */
    void compute_statistics(
        std::map<std::string, const double*> w_block,
        std::map<std::string, const double*> q_init = {}) override;

    /**
     * @brief Not yet implemented for Global Richardson.
     */
    void compute_stress() override;

    /**
     * @brief Get Richardson-extrapolated propagator at a specific contour point.
     *
     * Note: The contour index n is in full-step units (0..N).
     */
    void get_chain_propagator(double* q_out, int polymer, int v, int u, int n) override;

    /**
     * @brief Get total partition function (computed from Richardson propagators).
     */
    double get_total_partition(int polymer) override
    {
        return single_polymer_partitions[polymer];
    }

    /**
     * @brief Get solvent partition function.
     */
    double get_solvent_partition(int s) override
    {
        return single_solvent_partitions[s];
    }

    /**
     * @brief Get total concentration for a monomer type.
     */
    void get_total_concentration(std::string monomer_type, double* phi) override;

    /**
     * @brief Get concentration for a monomer type from specific polymer.
     */
    void get_total_concentration(int polymer, std::string monomer_type, double* phi) override;

    /**
     * @brief Get block concentration (not implemented).
     */
    void get_block_concentration(int, double*) override
    {
        throw_with_line_number("get_block_concentration not implemented for Global Richardson.");
    }

    /**
     * @brief Get solvent concentration.
     */
    void get_solvent_concentration(int s, double* phi) override;

    /**
     * @brief Get concentration for grand canonical ensemble.
     */
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, double* phi) override;

    /**
     * @brief Validate partition function consistency.
     */
    bool check_total_partition() override;
};

#endif
