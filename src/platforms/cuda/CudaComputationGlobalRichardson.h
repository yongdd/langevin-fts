/**
 * @file CudaComputationGlobalRichardson.h
 * @brief GPU propagator computation with Global Richardson extrapolation.
 *
 * This class implements Global Richardson extrapolation at the quadrature level,
 * maintaining two independent propagator chains that are combined via Richardson
 * extrapolation to compute physical quantities (Q, φ).
 *
 * **Global Richardson at Quadrature Level:**
 *
 * Unlike per-step Richardson (cn-adi4) which applies extrapolation at every step,
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
 * - d_propagator_full[key][n] for n = 0..N (N+1 values)
 * - d_propagator_half[key][n] for n = 0..N (N+1 values, stores only even positions)
 * - d_propagator_richardson[key][n] for n = 0..N (N+1 values)
 *
 * **Computational Cost:**
 *
 * - Full-step chain: N steps
 * - Half-step chain: 2N steps
 * - Total: 3N ADI steps per propagator (vs 3N for cn-adi4 per-step)
 *
 * **Accuracy:**
 *
 * - Q and φ: O(ds⁴) via Richardson extrapolation
 *
 * @see CudaSolverGlobalRichardsonBase for the base solver
 * @see CudaComputationContinuous for per-step methods
 */

#ifndef CUDA_COMPUTATION_GLOBAL_RICHARDSON_H_
#define CUDA_COMPUTATION_GLOBAL_RICHARDSON_H_

#include <string>
#include <vector>
#include <map>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputation.h"
#include "PropagatorComputationOptimizer.h"
#include "CudaSolverGlobalRichardsonBase.h"
#include "CudaCommon.h"
#include "Scheduler.h"

/**
 * @class CudaComputationGlobalRichardson
 * @brief GPU computation with Global Richardson at quadrature level.
 *
 * Manages two independent propagator chains and applies Richardson
 * extrapolation only when computing the partition function Q.
 */
class CudaComputationGlobalRichardson : public PropagatorComputation<double>
{
private:
    ComputationBox<double>* cb;      ///< Computation box
    Molecules* molecules;             ///< Molecules container
    PropagatorComputationOptimizer* propagator_computation_optimizer;

    CudaSolverGlobalRichardsonBase* solver;  ///< Base CN-ADI2 solver
    Scheduler* sc;                            ///< Computation scheduler
    int n_streams;                            ///< Number of parallel streams

    cudaStream_t streams[MAX_STREAMS][2];     ///< CUDA streams [kernel, memcpy]

    /// @name Full-step propagators (N+1 steps) on device
    /// @{
    std::map<std::string, double**> d_propagator_full;
    std::map<std::string, int> propagator_full_size;
    /// @}

    /// @name Half-step propagators (N+1 steps, only even positions stored) on device
    /// Only stores q_half at even positions (0, 2, 4, ..., 2N) which corresponds
    /// to full-step positions. Intermediate odd positions (1, 3, 5, ...) are
    /// computed on-the-fly but not stored.
    /// @{
    std::map<std::string, double**> d_propagator_half;
    std::map<std::string, int> propagator_half_size;
    /// @}

    /// @name Richardson-extrapolated propagators (N+1 steps, same size as full) on device
    /// q_rich[n] = (4·q_half[2n] - q_full[n]) / 3
    /// @{
    std::map<std::string, double**> d_propagator_richardson;
    /// @}

    /// @name Concentration fields on device
    /// @{
    std::map<std::tuple<int, std::string, std::string>, double*> d_phi_block;
    std::vector<double*> d_phi_solvent;
    /// @}

    /// @name Working arrays
    /// @{
    double* d_q_unity;               ///< Unity array
    double* d_q_mask;                ///< Mask for impenetrable regions
    double* d_phi;                   ///< Temporary phi
    double* d_q_half_temp[MAX_STREAMS];  ///< Temp buffer for half-step (per stream)
    /// @}

    /**
     * @brief Segment pairs for partition function calculation.
     * Tuple: (polymer_idx, d_q_richardson_left, d_q_richardson_right, n_aggregated)
     */
    std::vector<std::tuple<int, double*, double*, int>> partition_segment_info;

    #ifndef NDEBUG
    std::map<std::string, bool*> propagator_finished;  ///< Debug: tracks completed segments
    #endif

    /**
     * @brief Calculate concentration using Richardson-extrapolated propagators.
     *
     * Integrates: φ = Σ simpson_coeff[n] * q_rich_1 * q_rich_2
     */
    void calculate_phi_one_block(
        double* d_phi,
        double** d_q_1_richardson,
        double** d_q_2_richardson,
        const int N_LEFT,
        const int N_RIGHT
    );

public:
    /**
     * @brief Construct Global Richardson computation on GPU.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Propagator optimizer
     */
    CudaComputationGlobalRichardson(
        ComputationBox<double>* cb,
        Molecules* molecules,
        PropagatorComputationOptimizer* propagator_computation_optimizer);

    ~CudaComputationGlobalRichardson();

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
