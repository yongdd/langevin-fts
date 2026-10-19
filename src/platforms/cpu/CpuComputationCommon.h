/**
 * @file CpuComputationCommon.h
 * @brief State and accessors shared by every CPU propagator computation class.
 *
 * Class hierarchy:
 *
 *     PropagatorComputation<T>
 *       └─ CpuComputationCommon<T>                (solver, scheduler, φ storage, getters)
 *            ├─ CpuComputationBase<T>             (full propagator storage)
 *            │    ├─ CpuComputationContinuous<T>
 *            │    └─ CpuComputationDiscrete<T>
 *            └─ CpuComputationReduceMemoryBase<T> (checkpointed propagator storage)
 *                 ├─ CpuComputationReduceMemoryContinuous<T>
 *                 └─ CpuComputationReduceMemoryDiscrete<T>
 *
 * Only members whose meaning and implementation are identical for the full
 * and the reduce-memory variant live here: the concentration and partition
 * getters read phi_block, phi_solvent and the partition arrays, which both
 * variants fill in the same layout; compute_statistics() chains the two
 * compute_* calls; the destructor frees phi_block and phi_solvent, which the
 * derived constructors allocate (the sizes differ per variant). Propagator
 * storage and the compute_* algorithms stay in the variant classes so that
 * each can be changed on its own.
 */

#ifndef CPU_COMPUTATION_COMMON_H_
#define CPU_COMPUTATION_COMMON_H_

#include <string>
#include <vector>
#include <map>
#include <tuple>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "CpuSolver.h"
#include "Scheduler.h"

template <typename T>
class CpuComputationCommon : public PropagatorComputation<T>
{
protected:
    CpuSolver<T> *propagator_solver;  ///< Solver for the diffusion / bond-convolution equation
    Scheduler *sc;                     ///< Propagator computation scheduler
    int n_streams;                     ///< Number of parallel computation streams

    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;  ///< Debug: which propagator steps are computed
    #endif

    /// Block concentrations, key (polymer_id, key_left, key_right), each of size n_basis.
    std::map<std::tuple<int, std::string, std::string>, T *> phi_block;

    std::vector<T *> phi_solvent;  ///< One concentration array per solvent species

    /**
     * @brief Number of propagator streams from OMP_NUM_THREADS.
     *
     * Returns the leading integer of OMP_NUM_THREADS when it is positive, and
     * `default_n_streams` when the variable is unset, empty, or not a positive
     * integer.
     */
    static int n_streams_from_env(int default_n_streams);

public:
    /**
     * @param cb                               Computation box
     * @param molecules                        Molecules container
     * @param propagator_computation_optimizer Propagator optimizer with dependency info
     * @param n_streams                        Initial number of streams (the derived
     *                                         constructor may change it)
     */
    CpuComputationCommon(ComputationBox<T>* cb, Molecules *molecules,
                         PropagatorComputationOptimizer* propagator_computation_optimizer,
                         int n_streams);
    /// Frees phi_block and phi_solvent (allocated by the derived constructors).
    virtual ~CpuComputationCommon();

    /// Update the solver for new box dimensions.
    void update_laplacian_operator() override;

    /// compute_propagators() followed by compute_concentrations().
    void compute_statistics(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    /// Force off-diagonal stress computation; forwarded to the solver.
    void set_force_off_diagonal_stress(bool force) override
    {
        PropagatorComputation<T>::set_force_off_diagonal_stress(force);
        if (propagator_solver != nullptr)
            propagator_solver->set_force_off_diagonal_stress(force);
    }

    /// Q = V^{-1} ∫ q_forward q_backward dr of one polymer.
    T get_total_partition(int polymer) override;
    /// Total concentration of a monomer type (polymers and solvents).
    void get_total_concentration(std::string monomer_type, T *phi) override;
    /// Concentration of a monomer type from one polymer.
    void get_total_concentration(int polymer, std::string monomer_type, T *phi) override;
    /// Concentration with fugacity weighting (grand canonical ensemble).
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, T *phi) override;
    /// Block concentrations of one polymer, output size n_basis * n_blocks (aggregation must be off).
    void get_block_concentration(int polymer, T *phi) override;
    /// Partition function of one solvent species.
    T get_solvent_partition(int s) override;
    /// Concentration of one solvent species.
    void get_solvent_concentration(int s, T *phi) override;
};

#endif  // CPU_COMPUTATION_COMMON_H_
