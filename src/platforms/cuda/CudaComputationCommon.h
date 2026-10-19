/**
 * @file CudaComputationCommon.h
 * @brief State and accessors shared by every CUDA propagator computation class.
 *
 * Class hierarchy:
 *
 *     PropagatorComputation<T>
 *       └─ CudaComputationCommon<T>                (solver, scheduler, streams, unity/mask, partition getters)
 *            ├─ CudaComputationBase<T>             (full propagator storage, device-side φ)
 *            │    ├─ CudaComputationContinuous<T>
 *            │    └─ CudaComputationDiscrete<T>
 *            └─ CudaComputationReduceMemoryBase<T> (checkpointed storage, pinned host φ)
 *                 ├─ CudaComputationReduceMemoryContinuous<T>
 *                 └─ CudaComputationReduceMemoryDiscrete<T>
 *
 * The full variant keeps concentrations on the device and the reduce-memory
 * variant in pinned host memory, so the concentration getters stay in the
 * variant bases; only what is identical in both (solver handle, scheduler,
 * streams, unity and mask arrays, partition getters) lives here.
 */

#ifndef CUDA_COMPUTATION_COMMON_H_
#define CUDA_COMPUTATION_COMMON_H_

#include <string>
#include <map>

#include "ComputationBox.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "CudaCommon.h"
#include "CudaSolver.h"
#include "Scheduler.h"

template <typename T>
class CudaComputationCommon : public PropagatorComputation<T>
{
protected:
    CudaSolver<T> *propagator_solver;  ///< PDE solver (pseudo-spectral or real-space)
    Scheduler *sc;                      ///< Propagator execution scheduler
    int n_streams;                      ///< Number of parallel streams

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy]

    /**
     * @brief Per-stream device scratch for one segment's stress components (6 values).
     *
     * compute_stress() hands d_segment_stress_[STREAM] to
     * CudaSolver::compute_single_segment_stress() on that stream, so no two
     * streams share a buffer. Allocated once here, freed in the destructor.
     */
    CuDeviceData<T> *d_segment_stress_[MAX_STREAMS];

    CuDeviceData<T> *d_q_unity;  ///< Unity array for propagator initialization
    double *d_q_mask;            ///< Mask for impenetrable regions (nanoparticles)

    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;  ///< Debug: which propagator steps are computed
    #endif

public:
    /**
     * @param cb                               Computation box
     * @param molecules                        Molecules container
     * @param propagator_computation_optimizer Propagator optimizer with dependency info
     * @param n_streams                        Initial number of streams (the derived
     *                                         constructor may change it)
     */
    CudaComputationCommon(ComputationBox<T>* cb, Molecules *molecules,
                          PropagatorComputationOptimizer* propagator_computation_optimizer,
                          int n_streams);
    virtual ~CudaComputationCommon();

    /**
     * @brief Number of propagator streams from OMP_NUM_THREADS.
     *
     * Returns the leading integer of OMP_NUM_THREADS when it is positive, and
     * `default_n_streams` when the variable is unset, empty, or not a positive
     * integer (same rule as the CPU side).
     */
    static int n_streams_from_env(int default_n_streams);

    /// Update the solver for new box dimensions.
    void update_laplacian_operator() override;

    /// Force off-diagonal stress computation; forwarded to the solver.
    void set_force_off_diagonal_stress(bool force) override
    {
        PropagatorComputation<T>::set_force_off_diagonal_stress(force);
        if (propagator_solver != nullptr)
            propagator_solver->set_force_off_diagonal_stress(force);
    }

    /// Q = V^{-1} ∫ q_forward q_backward dr of one polymer.
    T get_total_partition(int polymer) override;
    /// Partition function of one solvent species.
    T get_solvent_partition(int s) override;
};

#endif  // CUDA_COMPUTATION_COMMON_H_
