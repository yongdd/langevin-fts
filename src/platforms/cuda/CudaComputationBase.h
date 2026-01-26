/**
 * @file CudaComputationBase.h
 * @brief Common base class for CUDA propagator computation classes.
 *
 * This header provides CudaComputationBase, a common base class that
 * consolidates shared functionality between CudaComputationContinuous
 * and CudaComputationDiscrete, including:
 *
 * - Common member variable declarations
 * - Shared concentration query methods
 * - Partition function accessors
 * - Laplacian operator updates
 *
 * @see CudaComputationContinuous for continuous chain implementation
 * @see CudaComputationDiscrete for discrete chain implementation
 */

#ifndef CUDA_COMPUTATION_BASE_H_
#define CUDA_COMPUTATION_BASE_H_

#include <string>
#include <vector>
#include <map>
#include <cufft.h>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "CudaCommon.h"
#include "CudaSolver.h"
#include "Scheduler.h"

/**
 * @class CudaComputationBase
 * @brief Common base class for CUDA propagator computation.
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
 */
template <typename T>
class CudaComputationBase : public PropagatorComputation<T>
{
protected:
    CudaSolver<T> *propagator_solver;  ///< PDE solver (pseudo-spectral or real-space)

    int n_streams;  ///< Number of parallel streams

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy]

    CuDeviceData<T> *d_q_unity;  ///< Unity array for propagator initialization

    double *d_q_mask;  ///< Mask for impenetrable regions (nanoparticles)

    CuDeviceData<T> *d_q_pair[MAX_STREAMS][2];  ///< Workspace [prev, next] per stream

    Scheduler *sc;  ///< Execution scheduler for propagator dependencies

    /**
     * @brief Propagator arrays on device.
     *
     * Key: dependency_code + monomer_type
     * Value: Array of device pointers [n_segments+1]
     */
    std::map<std::string, CuDeviceData<T> **> d_propagator;

    /** @brief Size of each propagator array for deallocation. */
    std::map<std::string, int> propagator_size;

    #ifndef NDEBUG
    /** @brief Debug: track propagator computation completion. */
    std::map<std::string, bool *> propagator_finished;
    #endif

    /**
     * @brief Block concentrations on device.
     *
     * Key: (polymer_id, key_left, key_right) with key_left <= key_right
     */
    std::map<std::tuple<int, std::string, std::string>, CuDeviceData<T> *> d_phi_block;

    CuDeviceData<T> *d_phi;  ///< Temporary for concentration computation

    std::vector<CuDeviceData<T> *> d_phi_solvent;  ///< Solvent concentrations

    // Reduced basis support (populated by derived class in set_space_group)
    int* d_full_to_reduced_map_base_;     ///< Map from full grid to reduced basis (device)
    int* d_reduced_basis_indices_base_;   ///< Indices of reduced basis points (device)
    CuDeviceData<T>* d_phi_full_buffer_;  ///< Buffer for expanding to full grid (device)

public:
    /**
     * @brief Construct CUDA computation base.
     *
     * @param cb                              Computation box
     * @param molecules                       Molecules container
     * @param propagator_computation_optimizer Propagator optimizer with dependency info
     */
    CudaComputationBase(ComputationBox<T>* cb, Molecules *molecules,
                        PropagatorComputationOptimizer* propagator_computation_optimizer);

    /**
     * @brief Virtual destructor.
     */
    virtual ~CudaComputationBase() {}

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
     * @param phi          Output concentration array (host)
     */
    void get_total_concentration(std::string monomer_type, T *phi) override;

    /**
     * @brief Get concentration of monomer type from a specific polymer.
     *
     * @param polymer      Polymer index
     * @param monomer_type Monomer type
     * @param phi          Output concentration array (host)
     */
    void get_total_concentration(int polymer, std::string monomer_type, T *phi) override;

    /**
     * @brief Get concentration with fugacity weighting.
     *
     * @param fugacity     Chemical activity
     * @param polymer      Polymer index
     * @param monomer_type Monomer type
     * @param phi          Output concentration array (host)
     */
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, T *phi) override;

    /**
     * @brief Get block concentration for a polymer.
     *
     * @param polymer Polymer index
     * @param phi     Output array (size n_grid * n_blocks, host)
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
     * @param phi Output concentration array (host)
     */
    void get_solvent_concentration(int s, T *phi) override;
};

#endif
