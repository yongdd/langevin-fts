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
#include "CudaComputationCommon.h"

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
class CudaComputationBase : public CudaComputationCommon<T>
{
protected:

    CuDeviceData<T> *d_q_pair[MAX_STREAMS][2];  ///< Workspace [prev, next] per stream

    /**
     * @brief Propagator arrays on device.
     *
     * Key: dependency_code + monomer_type
     * Value: Array of device pointers [n_segments+1]
     */
    std::map<std::string, CuDeviceData<T> **> d_propagator;

    /// Size of each propagator array for deallocation.
    std::map<std::string, int> propagator_size;

    /**
     * @brief Block concentrations on device.
     *
     * Key: (polymer_id, key_left, key_right) with key_left <= key_right
     */
    std::map<std::tuple<int, std::string, std::string>, CuDeviceData<T> *> d_phi_block;

    CuDeviceData<T> *d_phi;  ///< Temporary for concentration computation

    std::vector<CuDeviceData<T> *> d_phi_solvent;  ///< Solvent concentrations

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

    /// Virtual destructor.
    virtual ~CudaComputationBase() {}

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
     * @brief Get solvent concentration field.
     *
     * @param s   Solvent index
     * @param phi Output concentration array (host)
     */
    void get_solvent_concentration(int s, T *phi) override;
};

#endif
