/**
 * @file ComputationCommon.h
 * @brief Platform-neutral base of the propagator computation classes.
 *
 * Holds what every chain model and storage mode shares: the platform backend,
 * the propagator scheduler, the block and solvent concentrations, the fields
 * kept for the solvents, and the accessors that read them. The derived classes
 * own the propagator storage and the compute_* algorithms.
 *
 * @see ComputationBackend for the platform interface
 * @see ComputationContinuous for the continuous-chain computation
 */

#ifndef COMPUTATION_COMMON_H_
#define COMPUTATION_COMMON_H_

#include <array>
#include <memory>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "ComputationBackend.h"
#include "ComputationBox.h"
#include "Molecules.h"
#include "PropagatorComputation.h"
#include "PropagatorComputationOptimizer.h"
#include "Scheduler.h"

/**
 * @brief Carries an exception out of an OpenMP parallel loop.
 *
 * An exception cannot leave an OpenMP parallel region; the program would be
 * terminated. run() catches it inside the loop body and rethrow() rethrows
 * the first one after the loop, so that, for example, running out of GPU
 * memory during a computation reaches Python as an exception.
 */
class ParallelErrors
{
private:
    std::exception_ptr first;
    std::mutex mutex;

public:
    template <typename Body>
    void run(Body&& body)
    {
        try
        {
            body();
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!first)
                first = std::current_exception();
        }
    }

    void rethrow()
    {
        if (first)
            std::rethrow_exception(first);
    }
};

template <typename T>
class ComputationCommon : public PropagatorComputation<T>
{
protected:
    std::unique_ptr<ComputationBackend<T>> backend;  ///< Platform operations
    std::unique_ptr<Scheduler> sc;                   ///< Order in which the propagators are computed
    int n_streams;                   ///< Number of propagators computed concurrently

    /// Unnormalized segment-product integral of each block; getters apply the
    /// canonical or fugacity normalization, including when volume fraction is zero.
    std::map<std::tuple<int, std::string, std::string>, BackendArray<T>*> phi_block;
    std::vector<BackendArray<T>*> phi_solvent;     ///< Concentration of each solvent
    std::map<std::string, BackendArray<T>*> w;     ///< Fields of the last compute_propagators()

    /**
     * @param create_backend Called after the space group is set on the box,
     *                       because the solvers size their buffers from it.
     */
    ComputationCommon(ComputationBox<T>* cb, Molecules* molecules,
                      PropagatorComputationOptimizer* propagator_computation_optimizer,
                      SpaceGroup* space_group,
                      const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend);

    /**
     * @brief Check the inputs of compute_propagators(), keep a copy of the
     *        fields (solvents need them) and update the Boltzmann factors.
     *
     * Checked here, before any parallel loop: every monomer type has a field
     * and every grafting point has its q_init.
     */
    void set_fields(const std::map<std::string, const T*>& w_input,
                    const std::map<std::string, const T*>& q_init);

    /// Solvent concentrations and partition functions from the stored fields.
    void compute_solvent_concentrations();

    /// Contour step ds of the block that propagator `key` belongs to.
    double contour_step(const std::string& key) const;

    /// Normalization of a block concentration, φ = norm · Σ_n c_n q q†:
    /// norm = ds · volume fraction / alpha · n_repeated / Q.
    T concentration_norm(const std::tuple<int, std::string, std::string>& block_key) const;

    /// Stress of a block: the sum over its segments times the contour step ds.
    void scale_by_contour_step(std::array<T, 6>& stress, const std::string& key_right) const;

    /**
     * @brief Sum the stresses of the blocks of each polymer into dq_dl and
     *        convert them to lattice derivatives.
     * @param stress_of_block One entry per block, in the order of phi_block.
     */
    void set_stress_from_blocks(const std::vector<std::array<T, 6>>& stress_of_block);

    /**
     * @brief Turn the summed v⊗v components in dq_dl into lattice-parameter
     *        derivatives [dH/dL1, dH/dL2, dH/dL3, dH/dγ, dH/dβ, dH/dα].
     * @see docs/theory/StressTensor.md
     */
    void convert_stress_to_lattice_derivatives();

    /// Throw if stress is requested for a setup that does not support it.
    void check_stress_supported(const std::string& method) const;

private:
    /// Free the concentrations, the fields and the backend (also on a failed construction).
    void release();

public:
    virtual ~ComputationCommon();

    void update_laplacian_operator() override;
    void set_force_off_diagonal_stress(bool force) override;
    void compute_statistics(std::map<std::string, const T*> w_input,
                            std::map<std::string, const T*> q_init = {}) override;

    T get_total_partition(int polymer) override;
    T get_solvent_partition(int s) override;
    void get_total_concentration(std::string monomer_type, T* phi) override;
    void get_total_concentration(int polymer, std::string monomer_type, T* phi) override;
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, T* phi) override;
    void get_block_concentration(int polymer, T* phi) override;
    void get_solvent_concentration(int s, T* phi) override;
};

#endif
