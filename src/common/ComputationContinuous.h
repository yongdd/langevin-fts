/**
 * @file ComputationContinuous.h
 * @brief Propagator computation for continuous (Gaussian) chains, full storage.
 *
 * Solves the modified diffusion equation
 *
 *     ∂q/∂s = (b²/6)∇²q - w(r)q
 *
 * for every propagator the optimizer selected, keeping q(r, s) at every
 * contour step, and integrates the concentrations and stress along the
 * contour with Simpson's rule:
 *
 *     φ(r) = (1/Q) ∫ q(r,s) q†(r,N-s) ds
 *
 * The class is platform neutral: the grid work goes through a
 * ComputationBackend, which the platform factories (MklFactory, FftwFactory,
 * CudaFactory) pass in.
 *
 * **Propagator storage:** propagator[key][n] for n = 0..max_n_segment, where
 * n = 0 is the initial condition.
 *
 * @see ComputationCommon for the shared concentration accessors
 * @see ComputationBackend for the platform operations
 */

#ifndef COMPUTATION_CONTINUOUS_H_
#define COMPUTATION_CONTINUOUS_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ComputationCommon.h"

template <typename T>
class ComputationContinuous : public ComputationCommon<T>
{
private:
    std::string method;  ///< "pseudospectral" or "realspace"

    std::map<std::string, std::vector<BackendArray<T>*>> propagator;  ///< propagator[key][n], n = 0..max_n_segment

    /// One segment per polymer for its partition function: (polymer, q, q†, n_aggregated)
    std::vector<std::tuple<int, BackendArray<T>*, BackendArray<T>*, int>> single_partition_segment;

    /// Free the propagator storage (also on a failed construction).
    void free_propagators();

    /// Sources (q_i, n_i) of a junction or an aggregated propagator.
    std::vector<std::pair<BackendArray<T>*, int>> junction_sources(
        const std::vector<std::tuple<std::string, int, int>>& deps);

public:
    /**
     * @param method         "pseudospectral" or "realspace" (stress is pseudo-spectral only)
     * @param create_backend Creates the platform backend (see ComputationCommon)
     */
    ComputationContinuous(ComputationBox<T>* cb, Molecules* molecules,
                          PropagatorComputationOptimizer* propagator_computation_optimizer,
                          std::string method, SpaceGroup* space_group,
                          const std::function<std::unique_ptr<ComputationBackend<T>>()>& create_backend);
    ~ComputationContinuous();

    void compute_propagators(std::map<std::string, const T*> w_input,
                             std::map<std::string, const T*> q_init = {}) override;
    void advance_propagator_single_segment(T* q_init, T* q_out, int p, int v, int u) override;
    void compute_concentrations() override;
    void compute_stress() override;
    void get_chain_propagator(T* q_out, int polymer, int v, int u, int n) override;
    std::vector<double> compute_partition_errors() override;
};

#endif
