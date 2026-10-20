/**
 * @file ComputationDiscrete.h
 * @brief Propagator computation for discrete (bead-spring) chains, full storage.
 *
 * Solves the Chapman-Kolmogorov recursion
 *
 *     q(r, n+1) = exp(-w(r) ds) ∫ g(r-r') q(r', n) dr'
 *
 * where g is the bond function (Park et al., J. Chem. Phys. 150, 234901
 * (2019)), keeping q(r, n) at every segment, and sums the concentrations
 * and stress over the segments:
 *
 *     φ(r) = (1/Q) Σ_n q(r,n) q†(r,N+1-n) / exp(-w(r) ds)
 *
 * The division removes the Boltzmann factor of segment n, which both
 * propagators contain.
 *
 * **Half bonds.** A junction sits between segments, half a bond away from
 * each of them. The propagators that meet there are stored half a bond past
 * their last segment, q(r, n+1/2) = g^(1/2) * q(r, n), in
 * propagator_half_steps[key][n] for every n in the optimizer's junction_ends;
 * propagator_half_steps[key][0] holds the junction value a propagator starts
 * from.
 *
 * The class is platform neutral: the grid work goes through a
 * ComputationBackend, which the platform factories (MklFactory, FftwFactory,
 * CudaFactory) pass in.
 *
 * **Propagator storage:** propagator[key][n] for n = 1..max_n_segment
 * (segment n; index 0 is unused).
 *
 * @see ComputationCommon for the shared concentration accessors
 * @see ComputationBackend for the platform operations
 */

#ifndef COMPUTATION_DISCRETE_H_
#define COMPUTATION_DISCRETE_H_

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "ComputationCommon.h"

template <typename T>
class ComputationDiscrete : public ComputationCommon<T>
{
private:
    std::map<std::string, BackendArray<T>**> propagator;             ///< q(r, n), n = 1..max_n_segment
    std::map<std::string, BackendArray<T>**> propagator_half_steps;  ///< q(r, n+1/2) where needed, else nullptr
    std::map<std::string, int> propagator_size;                      ///< max_n_segment + 1 per key

    /// One segment per polymer for its partition function: (polymer, q, q†, monomer type, n_aggregated)
    std::vector<std::tuple<int, BackendArray<T>*, BackendArray<T>*, std::string, int>> single_partition_segment;

    /// Free the propagator storage (also on a failed construction).
    void free_propagators();

public:
    ComputationDiscrete(ComputationBox<T>* cb, Molecules* molecules,
                        PropagatorComputationOptimizer* propagator_computation_optimizer,
                        SpaceGroup* space_group,
                        const std::function<ComputationBackend<T>*()>& create_backend);
    ~ComputationDiscrete();

    void compute_propagators(std::map<std::string, const T*> w_input,
                             std::map<std::string, const T*> q_init = {}) override;
    void advance_propagator_single_segment(T* q_init, T* q_out, int p, int v, int u) override;
    void compute_concentrations() override;
    void compute_stress() override;
    void get_chain_propagator(T* q_out, int polymer, int v, int u, int n) override;
    bool check_total_partition() override;
};

#endif
