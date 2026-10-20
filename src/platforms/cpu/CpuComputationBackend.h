/**
 * @file CpuComputationBackend.h
 * @brief CPU implementation of ComputationBackend.
 *
 * Fields are host arrays; a "stream" is the OpenMP thread that issues the
 * call, and every call completes before it returns. The propagator solver
 * (RQM4, RK2, CN-ADI or the discrete-chain solver) is chosen from the chain
 * model and the numerical method.
 *
 * @see ComputationBackend for the operation contracts
 */

#ifndef CPU_COMPUTATION_BACKEND_H_
#define CPU_COMPUTATION_BACKEND_H_

#include <climits>
#include <memory>
#include <map>
#include <string>
#include <vector>

#include "ComputationBackend.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "FFT.h"
#include "Molecules.h"
#include "SpaceGroup.h"

template <typename T>
class CpuComputationBackend : public ComputationBackend<T>
{
private:
    ComputationBox<T>* cb;
    SpaceGroup* space_group;
    std::string chain_model;
    std::unique_ptr<CpuSolver<T>> solver;
    int n_streams;
    int n_basis;

    /// Discrete chains with a space group: exp(-w ds) in the reduced basis
    std::map<std::string, std::vector<T>> exp_dw_reduced;

    static T* raw(BackendArray<T>* a) { return reinterpret_cast<T*>(a); }
    static T* raw(CheckpointArray<T>* a) { return reinterpret_cast<T*>(a); }

    void sum_sources(T* q, const std::vector<std::pair<const T*, int>>& sources);
    void multiply_sources(T* q, const std::vector<std::pair<const T*, int>>& sources);

public:
    /**
     * @param method           "pseudospectral" or "realspace" (continuous chains)
     * @param numerical_method "rqm4", "rk2" or "cn-adi2" ("" selects the default)
     * @param max_n_streams    Upper bound on the number of concurrent propagators
     */
    CpuComputationBackend(ComputationBox<T>* cb, Molecules* molecules,
                          std::string method, std::string numerical_method,
                          FFTBackend fft_backend, SpaceGroup* space_group,
                          int max_n_streams = INT_MAX);
    ~CpuComputationBackend();

    int get_n_streams() const override { return n_streams; }
    void synchronize() override {}

    BackendArray<T>* allocate() override;
    void deallocate(BackendArray<T>* a) override;
    void copy_from_host(int stream, BackendArray<T>* dst, const T* src) override;
    void copy_to_host(T* dst, BackendArray<T>* src) override;
    CheckpointArray<T>* allocate_checkpoint() override;
    void deallocate_checkpoint(CheckpointArray<T>* a) override;
    void store_checkpoint(int stream, CheckpointArray<T>* dst, BackendArray<T>* src) override;
    void load_checkpoint(int stream, BackendArray<T>* dst, CheckpointArray<T>* src) override;

    void update_laplacian_operator() override;
    void update_dw(std::map<std::string, const T*> w_input) override;
    void set_force_off_diagonal_stress(bool force) override;
    void advance_propagator(int stream, BackendArray<T>* q_in, BackendArray<T>* q_out,
                            const std::string& monomer_type, int ds_index) override;
    void advance_propagator_half_bond_step(int stream, BackendArray<T>* q_in, BackendArray<T>* q_out,
                                           const std::string& monomer_type) override;
    BackendArray<T>* boltzmann_factor(const std::string& monomer_type) override;

    void set_unity(int stream, BackendArray<T>* q) override;
    void set_zero(int stream, BackendArray<T>* a) override;
    void copy(int stream, BackendArray<T>* dst, BackendArray<T>* src) override;
    void junction_sum(int stream, BackendArray<T>* q,
                      const std::vector<std::pair<BackendArray<T>*, int>>& sources) override;
    void junction_product(int stream, BackendArray<T>* q,
                          const std::vector<std::pair<BackendArray<T>*, int>>& sources) override;
    void junction_sum(int stream, BackendArray<T>* q,
                      const std::vector<std::pair<CheckpointArray<T>*, int>>& sources) override;
    void junction_product(int stream, BackendArray<T>* q,
                          const std::vector<std::pair<CheckpointArray<T>*, int>>& sources) override;
    void apply_mask(int stream, BackendArray<T>* q) override;
    void multiply_boltzmann_factor(int stream, BackendArray<T>* q, const std::string& monomer_type) override;

    T inner_product(BackendArray<T>* a, BackendArray<T>* b) override;
    T inner_product_inverse_weight(BackendArray<T>* a, BackendArray<T>* b, BackendArray<T>* w) override;
    void add_segment_products(int stream, BackendArray<T>* phi, const std::vector<SegmentPair<T>>& pairs) override;
    void normalize_concentration(int stream, BackendArray<T>* phi, T norm, BackendArray<T>* divisor) override;
    std::array<T, 6> block_stress(int stream, const std::vector<SegmentPair<T>>& pairs,
                                  const std::string& monomer_type) override;
    T solvent_concentration(BackendArray<T>* phi, BackendArray<T>* w, double ds, double volume_fraction) override;
    void sum_to_host(T* out, const std::vector<BackendArray<T>*>& arrays, const std::vector<T>& weights) override;
};

#endif
