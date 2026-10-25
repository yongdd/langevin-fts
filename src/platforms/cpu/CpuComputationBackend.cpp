/**
 * @file CpuComputationBackend.cpp
 * @brief CPU implementation of ComputationBackend.
 */

#include <algorithm>
#include <cmath>
#include <complex>

#include "CpuComputationBackend.h"
#include "NumericalMethods.h"
#include "CpuSolverCNADI.h"
#include "CpuSolverPseudoDiscrete.h"
#include "CpuSolverPseudoRK2.h"
#include "CpuSolverPseudoRQM4.h"

template <typename T>
CpuComputationBackend<T>::CpuComputationBackend(
    ComputationBox<T>* cb, Molecules* molecules,
    const NumericalMethodInfo& method_info,
    FFTBackend fft_backend, SpaceGroup* space_group, int max_n_streams)
    : cb(cb), space_group(space_group), chain_model(molecules->get_model_name())
{
    n_basis = cb->get_n_basis();
    n_streams = std::min(this->n_streams_from_env(/*default_n_streams=*/4), max_n_streams);

    switch (method_info.id)
    {
    case NumericalMethodId::Discrete:
        solver = std::make_unique<CpuSolverPseudoDiscrete<T>>(cb, molecules, fft_backend);
        break;
    case NumericalMethodId::RQM4:
        solver = std::make_unique<CpuSolverPseudoRQM4<T>>(cb, molecules, fft_backend);
        break;
    case NumericalMethodId::RK2:
        solver = std::make_unique<CpuSolverPseudoRK2<T>>(cb, molecules, fft_backend);
        break;
    case NumericalMethodId::CNADI2:
        if constexpr (std::is_same_v<T, double>)
            solver = std::make_unique<CpuSolverCNADI>(cb, molecules);
        break;
    }

    // The solver expands/reduces around its FFTs when a space group is set
    // (the owning member releases the solver if this throws)
    if (space_group != nullptr)
        solver->set_space_group(space_group);
    solver->update_laplacian_operator();
}

template <typename T>
CpuComputationBackend<T>::~CpuComputationBackend() = default;

// ==================== Memory ====================

template <typename T>
BackendArray<T>* CpuComputationBackend<T>::allocate()
{
    return reinterpret_cast<BackendArray<T>*>(new T[n_basis]());
}

template <typename T>
void CpuComputationBackend<T>::deallocate(BackendArray<T>* a)
{
    delete[] raw(a);
}

template <typename T>
void CpuComputationBackend<T>::copy_from_host(int, BackendArray<T>* dst, const T* src)
{
    std::copy(src, src + n_basis, raw(dst));
}

template <typename T>
void CpuComputationBackend<T>::copy_to_host(T* dst, BackendArray<T>* src)
{
    std::copy(raw(src), raw(src) + n_basis, dst);
}

template <typename T>
CheckpointArray<T>* CpuComputationBackend<T>::allocate_checkpoint()
{
    return reinterpret_cast<CheckpointArray<T>*>(new T[n_basis]);
}

template <typename T>
void CpuComputationBackend<T>::deallocate_checkpoint(CheckpointArray<T>* a)
{
    delete[] raw(a);
}

template <typename T>
void CpuComputationBackend<T>::store_checkpoint(int, CheckpointArray<T>* dst, BackendArray<T>* src)
{
    std::copy(raw(src), raw(src) + n_basis, raw(dst));
}

template <typename T>
void CpuComputationBackend<T>::load_checkpoint(int, BackendArray<T>* dst, CheckpointArray<T>* src)
{
    std::copy(raw(src), raw(src) + n_basis, raw(dst));
}

// ==================== Solver ====================

template <typename T>
void CpuComputationBackend<T>::update_laplacian_operator()
{
    solver->update_laplacian_operator();
}

template <typename T>
void CpuComputationBackend<T>::update_dw(std::map<std::string, const T*> w_input)
{
    solver->update_dw(w_input);

    // The discrete solver keeps exp(-w ds) on the full grid unless its CrysFFT
    // path stores it in the reduced basis already
    if (chain_model == "discrete" && space_group != nullptr)
    {
        for (const auto& [monomer_type, exp_dw_solver] : solver->exp_dw.at(0))
        {
            std::vector<T>& reduced = exp_dw_reduced[monomer_type];
            reduced.resize(n_basis);
            if (static_cast<int>(exp_dw_solver.size()) == n_basis)
                std::copy(exp_dw_solver.begin(), exp_dw_solver.end(), reduced.begin());
            else if constexpr (std::is_same_v<T, double>)
                space_group->to_reduced_basis(exp_dw_solver.data(), reduced.data(), 1);
            else
                throw_with_line_number("Discrete chains with a space group require real fields.");
        }
    }
}

template <typename T>
void CpuComputationBackend<T>::set_force_off_diagonal_stress(bool force)
{
    solver->set_force_off_diagonal_stress(force);
}

template <typename T>
void CpuComputationBackend<T>::advance_propagator(
    int, BackendArray<T>* q_in, BackendArray<T>* q_out, const std::string& monomer_type, int ds_index)
{
    solver->advance_propagator(raw(q_in), raw(q_out), monomer_type, cb->get_mask(), ds_index);
}

template <typename T>
void CpuComputationBackend<T>::advance_propagator_half_bond_step(
    int, BackendArray<T>* q_in, BackendArray<T>* q_out, const std::string& monomer_type)
{
    solver->advance_propagator_half_bond_step(raw(q_in), raw(q_out), monomer_type);
}

template <typename T>
BackendArray<T>* CpuComputationBackend<T>::boltzmann_factor(const std::string& monomer_type)
{
    if (space_group != nullptr)
        return reinterpret_cast<BackendArray<T>*>(exp_dw_reduced.at(monomer_type).data());
    return reinterpret_cast<BackendArray<T>*>(solver->exp_dw.at(0).at(monomer_type).data());
}

// ==================== Propagator initial conditions ====================

template <typename T>
void CpuComputationBackend<T>::set_unity(int, BackendArray<T>* q)
{
    std::fill(raw(q), raw(q) + n_basis, T(1.0));
}

template <typename T>
void CpuComputationBackend<T>::set_zero(int, BackendArray<T>* a)
{
    std::fill(raw(a), raw(a) + n_basis, T(0.0));
}

template <typename T>
void CpuComputationBackend<T>::copy(int, BackendArray<T>* dst, BackendArray<T>* src)
{
    std::copy(raw(src), raw(src) + n_basis, raw(dst));
}

template <typename T>
void CpuComputationBackend<T>::sum_sources(T* q, const std::vector<std::pair<const T*, int>>& sources)
{
    for (int i = 0; i < n_basis; i++)
        q[i] = 0.0;
    for (const auto& [src, n_repeated] : sources)
        for (int i = 0; i < n_basis; i++)
            q[i] += src[i]*static_cast<double>(n_repeated);
}

template <typename T>
void CpuComputationBackend<T>::multiply_sources(T* q, const std::vector<std::pair<const T*, int>>& sources)
{
    for (int i = 0; i < n_basis; i++)
        q[i] = 1.0;
    for (const auto& [src, n_repeated] : sources)
        for (int i = 0; i < n_basis; i++)
            q[i] *= pow(src[i], n_repeated);
}

template <typename T>
void CpuComputationBackend<T>::junction_sum(
    int, BackendArray<T>* q, const std::vector<std::pair<BackendArray<T>*, int>>& sources)
{
    std::vector<std::pair<const T*, int>> host_sources;
    for (const auto& [src, n_repeated] : sources)
        host_sources.push_back({raw(src), n_repeated});
    sum_sources(raw(q), host_sources);
}

template <typename T>
void CpuComputationBackend<T>::junction_product(
    int, BackendArray<T>* q, const std::vector<std::pair<BackendArray<T>*, int>>& sources)
{
    std::vector<std::pair<const T*, int>> host_sources;
    for (const auto& [src, n_repeated] : sources)
        host_sources.push_back({raw(src), n_repeated});
    multiply_sources(raw(q), host_sources);
}

template <typename T>
void CpuComputationBackend<T>::junction_sum(
    int, BackendArray<T>* q, const std::vector<std::pair<CheckpointArray<T>*, int>>& sources)
{
    std::vector<std::pair<const T*, int>> host_sources;
    for (const auto& [src, n_repeated] : sources)
        host_sources.push_back({raw(src), n_repeated});
    sum_sources(raw(q), host_sources);
}

template <typename T>
void CpuComputationBackend<T>::junction_product(
    int, BackendArray<T>* q, const std::vector<std::pair<CheckpointArray<T>*, int>>& sources)
{
    std::vector<std::pair<const T*, int>> host_sources;
    for (const auto& [src, n_repeated] : sources)
        host_sources.push_back({raw(src), n_repeated});
    multiply_sources(raw(q), host_sources);
}

template <typename T>
void CpuComputationBackend<T>::apply_mask(int, BackendArray<T>* q)
{
    const double* mask = cb->get_mask();
    if (mask == nullptr)
        return;
    T* _q = raw(q);
    for (int i = 0; i < n_basis; i++)
        _q[i] *= mask[i];
}

template <typename T>
void CpuComputationBackend<T>::multiply_boltzmann_factor(int, BackendArray<T>* q, const std::string& monomer_type)
{
    T* _q = raw(q);
    const T* exp_dw = raw(boltzmann_factor(monomer_type));
    for (int i = 0; i < n_basis; i++)
        _q[i] *= exp_dw[i];
}

// ==================== Statistics ====================

template <typename T>
T CpuComputationBackend<T>::inner_product(BackendArray<T>* a, BackendArray<T>* b)
{
    return cb->inner_product(raw(a), raw(b));
}

template <typename T>
T CpuComputationBackend<T>::inner_product_inverse_weight(BackendArray<T>* a, BackendArray<T>* b, BackendArray<T>* w)
{
    return cb->inner_product_inverse_weight(raw(a), raw(b), raw(w));
}

template <typename T>
void CpuComputationBackend<T>::add_segment_products(
    int, BackendArray<T>* phi, const std::vector<SegmentPair<T>>& pairs)
{
    T* _phi = raw(phi);
    for (const SegmentPair<T>& pair : pairs)
    {
        const T* q_1 = raw(pair.q_1);
        const T* q_2 = raw(pair.q_2);
        const double weight = pair.weight;
        for (int i = 0; i < n_basis; i++)
            _phi[i] += weight*q_1[i]*q_2[i];
    }
}

template <typename T>
void CpuComputationBackend<T>::normalize_concentration(int, BackendArray<T>* phi, T norm, BackendArray<T>* divisor)
{
    T* _phi = raw(phi);
    if (divisor != nullptr)
    {
        const T* _divisor = raw(divisor);
        for (int i = 0; i < n_basis; i++)
            _phi[i] /= _divisor[i];
    }
    for (int i = 0; i < n_basis; i++)
        _phi[i] *= norm;
}

template <typename T>
std::array<T, 6> CpuComputationBackend<T>::block_stress(
    int, const std::vector<SegmentPair<T>>& pairs, const std::string& monomer_type)
{
    std::array<T, 6> stress = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (const SegmentPair<T>& pair : pairs)
    {
        // The solver expands/reduces internally when a space group is set
        std::vector<T> segment_stress = solver->compute_single_segment_stress(
            raw(pair.q_1), raw(pair.q_2), monomer_type, pair.is_half_bond_length);
        for (int d = 0; d < 6; d++)
            stress[d] += segment_stress[d]*pair.weight;
    }
    return stress;
}

template <typename T>
T CpuComputationBackend<T>::solvent_concentration(
    BackendArray<T>* phi, BackendArray<T>* w, double ds, double volume_fraction)
{
    T* _phi = raw(phi);
    const T* _w = raw(w);
    for (int i = 0; i < n_basis; i++)
        _phi[i] = exp(-_w[i] * ds);

    const T partition = cb->mean(_phi);
    const T norm = volume_fraction / partition;
    for (int i = 0; i < n_basis; i++)
        _phi[i] *= norm;

    // No solvent inside impenetrable regions (the mask is in the field basis)
    const double* mask = cb->get_mask();
    if (mask != nullptr)
        for (int i = 0; i < n_basis; i++)
            _phi[i] *= mask[i];
    return partition;
}

template <typename T>
void CpuComputationBackend<T>::sum_to_host(
    T* out, const std::vector<BackendArray<T>*>& arrays, const std::vector<T>& weights)
{
    for (int i = 0; i < n_basis; i++)
        out[i] = 0.0;
    for (size_t a = 0; a < arrays.size(); a++)
    {
        const T* _a = raw(arrays[a]);
        const T weight = weights[a];
        for (int i = 0; i < n_basis; i++)
            out[i] += _a[i]*weight;
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(CpuComputationBackend);
