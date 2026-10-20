/**
 * @file ComputationCommon.cpp
 * @brief Platform-neutral base of the propagator computation classes.
 */

#include <cmath>
#include <complex>

#include "ComputationCommon.h"
#include "PropagatorCode.h"

template <typename T>
ComputationCommon<T>::ComputationCommon(
    ComputationBox<T>* cb,
    Molecules* molecules,
    PropagatorComputationOptimizer* propagator_computation_optimizer,
    SpaceGroup* space_group,
    const std::function<ComputationBackend<T>*()>& create_backend)
    : PropagatorComputation<T>(cb, molecules, propagator_computation_optimizer),
      backend(nullptr), sc(nullptr), n_streams(0)
{
    try
    {
        if (propagator_computation_optimizer->get_computation_propagators().size() == 0)
            throw_with_line_number("There is no propagator code. Add polymers first.");
        if (propagator_computation_optimizer->get_computation_blocks().size() == 0)
            throw_with_line_number("There is no block. Add polymers first.");

        // Set the space group first so that get_n_basis() returns the reduced size
        if (space_group != nullptr)
            PropagatorComputation<T>::set_space_group(space_group);

        backend = create_backend();
        n_streams = backend->get_n_streams();

        for (const auto& item : propagator_computation_optimizer->get_computation_blocks())
            phi_block[item.first] = backend->allocate();
        for (int s = 0; s < molecules->get_n_solvent_types(); s++)
            phi_solvent.push_back(backend->allocate());

        sc = new Scheduler(propagator_computation_optimizer->get_computation_propagators(), n_streams);
    }
    catch (std::exception& exc)
    {
        // The destructor does not run for a partly constructed object
        release();
        throw_without_line_number(exc.what());
    }
}

template <typename T>
ComputationCommon<T>::~ComputationCommon()
{
    release();
}

template <typename T>
void ComputationCommon<T>::release()
{
    delete sc;
    sc = nullptr;
    if (backend != nullptr)
    {
        for (const auto& item : phi_block)
            backend->deallocate(item.second);
        for (const auto& item : phi_solvent)
            backend->deallocate(item);
        for (const auto& item : w)
            backend->deallocate(item.second);
        delete backend;
        backend = nullptr;
    }
    phi_block.clear();
    phi_solvent.clear();
    w.clear();
}

template <typename T>
void ComputationCommon<T>::set_fields(
    const std::map<std::string, const T*>& w_input,
    const std::map<std::string, const T*>& q_init)
{
    for (const auto& [key, edge] : this->propagator_computation_optimizer->get_computation_propagators())
    {
        if (!w_input.contains(edge.monomer_type))
            throw_with_line_number("monomer_type \"" + edge.monomer_type + "\" is not in w_input.");
        if (key[0] == '{' && edge.deps.size() == 0)
        {
            const std::string g = PropagatorCode::get_q_input_idx_from_key(key);
            if (!q_init.contains(g))
                throw_with_line_number("Could not find q_init[\"" + g + "\"]. Pass q_init to run() for grafted polymers.");
        }
    }

    for (const auto& [monomer_type, w_monomer] : w_input)
    {
        if (!w.contains(monomer_type))
            w[monomer_type] = backend->allocate();
        backend->copy_from_host(0, w[monomer_type], w_monomer);
    }

    backend->update_dw(w_input);
}

template <typename T>
void ComputationCommon<T>::compute_solvent_concentrations()
{
    const double ds = this->molecules->get_contour_length_mapping().get_global_ds();
    for (int s = 0; s < this->molecules->get_n_solvent_types(); s++)
    {
        const double volume_fraction   = std::get<0>(this->molecules->get_solvent(s));
        const std::string monomer_type = std::get<1>(this->molecules->get_solvent(s));
        if (!w.contains(monomer_type))
            throw_with_line_number("monomer_type \"" + monomer_type + "\" of solvent " + std::to_string(s) + " is not in w_input.");

        this->single_solvent_partitions[s] = backend->solvent_concentration(
            phi_solvent[s], w[monomer_type], ds, volume_fraction);
    }
}

template <typename T>
void ComputationCommon<T>::check_stress_supported(const std::string& method) const
{
    if (method == "realspace")
        throw_with_line_number("Currently, the real-space method does not support stress computation.");

    // Non-periodic (DCT/DST) stress is supported for real fields; the
    // per-mode Parseval weights are folded into the fourier_basis tables
    // (Pseudo::update_weighted_fourier_basis_mixed). The real-coefficient
    // transform path drops the imaginary part, so complex fields with
    // non-periodic boundaries are rejected explicitly.
    if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        for (const auto& bc : this->cb->get_boundary_conditions())
            if (bc != BoundaryCondition::PERIODIC)
                throw_with_line_number("Stress computation with non-periodic boundary "
                    "conditions is not supported for complex fields.");
    }
}

template <typename T>
void ComputationCommon<T>::convert_stress_to_lattice_derivatives()
{
    // The solvers sum v⊗v components directly, where v = 2π g⁻¹ m is the
    // deformation vector (units: 1/L²): V₁₁ = Σ(kernel × v₁²), V₁₂ = Σ(kernel × v₁v₂), ...
    //
    // The metric tensor g = hᵀh has simple derivatives:
    //   g₁₁ = L₁², g₂₂ = L₂², g₃₃ = L₃²
    //   g₁₂ = L₁L₂cosγ, g₁₃ = L₁L₃cosβ, g₂₃ = L₂L₃cosα
    //
    // Lattice parameter derivatives:
    //   ∂H/∂L₁ ∝ L₁V₁₁ + L₂cosγ·V₁₂ + L₃cosβ·V₁₃
    //   ∂H/∂γ  ∝ -L₁L₂sinγ·V₁₂
    //
    // @see docs/theory/StressTensor.md for derivation
    const int DIM = this->cb->get_dim();
    const int M   = this->cb->get_total_grid();

    const double L1 = this->cb->get_lx(0);
    const double L2 = (DIM >= 2) ? this->cb->get_lx(1) : 1.0;
    const double L3 = (DIM >= 3) ? this->cb->get_lx(2) : 1.0;

    const std::vector<double> angles = this->cb->get_angles();
    const double cos_a = std::cos(angles[0]);  // alpha: between b and c
    const double cos_b = std::cos(angles[1]);  // beta: between a and c
    const double cos_g = std::cos(angles[2]);  // gamma: between a and b
    const double sin_a = std::sin(angles[0]);
    const double sin_b = std::sin(angles[1]);
    const double sin_g = std::sin(angles[2]);

    // Normalization factor (from the Boltzmann factor derivative); local_ds is
    // already multiplied per block by the caller.
    const double norm = -3.0 * M * M;

    for (int p = 0; p < this->molecules->get_n_polymer_types(); p++)
    {
        T V_11 = this->dq_dl[p][0];
        T V_22 = (DIM >= 2) ? this->dq_dl[p][1] : T(0.0);
        T V_33 = T(0.0), V_12 = T(0.0), V_13 = T(0.0), V_23 = T(0.0);
        if (DIM == 3)
        {
            V_33 = this->dq_dl[p][2];
            V_12 = this->dq_dl[p][3];
            V_13 = this->dq_dl[p][4];
            V_23 = this->dq_dl[p][5];
        }
        else if (DIM == 2)
        {
            V_12 = this->dq_dl[p][2];
        }

        // ∂H/∂Lᵢ = Σⱼₖ Vⱼₖ ∂gⱼₖ/∂Lᵢ (the factor 2 of the diagonal is in the normalization)
        this->dq_dl[p][0] = (L1*V_11 + L2*cos_g*V_12 + L3*cos_b*V_13) / norm;
        if (DIM >= 2)
            this->dq_dl[p][1] = (L2*V_22 + L1*cos_g*V_12 + L3*cos_a*V_23) / norm;
        if (DIM >= 3)
            this->dq_dl[p][2] = (L3*V_33 + L1*cos_b*V_13 + L2*cos_a*V_23) / norm;

        // ∂g₁₂/∂γ = -L₁L₂sinγ → ∂H/∂γ = -L₁L₂sinγ·V₁₂, and likewise for β and α
        if (DIM == 3)
        {
            this->dq_dl[p][3] = -L1 * L2 * sin_g * V_12 / norm;  // dH/dγ
            this->dq_dl[p][4] = -L1 * L3 * sin_b * V_13 / norm;  // dH/dβ
            this->dq_dl[p][5] = -L2 * L3 * sin_a * V_23 / norm;  // dH/dα
        }
        else if (DIM == 2)
        {
            this->dq_dl[p][2] = -L1 * L2 * sin_g * V_12 / norm;  // dH/dγ
        }
    }
}

template <typename T>
void ComputationCommon<T>::update_laplacian_operator()
{
    try
    {
        backend->update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationCommon<T>::set_force_off_diagonal_stress(bool force)
{
    PropagatorComputation<T>::set_force_off_diagonal_stress(force);
    backend->set_force_off_diagonal_stress(force);
}

template <typename T>
void ComputationCommon<T>::compute_statistics(
    std::map<std::string, const T*> w_input,
    std::map<std::string, const T*> q_init)
{
    this->compute_propagators(w_input, q_init);
    this->compute_concentrations();
}

template <typename T>
T ComputationCommon<T>::get_total_partition(int polymer)
{
    const int P = this->molecules->get_n_polymer_types();
    if (polymer < 0 || polymer > P-1)
        throw_with_line_number("Index (" + std::to_string(polymer) + ") must be in range [0, " + std::to_string(P-1) + "]");
    return this->single_polymer_partitions[polymer];
}

template <typename T>
T ComputationCommon<T>::get_solvent_partition(int s)
{
    const int S = this->molecules->get_n_solvent_types();
    if (s < 0 || s > S-1)
        throw_with_line_number("Index (" + std::to_string(s) + ") must be in range [0, " + std::to_string(S-1) + "]");
    return this->single_solvent_partitions[s];
}

template <typename T>
void ComputationCommon<T>::get_total_concentration(std::string monomer_type, T* phi)
{
    try
    {
        std::vector<BackendArray<T>*> arrays;
        std::vector<T> weights;
        for (const auto& [key, phi_b] : phi_block)
        {
            const std::string& key_left = std::get<1>(key);
            const int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            if (PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                arrays.push_back(phi_b);
                weights.push_back(1.0);
            }
        }
        for (int s = 0; s < this->molecules->get_n_solvent_types(); s++)
        {
            if (std::get<1>(this->molecules->get_solvent(s)) == monomer_type)
            {
                arrays.push_back(phi_solvent[s]);
                weights.push_back(1.0);
            }
        }
        backend->sum_to_host(phi, arrays, weights);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationCommon<T>::get_total_concentration(int p, std::string monomer_type, T* phi)
{
    try
    {
        const int P = this->molecules->get_n_polymer_types();
        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        std::vector<BackendArray<T>*> arrays;
        std::vector<T> weights;
        for (const auto& [key, phi_b] : phi_block)
        {
            const std::string& key_left = std::get<1>(key);
            const int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            if (std::get<0>(key) == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                arrays.push_back(phi_b);
                weights.push_back(1.0);
            }
        }
        backend->sum_to_host(phi, arrays, weights);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationCommon<T>::get_total_concentration_gce(double fugacity, int p, std::string monomer_type, T* phi)
{
    try
    {
        const int P = this->molecules->get_n_polymer_types();
        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        // phi_block is normalized by volume_fraction/(alpha Q); the grand-canonical
        // concentration is normalized by the fugacity instead
        Polymer& pc = this->molecules->get_polymer(p);
        const T norm = fugacity/pc.get_volume_fraction()*pc.get_alpha()*this->single_polymer_partitions[p];

        std::vector<BackendArray<T>*> arrays;
        std::vector<T> weights;
        for (const auto& [key, phi_b] : phi_block)
        {
            const std::string& key_left = std::get<1>(key);
            const int n_segment_right = this->propagator_computation_optimizer->get_computation_block(key).n_segment_right;
            if (std::get<0>(key) == p && PropagatorCode::get_monomer_type_from_key(key_left) == monomer_type && n_segment_right != 0)
            {
                arrays.push_back(phi_b);
                weights.push_back(norm);
            }
        }
        backend->sum_to_host(phi, arrays, weights);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationCommon<T>::get_block_concentration(int p, T* phi)
{
    try
    {
        const int N = this->cb->get_n_basis();
        const int P = this->molecules->get_n_polymer_types();
        if (p < 0 || p > P-1)
            throw_with_line_number("Index (" + std::to_string(p) + ") must be in range [0, " + std::to_string(P-1) + "]");

        if (this->propagator_computation_optimizer->use_aggregation())
            throw_with_line_number("Disable 'aggregation' option to invoke 'get_block_concentration'.");

        Polymer& pc = this->molecules->get_polymer(p);
        std::vector<Block>& blocks = pc.get_blocks();
        for (size_t b = 0; b < blocks.size(); b++)
        {
            std::string key_left  = pc.get_propagator_key(blocks[b].v, blocks[b].u);
            std::string key_right = pc.get_propagator_key(blocks[b].u, blocks[b].v);
            if (key_left < key_right)
                key_left.swap(key_right);

            auto block_it = phi_block.find(std::make_tuple(p, key_left, key_right));
            if (block_it == phi_block.end())
                throw_with_line_number("Block (" + std::to_string(p) + ", " + key_left + ", " + key_right + ") is not in phi_block.");
            backend->copy_to_host(&phi[b*N], block_it->second);
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

template <typename T>
void ComputationCommon<T>::get_solvent_concentration(int s, T* phi)
{
    try
    {
        const int S = this->molecules->get_n_solvent_types();
        if (s < 0 || s > S-1)
            throw_with_line_number("Index (" + std::to_string(s) + ") must be in range [0, " + std::to_string(S-1) + "]");
        backend->copy_to_host(phi, phi_solvent[s]);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
#include "TemplateInstantiations.h"
INSTANTIATE_CLASS(ComputationCommon);
