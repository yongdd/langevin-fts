/**
 * @file PropagatorComputation.cpp
 * @brief Implementation of PropagatorComputation base class.
 *
 * Provides the base implementation for computing chain propagators and
 * partition functions in polymer field theory. Platform-specific solvers
 * (CpuComputationContinuous, CudaComputationDiscrete, etc.) derive from
 * this class to implement the actual propagator integration.
 *
 * **Propagator Computation:**
 *
 * Chain propagators q(r,s) satisfy the modified diffusion equation:
 *     ∂q/∂s = (b²/6)∇²q - w(r)q
 *
 * where b is statistical segment length and w(r) is the potential field.
 *
 * **Stress Calculation:**
 *
 * Box stress is computed from the derivative of partition function:
 *     σ_d = -(1/Q) × Σ_p φ_p/α_p × dQ_p/dL_d
 *
 * where φ_p is volume fraction, α_p is chain length, and dQ_p/dL_d is
 * the box length derivative of single-chain partition function.
 *
 * **Template Instantiations:**
 *
 * - PropagatorComputation<double>: Real fields (periodic boundaries)
 * - PropagatorComputation<std::complex<double>>: Complex fields
 *
 * @see CpuComputationContinuous for CPU implementation
 * @see CudaComputationContinuous for GPU implementation
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <complex>

#include "PropagatorComputation.h"

/**
 * @brief Construct propagator computation engine.
 *
 * Initializes storage for partition functions and stress derivatives.
 * Actual propagator memory is allocated in derived classes.
 *
 * @param cb                             Computation box for grid operations
 * @param molecules                      Polymer/solvent species definitions
 * @param propagator_computation_optimizer Optimized computation schedule
 *
 * @throws Exception if cb or molecules is null
 */
template <typename T>
PropagatorComputation<T>::PropagatorComputation(
    ComputationBox<T>* cb,
    Molecules *molecules,
    PropagatorComputationOptimizer *propagator_computation_optimizer)
{
    if (cb == nullptr)
        throw_with_line_number("ComputationBox<double>* cb is a null pointer");
    if (molecules == nullptr)
        throw_with_line_number("Molecules *molecules is a null pointer");

    this->cb = cb;
    this->molecules = molecules;
    this->propagator_computation_optimizer = propagator_computation_optimizer;
    this->space_group_ = nullptr;  // No space group by default

    // Total partition functions for each polymer
    single_polymer_partitions.resize(molecules->get_n_polymer_types());

    // Total partition functions for each solvent
    single_solvent_partitions.resize(molecules->get_n_solvent_types());

    // Allocate memory for dq_dl (6 components: xx, yy, zz, xy, xz, yz)
    for(int p=0; p<molecules->get_n_polymer_types(); p++){
        dq_dl.push_back({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    }
}
template <typename T>
PropagatorComputation<T>::~PropagatorComputation()
{
    // Vectors handle memory cleanup automatically
}

/**
 * @brief Compute stress tensor from propagator derivatives.
 *
 * Calculates the stress tensor using the canonical ensemble formula:
 *     σ_ij = Σ_p (φ_p/α_p) × (dQ_p/dε_ij) / Q_p
 *
 * @return Vector of stress values [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
 *         For 1D/2D: cross-terms are zero, only relevant components used.
 *
 * @note dq_dl must be populated by compute_stress() in derived class
 */
template <typename T>
std::vector<T> PropagatorComputation<T>::get_stress()
{
    const int N_STRESS = 6;  // Full stress tensor: xx, yy, zz, xy, xz, yz
    std::vector<T> stress(N_STRESS);

    int n_polymer_types = this->molecules->get_n_polymer_types();
    for(int d=0; d<N_STRESS; d++)
        stress[d] = 0.0;

    for(int p=0; p<n_polymer_types; p++){
        Polymer& pc = this->molecules->get_polymer(p);
        for(int d=0; d<N_STRESS; d++){
            stress[d] += (this->dq_dl[p][d] * pc.get_volume_fraction() /
                 pc.get_alpha()) / this->single_polymer_partitions[p];
        }
    }
    return stress;
}

/**
 * @brief Compute stress tensor for grand canonical ensemble.
 *
 * Uses fugacities instead of volume fractions for stress calculation:
 *     σ_ij = Σ_p z_p × (dQ_p/dε_ij)
 *
 * where z_p is the fugacity of polymer species p.
 *
 * @param fugacities Vector of polymer fugacities
 * @return Vector of stress values [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
 */
template <typename T>
std::vector<T> PropagatorComputation<T>::get_stress_gce(std::vector<double> fugacities)
{
    const int N_STRESS = 6;  // Full stress tensor: xx, yy, zz, xy, xz, yz
    std::vector<T> stress(N_STRESS);

    int n_polymer_types = this->molecules->get_n_polymer_types();
    for(int d=0; d<N_STRESS; d++)
        stress[d] = 0.0;

    for(int p=0; p<n_polymer_types; p++){
        for(int d=0; d<N_STRESS; d++){
            stress[d] += static_cast<T>(fugacities[p]) * this->dq_dl[p][d];
        }
    }
    return stress;
}

// ==================== Space Group / Reduced Basis Methods ====================

/**
 * @brief Set space group for reduced basis representation.
 *
 * Allocates internal buffers for full grid field storage when working
 * with reduced basis fields. Also sets space group on ComputationBox
 * so that integral(), inner_product() etc. work with reduced basis.
 *
 * @param sg Pointer to SpaceGroup, or nullptr to disable reduced basis mode
 */
template <typename T>
void PropagatorComputation<T>::set_space_group(SpaceGroup* sg)
{
    space_group_ = sg;

    // Also set space group on ComputationBox for unified field operations
    this->cb->set_space_group(sg);

    if (sg != nullptr) {
        // Allocate buffers for full grid operations
        int total_grid = this->cb->get_total_grid();
        int n_monomer_types = this->molecules->get_bond_lengths().size();

        // Buffer for all monomer type fields
        w_full_buffer_.resize(n_monomer_types * total_grid);
        phi_full_buffer_.resize(total_grid);
    } else {
        // Release buffers
        w_full_buffer_.clear();
        w_full_buffer_.shrink_to_fit();
        phi_full_buffer_.clear();
        phi_full_buffer_.shrink_to_fit();
    }
}

/**
 * @brief Compute propagators using reduced basis input fields.
 *
 * Expands reduced basis fields to full grid before calling compute_propagators().
 * The propagator solver needs full grid fields for FFT operations.
 *
 * The derived class (e.g., CpuComputationContinuous) stores propagators in
 * reduced basis internally and handles expand/reduce around FFT operations.
 */
template <typename T>
void PropagatorComputation<T>::compute_propagators_reduced(
    std::map<std::string, const T*> w_reduced,
    std::map<std::string, const T*> q_init)
{
    if (space_group_ == nullptr) {
        throw_with_line_number("Space group not set. Call set_space_group() first.");
    }

    int total_grid = this->cb->get_total_grid();

    // Expand each reduced basis field to full grid for the solver
    std::map<std::string, const T*> w_full;
    int field_idx = 0;

    for (auto& pair : w_reduced) {
        T* full_ptr = w_full_buffer_.data() + field_idx * total_grid;
        if constexpr (std::is_same_v<T, double>) {
            space_group_->from_reduced_basis(pair.second, full_ptr, 1);
        }
        w_full[pair.first] = full_ptr;
        field_idx++;
    }

    // Call the standard compute_propagators with full grid fields
    // Propagators are stored in reduced basis by the derived class
    compute_propagators(w_full, q_init);
}

/**
 * @brief Get concentration in reduced basis.
 *
 * When concentrations are stored in reduced basis (after set_space_group()),
 * this method avoids the round-trip through full grid by getting the full
 * grid concentration and reducing it. The derived class stores concentrations
 * in reduced basis, so get_total_concentration() expands internally first.
 *
 * For efficiency, this still uses get_total_concentration() which expands
 * and accumulates, then we reduce the result. A more optimal implementation
 * would accumulate directly in reduced basis, but that requires access to
 * internal phi_block which is in the derived class.
 */
template <typename T>
void PropagatorComputation<T>::get_total_concentration_reduced(std::string monomer_type, T* phi_reduced)
{
    if (space_group_ == nullptr) {
        throw_with_line_number("Space group not set. Call set_space_group() first.");
    }

    // Get full grid concentration (derived class expands internally)
    get_total_concentration(monomer_type, phi_full_buffer_.data());

    // Convert to reduced basis
    if constexpr (std::is_same_v<T, double>) {
        space_group_->to_reduced_basis(phi_full_buffer_.data(), phi_reduced, 1);
    }
}

// Explicit template instantiation
template class PropagatorComputation<double>;
template class PropagatorComputation<std::complex<double>>;