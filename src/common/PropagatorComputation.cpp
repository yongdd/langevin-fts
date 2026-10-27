/**
 * @file PropagatorComputation.cpp
 * @brief Implementation of PropagatorComputation base class.
 *
 * Provides the base implementation for computing chain propagators and
 * partition functions in polymer field theory. The platform-neutral
 * computations (ComputationContinuous, ComputationDiscrete and their
 * reduce-memory variants) derive from this class.
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
 * The "stress" vector holds lattice-parameter derivatives of the free energy:
 *     stress = [dH/dL₁, dH/dL₂, dH/dL₃, dH/dγ, dH/dβ, dH/dα]
 *
 * Each polymer contributes (φ_p/α_p)·dq_dl_p/Q_p, where dq_dl_p is filled by
 * compute_stress() in the derived class (deformation-vector approach, see
 * docs/theory/StressTensor.md). Box relaxation performs gradient descent
 * with the same negative sign for lengths and angles.
 *
 * **Template Instantiations:**
 *
 * - PropagatorComputation<double>: Real fields (periodic boundaries)
 * - PropagatorComputation<std::complex<double>>: Complex fields
 *
 * @see ComputationContinuous for the continuous-chain implementation
 * @see ComputationBackend for the CPU and CUDA operations
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <complex>

#include "PropagatorComputation.h"
#include "ValidationUtils.h"

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

    // Allocate memory for dq_dl (6 components: dH/dL1..3, dH/dγ, dH/dβ, dH/dα)
    for(int p=0; p<molecules->get_n_polymer_types(); p++){
        dq_dl.push_back({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    }
}
template <typename T>
PropagatorComputation<T>::~PropagatorComputation()
{
    if (holds_orthogonal_cell_)
        cb->release_orthogonal_cell();
}

template <typename T>
void PropagatorComputation<T>::hold_orthogonal_cell()
{
    if (!holds_orthogonal_cell_)
    {
        cb->hold_orthogonal_cell();
        holds_orthogonal_cell_ = true;
    }
}

template <typename T>
void PropagatorComputation<T>::check_box_space_group() const
{
    if (cb->get_space_group() != space_group_)
        throw_with_line_number("The space group of the computation box changed after this computation "
            "was created; create a new computation or give each computation its own box.");
}

template <typename T>
void PropagatorComputation<T>::clear_computed()
{
    propagators_computed_ = false;
    concentrations_computed_ = false;
    stress_computed_ = false;
}

template <typename T>
void PropagatorComputation<T>::require_propagators() const
{
    if (!propagators_computed_)
        throw_with_line_number("Propagators not computed for the current fields and box. "
            "Call compute_propagators() first.");
}

template <typename T>
void PropagatorComputation<T>::require_concentrations() const
{
    require_propagators();
    if (!concentrations_computed_)
        throw_with_line_number("Concentrations not computed for the current propagators. "
            "Call compute_concentrations() after compute_propagators().");
}

template <typename T>
void PropagatorComputation<T>::require_stress() const
{
    require_propagators();
    if (!stress_computed_)
        throw_with_line_number("Stress not computed for the current propagators. "
            "Call compute_stress() after compute_propagators().");
}

/**
 * @brief Get lattice-parameter derivatives of the free energy (canonical ensemble).
 *
 * Combines per-polymer contributions:
 *     stress[d] = Σ_p (φ_p/α_p) × dq_dl_p[d] / Q_p
 *
 * @return Vector [dH/dL₁, dH/dL₂, dH/dL₃, dH/dγ, dH/dβ, dH/dα].
 *         For 2D: [dH/dL₁, dH/dL₂, dH/dγ, 0, 0, 0]. For 1D: only index 0.
 *
 * @note dq_dl must be populated by compute_stress() in derived class
 */
template <typename T>
std::vector<T> PropagatorComputation<T>::get_stress()
{
    this->check_box_space_group();
    this->require_stress();
    const int N_STRESS = 6;  // [dH/dL1, dH/dL2, dH/dL3, dH/dγ, dH/dβ, dH/dα]
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
 * @brief Get lattice-parameter derivatives for grand canonical ensemble.
 *
 * Uses fugacities instead of volume fractions:
 *     stress[d] = Σ_p z_p × dq_dl_p[d]
 *
 * where z_p is the fugacity of polymer species p.
 *
 * @param fugacities Vector of polymer fugacities
 * @return Vector [dH/dL₁, dH/dL₂, dH/dL₃, dH/dγ, dH/dβ, dH/dα]
 */
template <typename T>
std::vector<T> PropagatorComputation<T>::get_stress_gce(std::vector<double> fugacities)
{
    this->check_box_space_group();
    this->require_stress();
    const int N_STRESS = 6;  // [dH/dL1, dH/dL2, dH/dL3, dH/dγ, dH/dβ, dH/dα]
    std::vector<T> stress(N_STRESS);

    int n_polymer_types = this->molecules->get_n_polymer_types();
    if (static_cast<int>(fugacities.size()) != n_polymer_types)
        throw_with_line_number("get_stress_gce needs one fugacity per polymer type: got " +
            std::to_string(fugacities.size()) + " for " + std::to_string(n_polymer_types) + " polymer types.");
    for(int p=0; p<n_polymer_types; p++)
        validation::require_non_negative(fugacities[p], "fugacities[" + std::to_string(p) + "]");
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
 * The basis is fixed when the computation is constructed. Reattaching the same
 * group to its box is allowed, but changing it would invalidate the backend's
 * cached sizes and buffers.
 *
 * @param sg The group used at construction (nullptr if none)
 */
template <typename T>
void PropagatorComputation<T>::set_space_group(SpaceGroup* sg)
{
    if (sg != space_group_)
        throw_with_line_number("The space group of a computation cannot change after construction; "
            "create a new computation with the requested group.");

    // Also set space group on ComputationBox for unified field operations
    this->cb->set_space_group(sg);
}

template <typename T>
bool PropagatorComputation<T>::check_total_partition(double tolerance)
{
    for (const double error : compute_partition_errors())
        if (!std::isfinite(error) || error > tolerance)
            return false;
    return true;
}

// Explicit template instantiation
template class PropagatorComputation<double>;
template class PropagatorComputation<std::complex<double>>;
