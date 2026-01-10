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

/**
 * @brief Default implementation of add_checkpoint (no-op).
 *
 * In standard mode, all propagators are stored, so manual checkpoints
 * are not needed. This default implementation returns false.
 *
 * Memory-saving implementations override this to allocate checkpoint storage.
 *
 * @param polymer Polymer index
 * @param v       Starting vertex
 * @param u       Ending vertex
 * @param n       Contour step index
 * @return false (default: no checkpoint added)
 */
template <typename T>
bool PropagatorComputation<T>::add_checkpoint(int polymer, int v, int u, int n)
{
    // Default implementation: no-op for standard mode
    // Memory-saving implementations override this method
    return false;
}

// Explicit template instantiation
template class PropagatorComputation<double>;
template class PropagatorComputation<std::complex<double>>;