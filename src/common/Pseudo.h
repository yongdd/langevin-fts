/**
 * @file Pseudo.h
 * @brief Unified pseudo-spectral utilities for all boundary conditions.
 *
 * This header provides the Pseudo class for pseudo-spectral method
 * supporting periodic, reflecting, and absorbing boundary conditions,
 * as well as non-orthogonal crystal systems via reciprocal metric tensor.
 *
 * **Wavenumber Conventions:**
 *
 * - PERIODIC: k = 2π*n/L (n = 0, 1, ..., N/2, -N/2+1, ..., -1)
 * - REFLECTING (DCT): k = π*n/L (n = 0, 1, ..., N-1)
 * - ABSORBING (DST): k = π*(n+1)/L (n = 0, 1, ..., N-1)
 *
 * **Boltzmann Bond Factors:**
 *
 * Both chain models use the same mathematical formula but with different
 * physical interpretations:
 *
 * boltz_bond[k] = exp(-b²*|k|²*ds/6)
 *
 * - **Continuous chains**: This is the diffusion propagator from solving
 *   the modified diffusion equation ∂q/∂s = (b²/6)∇²q - wq.
 *
 * - **Discrete chains**: This is the bond function ĝ(k) = exp(-a²|k|²/6)
 *   from the Chapman-Kolmogorov integral equation. With the convention
 *   a² = b²ds (where a is the segment length and ds = 1/N), it takes
 *   the same form as the continuous case. See Park et al. J. Chem. Phys.
 *   150, 234901 (2019) for details.
 *
 * For non-orthogonal systems with periodic BC:
 * |k|² = g^{-1}_ij k_i k_j where g^{-1} is the reciprocal metric tensor.
 *
 * **Stress Calculation using Deformation Vector:**
 *
 * For stress calculation, we use the deformation vector v = 2π g⁻¹ m
 * where g⁻¹ is the inverse metric tensor and m is the Miller index vector.
 * The deformation vector components vᵢ have units of 1/L² (not 1/L like
 * Cartesian wavevector components).
 *
 * The stress arrays store v⊗v dyad product components in Voigt notation:
 * - Index 0: V₁₁ = Σ(kernel × v₁²) → drives L₁ optimization
 * - Index 1: V₂₂ = Σ(kernel × v₂²) → drives L₂ optimization
 * - Index 2: V₃₃ = Σ(kernel × v₃²) → drives L₃ optimization
 * - Index 3: V₁₂ = Σ(kernel × v₁v₂) → drives γ optimization
 * - Index 4: V₁₃ = Σ(kernel × v₁v₃) → drives β optimization
 * - Index 5: V₂₃ = Σ(kernel × v₂v₃) → drives α optimization
 *
 * For 2D: [V₁₁, V₂₂, V₁₂, 0, 0, 0]. For 1D: only index 0 used.
 *
 * @see CpuSolverPseudoRQM4 for continuous chain solver
 * @see CpuSolverPseudoDiscrete for discrete chain solver
 * @see docs/StressTensorCalculation.md for detailed derivation
 */

#ifndef PSEUDO_H_
#define PSEUDO_H_

#include <string>
#include <vector>
#include <map>
#include <array>

#include "Exception.h"
#include "ComputationBox.h"
#include "Molecules.h"

/**
 * @class Pseudo
 * @brief Pre-computed arrays for pseudo-spectral method.
 *
 * This class handles all boundary conditions:
 * periodic (FFT), reflecting (DCT), and absorbing (DST).
 * Also supports non-orthogonal crystal systems for periodic BC.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Array Storage:**
 *
 * - Periodic BC with r2c FFT: size = Nx × Ny × (Nz/2+1)
 * - Periodic BC with c2c FFT: size = Nx × Ny × Nz
 * - Non-periodic BC (DCT/DST): size = Nx × Ny × Nz (real-to-real)
 */
template <typename T>
class Pseudo
{
protected:
    std::vector<BoundaryCondition> bc;       ///< Boundary conditions per dimension
    std::map<std::string, double> bond_lengths;  ///< Segment lengths for each type
    std::vector<int> nx;                     ///< Grid dimensions [Nx, Ny, Nz]
    std::vector<double> dx;                  ///< Grid spacings [dx, dy, dz]
    double ds;                               ///< Contour step size

    /**
     * @brief Reciprocal metric tensor for wavenumber calculation.
     *
     * g^{-1}_ij = e*_i · e*_j where e* are reciprocal basis vectors.
     * Layout: [g^{-1}_11, g^{-1}_12, g^{-1}_13, g^{-1}_22, g^{-1}_23, g^{-1}_33]
     *
     * For orthogonal systems: g^{-1}_ii = 1/L_i², cross terms = 0.
     * Used to compute |k|² = g^{-1}_ij k_i k_j for non-orthogonal lattices.
     * Only applicable for periodic boundary conditions.
     */
    std::array<double, 6> recip_metric_;

    /**
     * @brief Reciprocal lattice vectors for k⊗k computation.
     *
     * Stores [a*|b*|c*] in column-major order where:
     * - a* = (b × c) / V  → recip_vec_[0..2]
     * - b* = (c × a) / V  → recip_vec_[3..5]
     * - c* = (a × b) / V  → recip_vec_[6..8]
     *
     * These are the "crystallographer's" reciprocal vectors (without 2π factor).
     * The wavevector is: k = 2π × (m₁·a* + m₂·b* + m₃·c*)
     */
    std::array<double, 9> recip_vec_;

    int *negative_k_idx;  ///< Mapping from k to -k indices (for complex fields)

    /**
     * @brief Boltzmann bond factors for full step (per ds_index).
     *
     * boltz_bond[ds_index][type][idx] = exp(-|k|² * b² * ds / 6)
     * where k depends on boundary condition type.
     *
     * For continuous chains: diffusion propagator.
     * For discrete chains: bond function (Fourier transform of Gaussian).
     *
     * ds_index is 1-based (from ContourLengthMapping).
     */
    std::map<int, std::map<std::string, double*>> boltz_bond;

    /**
     * @brief Boltzmann bond factors for half step (per ds_index).
     *
     * boltz_bond_half[ds_index][type][idx] = exp(-|k|² * b² * ds / 12)
     *
     * For continuous chains: half-step diffusion propagator.
     * For discrete chains: half-bond function (used at chain ends).
     *
     * ds_index is 1-based (from ContourLengthMapping).
     */
    std::map<int, std::map<std::string, double*>> boltz_bond_half;

    /**
     * @brief Unique ds values for each ds_index.
     *
     * ds_values[ds_index] = local_ds value for that index.
     * ds_index is 1-based.
     */
    std::map<int, double> ds_values;

    /**
     * @brief Deformation vector v⊗v diagonal components for stress calculation.
     *
     * The deformation vector is defined as v = 2π g⁻¹ m where g⁻¹ is the
     * inverse metric tensor and m is the Miller index vector.
     *
     * For periodic BC:
     * - fourier_basis_x[m] = v₁² where v₁ = 2π(G₁₁m₁ + G₁₂m₂ + G₁₃m₃)
     * - fourier_basis_y[m] = v₂² where v₂ = 2π(G₁₂m₁ + G₂₂m₂ + G₂₃m₃)
     * - fourier_basis_z[m] = v₃² where v₃ = 2π(G₁₃m₁ + G₂₃m₂ + G₃₃m₃)
     *
     * For non-periodic BC: uses k² = (π*n/L)² directly (orthogonal only).
     * Note: vᵢ has units of 1/L², not 1/L like Cartesian wavevector.
     */
    double *fourier_basis_x;
    double *fourier_basis_y;
    double *fourier_basis_z;

    /**
     * @brief Deformation vector v⊗v off-diagonal components for stress calculation.
     *
     * For non-orthogonal systems with periodic BC:
     * - fourier_basis_xy[m] = v₁ × v₂ (drives γ angle optimization)
     * - fourier_basis_xz[m] = v₁ × v₃ (drives β angle optimization)
     * - fourier_basis_yz[m] = v₂ × v₃ (drives α angle optimization)
     *
     * These are zero for non-periodic BC (orthogonal grids only).
     * Used for angle stress components in triclinic systems.
     */
    double *fourier_basis_xy;
    double *fourier_basis_xz;
    double *fourier_basis_yz;

    int total_grid;          ///< Total real grid size
    int total_complex_grid;  ///< Total "complex" grid size

    /**
     * @brief Update total_complex_grid based on boundary conditions.
     */
    void update_total_complex_grid();

    /**
     * @brief Recompute Boltzmann factors.
     */
    void update_boltz_bond();

    /**
     * @brief Recompute Fourier basis arrays for stress.
     */
    void update_weighted_fourier_basis();

    /**
     * @brief Update negative frequency index mapping.
     */
    void update_negative_frequency_mapping();

    /**
     * @brief Check if all BCs are periodic.
     */
    bool is_all_periodic() const;

    /**
     * @brief Update Boltzmann factors for periodic BC (with recip_metric).
     */
    void update_boltz_bond_periodic();

    /**
     * @brief Update Boltzmann factors for mixed BC.
     */
    void update_boltz_bond_mixed();

    /**
     * @brief Update Boltzmann factors for a specific ds_index (periodic BC).
     */
    void update_boltz_bond_periodic_for_ds_index(int ds_idx);

    /**
     * @brief Update Boltzmann factors for a specific ds_index (mixed BC).
     */
    void update_boltz_bond_mixed_for_ds_index(int ds_idx);

    /**
     * @brief Update Fourier basis for periodic BC (with cross-terms).
     */
    void update_weighted_fourier_basis_periodic();

    /**
     * @brief Update Fourier basis for mixed BC (no cross-terms).
     */
    void update_weighted_fourier_basis_mixed();

public:
    /**
     * @brief Construct Pseudo with given parameters.
     *
     * @param bond_lengths Segment lengths: {type: b}
     * @param bc          Boundary conditions (can be mixed)
     * @param nx          Grid dimensions
     * @param dx          Grid spacings
     * @param recip_metric Reciprocal metric tensor (only for periodic BC)
     *                     [g^{-1}_11, g^{-1}_12, g^{-1}_13, g^{-1}_22, g^{-1}_23, g^{-1}_33]
     *                     Default is identity for orthogonal systems.
     * @param recip_vec   Reciprocal lattice vectors [a*|b*|c*] in column-major order
     *                     for k⊗k computation. Default is Cartesian basis.
     *
     * @note After construction, call add_ds_value() for each unique ds value,
     *       then finalize_ds_values() to compute Boltzmann factors.
     */
    Pseudo(
        std::map<std::string, double> bond_lengths,
        std::vector<BoundaryCondition> bc,
        std::vector<int> nx, std::vector<double> dx,
        std::array<double, 6> recip_metric = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0},
        std::array<double, 9> recip_vec = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});

    /**
     * @brief Virtual destructor.
     */
    virtual ~Pseudo();

    /**
     * @brief Get total complex grid size.
     * @return Total grid size for Fourier coefficients
     */
    int get_total_complex_grid();

    /**
     * @brief Get Boltzmann factor for full step for given ds_index.
     *
     * @param monomer_type Monomer type (e.g., "A", "B")
     * @param ds_index     1-based index for ds value (default: 1 for global ds)
     * @return Pointer to Boltzmann factor array
     */
    virtual double* get_boltz_bond(std::string monomer_type, int ds_index);

    /**
     * @brief Get Boltzmann factor for half step for given ds_index.
     *
     * @param monomer_type Monomer type (e.g., "A", "B")
     * @param ds_index     1-based index for ds value
     * @return Pointer to Boltzmann factor array
     */
    virtual double* get_boltz_bond_half(std::string monomer_type, int ds_index);

    /**
     * @brief Add a ds value to be pre-computed.
     *
     * Call this for each unique local_ds value before calling
     * finalize_ds_values() to pre-compute Boltzmann factors.
     *
     * @param ds_index 1-based index for this ds value
     * @param ds_value The ds value
     */
    void add_ds_value(int ds_index, double ds_value);

    /**
     * @brief Finalize and compute Boltzmann factors for all added ds values.
     *
     * Must be called after all add_ds_value() calls and before
     * using get_boltz_bond() with ds_index parameter.
     *
     * This method is virtual to allow GPU implementations to allocate
     * device memory for the new ds_index values.
     */
    virtual void finalize_ds_values();

    /**
     * @brief Get x-direction Fourier basis for stress.
     */
    virtual const double* get_fourier_basis_x();

    /**
     * @brief Get y-direction Fourier basis for stress.
     */
    virtual const double* get_fourier_basis_y();

    /**
     * @brief Get z-direction Fourier basis for stress.
     */
    virtual const double* get_fourier_basis_z();

    /**
     * @brief Get xy cross-term Fourier basis for stress.
     */
    virtual const double* get_fourier_basis_xy();

    /**
     * @brief Get xz cross-term Fourier basis for stress.
     */
    virtual const double* get_fourier_basis_xz();

    /**
     * @brief Get yz cross-term Fourier basis for stress.
     */
    virtual const double* get_fourier_basis_yz();

    /**
     * @brief Get negative frequency index mapping.
     */
    virtual const int* get_negative_frequency_mapping();

    /**
     * @brief Get boundary conditions.
     */
    const std::vector<BoundaryCondition>& get_boundary_conditions() const { return bc; }

    /**
     * @brief Update all arrays after parameter changes.
     *
     * Updates Fourier basis arrays and recomputes Boltzmann factors for
     * all previously registered ds values.
     *
     * @param bc           New boundary conditions
     * @param bond_lengths New segment lengths
     * @param dx           New grid spacings
     * @param recip_metric New reciprocal metric tensor (for periodic BC)
     * @param recip_vec    New reciprocal lattice vectors for k⊗k computation
     */
    virtual void update(
        std::vector<BoundaryCondition> bc,
        std::map<std::string, double> bond_lengths,
        std::vector<double> dx,
        std::array<double, 6> recip_metric = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0},
        std::array<double, 9> recip_vec = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
};

#endif
