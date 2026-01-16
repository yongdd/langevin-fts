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
 * |k|² = G*_ij h_i h_j where G* is the reciprocal metric tensor.
 *
 * @see CpuSolverPseudoRQM4 for continuous chain solver
 * @see CpuSolverPseudoDiscrete for discrete chain solver
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
     * @brief Enable cell-averaged bond function (sinc filtering).
     *
     * When enabled, the Boltzmann bond factor is multiplied by sinc factors:
     * boltz_bond_filtered[k] = boltz_bond[k] × sinc(kx·dx/2) × sinc(ky·dy/2) × ...
     *
     * This corresponds to the cell-averaging approach from Park et al.,
     * J. Chem. Phys. 150, 234901 (2019), which ensures non-negativity of
     * the bond function in real space.
     *
     * Default: false (use standard analytical bond function)
     */
    bool use_cell_averaged_bond_;

    /**
     * @brief Reciprocal metric tensor for wavenumber calculation.
     *
     * G*_ij = e*_i · e*_j where e* are reciprocal basis vectors.
     * Layout: [G*_00, G*_01, G*_02, G*_11, G*_12, G*_22]
     *
     * For orthogonal systems: G*_ii = 1/L_i², cross terms = 0.
     * Used to compute |k|² = G*_ij k_i k_j for non-orthogonal lattices.
     * Only applicable for periodic boundary conditions.
     */
    std::array<double, 6> recip_metric_;

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
     * @brief Fourier basis for stress calculation (diagonal terms).
     */
    double *fourier_basis_x;
    double *fourier_basis_y;
    double *fourier_basis_z;

    /**
     * @brief Fourier basis for stress calculation (off-diagonal/cross terms).
     *
     * For non-orthogonal systems with periodic BC, cross-terms are needed
     * for the full stress tensor. For non-periodic BC or orthogonal systems,
     * these are zero.
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
     * @param ds          Contour step size
     * @param recip_metric Reciprocal metric tensor (only for periodic BC)
     *                     [G*_00, G*_01, G*_02, G*_11, G*_12, G*_22]
     *                     Default is identity for orthogonal systems.
     */
    Pseudo(
        std::map<std::string, double> bond_lengths,
        std::vector<BoundaryCondition> bc,
        std::vector<int> nx, std::vector<double> dx, double ds,
        std::array<double, 6> recip_metric = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0});

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
    virtual double* get_boltz_bond(std::string monomer_type, int ds_index = 1);

    /**
     * @brief Get Boltzmann factor for half step for given ds_index.
     *
     * @param monomer_type Monomer type (e.g., "A", "B")
     * @param ds_index     1-based index for ds value (default: 1 for global ds)
     * @return Pointer to Boltzmann factor array
     */
    virtual double* get_boltz_bond_half(std::string monomer_type, int ds_index = 1);

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
     * @brief Enable or disable cell-averaged bond function.
     *
     * When enabled, applies sinc filtering to the Boltzmann bond factor:
     * boltz_bond_filtered[k] = boltz_bond[k] × sinc(kx·dx/2) × sinc(ky·dy/2) × ...
     *
     * This ensures non-negativity of the bond function in real space,
     * following the cell-averaging approach from Park et al.,
     * J. Chem. Phys. 150, 234901 (2019).
     *
     * @param enabled True to enable cell-averaging, false for standard bond function
     *
     * @note After calling this method, call update() or finalize_ds_values()
     *       to recompute the Boltzmann factors with the new setting.
     */
    virtual void set_cell_averaged_bond(bool enabled);

    /**
     * @brief Check if cell-averaged bond function is enabled.
     * @return True if cell-averaging is enabled
     */
    bool is_cell_averaged_bond() const { return use_cell_averaged_bond_; }

    /**
     * @brief Update all arrays after parameter changes.
     *
     * @param bc           New boundary conditions
     * @param bond_lengths New segment lengths
     * @param dx           New grid spacings
     * @param ds           New contour step (usually unchanged)
     * @param recip_metric New reciprocal metric tensor (for periodic BC)
     */
    virtual void update(
        std::vector<BoundaryCondition> bc,
        std::map<std::string, double> bond_lengths,
        std::vector<double> dx, double ds,
        std::array<double, 6> recip_metric = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0});
};

#endif
