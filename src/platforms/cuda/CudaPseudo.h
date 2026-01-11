/**
 * @file CudaPseudo.h
 * @brief GPU implementation of pseudo-spectral method utilities.
 *
 * This header provides CudaPseudo, the GPU-specific implementation
 * of Pseudo that stores Boltzmann factors and Fourier basis vectors
 * in GPU device memory.
 *
 * **GPU Memory Contents:**
 *
 * - d_boltz_bond: exp(-k²b²ds/6) for each monomer type
 * - d_boltz_bond_half: exp(-k²b²ds/12) for half-bond steps
 * - d_fourier_basis_*: Weighted wavenumbers for stress calculation
 * - d_negative_k_idx: Index mapping for Hermitian symmetry
 *
 * @see Pseudo for the abstract interface
 * @see CudaSolverPseudoRQM4 for usage in continuous chains
 * @see CudaSolverPseudoDiscrete for usage in discrete chains
 */

#ifndef CUDA_PSEUDO_H_
#define CUDA_PSEUDO_H_

#include <string>
#include <vector>
#include <map>

#include "Pseudo.h"
#include "Exception.h"
#include "ComputationBox.h"
#include "Molecules.h"

/**
 * @class CudaPseudo
 * @brief GPU implementation of pseudo-spectral utilities.
 *
 * Stores pseudo-spectral operators in GPU device memory for efficient
 * access during propagator computation.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Boltzmann Factors:**
 *
 * Stored in Fourier space for efficient multiplication:
 * - boltz_bond[k] = exp(-k²b²ds/6) for continuous chains
 * - boltz_bond_half[k] = exp(-k²b²ds/12) for discrete chains
 *
 * **Fourier Basis:**
 *
 * Weighted wavenumbers for stress calculation:
 * - d_fourier_basis_x = kx² × weight
 * - Used in ∂H/∂Lx calculation
 */
template <typename T>
class CudaPseudo : public Pseudo<T>
{
private:
    /// @name Stress Calculation Arrays (Diagonal)
    /// @{
    double *d_fourier_basis_x;  ///< Weighted kx² in device memory
    double *d_fourier_basis_y;  ///< Weighted ky² in device memory
    double *d_fourier_basis_z;  ///< Weighted kz² in device memory
    /// @}

    /// @name Stress Calculation Arrays (Cross-terms for non-orthogonal)
    /// @{
    double *d_fourier_basis_xy;  ///< Weighted 2*kx*ky in device memory
    double *d_fourier_basis_xz;  ///< Weighted 2*kx*kz in device memory
    double *d_fourier_basis_yz;  ///< Weighted 2*ky*kz in device memory
    /// @}

    int *d_negative_k_idx;  ///< Index map for negative frequencies (device)

    /// @name Pseudo-Spectral Operators
    /// @{
    std::map<std::string, double*> d_boltz_bond;       ///< Full bond diffusion (device)
    std::map<std::string, double*> d_boltz_bond_half;  ///< Half bond diffusion (device)
    /// @}

    /**
     * @brief Upload Boltzmann factors to GPU.
     */
    void upload_boltz_bond();

    /**
     * @brief Upload Fourier basis to GPU.
     */
    void upload_fourier_basis();

public:
    /**
     * @brief Construct GPU pseudo-spectral utility.
     *
     * Computes and uploads Boltzmann factors and Fourier basis to GPU.
     *
     * @param bond_lengths Statistical segment lengths squared by monomer type
     * @param bc           Boundary conditions
     * @param nx           Grid dimensions
     * @param dx           Grid spacing [dx, dy, dz]
     * @param ds           Contour step size
     * @param recip_metric Reciprocal metric tensor [G*_00, G*_01, G*_02, G*_11, G*_12, G*_22]
     *                     Default is identity for orthogonal systems.
     */
    CudaPseudo(
        std::map<std::string, double> bond_lengths,
        std::vector<BoundaryCondition> bc, std::vector<int> nx, std::vector<double> dx, double ds,
        std::array<double, 6> recip_metric = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0});

    /**
     * @brief Destructor. Frees GPU memory.
     */
    ~CudaPseudo();

    /**
     * @brief Get full bond Boltzmann factor (device pointer).
     * @param monomer_type Monomer type
     * @return Device pointer to exp(-k²b²ds/6)
     */
    double* get_boltz_bond(std::string monomer_type) override { return d_boltz_bond[monomer_type]; };

    /**
     * @brief Get half bond Boltzmann factor (device pointer).
     * @param monomer_type Monomer type
     * @return Device pointer to exp(-k²b²ds/12)
     */
    double* get_boltz_bond_half(std::string monomer_type) override { return d_boltz_bond_half[monomer_type];};

    /** @brief Get x-direction Fourier basis (device pointer). */
    const double* get_fourier_basis_x() override { return d_fourier_basis_x;};

    /** @brief Get y-direction Fourier basis (device pointer). */
    const double* get_fourier_basis_y() override { return d_fourier_basis_y;};

    /** @brief Get z-direction Fourier basis (device pointer). */
    const double* get_fourier_basis_z() override { return d_fourier_basis_z;};

    /** @brief Get negative frequency mapping (device pointer). */
    const int* get_negative_frequency_mapping() override { return d_negative_k_idx;};

    /** @brief Get xy cross-term Fourier basis (device pointer). */
    const double* get_fourier_basis_xy() override { return d_fourier_basis_xy;};

    /** @brief Get xz cross-term Fourier basis (device pointer). */
    const double* get_fourier_basis_xz() override { return d_fourier_basis_xz;};

    /** @brief Get yz cross-term Fourier basis (device pointer). */
    const double* get_fourier_basis_yz() override { return d_fourier_basis_yz;};

    /**
     * @brief Update operators for new box dimensions.
     *
     * Called when box size changes during stress relaxation.
     *
     * @param bc           Boundary conditions
     * @param bond_lengths Segment lengths
     * @param dx           New grid spacing
     * @param ds           Contour step
     * @param recip_metric Reciprocal metric tensor
     */
    void update(
        std::vector<BoundaryCondition> bc,
        std::map<std::string, double> bond_lengths,
        std::vector<double> dx, double ds,
        std::array<double, 6> recip_metric = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0}) override;
};
#endif