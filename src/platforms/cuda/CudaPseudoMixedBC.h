/**
 * @file CudaPseudoMixedBC.h
 * @brief GPU implementation of pseudo-spectral utilities for mixed BCs.
 *
 * This header provides CudaPseudoMixedBC, the GPU-specific implementation
 * of PseudoMixedBC that stores Boltzmann factors and Fourier basis vectors
 * in GPU device memory for non-periodic boundary conditions.
 *
 * **GPU Memory Contents:**
 *
 * - d_boltz_bond: exp(-k²b²ds/6) where k depends on BC type
 * - d_boltz_bond_half: exp(-k²b²ds/12) for half-bond steps
 * - d_fourier_basis_*: Weighted wavenumbers for stress calculation
 *
 * **Wavenumber Conventions:**
 *
 * - REFLECTING (DCT): k = π*n/L
 * - ABSORBING (DST): k = π*(n+1)/L
 *
 * @see PseudoMixedBC for the base class
 * @see CudaSolverPseudoMixedBC for solver using this class
 */

#ifndef CUDA_PSEUDO_MIXED_BC_H_
#define CUDA_PSEUDO_MIXED_BC_H_

#include <string>
#include <vector>
#include <map>

#include "PseudoMixedBC.h"
#include "Exception.h"
#include "ComputationBox.h"
#include "Molecules.h"

/**
 * @class CudaPseudoMixedBC
 * @brief GPU implementation of pseudo-spectral utilities for mixed BCs.
 *
 * Stores pseudo-spectral operators in GPU device memory for efficient
 * access during propagator computation with non-periodic boundaries.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Boltzmann Factors:**
 *
 * For DCT/DST, the Boltzmann factors are computed using:
 * - REFLECTING: k² = (π*n/L)²
 * - ABSORBING: k² = (π*(n+1)/L)²
 *
 * The factors are: exp(-b²*k²*ds/6) and exp(-b²*k²*ds/12)
 */
template <typename T>
class CudaPseudoMixedBC : public PseudoMixedBC<T>
{
private:
    /// @name Stress Calculation Arrays (device memory)
    /// @{
    double *d_fourier_basis_x;  ///< Weighted kx² in device memory
    double *d_fourier_basis_y;  ///< Weighted ky² in device memory
    double *d_fourier_basis_z;  ///< Weighted kz² in device memory
    /// @}

    /// @name Pseudo-Spectral Operators (device memory)
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
     * @brief Construct GPU pseudo-spectral utility for mixed BCs.
     *
     * Computes Boltzmann factors on CPU, then uploads to GPU.
     *
     * @param bond_lengths Statistical segment lengths by monomer type
     * @param bc           Boundary conditions (can be mixed reflecting/absorbing)
     * @param nx           Grid dimensions
     * @param dx           Grid spacing [dx, dy, dz]
     * @param ds           Contour step size
     */
    CudaPseudoMixedBC(
        std::map<std::string, double> bond_lengths,
        std::vector<BoundaryCondition> bc,
        std::vector<int> nx, std::vector<double> dx, double ds);

    /**
     * @brief Destructor. Frees GPU memory.
     */
    ~CudaPseudoMixedBC();

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
    double* get_boltz_bond_half(std::string monomer_type) override { return d_boltz_bond_half[monomer_type]; };

    /** @brief Get x-direction Fourier basis (device pointer). */
    const double* get_fourier_basis_x() override { return d_fourier_basis_x; };

    /** @brief Get y-direction Fourier basis (device pointer). */
    const double* get_fourier_basis_y() override { return d_fourier_basis_y; };

    /** @brief Get z-direction Fourier basis (device pointer). */
    const double* get_fourier_basis_z() override { return d_fourier_basis_z; };

    /**
     * @brief Update operators for new box dimensions.
     *
     * Recomputes on CPU and uploads to GPU.
     *
     * @param bc           Boundary conditions
     * @param bond_lengths Segment lengths
     * @param dx           New grid spacing
     * @param ds           Contour step
     */
    void update(
        std::vector<BoundaryCondition> bc,
        std::map<std::string, double> bond_lengths,
        std::vector<double> dx, double ds) override;
};

#endif
