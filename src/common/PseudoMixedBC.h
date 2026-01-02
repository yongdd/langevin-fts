/**
 * @file PseudoMixedBC.h
 * @brief Pseudo-spectral utilities for mixed boundary conditions.
 *
 * This header provides the PseudoMixedBC class for pseudo-spectral method
 * with non-periodic boundary conditions (reflecting/absorbing).
 *
 * **Wavenumber Conventions:**
 *
 * - PERIODIC: k = 2π*n/L (n = 0, 1, ..., N/2, -N/2+1, ..., -1)
 * - REFLECTING (DCT): k = π*n/L (n = 0, 1, ..., N-1)
 * - ABSORBING (DST): k = π*(n+1)/L (n = 0, 1, ..., N-1)
 *
 * **Boltzmann Factors:**
 *
 * boltz_bond[k] = exp(-b²*k²*ds/6)
 *
 * For DCT/DST, the k² values are computed differently than for FFT.
 *
 * @see Pseudo for the periodic-only version
 * @see CpuSolverPseudoMixedBC for the solver using this class
 */

#ifndef PSEUDO_MIXED_BC_H_
#define PSEUDO_MIXED_BC_H_

#include <string>
#include <vector>
#include <map>
#include <array>

#include "Exception.h"
#include "ComputationBox.h"
#include "Molecules.h"

/**
 * @class PseudoMixedBC
 * @brief Pre-computed arrays for pseudo-spectral method with mixed BCs.
 *
 * Unlike the standard Pseudo class which only supports periodic BCs,
 * this class handles reflecting (DCT) and absorbing (DST) boundaries.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Array Storage:**
 *
 * For non-periodic BCs, all arrays have size M (not M/2+1) because
 * DCT/DST produces real coefficients.
 */
template <typename T>
class PseudoMixedBC
{
protected:
    std::vector<BoundaryCondition> bc;       ///< Boundary conditions per dimension
    std::map<std::string, double> bond_lengths;  ///< Segment lengths for each type
    std::vector<int> nx;                     ///< Grid dimensions [Nx, Ny, Nz]
    std::vector<double> dx;                  ///< Grid spacings [dx, dy, dz]
    double ds;                               ///< Contour step size

    /**
     * @brief Boltzmann factors for full contour step.
     *
     * boltz_bond[type][idx] = exp(-ds * |k|² * b²/6)
     * where k depends on boundary condition type.
     */
    std::map<std::string, double*> boltz_bond;

    /**
     * @brief Boltzmann factors for half contour step.
     *
     * boltz_bond_half[type][idx] = exp(-ds/2 * |k|² * b²/6)
     */
    std::map<std::string, double*> boltz_bond_half;

    /**
     * @brief Fourier basis for stress calculation in x-direction.
     */
    double *fourier_basis_x;
    double *fourier_basis_y;
    double *fourier_basis_z;

    int total_grid;          ///< Total real grid size
    int total_complex_grid;  ///< Total "complex" grid size (same as real for DCT/DST)

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
     * @brief Check if all BCs are periodic.
     */
    bool is_all_periodic() const;

public:
    /**
     * @brief Construct PseudoMixedBC with given parameters.
     *
     * @param bond_lengths Segment lengths: {type: b}
     * @param bc          Boundary conditions (can be mixed)
     * @param nx          Grid dimensions
     * @param dx          Grid spacings
     * @param ds          Contour step size
     */
    PseudoMixedBC(
        std::map<std::string, double> bond_lengths,
        std::vector<BoundaryCondition> bc,
        std::vector<int> nx, std::vector<double> dx, double ds);

    /**
     * @brief Virtual destructor.
     */
    virtual ~PseudoMixedBC();

    /**
     * @brief Get total complex grid size.
     * @return Total grid size for Fourier coefficients
     */
    int get_total_complex_grid();

    /**
     * @brief Get Boltzmann factor for full step.
     */
    virtual double* get_boltz_bond(std::string monomer_type);

    /**
     * @brief Get Boltzmann factor for half step.
     */
    virtual double* get_boltz_bond_half(std::string monomer_type);

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
     * @brief Get boundary conditions.
     */
    const std::vector<BoundaryCondition>& get_boundary_conditions() const { return bc; }

    /**
     * @brief Update all arrays after parameter changes.
     */
    virtual void update(
        std::vector<BoundaryCondition> bc,
        std::map<std::string, double> bond_lengths,
        std::vector<double> dx, double ds);
};

#endif
