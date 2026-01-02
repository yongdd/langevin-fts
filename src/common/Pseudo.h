/**
 * @file Pseudo.h
 * @brief Pseudo-spectral method utilities for solving the modified diffusion equation.
 *
 * This header provides the Pseudo class which contains pre-computed arrays and
 * methods for the pseudo-spectral (split-operator Fourier) method used to solve
 * the modified diffusion equation for polymer chain propagators.
 *
 * **Pseudo-Spectral Method:**
 *
 * The modified diffusion equation:
 *   dq/ds = (a²/6) ∇²q - w(r) q
 *
 * is solved using operator splitting:
 *   q(s+ds) ≈ exp(-ds*w/2) * FFT⁻¹{ exp(-ds*k²*a²/6) * FFT{ exp(-ds*w/2) * q(s) } }
 *
 * This requires pre-computed Boltzmann factors exp(-ds*k²*a²/6) in Fourier space.
 *
 * **Boltzmann Factors:**
 *
 * For each monomer type with segment length a:
 * - boltz_bond: exp(-ds * k² * a²/6) for full contour step
 * - boltz_bond_half: exp(-ds/2 * k² * a²/6) for half contour step
 *
 * Half-step factors are used for split-operator and Richardson extrapolation.
 *
 * @see PropagatorComputation for the high-level solver interface
 * @see CpuSolverPseudoContinuous, CudaSolverPseudoContinuous for implementations
 *
 * @example
 * @code
 * // Create Pseudo object with parameters
 * std::map<std::string, double> bonds = {{"A", 1.0}, {"B", 1.0}};
 * std::vector<BoundaryCondition> bc = {BoundaryCondition::PERIODIC, ...};
 * std::vector<int> nx = {32, 32, 32};
 * std::vector<double> dx = {0.125, 0.125, 0.125};
 * double ds = 0.01;
 *
 * Pseudo<double> pseudo(bonds, bc, nx, dx, ds);
 *
 * // Get Boltzmann factors for FFT-based propagator step
 * double* boltz_A = pseudo.get_boltz_bond("A");
 * double* boltz_A_half = pseudo.get_boltz_bond_half("A");
 * @endcode
 */

#ifndef PSEUDO_H_
#define PSEUDO_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "ComputationBox.h"
#include "Molecules.h"

/**
 * @class Pseudo
 * @brief Pre-computed arrays for pseudo-spectral propagator computation.
 *
 * This class stores Boltzmann factors and Fourier basis arrays needed for
 * the pseudo-spectral method. It handles the Fourier-space representation
 * of the diffusion operator.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Fourier Grid:**
 *
 * For a real-to-complex FFT of an Nx × Ny × Nz grid:
 * - Complex grid size: Nx × Ny × (Nz/2 + 1)
 * - Wave vectors: k_i = 2π n_i / L_i for n_i = 0, 1, ..., N_i/2, -N_i/2+1, ..., -1
 *
 * **Stress Calculation:**
 *
 * For box relaxation, we need d(ln Q)/dL which involves derivatives of
 * the propagator with respect to box dimensions. The fourier_basis arrays
 * store k_i / L_i for efficient stress computation.
 *
 * @see FiniteDifference for real-space method alternative
 */
template <typename T>
class Pseudo
{
protected:
    std::vector<BoundaryCondition> bc;       ///< Boundary conditions (must be PERIODIC)
    std::map<std::string, double> bond_lengths;  ///< Segment lengths squared for each type
    std::vector<int> nx;                     ///< Grid dimensions [Nx, Ny, Nz]
    std::vector<double> dx;                  ///< Grid spacings [dx, dy, dz]
    double ds;                               ///< Contour step size

    int *negative_k_idx;  ///< Mapping from k to -k indices for stress calculation

    /**
     * @brief Weighted Fourier basis for stress calculation.
     *
     * fourier_basis_x[k] = (2π/Lx)² * nx² for stress derivative computation.
     * Similar for y and z directions.
     */
    double *fourier_basis_x;
    double *fourier_basis_y;
    double *fourier_basis_z;

    /**
     * @brief Boltzmann factors for full contour step.
     *
     * boltz_bond[type][k] = exp(-ds * |k|² * a²/6) where a is segment length.
     */
    std::map<std::string, double*> boltz_bond;

    /**
     * @brief Boltzmann factors for half contour step.
     *
     * boltz_bond_half[type][k] = exp(-ds/2 * |k|² * a²/6)
     * Used in Richardson extrapolation for 4th-order accuracy.
     */
    std::map<std::string, double*> boltz_bond_half;

    int total_complex_grid;  ///< Total complex grid size for r2c FFT

    /**
     * @brief Update total_complex_grid based on grid dimensions.
     */
    void update_total_complex_grid();

    /**
     * @brief Recompute Boltzmann factors after parameter change.
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

public:
    /**
     * @brief Construct Pseudo with given parameters.
     *
     * @param bond_lengths Segment lengths squared: {type: (a/a_Ref)²}
     * @param bc          Boundary conditions (must all be PERIODIC)
     * @param nx          Grid dimensions
     * @param dx          Grid spacings
     * @param ds          Contour step size
     */
    Pseudo(
        std::map<std::string, double> bond_lengths,
        std::vector<BoundaryCondition> bc, std::vector<int> nx, std::vector<double> dx, double ds);

    /**
     * @brief Virtual destructor.
     */
    virtual ~Pseudo();

    /**
     * @brief Get total complex grid size.
     * @return Nx * Ny * (Nz/2 + 1) for 3D, similar for 1D/2D
     */
    int get_total_complex_grid();

    /**
     * @brief Get Boltzmann factor for full step.
     * @param monomer_type Monomer type label
     * @return Pointer to array of size total_complex_grid
     */
    virtual double* get_boltz_bond(std::string monomer_type);

    /**
     * @brief Get Boltzmann factor for half step.
     * @param monomer_type Monomer type label
     * @return Pointer to array of size total_complex_grid
     */
    virtual double* get_boltz_bond_half(std::string monomer_type);

    /**
     * @brief Get x-direction Fourier basis for stress.
     * @return Pointer to basis array
     */
    virtual const double* get_fourier_basis_x();

    /**
     * @brief Get y-direction Fourier basis for stress.
     * @return Pointer to basis array
     */
    virtual const double* get_fourier_basis_y();

    /**
     * @brief Get z-direction Fourier basis for stress.
     * @return Pointer to basis array
     */
    virtual const double* get_fourier_basis_z();

    /**
     * @brief Get negative frequency index mapping.
     * @return Pointer to index mapping array
     */
    virtual const int* get_negative_frequency_mapping();

    /**
     * @brief Update all arrays after parameter changes.
     *
     * Call this after changing box dimensions (for box relaxation).
     *
     * @param bc          New boundary conditions
     * @param bond_lengths New segment lengths
     * @param dx          New grid spacings
     * @param ds          New contour step (usually unchanged)
     */
    virtual void update(
        std::vector<BoundaryCondition> bc,
        std::map<std::string, double> bond_lengths,
        std::vector<double> dx, double ds)
    {
        this->bond_lengths = bond_lengths;
        this->bc = bc;
        this->dx = dx;
        this->ds = ds;

        update_total_complex_grid();
        update_boltz_bond();
        update_weighted_fourier_basis();
    };
};
#endif
