/**
 * @file ETDRK4Coefficients.h
 * @brief ETDRK4 coefficients for pseudo-spectral method.
 *
 * This header provides the ETDRK4Coefficients class for computing and storing
 * coefficients used in the ETDRK4 (Exponential Time Differencing Runge-Kutta 4)
 * time integration scheme for the modified diffusion equation.
 *
 * **ETDRK4 Algorithm (Cox & Matthews 2002):**
 *
 * For the equation: dq/ds = L*q + N(q) where:
 * - L = (b^2/6)*nabla^2 (linear diffusion, eigenvalue c = -k^2*b^2/6 in Fourier space)
 * - N(q) = -w*q (nonlinear/potential term)
 *
 * Stages:
 *   a = E2*q_hat + alpha*N_hat_n
 *   b = E2*q_hat + alpha*N_hat_a
 *   c = E2*a_hat + alpha*(2*N_hat_b - N_hat_n)
 *   q_hat_{n+1} = E*q_hat + f1*N_hat_n + f2*(N_hat_a + N_hat_b) + f3*N_hat_c
 *
 * **Coefficient Computation:**
 *
 * Uses Kassam-Trefethen (2005) contour integral method for numerical stability:
 * - Avoids catastrophic cancellation for small eigenvalues
 * - 32-point contour with radius 1.0 provides ~15 digits accuracy
 *
 * **References:**
 * - Cox & Matthews, J. Comput. Phys. 176, 430-455 (2002)
 * - Kassam & Trefethen, SIAM J. Sci. Comput. 26, 1214-1233 (2005)
 * - Song, Liu & Zhang, Chinese J. Polym. Sci. 36, 488-496 (2018)
 *
 * @see CpuSolverPseudoRQM4 for usage
 */

#ifndef ETDRK4_COEFFICIENTS_H_
#define ETDRK4_COEFFICIENTS_H_

#include <string>
#include <vector>
#include <map>
#include <array>

#include "Exception.h"
#include "ComputationBox.h"

/**
 * @class ETDRK4Coefficients
 * @brief Pre-computed ETDRK4 coefficient arrays for pseudo-spectral solver.
 *
 * This class computes and stores the ETDRK4 coefficients for each monomer type.
 * Supports all boundary conditions (periodic, reflecting, absorbing) and
 * non-orthogonal crystal systems.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Coefficient Storage:**
 *
 * Per monomer type, stores 6 arrays of size M_complex:
 * - E:     exp(c*h)        - full step exponential
 * - E2:    exp(c*h/2)      - half step exponential
 * - alpha: h*phi_1(c*h/2)  - stage coefficient
 * - f1, f2, f3: beta_1, beta_2, beta_3 (scaled by h) for final combination
 *
 * where h = ds (contour step) and c = -b^2*k^2/6 (eigenvalue).
 */
template <typename T>
class ETDRK4Coefficients
{
private:
    std::vector<BoundaryCondition> bc;           ///< Boundary conditions per dimension
    std::map<std::string, double> bond_lengths;  ///< Segment lengths for each type
    std::vector<int> nx;                         ///< Grid dimensions [Nx, Ny, Nz]
    std::vector<double> dx;                      ///< Grid spacings [dx, dy, dz]
    double ds;                                   ///< Contour step size
    std::array<double, 6> recip_metric_;         ///< Reciprocal metric tensor
    int total_complex_grid;                      ///< Size of coefficient arrays

    // ETDRK4 coefficient arrays per monomer type
    std::map<std::string, double*> E_;      ///< exp(c*h)
    std::map<std::string, double*> E2_;     ///< exp(c*h/2)
    std::map<std::string, double*> alpha_;  ///< h*phi_1(c*h/2)
    std::map<std::string, double*> f1_;     ///< h*beta_1(c*h)
    std::map<std::string, double*> f2_;     ///< h*beta_2(c*h)
    std::map<std::string, double*> f3_;     ///< h*beta_3(c*h)

    /**
     * @brief Check if all BCs are periodic.
     */
    bool is_all_periodic() const;

    /**
     * @brief Compute total complex grid size.
     */
    void compute_total_complex_grid();

    /**
     * @brief Compute coefficients for periodic BC.
     */
    void compute_coefficients_periodic();

    /**
     * @brief Compute coefficients for mixed BC.
     */
    void compute_coefficients_mixed();

public:
    /**
     * @brief Construct ETDRK4Coefficients with given parameters.
     *
     * @param bond_lengths  Segment lengths: {type: b}
     * @param bc            Boundary conditions (can be mixed)
     * @param nx            Grid dimensions
     * @param dx            Grid spacings
     * @param ds            Contour step size
     * @param recip_metric  Reciprocal metric tensor (for periodic BC)
     */
    ETDRK4Coefficients(
        std::map<std::string, double> bond_lengths,
        std::vector<BoundaryCondition> bc,
        std::vector<int> nx, std::vector<double> dx, double ds,
        std::array<double, 6> recip_metric = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0});

    /**
     * @brief Destructor. Frees coefficient arrays.
     */
    ~ETDRK4Coefficients();

    // Prevent copying
    ETDRK4Coefficients(const ETDRK4Coefficients&) = delete;
    ETDRK4Coefficients& operator=(const ETDRK4Coefficients&) = delete;

    /**
     * @brief Get total complex grid size.
     */
    int get_total_complex_grid() const { return total_complex_grid; }

    /**
     * @brief Get full-step exponential exp(c*h).
     */
    const double* get_E(const std::string& monomer_type) const;

    /**
     * @brief Get half-step exponential exp(c*h/2).
     */
    const double* get_E2(const std::string& monomer_type) const;

    /**
     * @brief Get stage coefficient h*phi_1(c*h/2).
     */
    const double* get_alpha(const std::string& monomer_type) const;

    /**
     * @brief Get beta_1 coefficient h*beta_1(c*h).
     */
    const double* get_f1(const std::string& monomer_type) const;

    /**
     * @brief Get beta_2 coefficient h*beta_2(c*h).
     */
    const double* get_f2(const std::string& monomer_type) const;

    /**
     * @brief Get beta_3 coefficient h*beta_3(c*h).
     */
    const double* get_f3(const std::string& monomer_type) const;

    /**
     * @brief Update coefficients after parameter changes.
     *
     * @param bc            New boundary conditions
     * @param bond_lengths  New segment lengths
     * @param dx            New grid spacings
     * @param ds            New contour step
     * @param recip_metric  New reciprocal metric tensor
     */
    void update(
        std::vector<BoundaryCondition> bc,
        std::map<std::string, double> bond_lengths,
        std::vector<double> dx, double ds,
        std::array<double, 6> recip_metric = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0});
};

#endif
