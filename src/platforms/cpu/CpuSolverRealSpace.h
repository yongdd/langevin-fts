/**
 * @file CpuSolverRealSpace.h
 * @brief Real-space finite difference solver for continuous chains on CPU.
 *
 * This header provides CpuSolverRealSpace, which implements the Crank-Nicolson
 * finite difference method for solving the modified diffusion equation.
 * Unlike pseudo-spectral methods, this supports non-periodic boundary conditions.
 *
 * **Numerical Method:**
 *
 * Uses Crank-Nicolson (implicit) scheme with operator splitting (ADI):
 *
 *     (1 + A_x/2)(1 + A_y/2)(1 + A_z/2) q^(n+1) =
 *     (1 - A_x/2)(1 - A_y/2)(1 - A_z/2) q^(n)
 *
 * where A_α = -(b²ds/12) ∂²/∂α² + (w·ds/6).
 *
 * **ADI (Alternating Direction Implicit):**
 *
 * Each spatial direction is solved implicitly in sequence, resulting in
 * a series of tridiagonal systems that are efficiently solvable.
 *
 * **Supported Boundary Conditions:**
 *
 * - Periodic: q(0) = q(L)
 * - Reflecting (Neumann): ∂q/∂n = 0 at boundary
 * - Absorbing (Dirichlet): q = 0 at boundary
 *
 * @see CpuSolver for the abstract interface
 * @see CpuSolverPseudoContinuous for spectral method (periodic only)
 * @see FiniteDifference for coefficient generation
 */

#ifndef CPU_SOLVER_REAL_SPACE_H_
#define CPU_SOLVER_REAL_SPACE_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "FiniteDifference.h"

/**
 * @class CpuSolverRealSpace
 * @brief CPU real-space solver using Crank-Nicolson finite differences.
 *
 * Implements the ADI (Alternating Direction Implicit) scheme for solving
 * the modified diffusion equation with various boundary conditions.
 *
 * **Tridiagonal Systems:**
 *
 * Each direction requires solving Ax = b where A is tridiagonal:
 *
 *     [ d_0  h_0   0    0   ...  0   ]   [ x_0 ]   [ b_0 ]
 *     [ l_1  d_1  h_1   0   ...  0   ]   [ x_1 ]   [ b_1 ]
 *     [  0   l_2  d_2  h_2  ...  0   ] × [ x_2 ] = [ b_2 ]
 *     [ ... ... ... ... ... ... ]   [ ... ]   [ ... ]
 *     [  0    0    0  l_n  d_n  h_n  ]   [ x_n ]   [ b_n ]
 *
 * For periodic boundaries, this becomes a cyclic tridiagonal system.
 *
 * **Memory per Direction per Monomer:**
 *
 * - xl, xd, xh: Lower, diagonal, upper coefficients for x-direction
 * - yl, yd, yh: Same for y-direction
 * - zl, zd, zh: Same for z-direction
 *
 * @note This is a beta feature. Stress calculation is not yet implemented.
 *
 * @example
 * @code
 * CpuSolverRealSpace solver(cb, molecules);
 * solver.update_dw(w_fields);
 *
 * // Advance with reflecting boundaries
 * solver.advance_propagator(q_in, q_out, "A", nullptr);
 * @endcode
 */
class CpuSolverRealSpace : public CpuSolver<double>
{
private:
    ComputationBox<double>* cb;  ///< Computation box for grid/boundary info
    Molecules *molecules;         ///< Molecules container

    /// @name Tridiagonal coefficients for x-direction
    /// @{
    std::map<std::string, double*> xl;  ///< Lower diagonal
    std::map<std::string, double*> xd;  ///< Main diagonal
    std::map<std::string, double*> xh;  ///< Upper diagonal
    /// @}

    /// @name Tridiagonal coefficients for y-direction
    /// @{
    std::map<std::string, double*> yl;  ///< Lower diagonal
    std::map<std::string, double*> yd;  ///< Main diagonal
    std::map<std::string, double*> yh;  ///< Upper diagonal
    /// @}

    /// @name Tridiagonal coefficients for z-direction
    /// @{
    std::map<std::string, double*> zl;  ///< Lower diagonal
    std::map<std::string, double*> zd;  ///< Main diagonal
    std::map<std::string, double*> zh;  ///< Upper diagonal
    /// @}

    /**
     * @brief Return maximum of two integers.
     */
    int max_of_two(int x, int y);

    /**
     * @brief Return minimum of two integers.
     */
    int min_of_two(int x, int y);

    /**
     * @brief Advance propagator in 3D using ADI splitting.
     *
     * @param bc          Boundary conditions for each direction
     * @param q_in        Input propagator
     * @param q_out       Output propagator
     * @param monomer_type Monomer type for coefficients
     */
    void advance_propagator_3d(
        std::vector<BoundaryCondition> bc,
        double *q_in, double *q_out, std::string monomer_type);

    /**
     * @brief Advance propagator in 2D using ADI splitting.
     */
    void advance_propagator_2d(
        std::vector<BoundaryCondition> bc,
        double *q_in, double *q_out, std::string monomer_type);

    /**
     * @brief Advance propagator in 1D (direct solve).
     */
    void advance_propagator_1d(
        std::vector<BoundaryCondition> bc,
        double *q_in, double *q_out, std::string monomer_type);

public:
    /**
     * @brief Construct real-space solver.
     *
     * Allocates tridiagonal coefficient arrays for each direction
     * and each monomer type.
     *
     * @param cb        Computation box with boundary conditions
     * @param molecules Molecules container with monomer types
     */
    CpuSolverRealSpace(ComputationBox<double>* cb, Molecules *molecules);

    /**
     * @brief Destructor. Frees tridiagonal coefficient arrays.
     */
    ~CpuSolverRealSpace();

    /**
     * @brief Update finite difference coefficients.
     *
     * Recomputes tridiagonal coefficients when box dimensions change.
     */
    void update_laplacian_operator() override;

    /**
     * @brief Update tridiagonal matrices from potential fields.
     *
     * Incorporates Boltzmann factors into the tridiagonal coefficients.
     *
     * @param w_input Map of potential fields by monomer type
     */
    void update_dw(std::map<std::string, const double*> w_input) override;

    /**
     * @brief Solve tridiagonal system (non-periodic).
     *
     * Uses Thomas algorithm for tridiagonal systems:
     *     l[i] x[i-1] + d[i] x[i] + h[i] x[i+1] = rhs[i]
     *
     * @param xl       Lower diagonal coefficients
     * @param xd       Main diagonal coefficients
     * @param xh       Upper diagonal coefficients
     * @param x        Output solution array
     * @param INTERVAL Stride between consecutive elements
     * @param d        Right-hand side vector
     * @param M        Number of unknowns
     *
     * @note Static function, can be called without instance.
     */
    static void tridiagonal(
        const double *xl, const double *xd, const double *xh,
        double *x, const int INTERVAL, const double *d, const int M);

    /**
     * @brief Solve cyclic tridiagonal system (periodic).
     *
     * Uses Sherman-Morrison formula for cyclic systems where
     * there are additional corner elements connecting first and last rows.
     *
     * @param xl       Lower diagonal (including wrap-around)
     * @param xd       Main diagonal
     * @param xh       Upper diagonal (including wrap-around)
     * @param x        Output solution array
     * @param INTERVAL Stride between consecutive elements
     * @param d        Right-hand side vector
     * @param M        Number of unknowns
     */
    static void tridiagonal_periodic(
        const double *xl, const double *xd, const double *xh,
        double *x, const int INTERVAL, const double *d, const int M);

    /**
     * @brief Advance propagator by one contour step.
     *
     * Uses ADI scheme appropriate for the grid dimensionality.
     *
     * @param q_in        Input propagator
     * @param q_out       Output propagator
     * @param monomer_type Monomer type
     * @param q_mask      Optional mask (currently ignored for real-space)
     */
    void advance_propagator(
                double *q_in, double *q_out, std::string monomer_type, const double *q_mask) override;

    /**
     * @brief Half-bond step (not used for continuous chains).
     *
     * Empty implementation - continuous chains don't use half-bond steps.
     */
    void advance_propagator_half_bond_step(double *, double *, std::string) override {};

    /**
     * @brief Compute stress from one segment.
     *
     * @warning Not yet implemented for real-space method.
     *
     * @return Empty vector (stress calculation not supported)
     */
    std::vector<double> compute_single_segment_stress(
                double *q_1, double *q_2, std::string monomer_type, bool is_half_bond_length) override;
};
#endif
