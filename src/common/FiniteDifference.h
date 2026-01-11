/**
 * @file FiniteDifference.h
 * @brief Finite difference Laplacian matrix construction for real-space method.
 *
 * This header provides the FiniteDifference class which constructs the
 * discretized Laplacian operator for the real-space (finite difference)
 * propagator solver. Unlike the pseudo-spectral method which uses FFT,
 * the real-space method directly solves the diffusion equation using
 * CN-ADI (Crank-Nicolson ADI) with tridiagonal matrix solvers.
 *
 * **Real-Space Method:**
 *
 * The modified diffusion equation:
 *   dq/ds = (a²/6) ∇²q - w(r) q
 *
 * is discretized as:
 *   (I - ds/2 * L) q^{n+1} = (I + ds/2 * L) q^n - ds * w * q^{n+1/2}
 *
 * where L is the Laplacian matrix constructed by this class.
 *
 * **Boundary Conditions:**
 *
 * The real-space method supports non-periodic boundary conditions:
 * - PERIODIC: q(0) = q(L), handled by matrix structure
 * - REFLECTING: dq/dn = 0 at boundary (Neumann BC)
 * - ABSORBING: q = 0 at boundary (Dirichlet BC)
 *
 * @note This is a beta feature. The pseudo-spectral method is generally
 *       preferred for periodic systems due to higher accuracy.
 *
 * @see CpuSolverRealSpace, CudaSolverRealSpace for implementations
 * @see Pseudo for the pseudo-spectral alternative
 *
 * @example
 * @code
 * // Get tridiagonal Laplacian coefficients for x-direction
 * int nx = 32;
 * double dx = 0.125;
 * double bond_length_sq = 1.0;
 * double ds = 0.01;
 *
 * double* xl = new double[nx];  // lower diagonal
 * double* xd = new double[nx];  // main diagonal
 * double* xh = new double[nx];  // upper diagonal
 * // Similar for y and z directions...
 *
 * FiniteDifference::get_laplacian_matrix(
 *     bc, {nx}, {dx},
 *     xl, xd, xh, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
 *     bond_length_sq, ds);
 * @endcode
 */

#ifndef FINITE_DIFFERENCE_H_
#define FINITE_DIFFERENCE_H_

// Toggle CN-ADI4 (4th-order) for real-space solver
// Set to 0 for CN-ADI2 (2nd-order, faster, more stable) - default
// Set to 1 for CN-ADI4 (4th-order, slower, may be unstable near absorbing boundaries)
#ifndef REALSPACE_CN_ADI4
#define REALSPACE_CN_ADI4 0
#endif

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "ComputationBox.h"
#include "Molecules.h"

/**
 * @class FiniteDifference
 * @brief Static utility for constructing finite difference Laplacian matrices.
 *
 * Constructs tridiagonal matrix coefficients for the discretized Laplacian
 * operator in each direction. The matrices are used in CN-ADI (Crank-Nicolson
 * Alternating Direction Implicit) time stepping for the real-space propagator solver.
 *
 * **Matrix Structure:**
 *
 * For each direction, the Laplacian is tridiagonal with coefficients:
 * - Lower diagonal (xl, yl, zl): coefficient for q_{i-1}
 * - Main diagonal (xd, yd, zd): coefficient for q_i
 * - Upper diagonal (xh, yh, zh): coefficient for q_{i+1}
 *
 * The coefficients incorporate:
 * - Grid spacing dx
 * - Segment length a² (bond_length_sq)
 * - Time step ds
 * - Boundary conditions
 */
class FiniteDifference
{
public:
    /**
     * @brief Construct Laplacian matrix coefficients.
     *
     * Fills the tridiagonal matrix arrays for the Laplacian operator
     * in each spatial direction, incorporating boundary conditions.
     *
     * @param bc             Boundary conditions for each face
     * @param nx             Grid dimensions [Nx, Ny, Nz]
     * @param dx             Grid spacings [dx, dy, dz]
     * @param xl             Output: x-direction lower diagonal (length Nx)
     * @param xd             Output: x-direction main diagonal (length Nx)
     * @param xh             Output: x-direction upper diagonal (length Nx)
     * @param yl             Output: y-direction lower diagonal (length Ny, or nullptr for 1D)
     * @param yd             Output: y-direction main diagonal
     * @param yh             Output: y-direction upper diagonal
     * @param zl             Output: z-direction lower diagonal (length Nz, or nullptr for 1D/2D)
     * @param zd             Output: z-direction main diagonal
     * @param zh             Output: z-direction upper diagonal
     * @param bond_length_sq Squared segment length (a/a_Ref)²
     * @param ds             Contour step size
     *
     * @note For 1D, pass nullptr for y and z arrays.
     *       For 2D, pass nullptr for z arrays.
     *
     * @note The coefficients include the factor (a²/6) * ds / dx² for
     *       CN-ADI integration.
     */
    static void get_laplacian_matrix(
        std::vector<BoundaryCondition> bc,
        std::vector<int> nx, std::vector<double> dx,
        double *xl, double *xd, double *xh,
        double *yl, double *yd, double *yh,
        double *zl, double *zd, double *zh,
        double bond_length_sq, double ds);
};
#endif