/**
 * @file CpuSolverCNADI.h
 * @brief CN-ADI finite difference solver for continuous chains on CPU.
 *
 * This header provides CpuSolverCNADI, which implements the CN-ADI
 * (Crank-Nicolson Alternating Direction Implicit) finite difference method
 * for solving the modified diffusion equation.
 * Unlike pseudo-spectral methods, this supports non-periodic boundary conditions.
 *
 * **Numerical Method:**
 *
 * Uses CN-ADI (implicit) scheme with operator splitting:
 *
 *     (1 + A_x/2)(1 + A_y/2)(1 + A_z/2) q^(n+1) =
 *     (1 - A_x/2)(1 - A_y/2)(1 - A_z/2) q^(n)
 *
 * where A_α = -(b²ds/12) ∂²/∂α² + (w·ds/6).
 *
 * **Accuracy:**
 *
 * This solver implements CN-ADI2 (2nd order accuracy in ds).
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
 * @see CpuSolverPseudoRQM4 for spectral method (periodic only)
 * @see FiniteDifference for coefficient generation
 */

#ifndef CPU_SOLVER_CNADI_H_
#define CPU_SOLVER_CNADI_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "FiniteDifference.h"
#include "SpaceGroup.h"

/**
 * @class CpuSolverCNADI
 * @brief CPU solver using CN-ADI (Crank-Nicolson ADI).
 *
 * Implements the CN-ADI scheme for solving the modified diffusion equation
 * with various boundary conditions. Provides CN-ADI2 (2nd order).
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
 * - xl, xd, xh: Lower, diagonal, upper coefficients for x-direction (full step)
 * - Similar for y and z directions
 *
 * @note This is a beta feature. Stress calculation is not yet implemented.
 *
 * @example
 * @code
 * CpuSolverCNADI solver(cb, molecules);
 * solver.update_dw(w_fields);
 *
 * // Advance with reflecting boundaries
 * solver.advance_propagator(q_in, q_out, "A", nullptr);
 * @endcode
 */
class CpuSolverCNADI : public CpuSolver<double>
{
private:
    ComputationBox<double>* cb;  ///< Computation box for grid/boundary info
    Molecules *molecules;         ///< Molecules container
    SpaceGroup* space_group_;     ///< Space group pointer (nullptr if not used)

    /// @name Tridiagonal coefficients per ds_index per monomer_type
    /// Nested map structure: [ds_index][monomer_type] -> coefficient array
    /// @{
    std::map<int, std::map<std::string, double*>> xl;  ///< x-direction lower diagonal
    std::map<int, std::map<std::string, double*>> xd;  ///< x-direction main diagonal
    std::map<int, std::map<std::string, double*>> xh;  ///< x-direction upper diagonal

    std::map<int, std::map<std::string, double*>> yl;  ///< y-direction lower diagonal
    std::map<int, std::map<std::string, double*>> yd;  ///< y-direction main diagonal
    std::map<int, std::map<std::string, double*>> yh;  ///< y-direction upper diagonal

    std::map<int, std::map<std::string, double*>> zl;  ///< z-direction lower diagonal
    std::map<int, std::map<std::string, double*>> zd;  ///< z-direction main diagonal
    std::map<int, std::map<std::string, double*>> zh;  ///< z-direction upper diagonal
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
     * @param ds_index    Index for ds value (1-based) from ContourLengthMapping
     */
    void advance_propagator_3d(
        std::vector<BoundaryCondition> bc,
        double *q_in, double *q_out, std::string monomer_type, int ds_index);

    /**
     * @brief Advance propagator in 2D using ADI splitting.
     * @param ds_index Index for ds value (1-based) from ContourLengthMapping
     */
    void advance_propagator_2d(
        std::vector<BoundaryCondition> bc,
        double *q_in, double *q_out, std::string monomer_type, int ds_index);

    /**
     * @brief Advance propagator in 1D (direct solve).
     * @param ds_index Index for ds value (1-based) from ContourLengthMapping
     */
    void advance_propagator_1d(
        std::vector<BoundaryCondition> bc,
        double *q_in, double *q_out, std::string monomer_type, int ds_index);

    /**
     * @brief Single ADI step with specified coefficients.
     *
     * Core implementation of the ADI step. This method performs:
     * 1. Apply potential Boltzmann factor: q_exp = exp_dw * q_in
     * 2. ADI diffusion solve
     * 3. Apply potential Boltzmann factor: q_out *= exp_dw
     *
     * @param q_in        Input propagator
     * @param q_out       Output propagator
     * @param exp_dw_ptr  Boltzmann factor for potential
     * @param _xl,_xd,_xh X-direction tridiagonal coefficients
     * @param _yl,_yd,_yh Y-direction tridiagonal coefficients
     * @param _zl,_zd,_zh Z-direction tridiagonal coefficients
     * @param q_mask      Optional mask for impenetrable regions
     */
    void advance_propagator_step(
        double *q_in, double *q_out,
        const double *exp_dw_ptr,
        double *_xl, double *_xd, double *_xh,
        double *_yl, double *_yd, double *_yh,
        double *_zl, double *_zd, double *_zh,
        const double *q_mask);

    /**
     * @brief 3D ADI step with specified coefficients.
     */
    void advance_propagator_3d_step(
        std::vector<BoundaryCondition> bc,
        double *q_in, double *q_out,
        double *_xl, double *_xd, double *_xh,
        double *_yl, double *_yd, double *_yh,
        double *_zl, double *_zd, double *_zh);

    /**
     * @brief 2D ADI step with specified coefficients.
     */
    void advance_propagator_2d_step(
        std::vector<BoundaryCondition> bc,
        double *q_in, double *q_out,
        double *_xl, double *_xd, double *_xh,
        double *_yl, double *_yd, double *_yh);

    /**
     * @brief 1D step with specified coefficients.
     */
    void advance_propagator_1d_step(
        std::vector<BoundaryCondition> bc,
        double *q_in, double *q_out,
        double *_xl, double *_xd, double *_xh);

public:
    /**
     * @brief Construct CN-ADI solver.
     *
     * Allocates tridiagonal coefficient arrays for each direction
     * and each monomer type.
     *
     * @param cb            Computation box with boundary conditions
     * @param molecules     Molecules container with monomer types
     */
    CpuSolverCNADI(ComputationBox<double>* cb, Molecules *molecules);

    /**
     * @brief Destructor. Frees tridiagonal coefficient arrays.
     */
    ~CpuSolverCNADI();

    /**
     * @brief Set space group for reduced basis operations.
     *
     * When set, q_in/q_out are in reduced basis and the solver expands/reduces
     * internally around finite-difference operations.
     *
     * @param sg Space group pointer (nullptr to disable)
     */
    void set_space_group(SpaceGroup* sg) override { space_group_ = sg; }

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
     * @param q_mask      Optional mask (currently ignored for CN-ADI)
     * @param ds_index    Index for the ds value (1-based) from ContourLengthMapping
     */
    void advance_propagator(
                double *q_in, double *q_out, std::string monomer_type, const double *q_mask, int ds_index) override;

    /**
     * @brief Half-bond step (not used for continuous chains).
     *
     * Empty implementation - continuous chains don't use half-bond steps.
     */
    void advance_propagator_half_bond_step(double *, double *, std::string) override {};

    /**
     * @brief Compute stress from one segment.
     *
     * @warning Not yet implemented for CN-ADI method.
     *
     * @return Empty vector (stress calculation not supported)
     */
    std::vector<double> compute_single_segment_stress(
                double *q_1, double *q_2, std::string monomer_type, bool is_half_bond_length) override;
};
#endif
