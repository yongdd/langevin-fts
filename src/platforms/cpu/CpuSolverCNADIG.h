/**
 * @file CpuSolverCNADIG.h
 * @brief Global Richardson extrapolation solver for continuous chains on CPU.
 *
 * This header provides CpuSolverCNADIG, which implements "Global Richardson"
 * extrapolation for solving the modified diffusion equation. Unlike CN-ADI4 (per-step
 * Richardson), this method maintains two independent propagator evolutions and applies
 * Richardson extrapolation to combine them.
 *
 * **Algorithm (Global Richardson):**
 *
 * Two independent propagator evolutions are maintained:
 * 1. Full-step evolution: advances by ds each step
 * 2. Half-step evolution: advances by ds/2 twice each step
 *
 * The evolutions are INDEPENDENT - each uses its own previous state, not the
 * Richardson-extrapolated result. At each step:
 * - q_full_{n+1} = A(q_full_n, ds)
 * - q_half_{n+1} = A(A(q_half_n, ds/2), ds/2)
 * - q_out = (4*q_half_{n+1} - q_full_{n+1}) / 3
 *
 * **Comparison with CN-ADI4 (Per-Step Richardson):**
 *
 * CN-ADI4:
 * - q_out = (4*A(A(q_in, ds/2), ds/2) - A(q_in, ds)) / 3
 * - Next step uses q_out as input
 * - Can have stability issues with sharp initial conditions
 *
 * Global Richardson (this class):
 * - Two independent evolutions from the same initial condition
 * - Richardson applied to combine independent results
 * - More stable for delta-function initial conditions
 *
 * **Usage:**
 *
 * The solver must be reset when starting a new propagator computation:
 * 1. Call reset_internal_state() before computing a new propagator
 * 2. First advance_propagator() call initializes internal states from q_in
 * 3. Subsequent calls use internal states (ignoring q_in)
 *
 * @see CpuSolver for the abstract interface
 * @see CpuSolverCNADI for per-step Richardson (CN-ADI4)
 */

#ifndef CPU_SOLVER_CN_ADI_G_H_
#define CPU_SOLVER_CN_ADI_G_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "FiniteDifference.h"

/**
 * @class CpuSolverCNADIG
 * @brief CPU solver using Global Richardson extrapolation.
 *
 * Implements Richardson extrapolation with two independent propagator evolutions.
 * This provides 4th-order accuracy while maintaining stability for challenging
 * initial conditions (e.g., delta functions for grafted polymers).
 *
 * **Internal State:**
 *
 * The solver maintains two independent propagator states:
 * - q_full_internal: Evolved with full step size (ds)
 * - q_half_internal: Evolved with half step size (ds/2, twice per step)
 *
 * These states are initialized from q_in on the first call after reset.
 *
 * **Computational Cost:**
 *
 * Per advance_propagator() call:
 * - 1 full step (ds) + 2 half steps (ds/2) = 3 ADI solves
 * - Same cost as CN-ADI4, but different algorithm
 */
class CpuSolverCNADIG : public CpuSolver<double>
{
private:
    ComputationBox<double>* cb;  ///< Computation box for grid/boundary info
    Molecules* molecules;         ///< Molecules container

    /// @name Tridiagonal coefficients for full step (ds)
    /// @{
    std::map<std::string, double*> xl_full;  ///< x lower diagonal
    std::map<std::string, double*> xd_full;  ///< x main diagonal
    std::map<std::string, double*> xh_full;  ///< x upper diagonal

    std::map<std::string, double*> yl_full;  ///< y lower diagonal
    std::map<std::string, double*> yd_full;  ///< y main diagonal
    std::map<std::string, double*> yh_full;  ///< y upper diagonal

    std::map<std::string, double*> zl_full;  ///< z lower diagonal
    std::map<std::string, double*> zd_full;  ///< z main diagonal
    std::map<std::string, double*> zh_full;  ///< z upper diagonal
    /// @}

    /// @name Tridiagonal coefficients for half step (ds/2)
    /// @{
    std::map<std::string, double*> xl_half;  ///< x lower diagonal
    std::map<std::string, double*> xd_half;  ///< x main diagonal
    std::map<std::string, double*> xh_half;  ///< x upper diagonal

    std::map<std::string, double*> yl_half;  ///< y lower diagonal
    std::map<std::string, double*> yd_half;  ///< y main diagonal
    std::map<std::string, double*> yh_half;  ///< y upper diagonal

    std::map<std::string, double*> zl_half;  ///< z lower diagonal
    std::map<std::string, double*> zd_half;  ///< z main diagonal
    std::map<std::string, double*> zh_half;  ///< z upper diagonal
    /// @}

    /// @name Boltzmann factors for potential field
    /// @{
    std::map<std::string, std::vector<double>> exp_dw_full;  ///< exp(-w*ds/2)
    std::map<std::string, std::vector<double>> exp_dw_half;  ///< exp(-w*ds/4)
    /// @}

    /// @name Internal propagator states for independent evolutions
    /// @{
    double* q_full_internal;  ///< Full-step propagator state
    double* q_half_internal;  ///< Half-step propagator state
    double* q_temp;           ///< Temporary for intermediate calculations
    bool is_initialized;      ///< Whether internal states have been initialized
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
     * @brief Advance propagator by one full step (ds).
     *
     * @param q_in   Input propagator
     * @param q_out  Output propagator
     * @param monomer_type Monomer type for coefficients
     */
    void advance_full_step(double* q_in, double* q_out, std::string monomer_type);

    /**
     * @brief Advance propagator by one half step (ds/2).
     *
     * @param q_in   Input propagator
     * @param q_out  Output propagator
     * @param monomer_type Monomer type for coefficients
     */
    void advance_half_step(double* q_in, double* q_out, std::string monomer_type);

    /**
     * @brief 3D ADI step with specified coefficients.
     */
    void advance_propagator_3d_step(
        std::vector<BoundaryCondition> bc,
        double* q_in, double* q_out,
        double* _xl, double* _xd, double* _xh,
        double* _yl, double* _yd, double* _yh,
        double* _zl, double* _zd, double* _zh);

    /**
     * @brief 2D ADI step with specified coefficients.
     */
    void advance_propagator_2d_step(
        std::vector<BoundaryCondition> bc,
        double* q_in, double* q_out,
        double* _xl, double* _xd, double* _xh,
        double* _yl, double* _yd, double* _yh);

    /**
     * @brief 1D step with specified coefficients.
     */
    void advance_propagator_1d_step(
        std::vector<BoundaryCondition> bc,
        double* q_in, double* q_out,
        double* _xl, double* _xd, double* _xh);

public:
    /**
     * @brief Construct Global Richardson solver.
     *
     * Allocates tridiagonal coefficient arrays for both full and half
     * step sizes, and internal state arrays.
     *
     * @param cb        Computation box with boundary conditions
     * @param molecules Molecules container with monomer types
     */
    CpuSolverCNADIG(ComputationBox<double>* cb, Molecules* molecules);

    /**
     * @brief Destructor. Frees all allocated arrays.
     */
    ~CpuSolverCNADIG();

    /**
     * @brief Update finite difference coefficients.
     *
     * Recomputes tridiagonal coefficients for both step sizes
     * when box dimensions change.
     */
    void update_laplacian_operator() override;

    /**
     * @brief Update Boltzmann factors from potential fields.
     *
     * Computes exp(-w*ds/2) for full step and exp(-w*ds/4) for half step.
     *
     * @param w_input Map of potential fields by monomer type
     */
    void update_dw(std::map<std::string, const double*> w_input) override;

    /**
     * @brief Reset internal propagator states.
     *
     * Must be called before starting a new propagator computation.
     * The next advance_propagator() call will initialize internal
     * states from q_in.
     */
    void reset_internal_state();

    /**
     * @brief Solve tridiagonal system (non-periodic).
     */
    static void tridiagonal(
        const double* xl, const double* xd, const double* xh,
        double* x, const int INTERVAL, const double* d, const int M);

    /**
     * @brief Solve cyclic tridiagonal system (periodic).
     */
    static void tridiagonal_periodic(
        const double* xl, const double* xd, const double* xh,
        double* x, const int INTERVAL, const double* d, const int M);

    /**
     * @brief Advance propagator by one contour step using Global Richardson.
     *
     * On first call after reset:
     * - Initializes q_full_internal and q_half_internal from q_in
     *
     * On all calls:
     * - Advances q_full_internal by one ds step (using its own state)
     * - Advances q_half_internal by two ds/2 steps (using its own state)
     * - Returns Richardson-extrapolated result: (4*q_half - q_full) / 3
     *
     * @param q_in        Input propagator (only used on first call after reset)
     * @param q_out       Output propagator (Richardson-extrapolated)
     * @param monomer_type Monomer type
     * @param q_mask      Optional mask for impenetrable regions
     * @param ds_index    Index for the ds value (default: 1)
     */
    void advance_propagator(
        double* q_in, double* q_out, std::string monomer_type,
        const double* q_mask, int ds_index = 1) override;

    /**
     * @brief Half-bond step (not used for continuous chains).
     */
    void advance_propagator_half_bond_step(double*, double*, std::string) override {};

    /**
     * @brief Compute stress from one segment.
     *
     * @warning Not yet implemented for Global Richardson method.
     *
     * @return Empty vector (stress calculation not supported)
     */
    std::vector<double> compute_single_segment_stress(
        double* q_1, double* q_2, std::string monomer_type,
        bool is_half_bond_length) override;
};

#endif  // CPU_SOLVER_CN_ADI_G_H_
