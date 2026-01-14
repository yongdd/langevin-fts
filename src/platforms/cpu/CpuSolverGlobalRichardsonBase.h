/**
 * @file CpuSolverGlobalRichardsonBase.h
 * @brief Base CN-ADI2 solver for Global Richardson extrapolation.
 *
 * This solver provides the base CN-ADI2 method with the ability to advance
 * propagators using either full step (ds) or half step (ds/2). It is designed
 * to be used with CpuComputationGlobalRichardson which maintains two independent
 * propagator chains and applies Richardson extrapolation at the quadrature level.
 *
 * **Design Philosophy:**
 *
 * Unlike CpuSolverRichardsonGlobal which applies Richardson at every step,
 * this solver is stateless and simply provides the base CN-ADI2 advancement.
 * The computation layer is responsible for:
 * - Maintaining two independent propagator chains
 * - Calling advance_full_step for the full-step chain
 * - Calling advance_half_step for the half-step chain
 * - Applying Richardson extrapolation only when computing Q
 *
 * **Global Richardson at Quadrature Level:**
 *
 * The key insight is that Richardson extrapolation is most effective when
 * applied to the final computed quantities (Q, φ) rather than intermediate
 * propagator values. This is because:
 * 1. The half-step chain already has 4× smaller error per step
 * 2. Richardson's power comes from canceling accumulated error at the endpoint
 * 3. Intermediate propagators are used for φ integration where errors average out
 *
 * @see CpuComputationGlobalRichardson for the computation layer
 * @see CpuSolverCNADI for per-step Richardson (cn-adi4-lr)
 */

#ifndef CPU_SOLVER_GLOBAL_RICHARDSON_BASE_H_
#define CPU_SOLVER_GLOBAL_RICHARDSON_BASE_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "FiniteDifference.h"

/**
 * @class CpuSolverGlobalRichardsonBase
 * @brief Stateless CN-ADI2 solver with full and half step support.
 *
 * Provides base CN-ADI2 propagator advancement for Global Richardson method.
 * This solver is stateless - it does not maintain internal propagator state.
 * The computation layer manages two independent propagator chains.
 *
 * **Step Sizes:**
 *
 * - Full step: ds (from molecules->get_ds())
 * - Half step: ds/2
 *
 * **Boltzmann Factors:**
 *
 * Uses symmetric Strang splitting:
 * - Full step: exp(-w*ds/2) at start and end
 * - Half step: exp(-w*ds/4) at start and end
 */
class CpuSolverGlobalRichardsonBase
{
private:
    ComputationBox<double>* cb;  ///< Computation box
    Molecules* molecules;         ///< Molecules container

    /// @name Boltzmann factors
    /// @{
    std::map<std::string, std::vector<double>> exp_dw_full;  ///< exp(-w*ds/2)
    std::map<std::string, std::vector<double>> exp_dw_half;  ///< exp(-w*ds/4)
    /// @}

    /// @name Full step tridiagonal coefficients (ds)
    /// @{
    std::map<std::string, double*> xl_full, xd_full, xh_full;
    std::map<std::string, double*> yl_full, yd_full, yh_full;
    std::map<std::string, double*> zl_full, zd_full, zh_full;
    /// @}

    /// @name Half step tridiagonal coefficients (ds/2)
    /// @{
    std::map<std::string, double*> xl_half, xd_half, xh_half;
    std::map<std::string, double*> yl_half, yd_half, yh_half;
    std::map<std::string, double*> zl_half, zd_half, zh_half;
    /// @}

    int max_of_two(int x, int y);
    int min_of_two(int x, int y);

    /// @name ADI step implementations
    /// @{
    void advance_propagator_3d_step(
        std::vector<BoundaryCondition> bc,
        double* q_in, double* q_out,
        double* _xl, double* _xd, double* _xh,
        double* _yl, double* _yd, double* _yh,
        double* _zl, double* _zd, double* _zh);

    void advance_propagator_2d_step(
        std::vector<BoundaryCondition> bc,
        double* q_in, double* q_out,
        double* _xl, double* _xd, double* _xh,
        double* _yl, double* _yd, double* _yh);

    void advance_propagator_1d_step(
        std::vector<BoundaryCondition> bc,
        double* q_in, double* q_out,
        double* _xl, double* _xd, double* _xh);
    /// @}

    /// @name Tridiagonal solvers
    /// @{
    static void tridiagonal(
        const double* xl, const double* xd, const double* xh,
        double* x, const int INTERVAL, const double* d, const int M);

    static void tridiagonal_periodic(
        const double* xl, const double* xd, const double* xh,
        double* x, const int INTERVAL, const double* d, const int M);
    /// @}

public:
    /**
     * @brief Construct base solver for Global Richardson.
     *
     * @param cb        Computation box
     * @param molecules Molecules container
     */
    CpuSolverGlobalRichardsonBase(ComputationBox<double>* cb, Molecules* molecules);

    ~CpuSolverGlobalRichardsonBase();

    /**
     * @brief Update tridiagonal coefficients for new box dimensions.
     */
    void update_laplacian_operator();

    /**
     * @brief Update Boltzmann factors from potential fields.
     *
     * @param w_input Map of potential fields by monomer type
     */
    void update_dw(std::map<std::string, const double*> w_input);

    /**
     * @brief Advance propagator by one full step (ds).
     *
     * Uses CN-ADI2 with step size ds.
     *
     * @param q_in        Input propagator
     * @param q_out       Output propagator
     * @param monomer_type Monomer type
     * @param q_mask      Optional mask for impenetrable regions
     */
    void advance_full_step(
        double* q_in, double* q_out,
        std::string monomer_type, const double* q_mask = nullptr);

    /**
     * @brief Advance propagator by one half step (ds/2).
     *
     * Uses CN-ADI2 with step size ds/2.
     *
     * @param q_in        Input propagator
     * @param q_out       Output propagator
     * @param monomer_type Monomer type
     * @param q_mask      Optional mask for impenetrable regions
     */
    void advance_half_step(
        double* q_in, double* q_out,
        std::string monomer_type, const double* q_mask = nullptr);
};

#endif
