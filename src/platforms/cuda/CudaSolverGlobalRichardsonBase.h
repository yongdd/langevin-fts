/**
 * @file CudaSolverGlobalRichardsonBase.h
 * @brief Base CN-ADI2 solver for Global Richardson extrapolation on GPU.
 *
 * This solver provides the base CN-ADI2 method with the ability to advance
 * propagators using either full step (ds) or half step (ds/2). It is designed
 * to be used with CudaComputationGlobalRichardson which maintains two independent
 * propagator chains and applies Richardson extrapolation at the quadrature level.
 *
 * **Design Philosophy:**
 *
 * Unlike CudaSolverRichardsonGlobal which applies Richardson at every step,
 * this solver is stateless and simply provides the base CN-ADI2 advancement.
 * The computation layer is responsible for:
 * - Maintaining two independent propagator chains
 * - Calling advance_full_step for the full-step chain
 * - Calling advance_half_step for the half-step chain
 * - Applying Richardson extrapolation only when computing Q
 *
 * @see CudaComputationGlobalRichardson for the computation layer
 * @see CpuSolverGlobalRichardsonBase for CPU version
 */

#ifndef CUDA_SOLVER_GLOBAL_RICHARDSON_BASE_H_
#define CUDA_SOLVER_GLOBAL_RICHARDSON_BASE_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "FiniteDifference.h"
#include "CudaCommon.h"

/**
 * @class CudaSolverGlobalRichardsonBase
 * @brief Stateless CN-ADI2 solver with full and half step support on GPU.
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
class CudaSolverGlobalRichardsonBase
{
private:
    ComputationBox<double>* cb;  ///< Computation box
    Molecules* molecules;         ///< Molecules container

    int dim;                      ///< Dimensionality (1, 2, or 3)
    int n_streams;                ///< Number of parallel streams

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy]

    /// @name Full step tridiagonal coefficients (ds)
    /// @{
    std::map<std::string, double*> d_xl_full, d_xd_full, d_xh_full;
    std::map<std::string, double*> d_yl_full, d_yd_full, d_yh_full;
    std::map<std::string, double*> d_zl_full, d_zd_full, d_zh_full;
    /// @}

    /// @name Half step tridiagonal coefficients (ds/2)
    /// @{
    std::map<std::string, double*> d_xl_half, d_xd_half, d_xh_half;
    std::map<std::string, double*> d_yl_half, d_yd_half, d_yh_half;
    std::map<std::string, double*> d_zl_half, d_zd_half, d_zh_half;
    /// @}

    /// @name Boltzmann factors on device
    /// @{
    std::map<std::string, double*> d_exp_dw_full;  ///< exp(-w*ds/2)
    std::map<std::string, double*> d_exp_dw_half;  ///< exp(-w*ds/4)
    /// @}

    /// @name Tridiagonal solver workspace (per stream)
    /// @{
    double* d_q_star[MAX_STREAMS];    ///< First intermediate solution
    double* d_q_dstar[MAX_STREAMS];   ///< Second intermediate (3D)
    double* d_c_star[MAX_STREAMS];    ///< Modified upper diagonal
    double* d_q_sparse[MAX_STREAMS];  ///< Sherman-Morrison correction
    double* d_temp[MAX_STREAMS];      ///< General temporary
    /// @}

    /// @name Index offset arrays
    /// @{
    int* d_offset_xy;  ///< 3D: XY-plane offsets
    int* d_offset_yz;  ///< 3D: YZ-plane offsets
    int* d_offset_xz;  ///< 3D: XZ-plane offsets
    int* d_offset_x;   ///< 2D: X-direction offsets
    int* d_offset_y;   ///< 2D: Y-direction offsets
    int* d_offset;     ///< 1D: System offsets
    /// @}

    /// @name ADI step implementations
    /// @{
    void advance_propagator_3d_step(
        std::vector<BoundaryCondition> bc,
        const int STREAM,
        double* d_q_in, double* d_q_out,
        double* _d_xl, double* _d_xd, double* _d_xh,
        double* _d_yl, double* _d_yd, double* _d_yh,
        double* _d_zl, double* _d_zd, double* _d_zh);

    void advance_propagator_2d_step(
        std::vector<BoundaryCondition> bc,
        const int STREAM,
        double* d_q_in, double* d_q_out,
        double* _d_xl, double* _d_xd, double* _d_xh,
        double* _d_yl, double* _d_yd, double* _d_yh);

    void advance_propagator_1d_step(
        std::vector<BoundaryCondition> bc,
        const int STREAM,
        double* d_q_in, double* d_q_out,
        double* _d_xl, double* _d_xd, double* _d_xh);
    /// @}

public:
    /**
     * @brief Construct base solver for Global Richardson on GPU.
     *
     * @param cb        Computation box
     * @param molecules Molecules container
     * @param n_streams Number of parallel CUDA streams
     * @param streams   Pre-created CUDA streams
     */
    CudaSolverGlobalRichardsonBase(
        ComputationBox<double>* cb,
        Molecules* molecules,
        int n_streams,
        cudaStream_t streams[MAX_STREAMS][2]);

    ~CudaSolverGlobalRichardsonBase();

    /**
     * @brief Update tridiagonal coefficients for new box dimensions.
     */
    void update_laplacian_operator();

    /**
     * @brief Update Boltzmann factors from potential fields.
     *
     * @param device   "host" or "device" indicating where w_input resides
     * @param w_input  Map of potential fields by monomer type
     */
    void update_dw(std::string device, std::map<std::string, const double*> w_input);

    /**
     * @brief Advance propagator by one full step (ds).
     *
     * Uses CN-ADI2 with step size ds.
     *
     * @param STREAM      CUDA stream index
     * @param d_q_in      Input propagator (device)
     * @param d_q_out     Output propagator (device)
     * @param monomer_type Monomer type
     * @param d_q_mask    Optional mask for impenetrable regions (device)
     */
    void advance_full_step(
        const int STREAM,
        double* d_q_in, double* d_q_out,
        std::string monomer_type, double* d_q_mask = nullptr);

    /**
     * @brief Advance propagator by one half step (ds/2).
     *
     * Uses CN-ADI2 with step size ds/2.
     *
     * @param STREAM      CUDA stream index
     * @param d_q_in      Input propagator (device)
     * @param d_q_out     Output propagator (device)
     * @param monomer_type Monomer type
     * @param d_q_mask    Optional mask for impenetrable regions (device)
     */
    void advance_half_step(
        const int STREAM,
        double* d_q_in, double* d_q_out,
        std::string monomer_type, double* d_q_mask = nullptr);

    /**
     * @brief Get CUDA stream for a given index.
     */
    cudaStream_t get_stream(int STREAM, int idx) { return streams[STREAM][idx]; }
};

#endif
