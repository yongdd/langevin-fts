/**
 * @file CudaSolverRichardsonGlobal.h
 * @brief GPU solver using Global Richardson extrapolation.
 *
 * This header provides CudaSolverRichardsonGlobal, the CUDA implementation
 * of the Global Richardson extrapolation method for propagator computation.
 *
 * **Algorithm (Global Richardson):**
 *
 * Two independent propagator evolutions are maintained per stream:
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
 * 1. Call reset_internal_state(STREAM) before computing a new propagator
 * 2. First advance_propagator() call initializes internal states from d_q_in
 * 3. Subsequent calls use internal states (ignoring d_q_in)
 *
 * @see CudaSolver for the abstract interface
 * @see CudaSolverCNADI for per-step Richardson (CN-ADI4)
 */

#ifndef CUDA_SOLVER_RICHARDSON_GLOBAL_H_
#define CUDA_SOLVER_RICHARDSON_GLOBAL_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "FiniteDifference.h"
#include "CudaSolver.h"
#include "CudaCommon.h"

/**
 * @class CudaSolverRichardsonGlobal
 * @brief GPU solver using Global Richardson extrapolation.
 *
 * Implements Richardson extrapolation with two independent propagator evolutions.
 * This provides 4th-order accuracy while maintaining stability for challenging
 * initial conditions (e.g., delta functions for grafted polymers).
 *
 * **Internal State (per stream):**
 *
 * Each CUDA stream maintains two independent propagator states:
 * - d_q_full_internal[STREAM]: Evolved with full step size (ds)
 * - d_q_half_internal[STREAM]: Evolved with half step size (ds/2, twice per step)
 *
 * These states are initialized from d_q_in on the first call after reset.
 *
 * **Computational Cost:**
 *
 * Per advance_propagator() call:
 * - 1 full step (ds) + 2 half steps (ds/2) = 3 ADI solves
 * - Same cost as CN-ADI4, but different algorithm
 */
class CudaSolverRichardsonGlobal : public CudaSolver<double>
{
private:
    ComputationBox<double>* cb;  ///< Computation box
    Molecules *molecules;        ///< Molecules container

    int dim;                     ///< Dimensionality (1, 2, or 3)

    std::string chain_model;     ///< Chain model type
    bool reduce_memory_usage;    ///< Memory-saving mode flag

    int n_streams;               ///< Number of parallel streams

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy]

    /// @name X-direction Tridiagonal Coefficients (full step)
    /// @{
    std::map<std::string, double*> d_xl;  ///< Lower diagonal by monomer type
    std::map<std::string, double*> d_xd;  ///< Main diagonal by monomer type
    std::map<std::string, double*> d_xh;  ///< Upper diagonal by monomer type
    /// @}

    /// @name Y-direction Tridiagonal Coefficients (full step)
    /// @{
    std::map<std::string, double*> d_yl;  ///< Lower diagonal
    std::map<std::string, double*> d_yd;  ///< Main diagonal
    std::map<std::string, double*> d_yh;  ///< Upper diagonal
    /// @}

    /// @name Z-direction Tridiagonal Coefficients (full step)
    /// @{
    std::map<std::string, double*> d_zl;  ///< Lower diagonal
    std::map<std::string, double*> d_zd;  ///< Main diagonal
    std::map<std::string, double*> d_zh;  ///< Upper diagonal
    /// @}

    /// @name Half-step Tridiagonal Coefficients
    /// @{
    std::map<std::string, double*> d_xl_half;  ///< X-direction lower (ds/2)
    std::map<std::string, double*> d_xd_half;  ///< X-direction diagonal (ds/2)
    std::map<std::string, double*> d_xh_half;  ///< X-direction upper (ds/2)

    std::map<std::string, double*> d_yl_half;  ///< Y-direction lower (ds/2)
    std::map<std::string, double*> d_yd_half;  ///< Y-direction diagonal (ds/2)
    std::map<std::string, double*> d_yh_half;  ///< Y-direction upper (ds/2)

    std::map<std::string, double*> d_zl_half;  ///< Z-direction lower (ds/2)
    std::map<std::string, double*> d_zd_half;  ///< Z-direction diagonal (ds/2)
    std::map<std::string, double*> d_zh_half;  ///< Z-direction upper (ds/2)
    /// @}

    /// @name Tridiagonal Solver Workspace (per stream)
    /// @{
    double *d_q_star  [MAX_STREAMS];  ///< First intermediate solution
    double *d_q_dstar [MAX_STREAMS];  ///< Second intermediate (3D)
    double *d_c_star  [MAX_STREAMS];  ///< Modified upper diagonal
    double *d_q_sparse[MAX_STREAMS];  ///< Sherman-Morrison correction
    double *d_temp    [MAX_STREAMS];  ///< General temporary
    /// @}

    /// @name Internal Propagator States for Global Richardson (per stream)
    /// @{
    double *d_q_full_internal[MAX_STREAMS];  ///< Full-step propagator state
    double *d_q_half_internal[MAX_STREAMS];  ///< Half-step propagator state
    double *d_q_work[MAX_STREAMS];           ///< Work array for intermediate results
    bool is_initialized[MAX_STREAMS];        ///< Whether internal states have been initialized
    /// @}

    /// @name Index Offset Arrays
    /// @{
    int* d_offset_xy;  ///< 3D: XY-plane offsets
    int* d_offset_yz;  ///< 3D: YZ-plane offsets
    int* d_offset_xz;  ///< 3D: XZ-plane offsets
    int* d_offset_x;   ///< 2D: X-direction offsets
    int* d_offset_y;   ///< 2D: Y-direction offsets
    int* d_offset;     ///< 1D: System offsets
    /// @}

    /**
     * @brief 3D ADI step with explicit coefficient arrays.
     */
    void advance_propagator_3d_step(
        std::vector<BoundaryCondition> bc,
        const int STREAM,
        double *d_q_in, double *d_q_out,
        double *_d_xl, double *_d_xd, double *_d_xh,
        double *_d_yl, double *_d_yd, double *_d_yh,
        double *_d_zl, double *_d_zd, double *_d_zh);

    /**
     * @brief 2D ADI step with explicit coefficient arrays.
     */
    void advance_propagator_2d_step(
        std::vector<BoundaryCondition> bc,
        const int STREAM,
        double *d_q_in, double *d_q_out,
        double *_d_xl, double *_d_xd, double *_d_xh,
        double *_d_yl, double *_d_yd, double *_d_yh);

    /**
     * @brief 1D step with explicit coefficient arrays.
     */
    void advance_propagator_1d_step(
        std::vector<BoundaryCondition> bc,
        const int STREAM,
        double *d_q_in, double *d_q_out,
        double *_d_xl, double *_d_xd, double *_d_xh);

    /**
     * @brief Advance propagator by one full step (ds).
     */
    void advance_full_step(const int STREAM, double* d_q_in, double* d_q_out, std::string monomer_type);

    /**
     * @brief Advance propagator by two half steps (ds/2 each).
     */
    void advance_two_half_steps(const int STREAM, double* d_q_in, double* d_q_out, std::string monomer_type);

public:
    /**
     * @brief Construct GPU Global Richardson solver.
     *
     * @param cb                  Computation box
     * @param molecules           Molecules container
     * @param n_streams           Number of parallel streams
     * @param streams             Pre-created CUDA streams
     * @param reduce_memory_usage Memory-saving mode
     */
    CudaSolverRichardsonGlobal(ComputationBox<double>* cb, Molecules *molecules,
        int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool reduce_memory_usage);

    /**
     * @brief Destructor. Frees GPU resources.
     */
    ~CudaSolverRichardsonGlobal();

    /** @brief Update finite difference coefficients for new box. */
    void update_laplacian_operator() override;

    /** @brief Update Boltzmann factors from potential fields. */
    void update_dw(std::string device, std::map<std::string, const double*> w_input) override;

    /**
     * @brief Reset internal propagator states for a stream.
     *
     * Must be called before starting a new propagator computation.
     * The next advance_propagator() call will initialize internal
     * states from d_q_in.
     *
     * @param STREAM Stream index to reset
     */
    void reset_internal_state(int STREAM);

    /**
     * @brief Advance propagator by one contour step using Global Richardson.
     *
     * On first call after reset:
     * - Initializes d_q_full_internal and d_q_half_internal from d_q_in
     *
     * On all calls:
     * - Advances d_q_full_internal by one ds step (using its own state)
     * - Advances d_q_half_internal by two ds/2 steps (using its own state)
     * - Returns Richardson-extrapolated result: (4*q_half - q_full) / 3
     *
     * @param STREAM       Stream index
     * @param d_q_in       Input propagator (only used on first call after reset)
     * @param d_q_out      Output propagator (Richardson-extrapolated)
     * @param monomer_type Monomer type
     * @param d_q_mask     Optional mask (device)
     * @param ds_index     Index for ds value (ignored, uses global ds)
     */
    void advance_propagator(
        const int STREAM,
        double *d_q_in, double *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index = 1) override;

    /** @brief Half-bond step (not applicable for continuous). */
    void advance_propagator_half_bond_step(
        const int, double *, double *, std::string) override {};

    /**
     * @brief Compute stress contribution from one segment.
     *
     * @warning Not yet implemented for Global Richardson method.
     */
    void compute_single_segment_stress(
        const int STREAM,
        double *d_q_pair, double *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) override;
};
#endif
