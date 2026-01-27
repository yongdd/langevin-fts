/**
 * @file CudaSolverCNADI.h
 * @brief GPU solver using CN-ADI finite difference.
 *
 * This header provides CudaSolverCNADI, the CUDA implementation
 * of the CN-ADI (Crank-Nicolson Alternating Direction Implicit) finite
 * difference method for propagator computation. Uses parallel tridiagonal
 * solves on GPU.
 *
 * **CN-ADI Method:**
 *
 * Implicit time-stepping with 2nd-order accuracy (CN-ADI2, default):
 *     (I - ds/2 L) q(n+1) = (I + ds/2 L) q(n)
 *
 * With Richardson extrapolation enabled, achieves 4th-order (CN-ADI4).
 *
 * **ADI Splitting:**
 *
 * Multi-dimensional problems split into 1D tridiagonal systems:
 * - 2D: X-sweep → Y-sweep
 * - 3D: X-sweep → Y-sweep → Z-sweep
 *
 * **GPU Parallelization:**
 *
 * Each tridiagonal system solved independently in parallel:
 * - 2D: ny systems for X-sweep, nx systems for Y-sweep
 * - 3D: ny×nz, nx×nz, nx×ny systems respectively
 *
 * **Boundary Conditions:**
 *
 * Supports non-periodic boundaries (reflecting, absorbing)
 * which are not available in pseudo-spectral methods.
 *
 * @see CudaSolver for the abstract interface
 * @see CpuSolverCNADI for CPU version
 * @see FiniteDifference for coefficient computation
 */

#ifndef CUDA_SOLVER_CNADI_H_
#define CUDA_SOLVER_CNADI_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "FiniteDifference.h"
#include "CudaSolver.h"
#include "CudaCommon.h"
#include "SpaceGroup.h"

/**
 * @brief Device function: max of two integers.
 */
__device__ int d_max_of_two(int x, int y);

/**
 * @brief Device function: min of two integers.
 */
__device__ int d_min_of_two(int x, int y);

/**
 * @brief CUDA kernel: 1D Crank-Nicolson RHS computation.
 *
 * Computes the right-hand side (I + ds/2 Lx) q for 1D problems.
 *
 * @param bc_xl  Boundary condition at x-low
 * @param bc_xh  Boundary condition at x-high
 * @param d_xl   Lower diagonal coefficients
 * @param d_xd   Main diagonal coefficients
 * @param d_xh   Upper diagonal coefficients
 * @param d_q_out Output RHS vector
 * @param d_q_in  Input propagator
 * @param M       Grid size
 */
__global__ void compute_crank_1d(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    const double *d_xl, const double *d_xd, const double *d_xh,
    double *d_q_out, const double *d_q_in, const int M);

/**
 * @brief CUDA kernel: 2D ADI step 1 (X-direction).
 *
 * Computes RHS for X-sweep: (I + ds/2 Lx) q
 *
 * @param bc_xl, bc_xh  X-direction boundary conditions
 * @param bc_yl, bc_yh  Y-direction boundary conditions
 * @param d_xl, d_xd, d_xh  X-direction tridiagonal coefficients
 * @param I             X grid size
 * @param d_yl, d_yd, d_yh  Y-direction tridiagonal coefficients
 * @param J             Y grid size
 * @param d_q_out       Output RHS
 * @param d_q_in        Input propagator
 * @param M             Total grid size
 */
__global__ void compute_crank_2d_step_1(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_xl, const double *d_xd, const double *d_xh, const int I,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    double *d_q_out, const double *d_q_in, const int M);

/**
 * @brief CUDA kernel: 2D ADI step 2 (Y-direction).
 *
 * Computes RHS for Y-sweep using intermediate solution q*.
 *
 * @param bc_yl, bc_yh  Y-direction boundary conditions
 * @param d_yl, d_yd, d_yh  Y-direction tridiagonal coefficients
 * @param J             Y grid size
 * @param d_q_out       Output RHS
 * @param d_q_star      Intermediate solution from step 1
 * @param d_q_in        Original input propagator
 * @param M             Total grid size
 */
__global__ void compute_crank_2d_step_2(
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    double *d_q_out, const double *d_q_star, const double *d_q_in, const int M);

/**
 * @brief CUDA kernel: 3D ADI step 1 (X-direction).
 *
 * First ADI sweep in X-direction for 3D problems.
 *
 * @param bc_xl, bc_xh  X-direction boundary conditions
 * @param bc_yl, bc_yh  Y-direction boundary conditions
 * @param bc_zl, bc_zh  Z-direction boundary conditions
 * @param d_xl, d_xd, d_xh  X-direction coefficients
 * @param I             X grid size
 * @param d_yl, d_yd, d_yh  Y-direction coefficients
 * @param J             Y grid size
 * @param d_zl, d_zd, d_zh  Z-direction coefficients
 * @param K             Z grid size
 * @param d_q_out       Output RHS
 * @param d_q_in        Input propagator
 * @param M             Total grid size
 */
__global__ void compute_crank_3d_step_1(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    const double *d_xl, const double *d_xd, const double *d_xh, const int I,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    const double *d_zl, const double *d_zd, const double *d_zh, const int K,
    double *d_q_out, const double *d_q_in, const int M);

/**
 * @brief CUDA kernel: 3D ADI step 2 (Y-direction).
 *
 * Second ADI sweep in Y-direction using intermediate q*.
 *
 * @param bc_yl, bc_yh  Y-direction boundary conditions
 * @param d_yl, d_yd, d_yh  Y-direction coefficients
 * @param J, K          Y and Z grid sizes
 * @param d_q_out       Output RHS
 * @param d_q_star      Intermediate from step 1
 * @param d_q_in        Original input
 * @param M             Total grid size
 */
__global__ void compute_crank_3d_step_2(
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J, const int K,
    double *d_q_out, const double *d_q_star, const double *d_q_in, const int M);

/**
 * @brief CUDA kernel: 3D ADI step 3 (Z-direction).
 *
 * Final ADI sweep in Z-direction using intermediate q**.
 *
 * @param bc_zl, bc_zh  Z-direction boundary conditions
 * @param d_zl, d_zd, d_zh  Z-direction coefficients
 * @param J, K          Y and Z grid sizes
 * @param d_q_out       Output RHS
 * @param d_q_dstar     Intermediate from step 2
 * @param d_q_in        Original input
 * @param M             Total grid size
 */
__global__ void compute_crank_3d_step_3(
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    const double *d_zl, const double *d_zd, const double *d_zh, const int J, const int K,
    double *d_q_out, const double *d_q_dstar, const double *d_q_in, const int M);

/**
 * @brief CUDA kernel: Parallel tridiagonal solver (Thomas algorithm).
 *
 * Solves multiple independent tridiagonal systems in parallel.
 * Uses shared memory for high performance.
 *
 * @param d_xl      Lower diagonal (subdiagonal)
 * @param d_xd      Main diagonal
 * @param d_xh      Upper diagonal (superdiagonal)
 * @param d_c_star  Workspace for modified upper diagonal
 * @param d_d       Right-hand side vector
 * @param d_x       Solution output
 * @param d_offset  Starting offset for each system
 * @param REPEAT    Number of systems to solve
 * @param INTERVAL  Stride between systems
 * @param M         System size
 */
__global__ void tridiagonal(
    const double* __restrict__ d_xl,
    const double* __restrict__ d_xd,
    const double* __restrict__ d_xh,
    double* __restrict__ d_c_star,
    const double* __restrict__ d_d,
    double* __restrict__ d_x,
    const int* __restrict__ d_offset,
    const int REPEAT, const int INTERVAL, const int M);

/**
 * @brief CUDA kernel: Parallel tridiagonal solver for periodic boundaries.
 *
 * Handles cyclic tridiagonal systems using Sherman-Morrison formula.
 * More complex than non-periodic due to corner coupling.
 *
 * @param d_xl       Lower diagonal
 * @param d_xd       Main diagonal
 * @param d_xh       Upper diagonal
 * @param d_c_star   Workspace for modified coefficients
 * @param d_q_sparse Workspace for Sherman-Morrison correction
 * @param d_d        Right-hand side
 * @param d_x        Solution output
 * @param d_offset   Starting offsets
 * @param REPEAT     Number of systems
 * @param INTERVAL   Stride
 * @param M          System size
 */
__global__ void tridiagonal_periodic(
    const double* __restrict__ d_xl,
    const double* __restrict__ d_xd,
    const double* __restrict__ d_xh,
    double* __restrict__ d_c_star,
    double* __restrict__ d_q_sparse,
    const double* __restrict__ d_d,
    double* __restrict__ d_x,
    const int* __restrict__ d_offset,
    const int REPEAT, const int INTERVAL, const int M);

/**
 * @class CudaSolverCNADI
 * @brief GPU solver using CN-ADI (Crank-Nicolson ADI) finite difference.
 *
 * Implements propagator advancement using ADI (Alternating Direction Implicit)
 * method with parallel tridiagonal solves on GPU. Supports non-periodic
 * boundary conditions.
 *
 * **Tridiagonal Systems:**
 *
 * For each direction (x, y, z), stores coefficients:
 * - d_xl: Lower diagonal (subdiagonal)
 * - d_xd: Main diagonal
 * - d_xh: Upper diagonal (superdiagonal)
 *
 * **Workspace Arrays:**
 *
 * Per-stream workspace for tridiagonal solves:
 * - d_q_star: Intermediate ADI solution (after first sweep)
 * - d_q_dstar: Second intermediate (3D only)
 * - d_c_star: Modified coefficients for Thomas algorithm
 * - d_q_sparse: Sherman-Morrison workspace (periodic BCs)
 *
 * **Offset Arrays:**
 *
 * Index mappings for parallel system solving:
 * - d_offset_xy, d_offset_yz, d_offset_xz: 3D sweeps
 * - d_offset_x, d_offset_y: 2D sweeps
 *
 * @note Only supports double type (not complex) as CN-ADI method
 *       is primarily for continuous chains with non-periodic BCs.
 */
class CudaSolverCNADI : public CudaSolver<double>
{
private:
    ComputationBox<double>* cb;  ///< Computation box
    Molecules *molecules;        ///< Molecules container

    int dim;                     ///< Dimensionality (1, 2, or 3)

    std::string chain_model;     ///< Chain model type
    bool reduce_memory;      ///< Checkpointing mode flag
    bool use_4th_order;          ///< Use CN-ADI4 (4th order) instead of CN-ADI2 (2nd order)

    int n_streams;               ///< Number of parallel streams

    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams [kernel, memcpy]

    /// @name Space group support (reduced basis)
    /// @{
    SpaceGroup* space_group_;           ///< Space group pointer (nullptr if not used)
    int* d_reduced_basis_indices_;      ///< Device array: reduced → full index mapping
    int* d_full_to_reduced_map_;        ///< Device array: full → reduced index mapping
    int n_basis_;                       ///< Number of reduced basis points
    double *d_q_full_in_[MAX_STREAMS];  ///< Work buffer: full grid input (per stream)
    double *d_q_full_out_[MAX_STREAMS]; ///< Work buffer: full grid output (per stream)
    /// @}

    /// @name Tridiagonal Coefficients per ds_index per monomer_type
    /// Nested map structure: [ds_index][monomer_type] -> coefficient array (device)
    /// @{
    std::map<int, std::map<std::string, double*>> d_xl;  ///< X-direction lower diagonal
    std::map<int, std::map<std::string, double*>> d_xd;  ///< X-direction main diagonal
    std::map<int, std::map<std::string, double*>> d_xh;  ///< X-direction upper diagonal

    std::map<int, std::map<std::string, double*>> d_yl;  ///< Y-direction lower diagonal
    std::map<int, std::map<std::string, double*>> d_yd;  ///< Y-direction main diagonal
    std::map<int, std::map<std::string, double*>> d_yh;  ///< Y-direction upper diagonal

    std::map<int, std::map<std::string, double*>> d_zl;  ///< Z-direction lower diagonal
    std::map<int, std::map<std::string, double*>> d_zd;  ///< Z-direction main diagonal
    std::map<int, std::map<std::string, double*>> d_zh;  ///< Z-direction upper diagonal
    /// @}

    /// @name Half-step Tridiagonal Coefficients for CN-ADI4
    /// @{
    std::map<int, std::map<std::string, double*>> d_xl_half;  ///< X-direction lower (ds/2)
    std::map<int, std::map<std::string, double*>> d_xd_half;  ///< X-direction diagonal (ds/2)
    std::map<int, std::map<std::string, double*>> d_xh_half;  ///< X-direction upper (ds/2)

    std::map<int, std::map<std::string, double*>> d_yl_half;  ///< Y-direction lower (ds/2)
    std::map<int, std::map<std::string, double*>> d_yd_half;  ///< Y-direction diagonal (ds/2)
    std::map<int, std::map<std::string, double*>> d_yh_half;  ///< Y-direction upper (ds/2)

    std::map<int, std::map<std::string, double*>> d_zl_half;  ///< Z-direction lower (ds/2)
    std::map<int, std::map<std::string, double*>> d_zd_half;  ///< Z-direction diagonal (ds/2)
    std::map<int, std::map<std::string, double*>> d_zh_half;  ///< Z-direction upper (ds/2)
    /// @}

    /// @name Tridiagonal Solver Workspace (per stream)
    /// @{
    double *d_q_star  [MAX_STREAMS];  ///< First intermediate solution
    double *d_q_dstar [MAX_STREAMS];  ///< Second intermediate (3D)
    double *d_c_star  [MAX_STREAMS];  ///< Modified upper diagonal
    double *d_q_sparse[MAX_STREAMS];  ///< Sherman-Morrison correction
    double *d_temp    [MAX_STREAMS];  ///< General temporary
    double *d_q_full  [MAX_STREAMS];  ///< Full step result for CN-ADI4
    double *d_q_half  [MAX_STREAMS];  ///< Half step result for CN-ADI4
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
     * @brief 3D ADI propagator advancement.
     * @param ds_index Index for ds value (1-based) from ContourLengthMapping
     */
    void advance_propagator_3d(
        std::vector<BoundaryCondition> bc,
        const int STREAM,
        double *d_q_in, double *d_q_out, std::string monomer_type, int ds_index);

    /**
     * @brief 2D ADI propagator advancement.
     * @param ds_index Index for ds value (1-based) from ContourLengthMapping
     */
    void advance_propagator_2d(
        std::vector<BoundaryCondition> bc,
        const int STREAM,
        double *d_q_in, double *d_q_out, std::string monomer_type, int ds_index);

    /**
     * @brief 1D Crank-Nicolson propagator advancement.
     * @param ds_index Index for ds value (1-based) from ContourLengthMapping
     */
    void advance_propagator_1d(
        std::vector<BoundaryCondition> bc,
        const int STREAM,
        double *d_q_in, double *d_q_out, std::string monomer_type, int ds_index);

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

public:
    /**
     * @brief Construct GPU CN-ADI solver.
     *
     * @param cb                  Computation box
     * @param molecules           Molecules container
     * @param n_streams           Number of parallel streams
     * @param streams             Pre-created CUDA streams
     * @param reduce_memory   Checkpointing mode
     * @param use_4th_order       Use CN-ADI4 (4th order accuracy via Richardson
     *                            extrapolation) instead of CN-ADI2 (2nd order, default)
     */
    CudaSolverCNADI(ComputationBox<double>* cb, Molecules *molecules,
        int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool reduce_memory, bool use_4th_order = false);

    /**
     * @brief Destructor. Frees GPU resources.
     */
    ~CudaSolverCNADI();

    /**
     * @brief Set space group for reduced basis operations.
     *
     * When set, input/output are in reduced basis and the solver expands/reduces
     * internally around finite-difference operations.
     */
    void set_space_group(
        SpaceGroup* sg,
        int* d_reduced_basis_indices,
        int* d_full_to_reduced_map,
        int n_basis) override;

    /** @brief Update finite difference coefficients for new box. */
    void update_laplacian_operator() override;

    /** @brief Update Boltzmann factors from potential fields. */
    void update_dw(std::string device, std::map<std::string, const double*> w_input) override;

    /**
     * @brief Advance propagator by one contour step.
     *
     * Uses ADI method: X-sweep → Y-sweep → Z-sweep (for 3D).
     *
     * @param STREAM       Stream index
     * @param d_q_in       Input propagator (device)
     * @param d_q_out      Output propagator (device)
     * @param monomer_type Monomer type
     * @param d_q_mask     Optional mask (device)
     * @param ds_index     Index for ds value (1-based) from ContourLengthMapping
     */
    void advance_propagator(
        const int STREAM,
        double *d_q_in, double *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index) override;

    /** @brief Half-bond step (not applicable for CN-ADI continuous). */
    void advance_propagator_half_bond_step(
        const int, double *, double *, std::string) override {};

    /**
     * @brief Compute stress contribution from one segment.
     *
     * @param STREAM            Stream index
     * @param d_q_pair          Propagator product (device)
     * @param d_segment_stress  Output stress (device)
     * @param monomer_type      Monomer type
     * @param is_half_bond_length Ignored for CN-ADI
     */
    void compute_single_segment_stress(
        const int STREAM,
        double *d_q_pair, double *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) override;
};
#endif
