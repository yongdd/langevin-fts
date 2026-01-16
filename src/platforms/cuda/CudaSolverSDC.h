/**
 * @file CudaSolverSDC.h
 * @brief GPU solver using SDC (Spectral Deferred Correction) method.
 *
 * This header provides CudaSolverSDC, which implements the Spectral Deferred
 * Correction method using Gauss-Lobatto quadrature nodes for solving the
 * modified diffusion equation on GPU.
 *
 * **Numerical Method:**
 *
 * SDC iteratively improves the solution using spectral quadrature:
 *
 *     q(t_{n+1}) = q(t_n) + ∫_{t_n}^{t_{n+1}} F(q) dt
 *
 * Starting from a low-order predictor, SDC applies K correction iterations
 * using M Gauss-Lobatto collocation nodes per contour step.
 *
 * **GPU Parallelization:**
 *
 * - 1D: Parallel tridiagonal solvers
 * - 2D/3D: PCG sparse solver with GPU kernels
 * - F computation uses element-wise GPU kernels
 * - Spectral integration uses parallel reductions
 *
 * **Configuration:**
 *
 * - M: Number of Gauss-Lobatto nodes (default: 4)
 * - K: Number of SDC correction iterations (default: 5)
 *
 * **Order of Accuracy:**
 *
 * The SDC order is min(K+1, 2M-2):
 * - M=3, K=2: order 3
 * - M=4, K=5: order 6 (default)
 * - M=5, K=7: order 8
 *
 * With PCG solver (no splitting error), high-order accuracy is achieved in all dimensions.
 *
 * **Limitations:**
 *
 * - Stress computation is not yet implemented for SDC method
 * - Only supports continuous chain model
 *
 * @see CudaSolver for the abstract interface
 * @see CudaSolverCNADI for tridiagonal solvers (1D only)
 */

#ifndef CUDA_SOLVER_SDC_H_
#define CUDA_SOLVER_SDC_H_

#include <string>
#include <vector>
#include <map>
#include <complex>

#include <cuda_runtime.h>
#include <cufft.h>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CudaSolver.h"
#include "CudaCommon.h"

/**
 * @struct CudaSparseMatrixCSR
 * @brief CSR sparse matrix for PCG solver with Jacobi preconditioner.
 */
struct CudaSparseMatrixCSR
{
    int* d_row_ptr;          ///< Row pointers (device, size n+1)
    int* d_col_idx;          ///< Column indices (device, size nnz)
    double* d_values;        ///< Matrix values (device, size nnz)
    double* d_diag_inv;      ///< Inverse diagonal for Jacobi preconditioner (device, size n)
    int n;                   ///< Matrix dimension
    int nnz;                 ///< Number of non-zeros
    bool built;              ///< Whether matrix is built
};

/**
 * @brief CUDA kernel: Compute F(q) = D∇²q - wq for SDC.
 *
 * @param d_q Input propagator (device)
 * @param d_w Potential field (device)
 * @param d_F Output F(q) (device)
 * @param alpha_x Diffusion coefficient for x-direction
 * @param alpha_y Diffusion coefficient for y-direction
 * @param alpha_z Diffusion coefficient for z-direction
 * @param nx_I X grid size
 * @param nx_J Y grid size
 * @param nx_K Z grid size
 * @param bc Boundary conditions array
 * @param M Total grid size
 */
__global__ void compute_F_kernel_3d(
    const double* d_q, const double* d_w, double* d_F,
    double alpha_x, double alpha_y, double alpha_z,
    int nx_I, int nx_J, int nx_K,
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    int M);

__global__ void compute_F_kernel_2d(
    const double* d_q, const double* d_w, double* d_F,
    double alpha_x, double alpha_y,
    int nx_I, int nx_J,
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    int M);

__global__ void compute_F_kernel_1d(
    const double* d_q, const double* d_w, double* d_F,
    double alpha_x, int nx_I,
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    int M);

/**
 * @brief CUDA kernel: Apply SDC spectral integral.
 *
 * Computes: rhs[i] = X[m][i] + ∫ F dt - dtau * F_old[m+1][i]
 */
__global__ void sdc_spectral_integral_kernel(
    double* d_rhs,
    const double* d_X_m,
    const double** d_F_old,  // Array of pointers to F at each GL node
    const double* d_S_row,   // Integration matrix row S[m][:]
    int M_nodes,             // Number of GL nodes
    double ds,
    double dtau,
    int m_plus_1,            // Index m+1 for F_old[m+1]
    int n_grid);

/**
 * @brief CUDA kernel: Apply Boltzmann factor.
 */
__global__ void apply_exp_dw_kernel(
    double* d_out, const double* d_in, const double* d_exp_dw, int n_grid);

/**
 * @brief CUDA kernel: Copy array.
 */
__global__ void copy_array_kernel(double* d_out, const double* d_in, int n_grid);

/**
 * @class CudaSolverSDC
 * @brief GPU solver using SDC (Spectral Deferred Correction) with Gauss-Lobatto nodes.
 *
 * Implements the SDC scheme for solving the modified diffusion equation on GPU.
 * Uses tridiagonal solvers (1D) or PCG sparse solver (2D/3D) for implicit diffusion.
 *
 * **Algorithm per contour step:**
 *
 * 1. Predictor: Backward Euler at each sub-interval
 *    - 1D: Parallel tridiagonal solve (exact)
 *    - 2D/3D: PCG sparse solve (no splitting error)
 * 2. Corrections (K iterations):
 *    - Compute F = D∇²q - wq at all GL nodes
 *    - Apply SDC update using spectral integration matrix S
 */
class CudaSolverSDC : public CudaSolver<double>
{
private:
    ComputationBox<double>* cb;   ///< Computation box
    Molecules *molecules;          ///< Molecules container

    int dim;                       ///< Dimensionality (1, 2, or 3)
    int M;                         ///< Number of Gauss-Lobatto nodes
    int K;                         ///< Number of SDC correction iterations

    int n_streams;                 ///< Number of parallel streams
    cudaStream_t streams[MAX_STREAMS][2];  ///< CUDA streams

    std::vector<double> tau;       ///< Gauss-Lobatto nodes on [0, 1]
    std::vector<std::vector<double>> S;  ///< Spectral integration matrix (host)
    double* d_S;                   ///< Integration matrix on device (flattened)

    // Tridiagonal coefficients for each sub-interval (device memory)
    std::vector<std::map<std::string, double*>> d_xl;
    std::vector<std::map<std::string, double*>> d_xd;
    std::vector<std::map<std::string, double*>> d_xd_base;  ///< Base diagonal (diffusion only, for 1D)
    std::vector<std::map<std::string, double*>> d_xh;

    std::vector<std::map<std::string, double*>> d_yl;
    std::vector<std::map<std::string, double*>> d_yd;
    std::vector<std::map<std::string, double*>> d_yh;

    std::vector<std::map<std::string, double*>> d_zl;
    std::vector<std::map<std::string, double*>> d_zd;
    std::vector<std::map<std::string, double*>> d_zh;

    // Sub-interval time steps (stored for use in update_dw)
    std::vector<double> dtau_sub;

    // Potential field storage (device memory)
    std::map<std::string, double*> d_w_field;

    // Per-stream workspace arrays (device memory)
    std::vector<double*> d_X[MAX_STREAMS];       ///< Solution at GL nodes
    std::vector<double*> d_F[MAX_STREAMS];       ///< F(q) at GL nodes
    std::vector<double*> d_X_old[MAX_STREAMS];   ///< Previous X for correction
    std::vector<double*> d_F_old[MAX_STREAMS];   ///< Previous F for correction
    double* d_temp[MAX_STREAMS];                  ///< Temporary workspace
    double* d_rhs[MAX_STREAMS];                   ///< RHS for implicit solves

    // Tridiagonal workspace (per stream, 1D only)
    double* d_q_star[MAX_STREAMS];
    double* d_q_dstar[MAX_STREAMS];
    double* d_c_star[MAX_STREAMS];
    double* d_q_sparse[MAX_STREAMS];
    double* d_q_in_saved[MAX_STREAMS];  ///< Saved input for tridiagonal solve

    // Offset arrays for tridiagonal solve (1D only)
    int* d_offset_xy;
    int* d_offset_yz;
    int* d_offset_xz;
    int* d_offset_x;
    int* d_offset_y;
    int* d_offset;

    // Sparse matrices for PCG solve (2D/3D only)
    std::vector<std::map<std::string, CudaSparseMatrixCSR>> sparse_matrices;

    // Matrix-free PCG: diagonal inverse storage [sub_interval][monomer_type]
    std::vector<std::map<std::string, double*>> d_diag_inv_free;
    std::vector<std::map<std::string, bool>> diag_inv_built;

    // IMEX: Diffusion-only diagonal inverse for non-periodic BC
    std::vector<std::map<std::string, double*>> d_diag_inv_diff_only_;
    std::vector<std::map<std::string, bool>> diag_inv_diff_only_built_;

    // PCG workspace arrays (per stream, allocated for 2D/3D)
    double* d_pcg_r[MAX_STREAMS];        ///< Residual vector
    double* d_pcg_z[MAX_STREAMS];        ///< Preconditioned residual
    double* d_pcg_p[MAX_STREAMS];        ///< Search direction
    double* d_pcg_Ap[MAX_STREAMS];       ///< Matrix-vector product A*p
    int pcg_max_iter;                    ///< Maximum PCG iterations
    double pcg_tol;                      ///< PCG convergence tolerance

    // Device-side scalars for PCG (avoid GPU-CPU sync per iteration)
    // Layout: [0]=alpha, [1]=beta, [2]=rz_old, [3]=rz_new, [4]=pAp, [5]=r_norm_sq
    double* d_pcg_scalars[MAX_STREAMS];
    double* d_pcg_partial[MAX_STREAMS];  ///< Partial reduction buffer for fused kernels
    int pcg_n_blocks;                     ///< Number of blocks for PCG reductions

    // PCG optimization parameters
    static constexpr int PCG_FIXED_ITER = 50;           ///< Fixed iterations (increased for larger 3D grids)
    static constexpr int PCG_BLOCK_SIZE = 256;          ///< Block size for PCG kernels
    static constexpr int PCG_CONV_CHECK_INTERVAL = 10;  ///< Check convergence every N iterations (reduces GPU-CPU sync)

    // IMEX SDC: FFT-based diffusion solve for periodic BC
    bool is_periodic_;                                  ///< True if all BCs are periodic (use FFT)
    cufftHandle plan_forward_[MAX_STREAMS];             ///< Forward FFT plans (D2Z)
    cufftHandle plan_backward_[MAX_STREAMS];            ///< Backward FFT plans (Z2D)
    cufftDoubleComplex* d_qk_[MAX_STREAMS];             ///< Fourier space workspace
    int n_complex_;                                      ///< Size of complex array for FFT

    // Diffusion operator inverse in Fourier space: 1/(1 + dtau*D*|k|²)
    // [sub_interval][monomer_type] -> device array
    std::vector<std::map<std::string, double*>> d_diffusion_inv_;

    // Spectral Laplacian operator: -D*|k|² for computing F = D∇²q - wq
    // [monomer_type] -> device array (independent of sub-interval)
    std::map<std::string, double*> d_spectral_laplacian_;

    // Workspace for spectral F computation
    double* d_laplacian_q_[MAX_STREAMS];  ///< D∇²q result from spectral computation

    // IMEX mode flag for faster computation with periodic BC in 2D/3D
    bool imex_mode_enabled_;  ///< When true, use IMEX SDC for periodic BC in 2D/3D

    /**
     * @brief Compute Gauss-Lobatto nodes on [0, 1].
     */
    void compute_gauss_lobatto_nodes();

    /**
     * @brief Compute spectral integration matrix S.
     */
    void compute_integration_matrix();

    /**
     * @brief Compute F(q) = D∇²q - wq on GPU using finite differences.
     */
    void compute_F_device(int STREAM, const double* d_q, const double* d_w,
                          double* d_F, std::string monomer_type);

    /**
     * @brief Compute F(q) = D∇²q - wq using spectral Laplacian (FFT-based).
     *
     * For periodic BC, computes D∇²q in Fourier space:
     * D∇²q = IFFT(-D*|k|² * FFT(q))
     */
    void compute_F_spectral(int STREAM, const double* d_q, const double* d_w,
                            double* d_F, std::string monomer_type);

    /**
     * @brief Compute F_diff = D∇²q (diffusion part only) using spectral Laplacian.
     *
     * Used for Strang splitting where reaction is handled via Boltzmann factors.
     * The diffusion operator is computed spectrally for consistency with FFT-based solve.
     */
    void compute_F_diff_spectral(int STREAM, const double* d_q,
                                  double* d_F_diff, std::string monomer_type);

    /**
     * @brief Compute spectral Laplacian operator -D*|k|² for a monomer type.
     */
    void compute_spectral_laplacian(std::string monomer_type);

    /**
     * @brief Implicit diffusion solve for a specific sub-interval on GPU.
     *
     * Solves (I - dtau * D∇² + dtau*w) q_out = rhs.
     * - 1D: Uses tridiagonal solver (exact)
     * - 2D/3D: Uses PCG sparse solver (no splitting error)
     */
    void implicit_solve_step(int STREAM, int sub_interval,
                             double* d_q_in, double* d_q_out, std::string monomer_type);

    /**
     * @brief Tridiagonal solve in 1D on GPU.
     */
    void tridiagonal_solve_1d(int STREAM, int sub_interval,
        std::vector<BoundaryCondition> bc,
        double* d_q_in, double* d_q_out, std::string monomer_type);

    /**
     * @brief Build sparse Laplacian matrix in CSR format for direct solve.
     *
     * Builds A = I - dtau * D * ∇² for 2D or 3D Laplacian.
     *
     * @param sub_interval Sub-interval index
     * @param monomer_type Monomer type for bond length
     */
    void build_sparse_matrix(int sub_interval, std::string monomer_type);

    /**
     * @brief Sparse matrix-vector product: y = A * x on GPU.
     *
     * Uses CUDA kernel for CSR sparse matrix-vector multiplication.
     *
     * @param mat Sparse matrix in CSR format
     * @param d_x Input vector (device)
     * @param d_y Output vector (device)
     * @param stream CUDA stream for asynchronous execution
     */
    void sparse_matvec(const CudaSparseMatrixCSR& mat, const double* d_x,
                       double* d_y, cudaStream_t stream);

    /**
     * @brief Solve sparse linear system using PCG on GPU.
     *
     * Solves A * q_out = q_in using Preconditioned Conjugate Gradient
     * with Jacobi preconditioner. Matrix A = I - dtau * D * ∇² is SPD.
     *
     * @param STREAM CUDA stream index
     * @param sub_interval Sub-interval index
     * @param d_q_in Input (RHS)
     * @param d_q_out Output (solution)
     * @param monomer_type Monomer type
     */
    void sparse_solve(int STREAM, int sub_interval,
                      double* d_q_in, double* d_q_out, std::string monomer_type);

    /**
     * @brief PCG solve step for 2D/3D on GPU.
     *
     * Solves (I - dtau * D∇² + dtau*w) q_out = q_in using PCG sparse solver.
     * This avoids splitting errors that would occur with ADI methods.
     */
    void pcg_solve_step(int STREAM, int sub_interval,
                        double* d_q_in, double* d_q_out, std::string monomer_type);

    /**
     * @brief Matrix-free PCG solve for 2D/3D on GPU.
     *
     * Solves (I - dtau * D∇² + dtau*w) q_out = q_in using PCG with
     * matrix-free matvec (no sparse matrix storage needed).
     * Better memory efficiency and cache utilization for large grids.
     */
    void matvec_free_solve(int STREAM, int sub_interval,
                           double* d_q_in, double* d_q_out, std::string monomer_type);

    /**
     * @brief IMEX diffusion solve using FFT for periodic BC.
     *
     * Solves (I - dtau * D∇²) q_out = q_in using FFT-based direct solve.
     * This is used for IMEX SDC where diffusion is treated implicitly
     * and reaction explicitly.
     *
     * For periodic BC: q_out = IFFT( FFT(q_in) / (1 + dtau*D*|k|²) )
     */
    void diffusion_solve_fft(int STREAM, int sub_interval,
                             double* d_q_in, double* d_q_out, std::string monomer_type);

    /**
     * @brief Compute diffusion operator inverse in Fourier space.
     *
     * Computes 1/(1 + dtau*D*|k|²) for each sub-interval and monomer type.
     */
    void compute_diffusion_inverse(int sub_interval, std::string monomer_type);

    /**
     * @brief Apply FFT-based preconditioner for PCG.
     *
     * Computes z = (I - dtau*D∇²)^{-1} * r using FFT.
     * This is a much better preconditioner than Jacobi for periodic BC
     * since it captures the diffusion operator structure exactly.
     *
     * M^{-1} ≈ (I - dtau*D∇²)^{-1} for full operator A = (I - dtau*D∇² + dtau*w)
     * Preconditioned system: M^{-1}A ≈ I + O(dtau*w), converges in 1-3 iterations.
     */
    void apply_fft_preconditioner(int STREAM, int sub_interval,
                                   double* d_r, double* d_z, std::string monomer_type);

    /**
     * @brief IMEX diffusion solve using PCG for non-periodic BC.
     *
     * Solves (I - dtau * D∇²) q_out = q_in using PCG with Jacobi preconditioner.
     * This is used for IMEX SDC where diffusion is treated implicitly
     * and reaction explicitly, for non-periodic boundary conditions in 2D/3D.
     */
    void diffusion_solve_pcg(int STREAM, int sub_interval,
                             double* d_q_in, double* d_q_out, std::string monomer_type);

    /**
     * @brief IMEX diffusion solve using tridiagonal solver for 1D.
     *
     * Solves (I - dtau * D∇²) q_out = q_in using tridiagonal solver.
     * Uses the base diagonal coefficients (diffusion only, no w term).
     */
    void diffusion_solve_tridiag_1d(int STREAM, int sub_interval,
                                     std::vector<BoundaryCondition> bc,
                                     double* d_q_in, double* d_q_out, std::string monomer_type);

    /**
     * @brief Compute F_diff = D∇²q (diffusion part only) using finite differences.
     *
     * Used for IMEX SDC corrections with non-periodic BC where FFT cannot be used.
     * The Laplacian is computed using 2nd-order central differences with
     * appropriate ghost cell handling for boundary conditions.
     */
    void compute_F_diff_fd(int STREAM, const double* d_q,
                           double* d_F_diff, std::string monomer_type);

public:
    /**
     * @brief Construct GPU SDC solver.
     *
     * @param cb Computation box with boundary conditions
     * @param molecules Molecules container
     * @param n_streams Number of parallel streams
     * @param streams Pre-created CUDA streams
     * @param M Number of Gauss-Lobatto nodes (default: 4 for 6th order quadrature)
     * @param K Number of SDC correction iterations (default: 5 for 6th order accuracy)
     *
     * **Order of Accuracy:**
     * The SDC order is min(K+1, 2M-2):
     * - M=3, K=2: order 3
     * - M=4, K=5: order 6 (default)
     * - M=5, K=7: order 8
     */
    CudaSolverSDC(ComputationBox<double>* cb, Molecules *molecules,
        int n_streams, cudaStream_t streams[MAX_STREAMS][2],
        int M = 4, int K = 5);

    /**
     * @brief Destructor. Frees GPU resources.
     */
    ~CudaSolverSDC();

    /**
     * @brief Update Laplacian coefficients.
     */
    void update_laplacian_operator() override;

    /**
     * @brief Update Boltzmann factors from potential fields.
     */
    void update_dw(std::string device, std::map<std::string, const double*> w_input) override;

    /**
     * @brief Advance propagator by one contour step using SDC.
     */
    void advance_propagator(
        const int STREAM,
        double *d_q_in, double *d_q_out,
        std::string monomer_type, double *d_q_mask, int ds_index = 1) override;

    /**
     * @brief Half-bond step (not used for continuous chains).
     */
    void advance_propagator_half_bond_step(
        const int, double *, double *, std::string) override {};

    /**
     * @brief Compute stress (not yet implemented).
     */
    void compute_single_segment_stress(
        const int STREAM,
        double *d_q_pair, double *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) override;

    /**
     * @brief Get the number of Gauss-Lobatto nodes.
     */
    int get_M() const { return M; }

    /**
     * @brief Get the number of SDC correction iterations.
     */
    int get_K() const { return K; }

    /**
     * @brief Enable or disable IMEX mode.
     *
     * IMEX (Implicit-Explicit) SDC treats diffusion implicitly and reaction
     * explicitly, which is faster for periodic BC in 2D/3D.
     *
     * @param enabled True to enable IMEX mode, false to use fully implicit
     */
    void set_imex_mode(bool enabled) { imex_mode_enabled_ = enabled; }

    /**
     * @brief Check if IMEX mode is enabled.
     */
    bool get_imex_mode() const { return imex_mode_enabled_; }
};
#endif
