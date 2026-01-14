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
 * - ADI sweeps use parallel tridiagonal solvers
 * - F computation uses element-wise GPU kernels
 * - Spectral integration uses parallel reductions
 *
 * **Configuration:**
 *
 * - M: Number of Gauss-Lobatto nodes (default: 3)
 * - K: Number of SDC correction iterations (default: 2)
 *
 * **Order of Accuracy:**
 *
 * - 1D: High order (up to 2K+1 with K corrections) - implicit solves are exact
 * - 2D/3D: Limited to 2nd-order due to O(ds²) ADI splitting error
 *
 * **Limitations:**
 *
 * ADI splitting solves (I - dt*Dx)(I - dt*Dy)q = RHS instead of (I - dt*(Dx+Dy))q = RHS.
 * The O(dt²*Dx*Dy) difference is an irreducible splitting error that persists
 * regardless of the number of SDC corrections. This limits 2D/3D to 2nd-order accuracy.
 *
 * @see CudaSolver for the abstract interface
 * @see CudaSolverCNADI for the underlying ADI solver
 */

#ifndef CUDA_SOLVER_SDC_H_
#define CUDA_SOLVER_SDC_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CudaSolver.h"
#include "CudaCommon.h"

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
 * @brief CUDA kernel: Apply Boltzmann factor and prepare for ADI.
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
 * Uses parallel ADI solvers for implicit diffusion at each sub-interval.
 *
 * **Algorithm per contour step:**
 *
 * 1. Predictor: Backward Euler at each sub-interval using ADI
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
    std::vector<std::map<std::string, double*>> d_xh;

    std::vector<std::map<std::string, double*>> d_yl;
    std::vector<std::map<std::string, double*>> d_yd;
    std::vector<std::map<std::string, double*>> d_yh;

    std::vector<std::map<std::string, double*>> d_zl;
    std::vector<std::map<std::string, double*>> d_zd;
    std::vector<std::map<std::string, double*>> d_zh;

    // Boltzmann factors for sub-intervals (device memory)
    std::vector<std::map<std::string, double*>> d_exp_dw_sub;

    // Potential field storage (device memory)
    std::map<std::string, double*> d_w_field;

    // Per-stream workspace arrays (device memory)
    std::vector<double*> d_X[MAX_STREAMS];       ///< Solution at GL nodes
    std::vector<double*> d_F[MAX_STREAMS];       ///< F(q) at GL nodes
    std::vector<double*> d_X_old[MAX_STREAMS];   ///< Previous X for correction
    std::vector<double*> d_F_old[MAX_STREAMS];   ///< Previous F for correction
    double* d_temp[MAX_STREAMS];                  ///< Temporary workspace
    double* d_rhs[MAX_STREAMS];                   ///< RHS for implicit solves

    // ADI workspace (per stream)
    double* d_q_star[MAX_STREAMS];
    double* d_q_dstar[MAX_STREAMS];
    double* d_c_star[MAX_STREAMS];
    double* d_q_sparse[MAX_STREAMS];
    double* d_q_in_saved[MAX_STREAMS];  ///< Saved input for ADI (fixes Y-direction bug)

    // Offset arrays for ADI
    int* d_offset_xy;
    int* d_offset_yz;
    int* d_offset_xz;
    int* d_offset_x;
    int* d_offset_y;
    int* d_offset;

    /**
     * @brief Compute Gauss-Lobatto nodes on [0, 1].
     */
    void compute_gauss_lobatto_nodes();

    /**
     * @brief Compute spectral integration matrix S.
     */
    void compute_integration_matrix();

    /**
     * @brief Compute F(q) = D∇²q - wq on GPU.
     */
    void compute_F_device(int STREAM, const double* d_q, const double* d_w,
                          double* d_F, std::string monomer_type);

    /**
     * @brief ADI step for a specific sub-interval on GPU.
     */
    void adi_step(int STREAM, int sub_interval,
                  double* d_q_in, double* d_q_out, std::string monomer_type);

    void adi_step_3d(int STREAM, int sub_interval,
        std::vector<BoundaryCondition> bc,
        double* d_q_in, double* d_q_out, std::string monomer_type);

    void adi_step_2d(int STREAM, int sub_interval,
        std::vector<BoundaryCondition> bc,
        double* d_q_in, double* d_q_out, std::string monomer_type);

    void adi_step_1d(int STREAM, int sub_interval,
        std::vector<BoundaryCondition> bc,
        double* d_q_in, double* d_q_out, std::string monomer_type);

public:
    /**
     * @brief Construct GPU SDC solver.
     *
     * @param cb Computation box with boundary conditions
     * @param molecules Molecules container
     * @param n_streams Number of parallel streams
     * @param streams Pre-created CUDA streams
     * @param M Number of Gauss-Lobatto nodes (default: 3)
     * @param K Number of SDC correction iterations (default: 2)
     */
    CudaSolverSDC(ComputationBox<double>* cb, Molecules *molecules,
        int n_streams, cudaStream_t streams[MAX_STREAMS][2],
        int M = 3, int K = 2);

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
};
#endif
