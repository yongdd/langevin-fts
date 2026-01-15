/**
 * @file CpuSolverSDC.h
 * @brief SDC (Spectral Deferred Correction) solver for continuous chains on CPU.
 *
 * This header provides CpuSolverSDC, which implements the Spectral Deferred
 * Correction method using Gauss-Lobatto quadrature nodes for solving the
 * modified diffusion equation.
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
 * **Gauss-Lobatto Nodes:**
 *
 * The Gauss-Lobatto nodes on [0, 1] include endpoints:
 * - M=3: [0, 0.5, 1]
 * - M=5: [0, 0.5-√21/14, 0.5, 0.5+√21/14, 1]
 *
 * **Order of Accuracy:**
 *
 * - 1D: High order (up to 2K+1 with K corrections) - tridiagonal solve is exact
 * - 2D/3D: High order using MKL PARDISO direct sparse solver (no ADI splitting)
 *
 * With IMEX predictor (Backward Euler) and K corrections, the theoretical order
 * is 2K+1 for problems where the implicit solve is exact.
 *
 * **Implementation Details:**
 *
 * - 1D: Uses tridiagonal solver (exact, fast)
 * - 2D/3D: Uses MKL PARDISO direct sparse solver to avoid ADI splitting error
 *   The sparse matrix A = I - dtau * L is built in CSR format and factorized once
 *   per sub-interval per monomer type. The factorization is reused for all solves.
 *
 * **References:**
 *
 * - Dutt, Greengard, Rokhlin (2000) "Spectral Deferred Correction Methods
 *   for Ordinary Differential Equations." BIT Numerical Mathematics 40, 241-266.
 * - Minion (2003) "Semi-implicit spectral deferred correction methods for
 *   ordinary differential equations." Comm. Math. Sci. 1(3), 471-500.
 *
 * @see CpuSolver for the abstract interface
 * @see CpuSolverCNADI for the underlying ADI solver (1D only)
 */

#ifndef CPU_SOLVER_SDC_H_
#define CPU_SOLVER_SDC_H_

#include <string>
#include <vector>
#include <map>

#include "mkl.h"
#include "mkl_pardiso.h"

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "CpuSolverCNADI.h"

/**
 * @struct SparseMatrixCSR
 * @brief CSR (Compressed Sparse Row) format sparse matrix for PCG solver.
 */
struct SparseMatrixCSR
{
    std::vector<MKL_INT> row_ptr;   ///< Row pointers (size n+1)
    std::vector<MKL_INT> col_idx;   ///< Column indices (size nnz)
    std::vector<double> values;      ///< Matrix values (size nnz)
    std::vector<double> diag_inv;    ///< Inverse diagonal for Jacobi preconditioner
    MKL_INT n;                       ///< Matrix dimension
    MKL_INT nnz;                     ///< Number of non-zeros
    bool built;                      ///< Whether matrix is built
};

/**
 * @class CpuSolverSDC
 * @brief CPU solver using SDC (Spectral Deferred Correction) with Gauss-Lobatto nodes.
 *
 * Implements the SDC scheme for solving the modified diffusion equation.
 * Uses CpuSolverCNADI internally for implicit diffusion solves at each
 * sub-interval within a contour step.
 *
 * **Algorithm per contour step:**
 *
 * 1. Predictor: Backward Euler at each sub-interval using ADI
 * 2. Corrections (K iterations):
 *    - Compute F = D∇²q - wq at all GL nodes
 *    - Apply SDC update using spectral integration matrix S
 *
 * **Configuration:**
 *
 * - M: Number of Gauss-Lobatto nodes (default: 3)
 * - K: Number of correction iterations (default: 2)
 *
 * For M=4, K=5: 6th order accuracy (default configuration).
 * For M=3, K=2: 3rd order accuracy.
 */
class CpuSolverSDC : public CpuSolver<double>
{
private:
    ComputationBox<double>* cb;   ///< Computation box for grid/boundary info
    Molecules *molecules;          ///< Molecules container

    int M;                         ///< Number of Gauss-Lobatto nodes
    int K;                         ///< Number of SDC correction iterations

    std::vector<double> tau;       ///< Gauss-Lobatto nodes on [0, 1]
    std::vector<std::vector<double>> S;  ///< Spectral integration matrix (M-1) x M

    // Tridiagonal coefficients for sub-intervals (indexed by sub-interval)
    // For sub-interval m: dtau[m] = (tau[m+1] - tau[m]) * ds
    std::vector<std::map<std::string, double*>> xl;  ///< Lower diagonal per sub-interval
    std::vector<std::map<std::string, double*>> xd;  ///< Main diagonal per sub-interval
    std::vector<std::map<std::string, double*>> xh;  ///< Upper diagonal per sub-interval

    std::vector<std::map<std::string, double*>> yl;  ///< Y-direction lower
    std::vector<std::map<std::string, double*>> yd;  ///< Y-direction main
    std::vector<std::map<std::string, double*>> yh;  ///< Y-direction upper

    std::vector<std::map<std::string, double*>> zl;  ///< Z-direction lower
    std::vector<std::map<std::string, double*>> zd;  ///< Z-direction main
    std::vector<std::map<std::string, double*>> zh;  ///< Z-direction upper

    // Boltzmann factors for sub-intervals
    // exp_dw_sub[m] = exp(-w * dtau[m]) for reaction term
    std::vector<std::map<std::string, std::vector<double>>> exp_dw_sub;

    // Store w field directly for SDC corrections (like CUDA)
    std::map<std::string, std::vector<double>> w_field_store;

    // Sparse matrices for direct solve (2D/3D only)
    // Indexed by [sub_interval][monomer_type]
    std::vector<std::map<std::string, SparseMatrixCSR>> sparse_matrices;

    // Number of threads for workspace allocation
    int n_threads;

    // Workspace arrays - indexed by [thread][m] for GL nodes
    std::vector<std::vector<double*>> X;        ///< Solution at GL nodes: X[thread][m] of size n_grid
    std::vector<std::vector<double*>> X_old;    ///< Old solution for SDC corrections
    std::vector<std::vector<double*>> F_diff;   ///< Diffusion term D∇²q at GL nodes
    std::vector<std::vector<double*>> F_old;    ///< Old F for SDC corrections
    std::vector<std::vector<double*>> F_react;  ///< Reaction term -wq at GL nodes
    std::vector<double*> temp_array;            ///< Temporary workspace per thread
    std::vector<double*> rhs_array;             ///< RHS for implicit solves per thread

    // PCG workspace arrays (allocated for 2D/3D) - indexed by thread
    std::vector<double*> pcg_r;                 ///< Residual vector per thread
    std::vector<double*> pcg_z;                 ///< Preconditioned residual per thread
    std::vector<double*> pcg_p;                 ///< Search direction per thread
    std::vector<double*> pcg_Ap;                ///< Matrix-vector product A*p per thread
    int pcg_max_iter;              ///< Maximum PCG iterations
    double pcg_tol;                ///< PCG convergence tolerance

    /**
     * @brief Compute Gauss-Lobatto nodes on [0, 1].
     */
    void compute_gauss_lobatto_nodes();

    /**
     * @brief Compute spectral integration matrix S.
     *
     * S[m][j] = ∫_{tau[m]}^{tau[m+1]} L_j(t) dt
     * where L_j is the j-th Lagrange basis polynomial.
     */
    void compute_integration_matrix();

    /**
     * @brief Compute F(q) = D∇²q - wq.
     *
     * @param q Input propagator
     * @param F_out Output array for F(q)
     * @param monomer_type Monomer type for bond length
     */
    void compute_F(const double* q, const double* w, double* F_out, std::string monomer_type);

    /**
     * @brief ADI step for a specific sub-interval.
     *
     * Solves (I - dtau * D∇²) q_out = rhs using ADI.
     *
     * @param sub_interval Index of sub-interval (0 to M-2)
     * @param q_in Input propagator
     * @param q_out Output propagator
     * @param monomer_type Monomer type
     */
    void adi_step(int sub_interval, double* q_in, double* q_out, std::string monomer_type);

    /**
     * @brief ADI step in 3D.
     */
    void adi_step_3d(int sub_interval,
        std::vector<BoundaryCondition> bc,
        double* q_in, double* q_out, std::string monomer_type);

    /**
     * @brief ADI step in 2D.
     */
    void adi_step_2d(int sub_interval,
        std::vector<BoundaryCondition> bc,
        double* q_in, double* q_out, std::string monomer_type);

    /**
     * @brief ADI step in 1D.
     */
    void adi_step_1d(int sub_interval,
        std::vector<BoundaryCondition> bc,
        double* q_in, double* q_out, std::string monomer_type);

    /**
     * @brief Build sparse Laplacian matrix in CSR format.
     *
     * Builds A = I - dtau * D * ∇² for the full 2D or 3D Laplacian.
     * Uses 0-based indexing for MKL sparse BLAS.
     *
     * @param sub_interval Sub-interval index
     * @param monomer_type Monomer type for bond length
     */
    void build_sparse_matrix(int sub_interval, std::string monomer_type);

    /**
     * @brief Sparse matrix-vector product: y = A * x.
     *
     * @param mat Sparse matrix in CSR format
     * @param x Input vector
     * @param y Output vector
     */
    void sparse_matvec(const SparseMatrixCSR& mat, const double* x, double* y);

    /**
     * @brief Solve sparse linear system using Preconditioned Conjugate Gradient.
     *
     * Solves A * q_out = q_in using PCG with Jacobi preconditioner.
     * The matrix A = I - dtau * D * ∇² is symmetric positive definite.
     *
     * @param sub_interval Sub-interval index
     * @param q_in Input (RHS)
     * @param q_out Output (solution)
     * @param monomer_type Monomer type
     */
    void sparse_solve(int sub_interval, double* q_in, double* q_out, std::string monomer_type);

    /**
     * @brief Iterative solve step for 2D/3D (replaces ADI).
     *
     * Uses PCG solver instead of ADI splitting to avoid O(dt²) error.
     */
    void direct_solve_step(int sub_interval, double* q_in, double* q_out, std::string monomer_type);

    /**
     * @brief Return maximum of two integers.
     */
    int max_of_two(int x, int y);

    /**
     * @brief Return minimum of two integers.
     */
    int min_of_two(int x, int y);

public:
    /**
     * @brief Construct SDC solver.
     *
     * @param cb Computation box with boundary conditions
     * @param molecules Molecules container with monomer types
     * @param M Number of Gauss-Lobatto nodes (default: 4 for 6th order quadrature)
     * @param K Number of SDC correction iterations (default: 5 for 6th order accuracy)
     *
     * **Order of Accuracy:**
     * The SDC order is min(K+1, 2M-2):
     * - M=3, K=2: order 3
     * - M=4, K=5: order 6 (default)
     * - M=5, K=7: order 8
     */
    CpuSolverSDC(ComputationBox<double>* cb, Molecules *molecules, int M = 4, int K = 5);

    /**
     * @brief Destructor. Frees allocated arrays.
     */
    ~CpuSolverSDC();

    /**
     * @brief Update Laplacian coefficients for new box dimensions.
     */
    void update_laplacian_operator() override;

    /**
     * @brief Update Boltzmann factors from potential fields.
     *
     * @param w_input Map of potential fields by monomer type
     */
    void update_dw(std::map<std::string, const double*> w_input) override;

    /**
     * @brief Advance propagator by one contour step using SDC.
     *
     * @param q_in Input propagator
     * @param q_out Output propagator
     * @param monomer_type Monomer type
     * @param q_mask Optional mask (currently ignored)
     * @param ds_index ds index (SDC uses global ds only)
     */
    void advance_propagator(
        double *q_in, double *q_out, std::string monomer_type,
        const double *q_mask, int ds_index = 1) override;

    /**
     * @brief Half-bond step (not used for continuous chains).
     */
    void advance_propagator_half_bond_step(double *, double *, std::string) override {};

    /**
     * @brief Compute stress (not yet implemented for SDC).
     */
    std::vector<double> compute_single_segment_stress(
        double *q_1, double *q_2, std::string monomer_type, bool is_half_bond_length) override;

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
