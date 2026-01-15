/**
 * @file CpuSolverSDC.cpp
 * @brief CPU SDC solver implementation.
 *
 * Implements the Spectral Deferred Correction method for solving the
 * modified diffusion equation using Gauss-Lobatto quadrature nodes.
 *
 * **SDC Algorithm:**
 *
 * For equation: ∂q/∂s = D∇²q - wq
 *
 * 1. Discretize contour [0, ds] using M Gauss-Lobatto nodes
 * 2. Predictor: Backward Euler for diffusion, explicit for reaction
 * 3. K correction iterations using spectral integration matrix
 *
 * **Integration Matrix:**
 *
 * The spectral integration matrix S is computed using high-order
 * Gauss quadrature (16 points) for accurate Lagrange polynomial
 * integration over each sub-interval.
 *
 * @see CpuSolverCNADI for ADI tridiagonal solvers
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <utility>
#include <omp.h>

#include "CpuSolverSDC.h"
#include "FiniteDifference.h"

CpuSolverSDC::CpuSolverSDC(ComputationBox<double>* cb, Molecules *molecules, int M, int K)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->M = M;
        this->K = K;

        if(molecules->get_model_name() != "continuous")
            throw_with_line_number("SDC method only supports 'continuous' chain model.");

        if(!cb->is_orthogonal())
            throw_with_line_number("SDC method only supports orthogonal unit cells.");

        if(M < 2)
            throw_with_line_number("SDC requires at least M=2 Gauss-Lobatto nodes.");

        if(K < 0)
            throw_with_line_number("SDC requires K >= 0 correction iterations.");

        const int n_grid = cb->get_total_grid();

        // Compute Gauss-Lobatto nodes and integration matrix
        compute_gauss_lobatto_nodes();
        compute_integration_matrix();

        // Allocate tridiagonal coefficient arrays for each sub-interval
        xl.resize(M - 1);
        xd.resize(M - 1);
        xh.resize(M - 1);
        yl.resize(M - 1);
        yd.resize(M - 1);
        yh.resize(M - 1);
        zl.resize(M - 1);
        zd.resize(M - 1);
        zh.resize(M - 1);

        exp_dw_sub.resize(M - 1);

        for(int m = 0; m < M - 1; m++)
        {
            for(const auto& item: molecules->get_bond_lengths())
            {
                std::string monomer_type = item.first;

                xl[m][monomer_type] = new double[n_grid];
                xd[m][monomer_type] = new double[n_grid];
                xh[m][monomer_type] = new double[n_grid];

                yl[m][monomer_type] = new double[n_grid];
                yd[m][monomer_type] = new double[n_grid];
                yh[m][monomer_type] = new double[n_grid];

                zl[m][monomer_type] = new double[n_grid];
                zd[m][monomer_type] = new double[n_grid];
                zh[m][monomer_type] = new double[n_grid];

                exp_dw_sub[m][monomer_type].resize(n_grid);
            }
        }

        // Allocate w field storage for each monomer type (like CUDA)
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            w_field_store[monomer_type].resize(n_grid);
        }

        // Initialize sparse matrices and PCG workspace for 2D/3D
        const int DIM = cb->get_dim();

        // Get number of threads for per-thread workspace allocation
        n_threads = omp_get_max_threads();

        if(DIM >= 2)
        {
            sparse_matrices.resize(M - 1);
            for(int m = 0; m < M - 1; m++)
            {
                for(const auto& item: molecules->get_bond_lengths())
                {
                    std::string monomer_type = item.first;
                    SparseMatrixCSR& mat = sparse_matrices[m][monomer_type];
                    mat.n = n_grid;
                    mat.built = false;
                }
            }

            // Allocate PCG workspace arrays per thread
            pcg_r.resize(n_threads);
            pcg_z.resize(n_threads);
            pcg_p.resize(n_threads);
            pcg_Ap.resize(n_threads);
            for(int t = 0; t < n_threads; t++)
            {
                pcg_r[t] = new double[n_grid];
                pcg_z[t] = new double[n_grid];
                pcg_p[t] = new double[n_grid];
                pcg_Ap[t] = new double[n_grid];
            }
            pcg_max_iter = 1000;  // Maximum iterations
            pcg_tol = 1e-10;      // Convergence tolerance
        }

        // Allocate workspace arrays for GL node solutions per thread
        X.resize(n_threads);
        X_old.resize(n_threads);
        F_diff.resize(n_threads);
        F_old.resize(n_threads);
        F_react.resize(n_threads);
        temp_array.resize(n_threads);
        rhs_array.resize(n_threads);

        for(int t = 0; t < n_threads; t++)
        {
            X[t].resize(M);
            X_old[t].resize(M);
            F_diff[t].resize(M);
            F_old[t].resize(M);
            F_react[t].resize(M);
            for(int m = 0; m < M; m++)
            {
                X[t][m] = new double[n_grid];
                X_old[t][m] = new double[n_grid];
                F_diff[t][m] = new double[n_grid];
                F_old[t][m] = new double[n_grid];
                F_react[t][m] = new double[n_grid];
            }
            temp_array[t] = new double[n_grid];
            rhs_array[t] = new double[n_grid];
        }

        // Initialize Laplacian operator
        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

CpuSolverSDC::~CpuSolverSDC()
{
    // Free PCG workspace (2D/3D only)
    const int DIM = cb->get_dim();
    if(DIM >= 2)
    {
        for(int t = 0; t < n_threads; t++)
        {
            delete[] pcg_r[t];
            delete[] pcg_z[t];
            delete[] pcg_p[t];
            delete[] pcg_Ap[t];
        }
    }

    // Free tridiagonal coefficients
    for(int m = 0; m < M - 1; m++)
    {
        for(const auto& item: xl[m])
            delete[] item.second;
        for(const auto& item: xd[m])
            delete[] item.second;
        for(const auto& item: xh[m])
            delete[] item.second;

        for(const auto& item: yl[m])
            delete[] item.second;
        for(const auto& item: yd[m])
            delete[] item.second;
        for(const auto& item: yh[m])
            delete[] item.second;

        for(const auto& item: zl[m])
            delete[] item.second;
        for(const auto& item: zd[m])
            delete[] item.second;
        for(const auto& item: zh[m])
            delete[] item.second;
    }

    // Free per-thread workspace
    for(int t = 0; t < n_threads; t++)
    {
        for(int m = 0; m < M; m++)
        {
            delete[] X[t][m];
            delete[] X_old[t][m];
            delete[] F_diff[t][m];
            delete[] F_old[t][m];
            delete[] F_react[t][m];
        }
        delete[] temp_array[t];
        delete[] rhs_array[t];
    }
}

void CpuSolverSDC::compute_gauss_lobatto_nodes()
{
    tau.resize(M);

    if(M == 2)
    {
        tau[0] = 0.0;
        tau[1] = 1.0;
    }
    else if(M == 3)
    {
        tau[0] = 0.0;
        tau[1] = 0.5;
        tau[2] = 1.0;
    }
    else if(M == 4)
    {
        tau[0] = 0.0;
        tau[1] = 0.5 - std::sqrt(5.0) / 10.0;
        tau[2] = 0.5 + std::sqrt(5.0) / 10.0;
        tau[3] = 1.0;
    }
    else if(M == 5)
    {
        tau[0] = 0.0;
        tau[1] = 0.5 - std::sqrt(21.0) / 14.0;
        tau[2] = 0.5;
        tau[3] = 0.5 + std::sqrt(21.0) / 14.0;
        tau[4] = 1.0;
    }
    else if(M == 6)
    {
        // Exact Gauss-Lobatto nodes for M=6
        // Interior nodes are roots of P'_5(x) where P_5 is Legendre polynomial
        // x^2 = (7 ± 2*sqrt(7))/21, transformed to [0,1]
        double x1 = std::sqrt((7.0 + 2.0 * std::sqrt(7.0)) / 21.0);
        double x2 = std::sqrt((7.0 - 2.0 * std::sqrt(7.0)) / 21.0);
        tau[0] = 0.0;
        tau[1] = 0.5 * (1.0 - x1);
        tau[2] = 0.5 * (1.0 - x2);
        tau[3] = 0.5 * (1.0 + x2);
        tau[4] = 0.5 * (1.0 + x1);
        tau[5] = 1.0;
    }
    else if(M == 7)
    {
        // Exact Gauss-Lobatto nodes for M=7
        // Interior nodes are roots of P'_6(x) = x * (polynomial)
        // One root is x=0, others from: x^2 = (15 ± 2*sqrt(15))/33
        double x1 = std::sqrt((15.0 + 2.0 * std::sqrt(15.0)) / 33.0);
        double x2 = std::sqrt((15.0 - 2.0 * std::sqrt(15.0)) / 33.0);
        tau[0] = 0.0;
        tau[1] = 0.5 * (1.0 - x1);
        tau[2] = 0.5 * (1.0 - x2);
        tau[3] = 0.5;  // x = 0 maps to tau = 0.5
        tau[4] = 0.5 * (1.0 + x2);
        tau[5] = 0.5 * (1.0 + x1);
        tau[6] = 1.0;
    }
    else
    {
        // General formula using Chebyshev nodes of second kind
        // Good approximation for large M
        for(int j = 0; j < M; j++)
        {
            tau[j] = 0.5 * (1.0 - std::cos(std::numbers::pi * j / (M - 1)));
        }
    }
}

void CpuSolverSDC::compute_integration_matrix()
{
    // S is (M-1) x M matrix
    // S[m][j] = ∫_{tau[m]}^{tau[m+1]} L_j(t) dt
    // where L_j(t) is the j-th Lagrange basis polynomial

    S.resize(M - 1);
    for(int m = 0; m < M - 1; m++)
        S[m].resize(M);

    // Use 16-point Gauss quadrature for accurate integration
    const int n_gauss = 16;
    std::vector<double> gauss_nodes(n_gauss);
    std::vector<double> gauss_weights(n_gauss);

    // 16-point Gauss-Legendre nodes and weights on [-1, 1]
    // (Pre-computed values)
    const double nodes_16[] = {
        -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030,
        -0.6178762444026437, -0.4580167776572274, -0.2816035507792589, -0.0950125098376374,
         0.0950125098376374,  0.2816035507792589,  0.4580167776572274,  0.6178762444026437,
         0.7554044083550030,  0.8656312023878318,  0.9445750230732326,  0.9894009349916499
    };
    const double weights_16[] = {
        0.0271524594117541, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
        0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685,
        0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
        0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541
    };

    for(int i = 0; i < n_gauss; i++)
    {
        gauss_nodes[i] = nodes_16[i];
        gauss_weights[i] = weights_16[i];
    }

    // Compute S[m][j] for each sub-interval m and basis j
    for(int m = 0; m < M - 1; m++)
    {
        double a = tau[m];
        double b = tau[m + 1];
        double h = b - a;

        for(int j = 0; j < M; j++)
        {
            double integral = 0.0;

            for(int k = 0; k < n_gauss; k++)
            {
                // Map Gauss node from [-1, 1] to [a, b]
                double t = a + 0.5 * h * (gauss_nodes[k] + 1.0);

                // Evaluate Lagrange basis L_j(t)
                double L_j = 1.0;
                for(int i = 0; i < M; i++)
                {
                    if(i != j)
                        L_j *= (t - tau[i]) / (tau[j] - tau[i]);
                }

                integral += gauss_weights[k] * L_j;
            }

            S[m][j] = 0.5 * h * integral;
        }
    }
}

int CpuSolverSDC::max_of_two(int x, int y)
{
    return (x > y) ? x : y;
}

int CpuSolverSDC::min_of_two(int x, int y)
{
    return (x < y) ? x : y;
}

void CpuSolverSDC::update_laplacian_operator()
{
    try
    {
        const double ds = this->molecules->get_ds();

        for(const auto& item: this->molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second * item.second;

            // Compute coefficients for each sub-interval
            // Use Backward Euler matrix (not Crank-Nicolson) for SDC
            for(int m = 0; m < M - 1; m++)
            {
                double dtau = (tau[m + 1] - tau[m]) * ds;

                FiniteDifference::get_backward_euler_matrix(
                    this->cb->get_boundary_conditions(),
                    this->cb->get_nx(), this->cb->get_dx(),
                    xl[m][monomer_type], xd[m][monomer_type], xh[m][monomer_type],
                    yl[m][monomer_type], yd[m][monomer_type], yh[m][monomer_type],
                    zl[m][monomer_type], zd[m][monomer_type], zh[m][monomer_type],
                    bond_length_sq, dtau);
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverSDC::update_dw(std::map<std::string, const double*> w_input)
{
    const int n_grid = this->cb->get_total_grid();
    const double ds = this->molecules->get_ds();

    for(const auto& item: w_input)
    {
        const std::string& monomer_type = item.first;
        const double* w = item.second;

        // Store w directly for SDC corrections (like CUDA)
        std::vector<double>& w_store = w_field_store[monomer_type];
        for(int i = 0; i < n_grid; i++)
        {
            w_store[i] = w[i];
        }

        // Compute exp(-w * dtau) for each sub-interval
        for(int m = 0; m < M - 1; m++)
        {
            double dtau = (tau[m + 1] - tau[m]) * ds;
            std::vector<double>& exp_dw_vec = exp_dw_sub[m][monomer_type];

            for(int i = 0; i < n_grid; i++)
            {
                exp_dw_vec[i] = std::exp(-w[i] * dtau);
            }
        }
    }
}

void CpuSolverSDC::compute_F(const double* q, const double* w, double* F_out, std::string monomer_type)
{
    const int DIM = this->cb->get_dim();
    const int n_grid = this->cb->get_total_grid();
    const std::vector<int> nx = this->cb->get_nx();
    const std::vector<double> dx = this->cb->get_dx();
    const std::vector<BoundaryCondition> bc = this->cb->get_boundary_conditions();

    double bond_length = this->molecules->get_bond_lengths().at(monomer_type);
    double bond_length_sq = bond_length * bond_length;
    const double D = bond_length_sq / 6.0;

    // Initialize output
    for(int i = 0; i < n_grid; i++)
        F_out[i] = 0.0;

    if(DIM == 1)
    {
        double alpha_x = D / (dx[0] * dx[0]);

        for(int i = 0; i < nx[0]; i++)
        {
            // Get neighbor values with proper BC handling
            double q_im, q_ip;

            // Lower neighbor (cell-centered: antisymmetric ghost for absorbing BC)
            if(bc[0] == BoundaryCondition::PERIODIC)
                q_im = q[(nx[0] + i - 1) % nx[0]];
            else if(bc[0] == BoundaryCondition::ABSORBING && i == 0)
                q_im = -q[i];  // Antisymmetric ghost: q_{-1} = -q_0
            else
                q_im = q[max_of_two(0, i - 1)];  // Reflecting: symmetric ghost (clamp index)

            // Upper neighbor (cell-centered: antisymmetric ghost for absorbing BC)
            if(bc[1] == BoundaryCondition::PERIODIC)
                q_ip = q[(i + 1) % nx[0]];
            else if(bc[1] == BoundaryCondition::ABSORBING && i == nx[0] - 1)
                q_ip = -q[i];  // Antisymmetric ghost: q_{N} = -q_{N-1}
            else
                q_ip = q[min_of_two(nx[0] - 1, i + 1)];  // Reflecting: symmetric ghost (clamp index)

            double Lx = alpha_x * (q_im + q_ip - 2.0 * q[i]);
            F_out[i] = Lx - w[i] * q[i];
        }
    }
    else if(DIM == 2)
    {
        double alpha_x = D / (dx[0] * dx[0]);
        double alpha_y = D / (dx[1] * dx[1]);

        for(int i = 0; i < nx[0]; i++)
        {
            for(int j = 0; j < nx[1]; j++)
            {
                int idx = i * nx[1] + j;

                // Get neighbor values with proper BC handling for x-direction
                // (cell-centered: antisymmetric ghost for absorbing BC)
                double q_im, q_ip;
                if(bc[0] == BoundaryCondition::PERIODIC)
                    q_im = q[((nx[0] + i - 1) % nx[0]) * nx[1] + j];
                else if(bc[0] == BoundaryCondition::ABSORBING && i == 0)
                    q_im = -q[idx];  // Antisymmetric ghost
                else
                    q_im = q[max_of_two(0, i - 1) * nx[1] + j];

                if(bc[1] == BoundaryCondition::PERIODIC)
                    q_ip = q[((i + 1) % nx[0]) * nx[1] + j];
                else if(bc[1] == BoundaryCondition::ABSORBING && i == nx[0] - 1)
                    q_ip = -q[idx];  // Antisymmetric ghost
                else
                    q_ip = q[min_of_two(nx[0] - 1, i + 1) * nx[1] + j];

                // Get neighbor values with proper BC handling for y-direction
                // (cell-centered: antisymmetric ghost for absorbing BC)
                double q_jm, q_jp;
                if(bc[2] == BoundaryCondition::PERIODIC)
                    q_jm = q[i * nx[1] + (nx[1] + j - 1) % nx[1]];
                else if(bc[2] == BoundaryCondition::ABSORBING && j == 0)
                    q_jm = -q[idx];  // Antisymmetric ghost
                else
                    q_jm = q[i * nx[1] + max_of_two(0, j - 1)];

                if(bc[3] == BoundaryCondition::PERIODIC)
                    q_jp = q[i * nx[1] + (j + 1) % nx[1]];
                else if(bc[3] == BoundaryCondition::ABSORBING && j == nx[1] - 1)
                    q_jp = -q[idx];  // Antisymmetric ghost
                else
                    q_jp = q[i * nx[1] + min_of_two(nx[1] - 1, j + 1)];

                double Lx = alpha_x * (q_im + q_ip - 2.0 * q[idx]);
                double Ly = alpha_y * (q_jm + q_jp - 2.0 * q[idx]);
                F_out[idx] = Lx + Ly - w[idx] * q[idx];
            }
        }
    }
    else // DIM == 3
    {
        double alpha_x = D / (dx[0] * dx[0]);
        double alpha_y = D / (dx[1] * dx[1]);
        double alpha_z = D / (dx[2] * dx[2]);

        for(int i = 0; i < nx[0]; i++)
        {
            for(int j = 0; j < nx[1]; j++)
            {
                for(int k = 0; k < nx[2]; k++)
                {
                    int idx = i * nx[1] * nx[2] + j * nx[2] + k;

                    // Get neighbor values with proper BC handling for x-direction
                    // (cell-centered: antisymmetric ghost for absorbing BC)
                    double q_im, q_ip;
                    if(bc[0] == BoundaryCondition::PERIODIC)
                        q_im = q[((nx[0] + i - 1) % nx[0]) * nx[1] * nx[2] + j * nx[2] + k];
                    else if(bc[0] == BoundaryCondition::ABSORBING && i == 0)
                        q_im = -q[idx];  // Antisymmetric ghost
                    else
                        q_im = q[max_of_two(0, i - 1) * nx[1] * nx[2] + j * nx[2] + k];

                    if(bc[1] == BoundaryCondition::PERIODIC)
                        q_ip = q[((i + 1) % nx[0]) * nx[1] * nx[2] + j * nx[2] + k];
                    else if(bc[1] == BoundaryCondition::ABSORBING && i == nx[0] - 1)
                        q_ip = -q[idx];  // Antisymmetric ghost
                    else
                        q_ip = q[min_of_two(nx[0] - 1, i + 1) * nx[1] * nx[2] + j * nx[2] + k];

                    // Get neighbor values with proper BC handling for y-direction
                    // (cell-centered: antisymmetric ghost for absorbing BC)
                    double q_jm, q_jp;
                    if(bc[2] == BoundaryCondition::PERIODIC)
                        q_jm = q[i * nx[1] * nx[2] + ((nx[1] + j - 1) % nx[1]) * nx[2] + k];
                    else if(bc[2] == BoundaryCondition::ABSORBING && j == 0)
                        q_jm = -q[idx];  // Antisymmetric ghost
                    else
                        q_jm = q[i * nx[1] * nx[2] + max_of_two(0, j - 1) * nx[2] + k];

                    if(bc[3] == BoundaryCondition::PERIODIC)
                        q_jp = q[i * nx[1] * nx[2] + ((j + 1) % nx[1]) * nx[2] + k];
                    else if(bc[3] == BoundaryCondition::ABSORBING && j == nx[1] - 1)
                        q_jp = -q[idx];  // Antisymmetric ghost
                    else
                        q_jp = q[i * nx[1] * nx[2] + min_of_two(nx[1] - 1, j + 1) * nx[2] + k];

                    // Get neighbor values with proper BC handling for z-direction
                    // (cell-centered: antisymmetric ghost for absorbing BC)
                    double q_km, q_kp;
                    if(bc[4] == BoundaryCondition::PERIODIC)
                        q_km = q[i * nx[1] * nx[2] + j * nx[2] + (nx[2] + k - 1) % nx[2]];
                    else if(bc[4] == BoundaryCondition::ABSORBING && k == 0)
                        q_km = -q[idx];  // Antisymmetric ghost
                    else
                        q_km = q[i * nx[1] * nx[2] + j * nx[2] + max_of_two(0, k - 1)];

                    if(bc[5] == BoundaryCondition::PERIODIC)
                        q_kp = q[i * nx[1] * nx[2] + j * nx[2] + (k + 1) % nx[2]];
                    else if(bc[5] == BoundaryCondition::ABSORBING && k == nx[2] - 1)
                        q_kp = -q[idx];  // Antisymmetric ghost
                    else
                        q_kp = q[i * nx[1] * nx[2] + j * nx[2] + min_of_two(nx[2] - 1, k + 1)];

                    double Lx = alpha_x * (q_im + q_ip - 2.0 * q[idx]);
                    double Ly = alpha_y * (q_jm + q_jp - 2.0 * q[idx]);
                    double Lz = alpha_z * (q_km + q_kp - 2.0 * q[idx]);
                    F_out[idx] = Lx + Ly + Lz - w[idx] * q[idx];
                }
            }
        }
    }
}

void CpuSolverSDC::adi_step(int sub_interval, double* q_in, double* q_out, std::string monomer_type)
{
    const int DIM = this->cb->get_dim();
    std::vector<BoundaryCondition> bc = this->cb->get_boundary_conditions();

    if(DIM >= 2)
    {
        // Use direct sparse solver for 2D/3D to avoid ADI splitting error
        direct_solve_step(sub_interval, q_in, q_out, monomer_type);
    }
    else // DIM == 1
    {
        adi_step_1d(sub_interval, bc, q_in, q_out, monomer_type);
    }
}

void CpuSolverSDC::adi_step_3d(int sub_interval,
    std::vector<BoundaryCondition> bc,
    double* q_in, double* q_out, std::string monomer_type)
{
    try
    {
        const int n_grid = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();
        double q_star[n_grid];
        double q_dstar[n_grid];
        double temp1[nx[0]];
        double temp2[nx[1]];
        double temp3[nx[2]];

        double *_xl = xl[sub_interval][monomer_type];
        double *_xd = xd[sub_interval][monomer_type];
        double *_xh = xh[sub_interval][monomer_type];

        double *_yl = yl[sub_interval][monomer_type];
        double *_yd = yd[sub_interval][monomer_type];
        double *_yh = yh[sub_interval][monomer_type];

        double *_zl = zl[sub_interval][monomer_type];
        double *_zd = zd[sub_interval][monomer_type];
        double *_zh = zh[sub_interval][monomer_type];

        // Backward Euler ADI: Solve (I - dt*Dx)(I - dt*Dy)(I - dt*Dz) q = rhs
        // Step 1: Solve (I - dt*Dx) q_star = q_in along x-direction
        // Step 2: Solve (I - dt*Dy) q_dstar = q_star along y-direction
        // Step 3: Solve (I - dt*Dz) q_out = q_dstar along z-direction

        // X-sweep: solve A_x * q_star = q_in for each (j,k) pencil
        for(int j = 0; j < nx[1]; j++)
        {
            for(int k = 0; k < nx[2]; k++)
            {
                for(int i = 0; i < nx[0]; i++)
                {
                    int i_j_k = i * nx[1] * nx[2] + j * nx[2] + k;
                    temp1[i] = q_in[i_j_k];
                }

                int j_k = j * nx[2] + k;
                if(bc[0] == BoundaryCondition::PERIODIC)
                    CpuSolverCNADI::tridiagonal_periodic(_xl, _xd, _xh, &q_star[j_k], nx[1] * nx[2], temp1, nx[0]);
                else
                    CpuSolverCNADI::tridiagonal(_xl, _xd, _xh, &q_star[j_k], nx[1] * nx[2], temp1, nx[0]);
            }
        }

        // Y-sweep: solve A_y * q_dstar = q_star for each (i,k) pencil
        for(int i = 0; i < nx[0]; i++)
        {
            for(int k = 0; k < nx[2]; k++)
            {
                for(int j = 0; j < nx[1]; j++)
                {
                    int i_j_k = i * nx[1] * nx[2] + j * nx[2] + k;
                    temp2[j] = q_star[i_j_k];
                }

                int i_k = i * nx[1] * nx[2] + k;
                if(bc[2] == BoundaryCondition::PERIODIC)
                    CpuSolverCNADI::tridiagonal_periodic(_yl, _yd, _yh, &q_dstar[i_k], nx[2], temp2, nx[1]);
                else
                    CpuSolverCNADI::tridiagonal(_yl, _yd, _yh, &q_dstar[i_k], nx[2], temp2, nx[1]);
            }
        }

        // Z-sweep: solve A_z * q_out = q_dstar for each (i,j) pencil
        for(int i = 0; i < nx[0]; i++)
        {
            for(int j = 0; j < nx[1]; j++)
            {
                for(int k = 0; k < nx[2]; k++)
                {
                    int i_j_k = i * nx[1] * nx[2] + j * nx[2] + k;
                    temp3[k] = q_dstar[i_j_k];
                }

                int i_j = i * nx[1] * nx[2] + j * nx[2];
                if(bc[4] == BoundaryCondition::PERIODIC)
                    CpuSolverCNADI::tridiagonal_periodic(_zl, _zd, _zh, &q_out[i_j], 1, temp3, nx[2]);
                else
                    CpuSolverCNADI::tridiagonal(_zl, _zd, _zh, &q_out[i_j], 1, temp3, nx[2]);
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverSDC::adi_step_2d(int sub_interval,
    std::vector<BoundaryCondition> bc,
    double* q_in, double* q_out, std::string monomer_type)
{
    try
    {
        const int n_grid = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();
        double q_star[n_grid];
        double temp1[nx[0]];
        double temp2[nx[1]];

        double *_xl = xl[sub_interval][monomer_type];
        double *_xd = xd[sub_interval][monomer_type];
        double *_xh = xh[sub_interval][monomer_type];

        double *_yl = yl[sub_interval][monomer_type];
        double *_yd = yd[sub_interval][monomer_type];
        double *_yh = yh[sub_interval][monomer_type];

        // Backward Euler ADI: Solve (I - dt*Dx)(I - dt*Dy) q = rhs
        // Step 1: Solve (I - dt*Dx) q_star = rhs along x-direction
        // Step 2: Solve (I - dt*Dy) q_out = q_star along y-direction

        // X-sweep: solve A_x * q_star = q_in for each row
        for(int j = 0; j < nx[1]; j++)
        {
            for(int i = 0; i < nx[0]; i++)
            {
                int i_j = i * nx[1] + j;
                temp1[i] = q_in[i_j];
            }

            if(bc[0] == BoundaryCondition::PERIODIC)
                CpuSolverCNADI::tridiagonal_periodic(_xl, _xd, _xh, &q_star[j], nx[1], temp1, nx[0]);
            else
                CpuSolverCNADI::tridiagonal(_xl, _xd, _xh, &q_star[j], nx[1], temp1, nx[0]);
        }

        // Y-sweep: solve A_y * q_out = q_star for each column
        for(int i = 0; i < nx[0]; i++)
        {
            for(int j = 0; j < nx[1]; j++)
            {
                int i_j = i * nx[1] + j;
                temp2[j] = q_star[i_j];
            }

            if(bc[2] == BoundaryCondition::PERIODIC)
                CpuSolverCNADI::tridiagonal_periodic(_yl, _yd, _yh, &q_out[i * nx[1]], 1, temp2, nx[1]);
            else
                CpuSolverCNADI::tridiagonal(_yl, _yd, _yh, &q_out[i * nx[1]], 1, temp2, nx[1]);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverSDC::adi_step_1d(int sub_interval,
    std::vector<BoundaryCondition> bc,
    double* q_in, double* q_out, std::string monomer_type)
{
    try
    {
        const std::vector<int> nx = this->cb->get_nx();

        double *_xl = xl[sub_interval][monomer_type];
        double *_xd = xd[sub_interval][monomer_type];
        double *_xh = xh[sub_interval][monomer_type];

        // For Backward Euler: solve A * q_out = q_in directly
        // No transformation of RHS (unlike Crank-Nicolson)
        if(bc[0] == BoundaryCondition::PERIODIC)
            CpuSolverCNADI::tridiagonal_periodic(_xl, _xd, _xh, q_out, 1, q_in, nx[0]);
        else
            CpuSolverCNADI::tridiagonal(_xl, _xd, _xh, q_out, 1, q_in, nx[0]);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverSDC::build_sparse_matrix(int sub_interval, std::string monomer_type)
{
    try
    {
        const int DIM = this->cb->get_dim();
        const int n_grid = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();
        const std::vector<double> dx = this->cb->get_dx();
        const std::vector<BoundaryCondition> bc = this->cb->get_boundary_conditions();
        const double ds = this->molecules->get_ds();

        double bond_length = this->molecules->get_bond_lengths().at(monomer_type);
        double bond_length_sq = bond_length * bond_length;
        const double D = bond_length_sq / 6.0;
        double dtau = (tau[sub_interval + 1] - tau[sub_interval]) * ds;

        SparseMatrixCSR& mat = sparse_matrices[sub_interval][monomer_type];

        // Build A = I - dtau * D * ∇² in CSR format (0-based indexing for MKL sparse BLAS)
        // For 2D: 5-point stencil (1 diagonal + 4 off-diagonals)
        // For 3D: 7-point stencil (1 diagonal + 6 off-diagonals)
        int stencil_size = (DIM == 2) ? 5 : 7;
        int max_nnz = n_grid * stencil_size;

        mat.row_ptr.resize(n_grid + 1);
        mat.col_idx.reserve(max_nnz);
        mat.values.reserve(max_nnz);
        mat.diag_inv.resize(n_grid);  // For Jacobi preconditioner
        mat.col_idx.clear();
        mat.values.clear();

        if(DIM == 2)
        {
            double rx = D * dtau / (dx[0] * dx[0]);
            double ry = D * dtau / (dx[1] * dx[1]);

            MKL_INT nnz_count = 0;
            for(int i = 0; i < nx[0]; i++)
            {
                int im = (bc[0] == BoundaryCondition::PERIODIC) ? (nx[0] + i - 1) % nx[0] : max_of_two(0, i - 1);
                int ip = (bc[1] == BoundaryCondition::PERIODIC) ? (i + 1) % nx[0] : min_of_two(nx[0] - 1, i + 1);

                for(int j = 0; j < nx[1]; j++)
                {
                    int jm = (bc[2] == BoundaryCondition::PERIODIC) ? (nx[1] + j - 1) % nx[1] : max_of_two(0, j - 1);
                    int jp = (bc[3] == BoundaryCondition::PERIODIC) ? (j + 1) % nx[1] : min_of_two(nx[1] - 1, j + 1);

                    int row = i * nx[1] + j;
                    mat.row_ptr[row] = nnz_count;  // 0-based

                    // Store entries in column order
                    std::vector<std::pair<int, double>> entries;

                    // Left neighbor (im, j) - skip if absorbing BC at left boundary
                    if(!(bc[0] == BoundaryCondition::ABSORBING && i == 0))
                    {
                        int col_im = im * nx[1] + j;
                        if(col_im != row)
                            entries.push_back({col_im, -rx});
                    }

                    // Bottom neighbor (i, jm) - skip if absorbing BC at bottom boundary
                    if(!(bc[2] == BoundaryCondition::ABSORBING && j == 0))
                    {
                        int col_jm = i * nx[1] + jm;
                        if(col_jm != row)
                            entries.push_back({col_jm, -ry});
                    }

                    // Diagonal (i, j)
                    double diag_val = 1.0 + 2.0 * rx + 2.0 * ry;
                    // Cell-centered boundary modifications using ghost cells:
                    // - Reflecting: symmetric ghost (q_{-1} = q_0) → diagonal decreases
                    // - Absorbing: antisymmetric ghost (q_{-1} = -q_0) → diagonal increases
                    if(bc[0] == BoundaryCondition::REFLECTING && i == 0)
                        diag_val -= rx;  // left boundary: symmetric
                    else if(bc[0] == BoundaryCondition::ABSORBING && i == 0)
                        diag_val += rx;  // left boundary: antisymmetric
                    if(bc[1] == BoundaryCondition::REFLECTING && i == nx[0] - 1)
                        diag_val -= rx;  // right boundary: symmetric
                    else if(bc[1] == BoundaryCondition::ABSORBING && i == nx[0] - 1)
                        diag_val += rx;  // right boundary: antisymmetric
                    if(bc[2] == BoundaryCondition::REFLECTING && j == 0)
                        diag_val -= ry;  // bottom boundary: symmetric
                    else if(bc[2] == BoundaryCondition::ABSORBING && j == 0)
                        diag_val += ry;  // bottom boundary: antisymmetric
                    if(bc[3] == BoundaryCondition::REFLECTING && j == nx[1] - 1)
                        diag_val -= ry;  // top boundary: symmetric
                    else if(bc[3] == BoundaryCondition::ABSORBING && j == nx[1] - 1)
                        diag_val += ry;  // top boundary: antisymmetric
                    entries.push_back({row, diag_val});

                    // Store inverse diagonal for Jacobi preconditioner
                    mat.diag_inv[row] = 1.0 / diag_val;

                    // Top neighbor (i, jp) - skip if absorbing BC at top boundary
                    if(!(bc[3] == BoundaryCondition::ABSORBING && j == nx[1] - 1))
                    {
                        int col_jp = i * nx[1] + jp;
                        if(col_jp != row)
                            entries.push_back({col_jp, -ry});
                    }

                    // Right neighbor (ip, j) - skip if absorbing BC at right boundary
                    if(!(bc[1] == BoundaryCondition::ABSORBING && i == nx[0] - 1))
                    {
                        int col_ip = ip * nx[1] + j;
                        if(col_ip != row)
                            entries.push_back({col_ip, -rx});
                    }

                    // Sort by column index
                    std::sort(entries.begin(), entries.end());

                    // Add to CSR (0-based indexing)
                    for(const auto& e: entries)
                    {
                        mat.col_idx.push_back(e.first);  // 0-based
                        mat.values.push_back(e.second);
                        nnz_count++;
                    }
                }
            }
            mat.row_ptr[n_grid] = nnz_count;
            mat.nnz = nnz_count;
        }
        else // DIM == 3
        {
            double rx = D * dtau / (dx[0] * dx[0]);
            double ry = D * dtau / (dx[1] * dx[1]);
            double rz = D * dtau / (dx[2] * dx[2]);

            MKL_INT nnz_count = 0;
            for(int i = 0; i < nx[0]; i++)
            {
                int im = (bc[0] == BoundaryCondition::PERIODIC) ? (nx[0] + i - 1) % nx[0] : max_of_two(0, i - 1);
                int ip = (bc[1] == BoundaryCondition::PERIODIC) ? (i + 1) % nx[0] : min_of_two(nx[0] - 1, i + 1);

                for(int j = 0; j < nx[1]; j++)
                {
                    int jm = (bc[2] == BoundaryCondition::PERIODIC) ? (nx[1] + j - 1) % nx[1] : max_of_two(0, j - 1);
                    int jp = (bc[3] == BoundaryCondition::PERIODIC) ? (j + 1) % nx[1] : min_of_two(nx[1] - 1, j + 1);

                    for(int k = 0; k < nx[2]; k++)
                    {
                        int km = (bc[4] == BoundaryCondition::PERIODIC) ? (nx[2] + k - 1) % nx[2] : max_of_two(0, k - 1);
                        int kp = (bc[5] == BoundaryCondition::PERIODIC) ? (k + 1) % nx[2] : min_of_two(nx[2] - 1, k + 1);

                        int row = i * nx[1] * nx[2] + j * nx[2] + k;
                        mat.row_ptr[row] = nnz_count;  // 0-based

                        // Store entries in column order
                        std::vector<std::pair<int, double>> entries;

                        // Left neighbor (im, j, k) - skip if absorbing BC at left boundary
                        if(!(bc[0] == BoundaryCondition::ABSORBING && i == 0))
                        {
                            int col_im = im * nx[1] * nx[2] + j * nx[2] + k;
                            if(col_im != row)
                                entries.push_back({col_im, -rx});
                        }

                        // Back neighbor (i, jm, k) - skip if absorbing BC at back boundary
                        if(!(bc[2] == BoundaryCondition::ABSORBING && j == 0))
                        {
                            int col_jm = i * nx[1] * nx[2] + jm * nx[2] + k;
                            if(col_jm != row)
                                entries.push_back({col_jm, -ry});
                        }

                        // Bottom neighbor (i, j, km) - skip if absorbing BC at bottom boundary
                        if(!(bc[4] == BoundaryCondition::ABSORBING && k == 0))
                        {
                            int col_km = i * nx[1] * nx[2] + j * nx[2] + km;
                            if(col_km != row)
                                entries.push_back({col_km, -rz});
                        }

                        // Diagonal (i, j, k)
                        double diag_val = 1.0 + 2.0 * rx + 2.0 * ry + 2.0 * rz;
                        // Cell-centered boundary modifications using ghost cells:
                        // - Reflecting: symmetric ghost (q_{-1} = q_0) → diagonal decreases
                        // - Absorbing: antisymmetric ghost (q_{-1} = -q_0) → diagonal increases
                        if(bc[0] == BoundaryCondition::REFLECTING && i == 0)
                            diag_val -= rx;
                        else if(bc[0] == BoundaryCondition::ABSORBING && i == 0)
                            diag_val += rx;
                        if(bc[1] == BoundaryCondition::REFLECTING && i == nx[0] - 1)
                            diag_val -= rx;
                        else if(bc[1] == BoundaryCondition::ABSORBING && i == nx[0] - 1)
                            diag_val += rx;
                        if(bc[2] == BoundaryCondition::REFLECTING && j == 0)
                            diag_val -= ry;
                        else if(bc[2] == BoundaryCondition::ABSORBING && j == 0)
                            diag_val += ry;
                        if(bc[3] == BoundaryCondition::REFLECTING && j == nx[1] - 1)
                            diag_val -= ry;
                        else if(bc[3] == BoundaryCondition::ABSORBING && j == nx[1] - 1)
                            diag_val += ry;
                        if(bc[4] == BoundaryCondition::REFLECTING && k == 0)
                            diag_val -= rz;
                        else if(bc[4] == BoundaryCondition::ABSORBING && k == 0)
                            diag_val += rz;
                        if(bc[5] == BoundaryCondition::REFLECTING && k == nx[2] - 1)
                            diag_val -= rz;
                        else if(bc[5] == BoundaryCondition::ABSORBING && k == nx[2] - 1)
                            diag_val += rz;
                        entries.push_back({row, diag_val});

                        // Store inverse diagonal for Jacobi preconditioner
                        mat.diag_inv[row] = 1.0 / diag_val;

                        // Top neighbor (i, j, kp) - skip if absorbing BC at top boundary
                        if(!(bc[5] == BoundaryCondition::ABSORBING && k == nx[2] - 1))
                        {
                            int col_kp = i * nx[1] * nx[2] + j * nx[2] + kp;
                            if(col_kp != row)
                                entries.push_back({col_kp, -rz});
                        }

                        // Front neighbor (i, jp, k) - skip if absorbing BC at front boundary
                        if(!(bc[3] == BoundaryCondition::ABSORBING && j == nx[1] - 1))
                        {
                            int col_jp = i * nx[1] * nx[2] + jp * nx[2] + k;
                            if(col_jp != row)
                                entries.push_back({col_jp, -ry});
                        }

                        // Right neighbor (ip, j, k) - skip if absorbing BC at right boundary
                        if(!(bc[1] == BoundaryCondition::ABSORBING && i == nx[0] - 1))
                        {
                            int col_ip = ip * nx[1] * nx[2] + j * nx[2] + k;
                            if(col_ip != row)
                                entries.push_back({col_ip, -rx});
                        }

                        // Sort by column index
                        std::sort(entries.begin(), entries.end());

                        // Add to CSR (0-based indexing)
                        for(const auto& e: entries)
                        {
                            mat.col_idx.push_back(e.first);  // 0-based
                            mat.values.push_back(e.second);
                            nnz_count++;
                        }
                    }
                }
            }
            mat.row_ptr[n_grid] = nnz_count;
            mat.nnz = nnz_count;
        }

        mat.built = true;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverSDC::sparse_matvec(const SparseMatrixCSR& mat, const double* x, double* y)
{
    // Compute y = A * x using CSR format (0-based indexing)
    const int n = mat.n;
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
    {
        double sum = 0.0;
        for(MKL_INT j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; j++)
        {
            sum += mat.values[j] * x[mat.col_idx[j]];
        }
        y[i] = sum;
    }
}

void CpuSolverSDC::sparse_solve(int sub_interval, double* q_in, double* q_out, std::string monomer_type)
{
    try
    {
        SparseMatrixCSR& mat = sparse_matrices[sub_interval][monomer_type];

        // Build matrix if not yet done (MUST be done BEFORE accessing mat.n)
        if(!mat.built)
        {
            build_sparse_matrix(sub_interval, monomer_type);
        }

        const int n = mat.n;

        // Get thread-local workspace (this function is called from parallel region)
        int tid = omp_get_thread_num();
        double* _pcg_r = pcg_r[tid];
        double* _pcg_z = pcg_z[tid];
        double* _pcg_p = pcg_p[tid];
        double* _pcg_Ap = pcg_Ap[tid];

        // Preconditioned Conjugate Gradient (PCG) with Jacobi preconditioner
        // Solve A * q_out = q_in where A is SPD
        // Note: No nested OpenMP since this is already called from parallel region

        // Initial guess: x0 = 0, so r0 = b = q_in
        double rz_old = 0.0;
        for(int i = 0; i < n; i++)
        {
            q_out[i] = 0.0;
            _pcg_r[i] = q_in[i];
            _pcg_z[i] = _pcg_r[i] * mat.diag_inv[i];  // Jacobi preconditioner
            _pcg_p[i] = _pcg_z[i];
            rz_old += _pcg_r[i] * _pcg_z[i];
        }

        for(int iter = 0; iter < pcg_max_iter; iter++)
        {
            // Ap = A * p (no nested OMP)
            for(int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for(MKL_INT j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; j++)
                {
                    sum += mat.values[j] * _pcg_p[mat.col_idx[j]];
                }
                _pcg_Ap[i] = sum;
            }

            // pAp = (p, Ap)
            double pAp = 0.0;
            for(int i = 0; i < n; i++)
            {
                pAp += _pcg_p[i] * _pcg_Ap[i];
            }

            double alpha = rz_old / pAp;

            // x_{k+1} = x_k + alpha * p
            // r_{k+1} = r_k - alpha * Ap
            double r_norm_sq = 0.0;
            for(int i = 0; i < n; i++)
            {
                q_out[i] += alpha * _pcg_p[i];
                _pcg_r[i] -= alpha * _pcg_Ap[i];
                r_norm_sq += _pcg_r[i] * _pcg_r[i];
            }

            // Check convergence: ||r||
            if(std::sqrt(r_norm_sq) < pcg_tol)
                break;

            // z_{k+1} = M^{-1} * r_{k+1}
            // rz_new = (r_{k+1}, z_{k+1})
            double rz_new = 0.0;
            for(int i = 0; i < n; i++)
            {
                _pcg_z[i] = _pcg_r[i] * mat.diag_inv[i];
                rz_new += _pcg_r[i] * _pcg_z[i];
            }

            double beta = rz_new / rz_old;

            // p_{k+1} = z_{k+1} + beta * p_k
            for(int i = 0; i < n; i++)
            {
                _pcg_p[i] = _pcg_z[i] + beta * _pcg_p[i];
            }

            rz_old = rz_new;
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverSDC::direct_solve_step(int sub_interval, double* q_in, double* q_out, std::string monomer_type)
{
    // Solve using PCG (no ADI splitting error)
    sparse_solve(sub_interval, q_in, q_out, monomer_type);
}

void CpuSolverSDC::advance_propagator(
    double *q_in, double *q_out, std::string monomer_type,
    const double *q_mask, int /*ds_index*/)
{
    try
    {
        const int n_grid = this->cb->get_total_grid();
        const double ds = this->molecules->get_ds();

        // Get thread-local workspace (this function may be called from parallel region)
        int tid = omp_get_thread_num();
        std::vector<double*>& _X = X[tid];
        std::vector<double*>& _X_old = X_old[tid];
        std::vector<double*>& _F_diff = F_diff[tid];
        std::vector<double*>& _F_old = F_old[tid];
        double* _temp_array = temp_array[tid];
        double* _rhs_array = rhs_array[tid];

        // Get stored w field for this monomer type
        const std::vector<double>& w_field = w_field_store[monomer_type];

        // Initialize _X[0] = q_in
        for(int i = 0; i < n_grid; i++)
            _X[0][i] = q_in[i];

        //=================================================================
        // Predictor: Backward Euler for diffusion, explicit for reaction
        //=================================================================
        for(int m = 0; m < M - 1; m++)
        {
            // Apply reaction term explicitly: temp = exp(-w*dtau) * _X[m]
            const std::vector<double>& exp_dw_vec = exp_dw_sub[m][monomer_type];
            for(int i = 0; i < n_grid; i++)
                _temp_array[i] = exp_dw_vec[i] * _X[m][i];

            // Apply diffusion implicitly using ADI
            adi_step(m, _temp_array, _X[m + 1], monomer_type);
        }

        //=================================================================
        // SDC Corrections (K iterations)
        //=================================================================
        for(int k_iter = 0; k_iter < K; k_iter++)
        {
            // Compute F at all GL nodes: F = D∇²q - wq
            for(int m = 0; m < M; m++)
            {
                compute_F(_X[m], w_field.data(), _F_diff[m], monomer_type);
            }

            // Store old X and F values
            for(int m = 0; m < M; m++)
            {
                for(int i = 0; i < n_grid; i++)
                {
                    _X_old[m][i] = _X[m][i];
                    _F_old[m][i] = _F_diff[m][i];
                }
            }

            // Reset _X[0]
            for(int i = 0; i < n_grid; i++)
                _X[0][i] = q_in[i];

            // SDC correction sweep
            for(int m = 0; m < M - 1; m++)
            {
                double dtau = (tau[m + 1] - tau[m]) * ds;

                // Compute spectral integral: ∫ F dt
                for(int i = 0; i < n_grid; i++)
                {
                    double integral = 0.0;
                    for(int j = 0; j < M; j++)
                    {
                        integral += S[m][j] * _F_old[j][i] * ds;
                    }
                    // Note: F_old = D∇²q - wq, so D∇²q = F_old + w*q
                    // We subtract only the diffusion part: dtau * D∇²q_old[m+1]
                    double diff_term = _F_old[m + 1][i] + w_field[i] * _X_old[m + 1][i];
                    _rhs_array[i] = _X[m][i] + integral - dtau * diff_term;
                }

                // Semi-implicit SDC: solve (I - dtau*D∇²) X[m+1] = rhs
                adi_step(m, _rhs_array, _X[m + 1], monomer_type);
            }
        }

        // Output is the final GL node
        for(int i = 0; i < n_grid; i++)
            q_out[i] = _X[M - 1][i];

        // Apply mask if provided
        if(q_mask != nullptr)
        {
            for(int i = 0; i < n_grid; i++)
                q_out[i] *= q_mask[i];
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

std::vector<double> CpuSolverSDC::compute_single_segment_stress(
    [[maybe_unused]] double *q_1, [[maybe_unused]] double *q_2,
    [[maybe_unused]] std::string monomer_type, [[maybe_unused]] bool is_half_bond_length)
{
    try
    {
        const int DIM = this->cb->get_dim();
        std::vector<double> stress(DIM);

        throw_with_line_number("SDC method does not support stress computation.");

        return stress;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
