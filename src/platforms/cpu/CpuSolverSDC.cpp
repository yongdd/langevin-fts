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

        // Allocate workspace arrays for GL node solutions
        X.resize(M);
        F_diff.resize(M);
        F_react.resize(M);
        for(int m = 0; m < M; m++)
        {
            X[m] = new double[n_grid];
            F_diff[m] = new double[n_grid];
            F_react[m] = new double[n_grid];
        }

        temp_array = new double[n_grid];
        rhs_array = new double[n_grid];

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

    // Free workspace
    for(int m = 0; m < M; m++)
    {
        delete[] X[m];
        delete[] F_diff[m];
        delete[] F_react[m];
    }

    delete[] temp_array;
    delete[] rhs_array;
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
    else
    {
        // General formula using Chebyshev nodes of second kind
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
            for(int m = 0; m < M - 1; m++)
            {
                double dtau = (tau[m + 1] - tau[m]) * ds;

                FiniteDifference::get_laplacian_matrix(
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
            int im = (bc[0] == BoundaryCondition::PERIODIC) ? (nx[0] + i - 1) % nx[0] : max_of_two(0, i - 1);
            int ip = (bc[1] == BoundaryCondition::PERIODIC) ? (i + 1) % nx[0] : min_of_two(nx[0] - 1, i + 1);

            double Lx = alpha_x * (q[im] + q[ip] - 2.0 * q[i]);
            F_out[i] = Lx - w[i] * q[i];
        }
    }
    else if(DIM == 2)
    {
        double alpha_x = D / (dx[0] * dx[0]);
        double alpha_y = D / (dx[1] * dx[1]);

        for(int i = 0; i < nx[0]; i++)
        {
            int im = (bc[0] == BoundaryCondition::PERIODIC) ? (nx[0] + i - 1) % nx[0] : max_of_two(0, i - 1);
            int ip = (bc[1] == BoundaryCondition::PERIODIC) ? (i + 1) % nx[0] : min_of_two(nx[0] - 1, i + 1);

            for(int j = 0; j < nx[1]; j++)
            {
                int jm = (bc[2] == BoundaryCondition::PERIODIC) ? (nx[1] + j - 1) % nx[1] : max_of_two(0, j - 1);
                int jp = (bc[3] == BoundaryCondition::PERIODIC) ? (j + 1) % nx[1] : min_of_two(nx[1] - 1, j + 1);

                int idx = i * nx[1] + j;
                int idx_im = im * nx[1] + j;
                int idx_ip = ip * nx[1] + j;
                int idx_jm = i * nx[1] + jm;
                int idx_jp = i * nx[1] + jp;

                double Lx = alpha_x * (q[idx_im] + q[idx_ip] - 2.0 * q[idx]);
                double Ly = alpha_y * (q[idx_jm] + q[idx_jp] - 2.0 * q[idx]);
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

                    int idx = i * nx[1] * nx[2] + j * nx[2] + k;
                    int idx_im = im * nx[1] * nx[2] + j * nx[2] + k;
                    int idx_ip = ip * nx[1] * nx[2] + j * nx[2] + k;
                    int idx_jm = i * nx[1] * nx[2] + jm * nx[2] + k;
                    int idx_jp = i * nx[1] * nx[2] + jp * nx[2] + k;
                    int idx_km = i * nx[1] * nx[2] + j * nx[2] + km;
                    int idx_kp = i * nx[1] * nx[2] + j * nx[2] + kp;

                    double Lx = alpha_x * (q[idx_im] + q[idx_ip] - 2.0 * q[idx]);
                    double Ly = alpha_y * (q[idx_jm] + q[idx_jp] - 2.0 * q[idx]);
                    double Lz = alpha_z * (q[idx_km] + q[idx_kp] - 2.0 * q[idx]);
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

    if(DIM == 3)
        adi_step_3d(sub_interval, bc, q_in, q_out, monomer_type);
    else if(DIM == 2)
        adi_step_2d(sub_interval, bc, q_in, q_out, monomer_type);
    else if(DIM == 1)
        adi_step_1d(sub_interval, bc, q_in, q_out, monomer_type);
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

        int im, ip, jm, jp, km, kp;

        // Calculate q_star (X-sweep)
        for(int j = 0; j < nx[1]; j++)
        {
            if(bc[2] == BoundaryCondition::PERIODIC)
                jm = (nx[1] + j - 1) % nx[1];
            else
                jm = max_of_two(0, j - 1);
            if(bc[3] == BoundaryCondition::PERIODIC)
                jp = (j + 1) % nx[1];
            else
                jp = min_of_two(nx[1] - 1, j + 1);

            for(int k = 0; k < nx[2]; k++)
            {
                if(bc[4] == BoundaryCondition::PERIODIC)
                    km = (nx[2] + k - 1) % nx[2];
                else
                    km = max_of_two(0, k - 1);
                if(bc[5] == BoundaryCondition::PERIODIC)
                    kp = (k + 1) % nx[2];
                else
                    kp = min_of_two(nx[2] - 1, k + 1);

                for(int i = 0; i < nx[0]; i++)
                {
                    if(bc[0] == BoundaryCondition::PERIODIC)
                        im = (nx[0] + i - 1) % nx[0];
                    else
                        im = max_of_two(0, i - 1);
                    if(bc[1] == BoundaryCondition::PERIODIC)
                        ip = (i + 1) % nx[0];
                    else
                        ip = min_of_two(nx[0] - 1, i + 1);

                    int i_j_k = i * nx[1] * nx[2] + j * nx[2] + k;
                    int im_j_k = im * nx[1] * nx[2] + j * nx[2] + k;
                    int ip_j_k = ip * nx[1] * nx[2] + j * nx[2] + k;
                    int i_jm_k = i * nx[1] * nx[2] + jm * nx[2] + k;
                    int i_jp_k = i * nx[1] * nx[2] + jp * nx[2] + k;
                    int i_j_km = i * nx[1] * nx[2] + j * nx[2] + km;
                    int i_j_kp = i * nx[1] * nx[2] + j * nx[2] + kp;

                    temp1[i] = 2.0 * ((3.0 - 0.5 * _xd[i] - _yd[j] - _zd[k]) * q_in[i_j_k]
                            - _zl[k] * q_in[i_j_km] - _zh[k] * q_in[i_j_kp]
                            - _yl[j] * q_in[i_jm_k] - _yh[j] * q_in[i_jp_k])
                            - _xl[i] * q_in[im_j_k] - _xh[i] * q_in[ip_j_k];
                }

                int j_k = j * nx[2] + k;
                if(bc[0] == BoundaryCondition::PERIODIC)
                    CpuSolverCNADI::tridiagonal_periodic(_xl, _xd, _xh, &q_star[j_k], nx[1] * nx[2], temp1, nx[0]);
                else
                    CpuSolverCNADI::tridiagonal(_xl, _xd, _xh, &q_star[j_k], nx[1] * nx[2], temp1, nx[0]);
            }
        }

        // Calculate q_dstar (Y-sweep)
        for(int i = 0; i < nx[0]; i++)
        {
            for(int k = 0; k < nx[2]; k++)
            {
                for(int j = 0; j < nx[1]; j++)
                {
                    if(bc[2] == BoundaryCondition::PERIODIC)
                        jm = (nx[1] + j - 1) % nx[1];
                    else
                        jm = max_of_two(0, j - 1);
                    if(bc[3] == BoundaryCondition::PERIODIC)
                        jp = (j + 1) % nx[1];
                    else
                        jp = min_of_two(nx[1] - 1, j + 1);

                    int i_j_k = i * nx[1] * nx[2] + j * nx[2] + k;
                    int i_jm_k = i * nx[1] * nx[2] + jm * nx[2] + k;
                    int i_jp_k = i * nx[1] * nx[2] + jp * nx[2] + k;

                    temp2[j] = q_star[i_j_k] + (_yd[j] - 1.0) * q_in[i_j_k]
                        + _yl[j] * q_in[i_jm_k] + _yh[j] * q_in[i_jp_k];
                }

                int i_k = i * nx[1] * nx[2] + k;
                if(bc[2] == BoundaryCondition::PERIODIC)
                    CpuSolverCNADI::tridiagonal_periodic(_yl, _yd, _yh, &q_dstar[i_k], nx[2], temp2, nx[1]);
                else
                    CpuSolverCNADI::tridiagonal(_yl, _yd, _yh, &q_dstar[i_k], nx[2], temp2, nx[1]);
            }
        }

        // Calculate q_out (Z-sweep)
        for(int i = 0; i < nx[0]; i++)
        {
            for(int j = 0; j < nx[1]; j++)
            {
                for(int k = 0; k < nx[2]; k++)
                {
                    if(bc[4] == BoundaryCondition::PERIODIC)
                        km = (nx[2] + k - 1) % nx[2];
                    else
                        km = max_of_two(0, k - 1);
                    if(bc[5] == BoundaryCondition::PERIODIC)
                        kp = (k + 1) % nx[2];
                    else
                        kp = min_of_two(nx[2] - 1, k + 1);

                    int i_j_k = i * nx[1] * nx[2] + j * nx[2] + k;
                    int i_j_km = i * nx[1] * nx[2] + j * nx[2] + km;
                    int i_j_kp = i * nx[1] * nx[2] + j * nx[2] + kp;

                    temp3[k] = q_dstar[i_j_k] + (_zd[k] - 1.0) * q_in[i_j_k]
                        + _zl[k] * q_in[i_j_km] + _zh[k] * q_in[i_j_kp];
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

        int im, ip, jm, jp;

        // Calculate q_star (X-sweep)
        for(int j = 0; j < nx[1]; j++)
        {
            if(bc[2] == BoundaryCondition::PERIODIC)
                jm = (nx[1] + j - 1) % nx[1];
            else
                jm = max_of_two(0, j - 1);
            if(bc[3] == BoundaryCondition::PERIODIC)
                jp = (j + 1) % nx[1];
            else
                jp = min_of_two(nx[1] - 1, j + 1);

            for(int i = 0; i < nx[0]; i++)
            {
                if(bc[0] == BoundaryCondition::PERIODIC)
                    im = (nx[0] + i - 1) % nx[0];
                else
                    im = max_of_two(0, i - 1);
                if(bc[1] == BoundaryCondition::PERIODIC)
                    ip = (i + 1) % nx[0];
                else
                    ip = min_of_two(nx[0] - 1, i + 1);

                int i_j = i * nx[1] + j;
                int i_jm = i * nx[1] + jm;
                int i_jp = i * nx[1] + jp;
                int im_j = im * nx[1] + j;
                int ip_j = ip * nx[1] + j;

                temp1[i] = 2.0 * ((2.0 - 0.5 * _xd[i] - _yd[j]) * q_in[i_j]
                          - _yl[j] * q_in[i_jm] - _yh[j] * q_in[i_jp])
                          - _xl[i] * q_in[im_j] - _xh[i] * q_in[ip_j];
            }

            if(bc[0] == BoundaryCondition::PERIODIC)
                CpuSolverCNADI::tridiagonal_periodic(_xl, _xd, _xh, &q_star[j], nx[1], temp1, nx[0]);
            else
                CpuSolverCNADI::tridiagonal(_xl, _xd, _xh, &q_star[j], nx[1], temp1, nx[0]);
        }

        // Calculate q_out (Y-sweep)
        for(int i = 0; i < nx[0]; i++)
        {
            for(int j = 0; j < nx[1]; j++)
            {
                if(bc[2] == BoundaryCondition::PERIODIC)
                    jm = (nx[1] + j - 1) % nx[1];
                else
                    jm = max_of_two(0, j - 1);
                if(bc[3] == BoundaryCondition::PERIODIC)
                    jp = (j + 1) % nx[1];
                else
                    jp = min_of_two(nx[1] - 1, j + 1);

                int i_j = i * nx[1] + j;
                int i_jm = i * nx[1] + jm;
                int i_jp = i * nx[1] + jp;

                temp2[j] = q_star[i_j] + (_yd[j] - 1.0) * q_in[i_j]
                    + _yl[j] * q_in[i_jm] + _yh[j] * q_in[i_jp];
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
        double q_star[nx[0]];

        double *_xl = xl[sub_interval][monomer_type];
        double *_xd = xd[sub_interval][monomer_type];
        double *_xh = xh[sub_interval][monomer_type];

        int im, ip;

        for(int i = 0; i < nx[0]; i++)
        {
            if(bc[0] == BoundaryCondition::PERIODIC)
                im = (nx[0] + i - 1) % nx[0];
            else
                im = max_of_two(0, i - 1);
            if(bc[1] == BoundaryCondition::PERIODIC)
                ip = (i + 1) % nx[0];
            else
                ip = min_of_two(nx[0] - 1, i + 1);

            q_star[i] = (2.0 - _xd[i]) * q_in[i] - _xl[i] * q_in[im] - _xh[i] * q_in[ip];
        }

        if(bc[0] == BoundaryCondition::PERIODIC)
            CpuSolverCNADI::tridiagonal_periodic(_xl, _xd, _xh, q_out, 1, q_star, nx[0]);
        else
            CpuSolverCNADI::tridiagonal(_xl, _xd, _xh, q_out, 1, q_star, nx[0]);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverSDC::advance_propagator(
    double *q_in, double *q_out, std::string monomer_type,
    const double *q_mask, int /*ds_index*/)
{
    try
    {
        const int n_grid = this->cb->get_total_grid();
        const double ds = this->molecules->get_ds();

        // Get stored w field for this monomer type
        // We need the w field to compute F = D∇²q - wq
        // The w field is implicitly stored via exp_dw_sub
        // We need to recover it: w = -log(exp_dw_sub) / dtau
        // For simplicity, we'll use a temporary approach

        // Initialize X[0] = q_in
        for(int i = 0; i < n_grid; i++)
            X[0][i] = q_in[i];

        //=================================================================
        // Predictor: Backward Euler for diffusion, explicit for reaction
        //=================================================================
        for(int m = 0; m < M - 1; m++)
        {
            // Apply reaction term explicitly: temp = exp(-w*dtau) * X[m]
            const std::vector<double>& exp_dw_vec = exp_dw_sub[m][monomer_type];
            for(int i = 0; i < n_grid; i++)
                temp_array[i] = exp_dw_vec[i] * X[m][i];

            // Apply diffusion implicitly using ADI
            adi_step(m, temp_array, X[m + 1], monomer_type);
        }

        //=================================================================
        // SDC Corrections (K iterations)
        //=================================================================
        // Use directly stored w field (like CUDA) to avoid numerical errors
        // from recovering w via -log(exp(-w*dtau))/dtau
        const std::vector<double>& w_field = w_field_store[monomer_type];

        for(int k_iter = 0; k_iter < K; k_iter++)
        {
            // Compute F at all GL nodes: F = D∇²q - wq
            for(int m = 0; m < M; m++)
            {
                compute_F(X[m], w_field.data(), F_diff[m], monomer_type);
            }

            // Store old X values
            double X_old[M][n_grid];
            double F_diff_old[M][n_grid];
            for(int m = 0; m < M; m++)
            {
                for(int i = 0; i < n_grid; i++)
                {
                    X_old[m][i] = X[m][i];
                    F_diff_old[m][i] = F_diff[m][i];
                }
            }

            // Reset X[0]
            for(int i = 0; i < n_grid; i++)
                X[0][i] = q_in[i];

            // SDC correction sweep
            for(int m = 0; m < M - 1; m++)
            {
                double dtau = (tau[m + 1] - tau[m]) * ds;

                // Compute spectral integral: ∫ F dt
                // F_diff_old contains FULL F = D∇²q - wq
                // For IMEX-SDC, we need:
                //   rhs = X[m] + ∫F dt - dtau * D∇²q_old[m+1]
                // Since F = D∇²q - wq, we have: D∇²q = F + wq
                // So: rhs = X[m] + ∫F dt - dtau * (F_old[m+1] + w*X_old[m+1])
                //         = X[m] + ∫F dt - dtau * F_old[m+1] - dtau * w * X_old[m+1]
                // But we want the reaction term in the RHS, so:
                //   rhs = X[m] + ∫F dt - dtau * D∇²q_old[m+1]
                //       = X[m] + ∫F dt - dtau * (F_old[m+1] + w * X_old[m+1])
                for(int i = 0; i < n_grid; i++)
                {
                    double integral = 0.0;
                    for(int j = 0; j < M; j++)
                    {
                        integral += S[m][j] * F_diff_old[j][i] * ds;
                    }
                    // Note: F_diff_old = D∇²q - wq, so D∇²q = F_diff_old + w*q
                    // We subtract only the diffusion part: dtau * D∇²q_old[m+1]
                    double diff_term = F_diff_old[m + 1][i] + w_field[i] * X_old[m + 1][i];
                    rhs_array[i] = X[m][i] + integral - dtau * diff_term;
                }

                // Semi-implicit SDC: solve (I - dtau*D∇²) X[m+1] = rhs
                adi_step(m, rhs_array, X[m + 1], monomer_type);
            }
        }

        // Output is the final GL node
        for(int i = 0; i < n_grid; i++)
            q_out[i] = X[M - 1][i];

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
