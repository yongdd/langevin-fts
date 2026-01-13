/**
 * @file CpuSolverGlobalRichardsonBase.cpp
 * @brief Implementation of base CN-ADI2 solver for Global Richardson.
 *
 * Provides stateless CN-ADI2 propagator advancement with full and half step
 * support. This solver is used by CpuComputationGlobalRichardson which manages
 * two independent propagator chains and applies Richardson at the quadrature level.
 */

#include <iostream>
#include <cmath>
#include <cstring>

#include "CpuSolverGlobalRichardsonBase.h"

CpuSolverGlobalRichardsonBase::CpuSolverGlobalRichardsonBase(
    ComputationBox<double>* cb, Molecules* molecules)
{
    try
    {
        this->cb = cb;
        this->molecules = molecules;

        if (molecules->get_model_name() != "continuous")
            throw_with_line_number("Global Richardson method only supports 'continuous' chain model.");

        if (!cb->is_orthogonal())
            throw_with_line_number("Global Richardson method only supports orthogonal unit cells.");

        const int M = cb->get_total_grid();

        // Allocate coefficient arrays for each monomer type
        for (const auto& item : molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;

            // Boltzmann factors
            exp_dw_full[monomer_type].resize(M);
            exp_dw_half[monomer_type].resize(M);

            // Full step coefficients
            xl_full[monomer_type] = new double[M];
            xd_full[monomer_type] = new double[M];
            xh_full[monomer_type] = new double[M];

            yl_full[monomer_type] = new double[M];
            yd_full[monomer_type] = new double[M];
            yh_full[monomer_type] = new double[M];

            zl_full[monomer_type] = new double[M];
            zd_full[monomer_type] = new double[M];
            zh_full[monomer_type] = new double[M];

            // Half step coefficients
            xl_half[monomer_type] = new double[M];
            xd_half[monomer_type] = new double[M];
            xh_half[monomer_type] = new double[M];

            yl_half[monomer_type] = new double[M];
            yd_half[monomer_type] = new double[M];
            yh_half[monomer_type] = new double[M];

            zl_half[monomer_type] = new double[M];
            zd_half[monomer_type] = new double[M];
            zh_half[monomer_type] = new double[M];
        }

        update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

CpuSolverGlobalRichardsonBase::~CpuSolverGlobalRichardsonBase()
{
    // Free full step coefficients
    for (const auto& item : xl_full) delete[] item.second;
    for (const auto& item : xd_full) delete[] item.second;
    for (const auto& item : xh_full) delete[] item.second;

    for (const auto& item : yl_full) delete[] item.second;
    for (const auto& item : yd_full) delete[] item.second;
    for (const auto& item : yh_full) delete[] item.second;

    for (const auto& item : zl_full) delete[] item.second;
    for (const auto& item : zd_full) delete[] item.second;
    for (const auto& item : zh_full) delete[] item.second;

    // Free half step coefficients
    for (const auto& item : xl_half) delete[] item.second;
    for (const auto& item : xd_half) delete[] item.second;
    for (const auto& item : xh_half) delete[] item.second;

    for (const auto& item : yl_half) delete[] item.second;
    for (const auto& item : yd_half) delete[] item.second;
    for (const auto& item : yh_half) delete[] item.second;

    for (const auto& item : zl_half) delete[] item.second;
    for (const auto& item : zd_half) delete[] item.second;
    for (const auto& item : zh_half) delete[] item.second;
}

int CpuSolverGlobalRichardsonBase::max_of_two(int x, int y)
{
    return (x > y) ? x : y;
}

int CpuSolverGlobalRichardsonBase::min_of_two(int x, int y)
{
    return (x < y) ? x : y;
}

void CpuSolverGlobalRichardsonBase::update_laplacian_operator()
{
    try
    {
        const double ds = this->molecules->get_ds();

        for (const auto& item : this->molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second * item.second;

            // Full step coefficients (ds)
            FiniteDifference::get_laplacian_matrix(
                this->cb->get_boundary_conditions(),
                this->cb->get_nx(), this->cb->get_dx(),
                xl_full[monomer_type], xd_full[monomer_type], xh_full[monomer_type],
                yl_full[monomer_type], yd_full[monomer_type], yh_full[monomer_type],
                zl_full[monomer_type], zd_full[monomer_type], zh_full[monomer_type],
                bond_length_sq, ds);

            // Half step coefficients (ds/2)
            FiniteDifference::get_laplacian_matrix(
                this->cb->get_boundary_conditions(),
                this->cb->get_nx(), this->cb->get_dx(),
                xl_half[monomer_type], xd_half[monomer_type], xh_half[monomer_type],
                yl_half[monomer_type], yd_half[monomer_type], yh_half[monomer_type],
                zl_half[monomer_type], zd_half[monomer_type], zh_half[monomer_type],
                bond_length_sq, ds * 0.5);
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverGlobalRichardsonBase::update_dw(std::map<std::string, const double*> w_input)
{
    const int M = this->cb->get_total_grid();
    const double ds = this->molecules->get_ds();

    for (const auto& item : w_input)
    {
        const std::string& monomer_type = item.first;
        const double* w = item.second;

        if (!exp_dw_full.contains(monomer_type))
            throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in exp_dw_full.");

        std::vector<double>& exp_full = exp_dw_full[monomer_type];
        std::vector<double>& exp_half = exp_dw_half[monomer_type];

        for (int i = 0; i < M; i++)
        {
            // Full step: exp(-w*ds/2) for symmetric Strang splitting
            exp_full[i] = exp(-w[i] * ds * 0.5);
            // Half step: exp(-w*ds/4) for symmetric Strang splitting with ds/2
            exp_half[i] = exp(-w[i] * ds * 0.25);
        }
    }
}

void CpuSolverGlobalRichardsonBase::advance_full_step(
    double* q_in, double* q_out,
    std::string monomer_type, const double* q_mask)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int DIM = this->cb->get_dim();
        const double* _exp_dw = exp_dw_full[monomer_type].data();

        // Thread-local temporary array to avoid race conditions
        std::vector<double> q_temp_local(M);

        // Apply starting Boltzmann factor: exp(-w*ds/2)
        for (int i = 0; i < M; i++)
            q_temp_local[i] = _exp_dw[i] * q_in[i];

        // ADI diffusion step with full step coefficients
        if (DIM == 3)
            advance_propagator_3d_step(
                this->cb->get_boundary_conditions(), q_temp_local.data(), q_out,
                xl_full[monomer_type], xd_full[monomer_type], xh_full[monomer_type],
                yl_full[monomer_type], yd_full[monomer_type], yh_full[monomer_type],
                zl_full[monomer_type], zd_full[monomer_type], zh_full[monomer_type]);
        else if (DIM == 2)
            advance_propagator_2d_step(
                this->cb->get_boundary_conditions(), q_temp_local.data(), q_out,
                xl_full[monomer_type], xd_full[monomer_type], xh_full[monomer_type],
                yl_full[monomer_type], yd_full[monomer_type], yh_full[monomer_type]);
        else if (DIM == 1)
            advance_propagator_1d_step(
                this->cb->get_boundary_conditions(), q_temp_local.data(), q_out,
                xl_full[monomer_type], xd_full[monomer_type], xh_full[monomer_type]);

        // Apply ending Boltzmann factor: exp(-w*ds/2)
        for (int i = 0; i < M; i++)
            q_out[i] *= _exp_dw[i];

        // Apply mask if provided
        if (q_mask != nullptr)
        {
            for (int i = 0; i < M; i++)
                q_out[i] *= q_mask[i];
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverGlobalRichardsonBase::advance_half_step(
    double* q_in, double* q_out,
    std::string monomer_type, const double* q_mask)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int DIM = this->cb->get_dim();
        const double* _exp_dw = exp_dw_half[monomer_type].data();

        // Thread-local temporary array to avoid race conditions
        std::vector<double> q_temp_local(M);

        // Apply starting Boltzmann factor: exp(-w*ds/4)
        for (int i = 0; i < M; i++)
            q_temp_local[i] = _exp_dw[i] * q_in[i];

        // ADI diffusion step with half step coefficients
        if (DIM == 3)
            advance_propagator_3d_step(
                this->cb->get_boundary_conditions(), q_temp_local.data(), q_out,
                xl_half[monomer_type], xd_half[monomer_type], xh_half[monomer_type],
                yl_half[monomer_type], yd_half[monomer_type], yh_half[monomer_type],
                zl_half[monomer_type], zd_half[monomer_type], zh_half[monomer_type]);
        else if (DIM == 2)
            advance_propagator_2d_step(
                this->cb->get_boundary_conditions(), q_temp_local.data(), q_out,
                xl_half[monomer_type], xd_half[monomer_type], xh_half[monomer_type],
                yl_half[monomer_type], yd_half[monomer_type], yh_half[monomer_type]);
        else if (DIM == 1)
            advance_propagator_1d_step(
                this->cb->get_boundary_conditions(), q_temp_local.data(), q_out,
                xl_half[monomer_type], xd_half[monomer_type], xh_half[monomer_type]);

        // Apply ending Boltzmann factor: exp(-w*ds/4)
        for (int i = 0; i < M; i++)
            q_out[i] *= _exp_dw[i];

        // Apply mask if provided
        if (q_mask != nullptr)
        {
            for (int i = 0; i < M; i++)
                q_out[i] *= q_mask[i];
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// ============================================================================
// ADI step methods (same as CpuSolverCNADI)
// ============================================================================

void CpuSolverGlobalRichardsonBase::advance_propagator_3d_step(
    std::vector<BoundaryCondition> bc,
    double* q_in, double* q_out,
    double* _xl, double* _xd, double* _xh,
    double* _yl, double* _yd, double* _yh,
    double* _zl, double* _zd, double* _zh)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();
        double q_star[M];
        double q_dstar[M];
        double temp1[nx[0]];
        double temp2[nx[1]];
        double temp3[nx[2]];

        int im, ip, jm, jp, km, kp;

        // Calculate q_star
        for (int j = 0; j < nx[1]; j++)
        {
            if (bc[2] == BoundaryCondition::PERIODIC)
                jm = (nx[1] + j - 1) % nx[1];
            else
                jm = max_of_two(0, j - 1);
            if (bc[3] == BoundaryCondition::PERIODIC)
                jp = (j + 1) % nx[1];
            else
                jp = min_of_two(nx[1] - 1, j + 1);

            for (int k = 0; k < nx[2]; k++)
            {
                if (bc[4] == BoundaryCondition::PERIODIC)
                    km = (nx[2] + k - 1) % nx[2];
                else
                    km = max_of_two(0, k - 1);
                if (bc[5] == BoundaryCondition::PERIODIC)
                    kp = (k + 1) % nx[2];
                else
                    kp = min_of_two(nx[2] - 1, k + 1);

                // B part of Ax=B matrix equation
                for (int i = 0; i < nx[0]; i++)
                {
                    if (bc[0] == BoundaryCondition::PERIODIC)
                        im = (nx[0] + i - 1) % nx[0];
                    else
                        im = max_of_two(0, i - 1);
                    if (bc[1] == BoundaryCondition::PERIODIC)
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
                if (bc[0] == BoundaryCondition::PERIODIC)
                    tridiagonal_periodic(_xl, _xd, _xh, &q_star[j_k], nx[1] * nx[2], temp1, nx[0]);
                else
                    tridiagonal(_xl, _xd, _xh, &q_star[j_k], nx[1] * nx[2], temp1, nx[0]);
            }
        }

        // Calculate q_dstar
        for (int i = 0; i < nx[0]; i++)
        {
            for (int k = 0; k < nx[2]; k++)
            {
                for (int j = 0; j < nx[1]; j++)
                {
                    if (bc[2] == BoundaryCondition::PERIODIC)
                        jm = (nx[1] + j - 1) % nx[1];
                    else
                        jm = max_of_two(0, j - 1);
                    if (bc[3] == BoundaryCondition::PERIODIC)
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
                if (bc[2] == BoundaryCondition::PERIODIC)
                    tridiagonal_periodic(_yl, _yd, _yh, &q_dstar[i_k], nx[2], temp2, nx[1]);
                else
                    tridiagonal(_yl, _yd, _yh, &q_dstar[i_k], nx[2], temp2, nx[1]);
            }
        }

        // Calculate q^(n+1)
        for (int i = 0; i < nx[0]; i++)
        {
            for (int j = 0; j < nx[1]; j++)
            {
                for (int k = 0; k < nx[2]; k++)
                {
                    if (bc[4] == BoundaryCondition::PERIODIC)
                        km = (nx[2] + k - 1) % nx[2];
                    else
                        km = max_of_two(0, k - 1);
                    if (bc[5] == BoundaryCondition::PERIODIC)
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
                if (bc[4] == BoundaryCondition::PERIODIC)
                    tridiagonal_periodic(_zl, _zd, _zh, &q_out[i_j], 1, temp3, nx[2]);
                else
                    tridiagonal(_zl, _zd, _zh, &q_out[i_j], 1, temp3, nx[2]);
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverGlobalRichardsonBase::advance_propagator_2d_step(
    std::vector<BoundaryCondition> bc,
    double* q_in, double* q_out,
    double* _xl, double* _xd, double* _xh,
    double* _yl, double* _yd, double* _yh)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();
        double q_star[M];
        double temp1[nx[0]];
        double temp2[nx[1]];

        int im, ip, jm, jp;

        // Calculate q_star
        for (int j = 0; j < nx[1]; j++)
        {
            if (bc[2] == BoundaryCondition::PERIODIC)
                jm = (nx[1] + j - 1) % nx[1];
            else
                jm = max_of_two(0, j - 1);
            if (bc[3] == BoundaryCondition::PERIODIC)
                jp = (j + 1) % nx[1];
            else
                jp = min_of_two(nx[1] - 1, j + 1);

            // B part of Ax=B matrix equation
            for (int i = 0; i < nx[0]; i++)
            {
                if (bc[0] == BoundaryCondition::PERIODIC)
                    im = (nx[0] + i - 1) % nx[0];
                else
                    im = max_of_two(0, i - 1);
                if (bc[1] == BoundaryCondition::PERIODIC)
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
            if (bc[0] == BoundaryCondition::PERIODIC)
                tridiagonal_periodic(_xl, _xd, _xh, &q_star[j], nx[1], temp1, nx[0]);
            else
                tridiagonal(_xl, _xd, _xh, &q_star[j], nx[1], temp1, nx[0]);
        }

        // Calculate q^(n+1)
        for (int i = 0; i < nx[0]; i++)
        {
            for (int j = 0; j < nx[1]; j++)
            {
                if (bc[2] == BoundaryCondition::PERIODIC)
                    jm = (nx[1] + j - 1) % nx[1];
                else
                    jm = max_of_two(0, j - 1);
                if (bc[3] == BoundaryCondition::PERIODIC)
                    jp = (j + 1) % nx[1];
                else
                    jp = min_of_two(nx[1] - 1, j + 1);

                int i_j = i * nx[1] + j;
                int i_jm = i * nx[1] + jm;
                int i_jp = i * nx[1] + jp;

                temp2[j] = q_star[i_j] + (_yd[j] - 1.0) * q_in[i_j]
                         + _yl[j] * q_in[i_jm] + _yh[j] * q_in[i_jp];
            }
            if (bc[2] == BoundaryCondition::PERIODIC)
                tridiagonal_periodic(_yl, _yd, _yh, &q_out[i * nx[1]], 1, temp2, nx[1]);
            else
                tridiagonal(_yl, _yd, _yh, &q_out[i * nx[1]], 1, temp2, nx[1]);
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverGlobalRichardsonBase::advance_propagator_1d_step(
    std::vector<BoundaryCondition> bc,
    double* q_in, double* q_out,
    double* _xl, double* _xd, double* _xh)
{
    try
    {
        const std::vector<int> nx = this->cb->get_nx();
        double q_star[nx[0]];

        int im, ip;

        for (int i = 0; i < nx[0]; i++)
        {
            if (bc[0] == BoundaryCondition::PERIODIC)
                im = (nx[0] + i - 1) % nx[0];
            else
                im = max_of_two(0, i - 1);
            if (bc[1] == BoundaryCondition::PERIODIC)
                ip = (i + 1) % nx[0];
            else
                ip = min_of_two(nx[0] - 1, i + 1);

            // B part of Ax=B matrix equation
            q_star[i] = (2.0 - _xd[i]) * q_in[i] - _xl[i] * q_in[im] - _xh[i] * q_in[ip];
        }
        if (bc[0] == BoundaryCondition::PERIODIC)
            tridiagonal_periodic(_xl, _xd, _xh, q_out, 1, q_star, nx[0]);
        else
            tridiagonal(_xl, _xd, _xh, q_out, 1, q_star, nx[0]);
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// ============================================================================
// Tridiagonal solvers (same as CpuSolverCNADI)
// ============================================================================

void CpuSolverGlobalRichardsonBase::tridiagonal(
    const double* xl, const double* xd, const double* xh,
    double* x, const int INTERVAL, const double* d, const int M)
{
    double c_star[M - 1];
    double temp;

    // Forward sweep
    temp = xd[0];
    c_star[0] = xh[0] / xd[0];
    x[0] = d[0] / xd[0];

    for (int i = 1; i < M; i++)
    {
        c_star[i - 1] = xh[i - 1] / temp;
        temp = xd[i] - xl[i] * c_star[i - 1];
        x[i * INTERVAL] = (d[i] - xl[i] * x[(i - 1) * INTERVAL]) / temp;
    }

    // Backward substitution
    for (int i = M - 2; i >= 0; i--)
        x[i * INTERVAL] = x[i * INTERVAL] - c_star[i] * x[(i + 1) * INTERVAL];
}

void CpuSolverGlobalRichardsonBase::tridiagonal_periodic(
    const double* xl, const double* xd, const double* xh,
    double* x, const int INTERVAL, const double* d, const int M)
{
    double c_star[M - 1];
    double q[M];
    double temp, value;

    // Forward sweep
    temp = xd[0] - 1.0;
    c_star[0] = xh[0] / temp;
    x[0] = d[0] / temp;
    q[0] = 1.0 / temp;

    for (int i = 1; i < M - 1; i++)
    {
        c_star[i - 1] = xh[i - 1] / temp;
        temp = xd[i] - xl[i] * c_star[i - 1];
        x[i * INTERVAL] = (d[i] - xl[i] * x[(i - 1) * INTERVAL]) / temp;
        q[i] = (-xl[i] * q[i - 1]) / temp;
    }
    c_star[M - 2] = xh[M - 2] / temp;
    temp = xd[M - 1] - xh[M - 1] * xl[0] - xl[M - 1] * c_star[M - 2];
    x[(M - 1) * INTERVAL] = (d[M - 1] - xl[M - 1] * x[(M - 2) * INTERVAL]) / temp;
    q[M - 1] = (xh[M - 1] - xl[M - 1] * q[M - 2]) / temp;

    // Backward substitution
    for (int i = M - 2; i >= 0; i--)
    {
        x[i * INTERVAL] = x[i * INTERVAL] - c_star[i] * x[(i + 1) * INTERVAL];
        q[i] = q[i] - c_star[i] * q[i + 1];
    }

    value = (x[0] + xl[0] * x[(M - 1) * INTERVAL]) / (1.0 + q[0] + xl[0] * q[M - 1]);
    for (int i = 0; i < M; i++)
        x[i * INTERVAL] = x[i * INTERVAL] - q[i] * value;
}
