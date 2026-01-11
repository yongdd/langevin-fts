/**
 * @file CpuSolverRealSpace.cpp
 * @brief CPU real-space Crank-Nicolson solver for chain propagators.
 *
 * Implements the Crank-Nicolson finite difference method with ADI
 * (Alternating Direction Implicit) splitting for solving the modified
 * diffusion equation in real space.
 *
 * **Crank-Nicolson Scheme:**
 *
 * Semi-implicit discretization:
 *     (I - ds/2·L) q^(n+1) = (I + ds/2·L) q^n
 *
 * where L is the discrete Laplacian operator.
 *
 * **ADI Splitting (3D):**
 *
 * The 3D solve is split into three 1D tridiagonal solves:
 * 1. X-direction sweep: compute q*
 * 2. Y-direction sweep: compute q**
 * 3. Z-direction sweep: compute q^(n+1)
 *
 * **Boundary Conditions:**
 *
 * Supports periodic, reflecting (Neumann), and absorbing (Dirichlet)
 * boundaries. Uses cyclic Thomas algorithm for periodic boundaries.
 *
 * **Limitation:**
 *
 * Only supports continuous chain model (not discrete).
 * Stress computation not yet implemented.
 *
 * @see FiniteDifference for matrix coefficient computation
 */

#include <iostream>
#include <cmath>
#include <numbers>

#include "CpuSolverRealSpace.h"

/**
 * @brief Construct real-space solver for continuous chains.
 *
 * Allocates tridiagonal matrix coefficients for each dimension
 * and monomer type.
 *
 * @param cb        Computation box for grid and boundary info
 * @param molecules Molecular system (must use continuous model)
 *
 * @throws Exception if discrete chain model is specified
 */
CpuSolverRealSpace::CpuSolverRealSpace(ComputationBox<double>* cb, Molecules *molecules)
{
    try{
        this->cb = cb;
        this->molecules = molecules;

        if(molecules->get_model_name() != "continuous")
            throw_with_line_number("Real-space method only support 'continuous' chain model.");

        if(!cb->is_orthogonal())
            throw_with_line_number("Real-space method only supports orthogonal unit cells. "
                                   "Use pseudo-spectral method (chain_model='continuous') for non-orthogonal systems.");

        // for(size_t i=0; i<cb->get_boundary_conditions().size(); i++)
        // {
        //     if (cb->get_boundary_condition(i) == BoundaryCondition::PERIODIC)
        //         throw_with_line_number("Currently, we do not support periodic boundary conditions in real-space method");
        // }

        const int M = cb->get_total_grid();

        // Create exp_dw vectors and tridiagonal matrix coefficient arrays
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            exp_dw     [monomer_type].resize(M);
            exp_dw_half[monomer_type].resize(M);

            // Full step coefficients
            xl[monomer_type] = new double[M];
            xd[monomer_type] = new double[M];
            xh[monomer_type] = new double[M];

            yl[monomer_type] = new double[M];
            yd[monomer_type] = new double[M];
            yh[monomer_type] = new double[M];

            zl[monomer_type] = new double[M];
            zd[monomer_type] = new double[M];
            zh[monomer_type] = new double[M];

            // Half step coefficients for Richardson extrapolation
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
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CpuSolverRealSpace::~CpuSolverRealSpace()
{
    // exp_dw, exp_dw_half, exp_dw_quarter vectors are automatically cleaned up

    // Full step coefficients
    for(const auto& item: xl)
        delete[] item.second;
    for(const auto& item: xd)
        delete[] item.second;
    for(const auto& item: xh)
        delete[] item.second;

    for(const auto& item: yl)
        delete[] item.second;
    for(const auto& item: yd)
        delete[] item.second;
    for(const auto& item: yh)
        delete[] item.second;

    for(const auto& item: zl)
        delete[] item.second;
    for(const auto& item: zd)
        delete[] item.second;
    for(const auto& item: zh)
        delete[] item.second;

    // Half step coefficients
    for(const auto& item: xl_half)
        delete[] item.second;
    for(const auto& item: xd_half)
        delete[] item.second;
    for(const auto& item: xh_half)
        delete[] item.second;

    for(const auto& item: yl_half)
        delete[] item.second;
    for(const auto& item: yd_half)
        delete[] item.second;
    for(const auto& item: yh_half)
        delete[] item.second;

    for(const auto& item: zl_half)
        delete[] item.second;
    for(const auto& item: zd_half)
        delete[] item.second;
    for(const auto& item: zh_half)
        delete[] item.second;
}
int CpuSolverRealSpace::max_of_two(int x, int y)
{
   return (x > y) ? x : y;
}
int CpuSolverRealSpace::min_of_two(int x, int y)
{
   return (x < y) ? x : y;
}
void CpuSolverRealSpace::update_laplacian_operator()
{
    try
    {
        const double ds = this->molecules->get_ds();

        for(const auto& item: this->molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;

            // Full step coefficients (ds)
            FiniteDifference::get_laplacian_matrix(
                this->cb->get_boundary_conditions(),
                this->cb->get_nx(), this->cb->get_dx(),
                xl[monomer_type], xd[monomer_type], xh[monomer_type],
                yl[monomer_type], yd[monomer_type], yh[monomer_type],
                zl[monomer_type], zd[monomer_type], zh[monomer_type],
                bond_length_sq, ds);

            // Half step coefficients (ds/2) for Richardson extrapolation
            FiniteDifference::get_laplacian_matrix(
                this->cb->get_boundary_conditions(),
                this->cb->get_nx(), this->cb->get_dx(),
                xl_half[monomer_type], xd_half[monomer_type], xh_half[monomer_type],
                yl_half[monomer_type], yd_half[monomer_type], yh_half[monomer_type],
                zl_half[monomer_type], zd_half[monomer_type], zh_half[monomer_type],
                bond_length_sq, ds*0.5);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuSolverRealSpace::update_dw(std::map<std::string, const double*> w_input)
{
    const int M = this->cb->get_total_grid();
    const double ds = this->molecules->get_ds();

    for(const auto& item: w_input)
    {
        if( !exp_dw.contains(item.first))
            throw_with_line_number("monomer_type \"" + item.first + "\" is not in exp_dw.");
    }

    for(const auto& item: w_input)
    {
        const std::string& monomer_type = item.first;
        const double *w = item.second;
        std::vector<double>& exp_dw_vec = exp_dw[monomer_type];
        std::vector<double>& exp_dw_half_vec = exp_dw_half[monomer_type];

        for(int i=0; i<M; i++)
        {
            // Full step: exp(-w*ds/2) for symmetric splitting
            exp_dw_vec[i] = exp(-w[i]*ds*0.5);
            // Half step: exp(-w*ds/4) for symmetric splitting with ds/2
            exp_dw_half_vec[i] = exp(-w[i]*ds*0.25);
        }
    }
}
void CpuSolverRealSpace::advance_propagator(
    double *q_in, double *q_out, std::string monomer_type, const double *q_mask)
{
    try
    {
        const int M = this->cb->get_total_grid();

        // Get Boltzmann factors for full and half steps
        const double *_exp_dw = exp_dw[monomer_type].data();           // exp(-w*ds/2)

#if USE_RICHARDSON_EXTRAPOLATION
        const double *_exp_dw_half = exp_dw_half[monomer_type].data(); // exp(-w*ds/4)

        // Temporary arrays
        double q_out1[M];  // Full step result
        double q_out2[M];  // Two half-steps result

        //=====================================================================
        // Full step (ds): exp(-w*ds/2) * Diffusion(ds) * exp(-w*ds/2)
        //=====================================================================
        advance_propagator_step(
            q_in, q_out1,
            _exp_dw,
            xl[monomer_type], xd[monomer_type], xh[monomer_type],
            yl[monomer_type], yd[monomer_type], yh[monomer_type],
            zl[monomer_type], zd[monomer_type], zh[monomer_type],
            nullptr);

        //=====================================================================
        // Two half-steps (ds/2 each)
        // Structure: exp(-w*ds/4) * D(ds/2) * exp(-w*ds/2) * D(ds/2) * exp(-w*ds/4)
        // where the middle exp(-w*ds/2) combines end of step1 and start of step2
        //=====================================================================
        {
            const int DIM = this->cb->get_dim();
            double q_temp[M];

            // First half-step: exp(-w*ds/4) * Diffusion(ds/2) * exp(-w*ds/2)
            // Apply exp(-w*ds/4) at start
            for(int i=0; i<M; i++)
                q_temp[i] = _exp_dw_half[i] * q_in[i];

            // ADI diffusion with half-step coefficients
            if(DIM == 3)
                advance_propagator_3d_step(
                    this->cb->get_boundary_conditions(), q_temp, q_out2,
                    xl_half[monomer_type], xd_half[monomer_type], xh_half[monomer_type],
                    yl_half[monomer_type], yd_half[monomer_type], yh_half[monomer_type],
                    zl_half[monomer_type], zd_half[monomer_type], zh_half[monomer_type]);
            else if(DIM == 2)
                advance_propagator_2d_step(
                    this->cb->get_boundary_conditions(), q_temp, q_out2,
                    xl_half[monomer_type], xd_half[monomer_type], xh_half[monomer_type],
                    yl_half[monomer_type], yd_half[monomer_type], yh_half[monomer_type]);
            else if(DIM == 1)
                advance_propagator_1d_step(
                    this->cb->get_boundary_conditions(), q_temp, q_out2,
                    xl_half[monomer_type], xd_half[monomer_type], xh_half[monomer_type]);

            // Apply exp(-w*ds/2) at junction (combines ds/4 from end of step1 + ds/4 from start of step2)
            for(int i=0; i<M; i++)
                q_out2[i] *= _exp_dw[i];

            // Second half-step: Diffusion(ds/2) * exp(-w*ds/4)
            // Copy for diffusion step
            for(int i=0; i<M; i++)
                q_temp[i] = q_out2[i];

            if(DIM == 3)
                advance_propagator_3d_step(
                    this->cb->get_boundary_conditions(), q_temp, q_out2,
                    xl_half[monomer_type], xd_half[monomer_type], xh_half[monomer_type],
                    yl_half[monomer_type], yd_half[monomer_type], yh_half[monomer_type],
                    zl_half[monomer_type], zd_half[monomer_type], zh_half[monomer_type]);
            else if(DIM == 2)
                advance_propagator_2d_step(
                    this->cb->get_boundary_conditions(), q_temp, q_out2,
                    xl_half[monomer_type], xd_half[monomer_type], xh_half[monomer_type],
                    yl_half[monomer_type], yd_half[monomer_type], yh_half[monomer_type]);
            else if(DIM == 1)
                advance_propagator_1d_step(
                    this->cb->get_boundary_conditions(), q_temp, q_out2,
                    xl_half[monomer_type], xd_half[monomer_type], xh_half[monomer_type]);

            // Apply exp(-w*ds/4) at end
            for(int i=0; i<M; i++)
                q_out2[i] *= _exp_dw_half[i];
        }

        //=====================================================================
        // Richardson extrapolation: q_out = (4*q_half - q_full) / 3
        //=====================================================================
        for(int i=0; i<M; i++)
            q_out[i] = (4.0*q_out2[i] - q_out1[i]) / 3.0;

#else  // 2nd-order: single full step only
        //=====================================================================
        // Full step (ds): exp(-w*ds/2) * Diffusion(ds) * exp(-w*ds/2)
        //=====================================================================
        advance_propagator_step(
            q_in, q_out,
            _exp_dw,
            xl[monomer_type], xd[monomer_type], xh[monomer_type],
            yl[monomer_type], yd[monomer_type], yh[monomer_type],
            zl[monomer_type], zd[monomer_type], zh[monomer_type],
            nullptr);
#endif

        // Multiply mask
        if(q_mask != nullptr)
        {
            for(int i=0; i<M; i++)
                q_out[i] *= q_mask[i];
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverRealSpace::advance_propagator_step(
    double *q_in, double *q_out,
    const double *exp_dw_ptr,
    double *_xl, double *_xd, double *_xh,
    double *_yl, double *_yd, double *_yh,
    double *_zl, double *_zd, double *_zh,
    const double *q_mask)
{
    const int M = this->cb->get_total_grid();
    const int DIM = this->cb->get_dim();
    double q_temp[M];

    // Apply starting Boltzmann factor: exp(-w*h/2)
    for(int i=0; i<M; i++)
        q_temp[i] = exp_dw_ptr[i] * q_in[i];

    // ADI diffusion step
    if(DIM == 3)
        advance_propagator_3d_step(this->cb->get_boundary_conditions(), q_temp, q_out,
            _xl, _xd, _xh, _yl, _yd, _yh, _zl, _zd, _zh);
    else if(DIM == 2)
        advance_propagator_2d_step(this->cb->get_boundary_conditions(), q_temp, q_out,
            _xl, _xd, _xh, _yl, _yd, _yh);
    else if(DIM == 1)
        advance_propagator_1d_step(this->cb->get_boundary_conditions(), q_temp, q_out,
            _xl, _xd, _xh);

    // Apply ending Boltzmann factor: exp(-w*h/2)
    for(int i=0; i<M; i++)
        q_out[i] *= exp_dw_ptr[i];

    // Apply mask if provided
    if(q_mask != nullptr)
    {
        for(int i=0; i<M; i++)
            q_out[i] *= q_mask[i];
    }
}
void CpuSolverRealSpace::advance_propagator_3d(
    std::vector<BoundaryCondition> bc,
    double *q_in, double *q_out, std::string monomer_type)
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

        double *_xl = xl[monomer_type];
        double *_xd = xd[monomer_type];
        double *_xh = xh[monomer_type];

        double *_yl = yl[monomer_type];
        double *_yd = yd[monomer_type];
        double *_yh = yh[monomer_type];

        double *_zl = zl[monomer_type];
        double *_zd = zd[monomer_type];
        double *_zh = zh[monomer_type];

        int im, ip, jm, jp, km, kp;

        // Calculate q_star
        for(int j=0;j<nx[1];j++)
        {
            if (bc[2] == BoundaryCondition::PERIODIC)
                jm = (nx[1]+j-1) % nx[1];
            else
                jm = max_of_two(0,j-1);
            if (bc[3] == BoundaryCondition::PERIODIC)
                jp = (j+1) % nx[1];
            else
                jp = min_of_two(nx[1]-1,j+1);

            for(int k=0;k<nx[2];k++)
            {
                if (bc[4] == BoundaryCondition::PERIODIC)
                    km = (nx[2]+k-1) % nx[2];
                else
                    km = max_of_two(0,k-1);
                if (bc[5] == BoundaryCondition::PERIODIC)
                    kp = (k+1) % nx[2];
                else
                    kp = min_of_two(nx[2]-1,k+1);

                // B part of Ax=B matrix equation
                for(int i=0;i<nx[0];i++)
                {
                    if (bc[0] == BoundaryCondition::PERIODIC)
                        im = (nx[0]+i-1) % nx[0];
                    else
                        im = max_of_two(0,i-1);
                    if (bc[1] == BoundaryCondition::PERIODIC)
                        ip = (i+1) % nx[0];
                    else
                        ip = min_of_two(nx[0]-1,i+1);

                    int i_j_k  = i*nx[1]*nx[2] + j*nx[2] + k;
                    int im_j_k = im*nx[1]*nx[2] + j*nx[2] + k;
                    int ip_j_k = ip*nx[1]*nx[2] + j*nx[2] + k;
                    int i_jm_k = i*nx[1]*nx[2] + jm*nx[2] + k;
                    int i_jp_k = i*nx[1]*nx[2] + jp*nx[2] + k;
                    int i_j_km = i*nx[1]*nx[2] + j*nx[2] + km;
                    int i_j_kp = i*nx[1]*nx[2] + j*nx[2] + kp;

                    temp1[i] = 2.0*((3.0-0.5*_xd[i]-_yd[j]-_zd[k])*q_in[i_j_k]
                            - _zl[k]*q_in[i_j_km] - _zh[k]*q_in[i_j_kp]
                            - _yl[j]*q_in[i_jm_k] - _yh[j]*q_in[i_jp_k])
                            - _xl[i]*q_in[im_j_k] - _xh[i]*q_in[ip_j_k];
                }
                int j_k = j*nx[2] + k;
                if (bc[0] == BoundaryCondition::PERIODIC)
                    tridiagonal_periodic(_xl, _xd, _xh, &q_star[j_k], nx[1]*nx[2], temp1, nx[0]);
                else
                    tridiagonal         (_xl, _xd, _xh, &q_star[j_k], nx[1]*nx[2], temp1, nx[0]);
            }
        }
        // Calculate q_dstar
        for(int i=0;i<nx[0];i++)
        {
            for(int k=0;k<nx[2];k++)
            {
                for(int j=0;j<nx[1];j++)
                {
                    if (bc[2] == BoundaryCondition::PERIODIC)
                        jm = (nx[1]+j-1) % nx[1];
                    else
                        jm = max_of_two(0,j-1);
                    if (bc[3] == BoundaryCondition::PERIODIC)
                        jp = (j+1) % nx[1];
                    else
                        jp = min_of_two(nx[1]-1,j+1);

                    int i_j_k  = i*nx[1]*nx[2] + j*nx[2] + k;
                    int i_jm_k = i*nx[1]*nx[2] + jm*nx[2] + k;
                    int i_jp_k = i*nx[1]*nx[2] + jp*nx[2] + k;

                    temp2[j] = q_star[i_j_k] + (_yd[j]-1.0)*q_in[i_j_k]
                        + _yl[j]*q_in[i_jm_k] + _yh[j]*q_in[i_jp_k];
                }
                int i_k = i*nx[1]*nx[2] + k;
                if (bc[2] == BoundaryCondition::PERIODIC)
                    tridiagonal_periodic(_yl, _yd, _yh, &q_dstar[i_k], nx[2], temp2, nx[1]);
                else
                    tridiagonal         (_yl, _yd, _yh, &q_dstar[i_k], nx[2], temp2, nx[1]);
            }
        }

        // Calculate q^(n+1)
        for(int i=0;i<nx[0];i++)
        {
            for(int j=0;j<nx[1];j++)
            {
                for(int k=0;k<nx[2];k++)
                {
                    if (bc[4] == BoundaryCondition::PERIODIC)
                        km = (nx[2]+k-1) % nx[2];
                    else
                        km = max_of_two(0,k-1);
                    if (bc[5] == BoundaryCondition::PERIODIC)
                        kp = (k+1) % nx[2];
                    else
                        kp = min_of_two(nx[2]-1,k+1);

                    int i_j_k  = i*nx[1]*nx[2] + j*nx[2] + k;
                    int i_j_km = i*nx[1]*nx[2] + j*nx[2] + km;
                    int i_j_kp = i*nx[1]*nx[2] + j*nx[2] + kp;

                    temp3[k] = q_dstar[i_j_k] + (_zd[k]-1.0)*q_in[i_j_k]
                        + _zl[k]*q_in[i_j_km] + _zh[k]*q_in[i_j_kp];
                }
                int i_j = i*nx[1]*nx[2] + j*nx[2];
                if (bc[4] == BoundaryCondition::PERIODIC)
                    tridiagonal_periodic(_zl, _zd, _zh, &q_out[i_j], 1, temp3, nx[2]);
                else
                    tridiagonal         (_zl, _zd, _zh, &q_out[i_j], 1, temp3, nx[2]);
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuSolverRealSpace::advance_propagator_2d(
    std::vector<BoundaryCondition> bc,
    double *q_in, double *q_out, std::string monomer_type)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const std::vector<int> nx = this->cb->get_nx();
        double q_star[M];
        double temp1[nx[0]];
        double temp2[nx[1]];

        double *_xl = xl[monomer_type];
        double *_xd = xd[monomer_type];
        double *_xh = xh[monomer_type];

        double *_yl = yl[monomer_type];
        double *_yd = yd[monomer_type];
        double *_yh = yh[monomer_type];

        int im, ip, jm, jp;

        // Calculate q_star
        for(int j=0;j<nx[1];j++)
        {
            if (bc[2] == BoundaryCondition::PERIODIC)
                jm = (nx[1]+j-1) % nx[1];
            else
                jm = max_of_two(0,j-1);
            if (bc[3] == BoundaryCondition::PERIODIC)
                jp = (j+1) % nx[1];
            else
                jp = min_of_two(nx[1]-1,j+1);

            // B part of Ax=B matrix equation
            for(int i=0;i<nx[0];i++)
            {
                if (bc[0] == BoundaryCondition::PERIODIC)
                    im = (nx[0]+i-1) % nx[0];
                else
                    im = max_of_two(0,i-1);
                if (bc[1] == BoundaryCondition::PERIODIC)
                    ip = (i+1) % nx[0];
                else
                    ip = min_of_two(nx[0]-1,i+1);

                int i_j = i*nx[1] + j;
                int i_jm = i*nx[1] + jm;
                int i_jp = i*nx[1] + jp;
                int im_j = im*nx[1] + j;
                int ip_j = ip*nx[1] + j;

                temp1[i] = 2.0*((2.0-0.5*_xd[i]-_yd[j])*q_in[i_j]
                          - _yl[j]*q_in[i_jm] - _yh[j]*q_in[i_jp])
                          - _xl[i]*q_in[im_j] - _xh[i]*q_in[ip_j];
            }
            if (bc[0] == BoundaryCondition::PERIODIC)
                tridiagonal_periodic(_xl, _xd, _xh, &q_star[j], nx[1], temp1, nx[0]);
            else
                tridiagonal         (_xl, _xd, _xh, &q_star[j], nx[1], temp1, nx[0]);
        }
        
        // for(int i=0;i<M; i++)
        //     q_out[i] = q_star[i];

        // Calculate q_dstar
        for(int i=0;i<nx[0];i++)
        {
            for(int j=0;j<nx[1];j++)
            {
                if (bc[2] == BoundaryCondition::PERIODIC)
                    jm = (nx[1]+j-1) % nx[1];
                else
                    jm = max_of_two(0,j-1);
                if (bc[3] == BoundaryCondition::PERIODIC)
                    jp = (j+1) % nx[1];
                else
                    jp = min_of_two(nx[1]-1,j+1);

                int i_j = i*nx[1] + j;
                int i_jm = i*nx[1] + jm;
                int i_jp = i*nx[1] + jp;

                temp2[j] = q_star[i_j] + (_yd[j]-1.0)*q_in[i_j]
                    + _yl[j]*q_in[i_jm] + _yh[j]*q_in[i_jp];
            }
            if (bc[2] == BoundaryCondition::PERIODIC)
                tridiagonal_periodic(_yl, _yd, _yh, &q_out[i*nx[1]], 1, temp2, nx[1]);
            else
                tridiagonal         (_yl, _yd, _yh, &q_out[i*nx[1]], 1, temp2, nx[1]);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuSolverRealSpace::advance_propagator_1d(
    std::vector<BoundaryCondition> bc,
    double *q_in, double *q_out, std::string monomer_type)
{
    try
    {
        const std::vector<int> nx = this->cb->get_nx();
        double q_star[nx[0]];

        double *_xl = xl[monomer_type];
        double *_xd = xd[monomer_type];
        double *_xh = xh[monomer_type];

        int im, ip;

        for(int i=0;i<nx[0];i++)
        {
            if (bc[0] == BoundaryCondition::PERIODIC)
                im = (nx[0]+i-1) % nx[0];
            else
                im = max_of_two(0,i-1);
            if (bc[1] == BoundaryCondition::PERIODIC)
                ip = (i+1) % nx[0];
            else
                ip = min_of_two(nx[0]-1,i+1);

            // B part of Ax=B matrix equation
            q_star[i] = (2.0-_xd[i])*q_in[i] - _xl[i]*q_in[im] - _xh[i]*q_in[ip];
        }
        if (bc[0] == BoundaryCondition::PERIODIC)
            tridiagonal_periodic(_xl, _xd, _xh, q_out, 1, q_star, nx[0]);
        else
            tridiagonal         (_xl, _xd, _xh, q_out, 1, q_star, nx[0]);

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// ============================================================================
// Step methods with explicit coefficient parameters for Richardson extrapolation
// ============================================================================

void CpuSolverRealSpace::advance_propagator_3d_step(
    std::vector<BoundaryCondition> bc,
    double *q_in, double *q_out,
    double *_xl, double *_xd, double *_xh,
    double *_yl, double *_yd, double *_yh,
    double *_zl, double *_zd, double *_zh)
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
        for(int j=0;j<nx[1];j++)
        {
            if (bc[2] == BoundaryCondition::PERIODIC)
                jm = (nx[1]+j-1) % nx[1];
            else
                jm = max_of_two(0,j-1);
            if (bc[3] == BoundaryCondition::PERIODIC)
                jp = (j+1) % nx[1];
            else
                jp = min_of_two(nx[1]-1,j+1);

            for(int k=0;k<nx[2];k++)
            {
                if (bc[4] == BoundaryCondition::PERIODIC)
                    km = (nx[2]+k-1) % nx[2];
                else
                    km = max_of_two(0,k-1);
                if (bc[5] == BoundaryCondition::PERIODIC)
                    kp = (k+1) % nx[2];
                else
                    kp = min_of_two(nx[2]-1,k+1);

                // B part of Ax=B matrix equation
                for(int i=0;i<nx[0];i++)
                {
                    if (bc[0] == BoundaryCondition::PERIODIC)
                        im = (nx[0]+i-1) % nx[0];
                    else
                        im = max_of_two(0,i-1);
                    if (bc[1] == BoundaryCondition::PERIODIC)
                        ip = (i+1) % nx[0];
                    else
                        ip = min_of_two(nx[0]-1,i+1);

                    int i_j_k  = i*nx[1]*nx[2] + j*nx[2] + k;
                    int im_j_k = im*nx[1]*nx[2] + j*nx[2] + k;
                    int ip_j_k = ip*nx[1]*nx[2] + j*nx[2] + k;
                    int i_jm_k = i*nx[1]*nx[2] + jm*nx[2] + k;
                    int i_jp_k = i*nx[1]*nx[2] + jp*nx[2] + k;
                    int i_j_km = i*nx[1]*nx[2] + j*nx[2] + km;
                    int i_j_kp = i*nx[1]*nx[2] + j*nx[2] + kp;

                    temp1[i] = 2.0*((3.0-0.5*_xd[i]-_yd[j]-_zd[k])*q_in[i_j_k]
                            - _zl[k]*q_in[i_j_km] - _zh[k]*q_in[i_j_kp]
                            - _yl[j]*q_in[i_jm_k] - _yh[j]*q_in[i_jp_k])
                            - _xl[i]*q_in[im_j_k] - _xh[i]*q_in[ip_j_k];
                }
                int j_k = j*nx[2] + k;
                if (bc[0] == BoundaryCondition::PERIODIC)
                    tridiagonal_periodic(_xl, _xd, _xh, &q_star[j_k], nx[1]*nx[2], temp1, nx[0]);
                else
                    tridiagonal         (_xl, _xd, _xh, &q_star[j_k], nx[1]*nx[2], temp1, nx[0]);
            }
        }
        // Calculate q_dstar
        for(int i=0;i<nx[0];i++)
        {
            for(int k=0;k<nx[2];k++)
            {
                for(int j=0;j<nx[1];j++)
                {
                    if (bc[2] == BoundaryCondition::PERIODIC)
                        jm = (nx[1]+j-1) % nx[1];
                    else
                        jm = max_of_two(0,j-1);
                    if (bc[3] == BoundaryCondition::PERIODIC)
                        jp = (j+1) % nx[1];
                    else
                        jp = min_of_two(nx[1]-1,j+1);

                    int i_j_k  = i*nx[1]*nx[2] + j*nx[2] + k;
                    int i_jm_k = i*nx[1]*nx[2] + jm*nx[2] + k;
                    int i_jp_k = i*nx[1]*nx[2] + jp*nx[2] + k;

                    temp2[j] = q_star[i_j_k] + (_yd[j]-1.0)*q_in[i_j_k]
                        + _yl[j]*q_in[i_jm_k] + _yh[j]*q_in[i_jp_k];
                }
                int i_k = i*nx[1]*nx[2] + k;
                if (bc[2] == BoundaryCondition::PERIODIC)
                    tridiagonal_periodic(_yl, _yd, _yh, &q_dstar[i_k], nx[2], temp2, nx[1]);
                else
                    tridiagonal         (_yl, _yd, _yh, &q_dstar[i_k], nx[2], temp2, nx[1]);
            }
        }

        // Calculate q^(n+1)
        for(int i=0;i<nx[0];i++)
        {
            for(int j=0;j<nx[1];j++)
            {
                for(int k=0;k<nx[2];k++)
                {
                    if (bc[4] == BoundaryCondition::PERIODIC)
                        km = (nx[2]+k-1) % nx[2];
                    else
                        km = max_of_two(0,k-1);
                    if (bc[5] == BoundaryCondition::PERIODIC)
                        kp = (k+1) % nx[2];
                    else
                        kp = min_of_two(nx[2]-1,k+1);

                    int i_j_k  = i*nx[1]*nx[2] + j*nx[2] + k;
                    int i_j_km = i*nx[1]*nx[2] + j*nx[2] + km;
                    int i_j_kp = i*nx[1]*nx[2] + j*nx[2] + kp;

                    temp3[k] = q_dstar[i_j_k] + (_zd[k]-1.0)*q_in[i_j_k]
                        + _zl[k]*q_in[i_j_km] + _zh[k]*q_in[i_j_kp];
                }
                int i_j = i*nx[1]*nx[2] + j*nx[2];
                if (bc[4] == BoundaryCondition::PERIODIC)
                    tridiagonal_periodic(_zl, _zd, _zh, &q_out[i_j], 1, temp3, nx[2]);
                else
                    tridiagonal         (_zl, _zd, _zh, &q_out[i_j], 1, temp3, nx[2]);
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverRealSpace::advance_propagator_2d_step(
    std::vector<BoundaryCondition> bc,
    double *q_in, double *q_out,
    double *_xl, double *_xd, double *_xh,
    double *_yl, double *_yd, double *_yh)
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
        for(int j=0;j<nx[1];j++)
        {
            if (bc[2] == BoundaryCondition::PERIODIC)
                jm = (nx[1]+j-1) % nx[1];
            else
                jm = max_of_two(0,j-1);
            if (bc[3] == BoundaryCondition::PERIODIC)
                jp = (j+1) % nx[1];
            else
                jp = min_of_two(nx[1]-1,j+1);

            // B part of Ax=B matrix equation
            for(int i=0;i<nx[0];i++)
            {
                if (bc[0] == BoundaryCondition::PERIODIC)
                    im = (nx[0]+i-1) % nx[0];
                else
                    im = max_of_two(0,i-1);
                if (bc[1] == BoundaryCondition::PERIODIC)
                    ip = (i+1) % nx[0];
                else
                    ip = min_of_two(nx[0]-1,i+1);

                int i_j = i*nx[1] + j;
                int i_jm = i*nx[1] + jm;
                int i_jp = i*nx[1] + jp;
                int im_j = im*nx[1] + j;
                int ip_j = ip*nx[1] + j;

                temp1[i] = 2.0*((2.0-0.5*_xd[i]-_yd[j])*q_in[i_j]
                          - _yl[j]*q_in[i_jm] - _yh[j]*q_in[i_jp])
                          - _xl[i]*q_in[im_j] - _xh[i]*q_in[ip_j];
            }
            if (bc[0] == BoundaryCondition::PERIODIC)
                tridiagonal_periodic(_xl, _xd, _xh, &q_star[j], nx[1], temp1, nx[0]);
            else
                tridiagonal         (_xl, _xd, _xh, &q_star[j], nx[1], temp1, nx[0]);
        }

        // Calculate q^(n+1)
        for(int i=0;i<nx[0];i++)
        {
            for(int j=0;j<nx[1];j++)
            {
                if (bc[2] == BoundaryCondition::PERIODIC)
                    jm = (nx[1]+j-1) % nx[1];
                else
                    jm = max_of_two(0,j-1);
                if (bc[3] == BoundaryCondition::PERIODIC)
                    jp = (j+1) % nx[1];
                else
                    jp = min_of_two(nx[1]-1,j+1);

                int i_j = i*nx[1] + j;
                int i_jm = i*nx[1] + jm;
                int i_jp = i*nx[1] + jp;

                temp2[j] = q_star[i_j] + (_yd[j]-1.0)*q_in[i_j]
                    + _yl[j]*q_in[i_jm] + _yh[j]*q_in[i_jp];
            }
            if (bc[2] == BoundaryCondition::PERIODIC)
                tridiagonal_periodic(_yl, _yd, _yh, &q_out[i*nx[1]], 1, temp2, nx[1]);
            else
                tridiagonal         (_yl, _yd, _yh, &q_out[i*nx[1]], 1, temp2, nx[1]);
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

void CpuSolverRealSpace::advance_propagator_1d_step(
    std::vector<BoundaryCondition> bc,
    double *q_in, double *q_out,
    double *_xl, double *_xd, double *_xh)
{
    try
    {
        const std::vector<int> nx = this->cb->get_nx();
        double q_star[nx[0]];

        int im, ip;

        for(int i=0;i<nx[0];i++)
        {
            if (bc[0] == BoundaryCondition::PERIODIC)
                im = (nx[0]+i-1) % nx[0];
            else
                im = max_of_two(0,i-1);
            if (bc[1] == BoundaryCondition::PERIODIC)
                ip = (i+1) % nx[0];
            else
                ip = min_of_two(nx[0]-1,i+1);

            // B part of Ax=B matrix equation
            q_star[i] = (2.0-_xd[i])*q_in[i] - _xl[i]*q_in[im] - _xh[i]*q_in[ip];
        }
        if (bc[0] == BoundaryCondition::PERIODIC)
            tridiagonal_periodic(_xl, _xd, _xh, q_out, 1, q_star, nx[0]);
        else
            tridiagonal         (_xl, _xd, _xh, q_out, 1, q_star, nx[0]);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

std::vector<double> CpuSolverRealSpace::compute_single_segment_stress(
    [[maybe_unused]] double *q_1, [[maybe_unused]] double *q_2,
    [[maybe_unused]] std::string monomer_type, [[maybe_unused]] bool is_half_bond_length)
{
    try
    {
        const int DIM  = this->cb->get_dim();
        std::vector<double> stress(DIM); 

        throw_with_line_number("Currently, the real-space method does not support stress computation.");
        
        return stress;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// This method solves CX=Y, where C is a tridiagonal matrix
void CpuSolverRealSpace::tridiagonal(
    const double *xl, const double *xd, const double *xh,
    double *x, const int INTERVAL, const double *d, const int M)
{
    // xl: a
    // xd: b
    // xh: c

    double c_star[M-1];
    double temp;

    // Forward sweep
    temp = xd[0];
    c_star[0] = xh[0]/xd[0];
    x[0] = d[0]/xd[0];

    for(int i=1; i<M; i++)
    {
        c_star[i-1] = xh[i-1]/temp;
        temp = xd[i]-xl[i]*c_star[i-1];
        x[i*INTERVAL] = (d[i]-xl[i]*x[(i-1)*INTERVAL])/temp;
    }

    // Backward substitution
    for(int i=M-2;i>=0; i--)
        x[i*INTERVAL] = x[i*INTERVAL] - c_star[i]*x[(i+1)*INTERVAL];
}

// This method solves CX=Y, where C is a near-tridiagonal matrix with periodic boundary condition
void CpuSolverRealSpace::tridiagonal_periodic(
    const double *xl, const double *xd, const double *xh,
    double *x, const int INTERVAL, const double *d, const int M)
{
    // xl: a
    // xd: b
    // xh: c
    // gamma = 1.0

    double c_star[M-1];
    double q[M];
    double temp, value;

    // Forward sweep
    temp = xd[0] - 1.0 ; 
    c_star[0] = xh[0]/temp;
    x[0] = d[0]/temp;
    q[0] =  1.0/temp;

    for(int i=1; i<M-1; i++)
    {
        c_star[i-1] = xh[i-1]/temp;
        temp = xd[i]-xl[i]*c_star[i-1];
        x[i*INTERVAL] = (d[i]-xl[i]*x[(i-1)*INTERVAL])/temp;
        q[i]        =     (-xl[i]*q[i-1])         /temp;
    }
    c_star[M-2] = xh[M-2]/temp;
    temp = xd[M-1]-xh[M-1]*xl[0] - xl[M-1]*c_star[M-2];
    x[(M-1)*INTERVAL] = ( d[M-1]-xl[M-1]*x[(M-2)*INTERVAL])/temp;
    q[M-1]          = (xh[M-1]-xl[M-1]*q[M-2])         /temp;

    // Backward substitution
    for(int i=M-2;i>=0; i--)
    {
        x[i*INTERVAL] = x[i*INTERVAL] - c_star[i]*x[(i+1)*INTERVAL];
        q[i]        = q[i]        - c_star[i]*q[i+1];
    }

    value = (x[0]+xl[0]*x[(M-1)*INTERVAL])/(1.0+q[0]+xl[0]*q[M-1]);
    for(int i=0; i<M; i++)
        x[i*INTERVAL] = x[i*INTERVAL] - q[i]*value;
}
