#include <iostream>
#include <cmath>
#include "CpuSolverReal.h"

#include "MklFFT3D.h"
#include "MklFFT2D.h"
#include "MklFFT1D.h"

CpuSolverReal::CpuSolverReal(ComputationBox *cb, Molecules *molecules)
{
    try{
        this->cb = cb;
        this->molecules = molecules;

        if(molecules->get_model_name() != "continuous")
            throw_with_line_number("Real-space method only support 'continuous' chain model.");     
        const int M = cb->get_n_grid();

        // Create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            exp_dw         [monomer_type] = new double[M];
            exp_dw_half    [monomer_type] = new double[M]; 
        }

        xl = new double[M];
        xd = new double[M];
        xh = new double[M];

        yl = new double[M];
        yd = new double[M];
        yh = new double[M];

        zl = new double[M];
        zd = new double[M];
        zh = new double[M];

        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CpuSolverReal::~CpuSolverReal()
{
    for(const auto& item: exp_dw)
        delete[] item.second;
    for(const auto& item: exp_dw_half)
        delete[] item.second;
    
    delete[] xl;
    delete[] xd;
    delete[] xh;

    delete[] yl;
    delete[] yd;
    delete[] yh;

    delete[] zl;
    delete[] zd;
    delete[] zh;
}
void CpuSolverReal::update_laplacian_operator()
{
    try
    {
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;

            FiniteDifference::get_laplacian_matrix(
                cb->get_boundary_conditions(),
                cb->get_nx(), cb->get_dx(),
                xl, xd, xh,
                yl, yd, yh,
                zl, zd, zh,
                molecules->get_ds());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuSolverReal::update_dw(std::map<std::string, const double*> w_input)
{
    const int M = cb->get_n_grid();
    const double ds = molecules->get_ds();

    for(const auto& item: w_input)
    {
        if( exp_dw.find(item.first) == exp_dw.end())
            throw_with_line_number("monomer_type \"" + item.first + "\" is not in exp_dw.");     
    }

    for(const auto& item: w_input)
    {
        std::string monomer_type = item.first;
        const double *w = item.second;

        for(int i=0; i<M; i++)
        { 
            exp_dw     [monomer_type][i] = exp(-w[i]*ds*0.5);
            exp_dw_half[monomer_type][i] = exp(-w[i]*ds*0.25);
        }
    }
}
void CpuSolverReal::advance_propagator_continuous(
    double *q_in, double *q_out, std::string monomer_type, const double *q_mask)
{
    try
    {
        const int M = cb->get_n_grid();
        double q_out1[M], q_out2[M];

        double *_exp_dw = exp_dw[monomer_type];
        double *_exp_dw_half = exp_dw_half[monomer_type];

        // Multiply mask
        if (q_mask != nullptr)
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
std::vector<double> CpuSolverReal::compute_single_segment_stress_continuous(
                double *q_1, double *q_2, std::string monomer_type)
{
    try
    {
        const int DIM  = cb->get_dim();
        const int M    = cb->get_n_grid();
        std::vector<double> stress(DIM);

        throw_with_line_number("Currently, real-space does not support stress computation.");   

        return stress;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// this method solves CX=Y, where C is a tridiagonal matrix 
void CpuSolverReal::tridiagonal(
    double *cu, double *cd, double *cl,
    double *x,  double *y,  const int M)
{
    // cu: a
    // cd: b
    // cl: c

    double c_star[M-1];
    double temp;

    // Forward sweep
    temp = cd[0];
    c_star[0] = cl[0]/cd[0];
    x[0] = y[0]/cd[0];

    for(int i=1; i<M; i++)
    {
        c_star[i-1] = cl[i-1]/temp;
        temp = cd[i]-cu[i]*c_star[i-1];
        x[i] = (y[i]-cu[i]*x[i-1])/temp;
    }

    // Backward substitution
    for(int j=M-2;j>=0; j--)
        x[j] = x[j] - c_star[j]*x[j+1];
}

// // this method solves CX=Y, where C is a near-tridiagonal nm by nm symmetric
// // matrix (indices from 1 to nm) with periodic boundary condition.
// // The off-diagonal terms are normalized to 1, and y are also pre-normalized
// // cd is the 1d array of diagonal terms
// void CpuSolverReal::periodic_tridiagonal(
//     double *cd,
//     double *x,
//     double *y,
//     const int nm)
// {
//   implicit none
//   integer, intent(in) :: nm
//   double precision, intent(in) :: cd(1:nm), y(1:nm)
//   double precision, intent(out) :: x(1:nm)
//   double precision :: l(1:nm), w(1:nm-1)
//   integer :: i
// !
//     ! Boisvert's method (see page 431 of his 1991 paper)
//     ! factorization
//     l(1) = 1.0d0/cd(1)
//     do i=2,nm-1
//     l(i) = 1.0d0/(cd(i)-l(i-1))
//     end do
//     !
//     w(1) = l(1)
//     do i=2,nm-2
//     w(i) = -w(i-1)*l(i)
//     end do
//     w(nm-1) = (1.0d0-w(nm-2))*l(nm-1)
//     !
//     l(nm) = 1.0d0/(cd(nm)-w(1)+sum(w(1:nm-3)*w(2:nm-2))-(1.0d0-w(nm-2))*w(nm-1))
//     !
//     ! forward substitution
//     x(1)=y(1)
//     do i=2,nm-1
//     x(i) = y(i)-l(i-1)*x(i-1)
//     end do
//     x(nm) = l(nm)*(y(nm)-sum(w(1:nm-1)*x(1:nm-1)))
//     !
//     ! backward substitution
//     x(nm-1) = l(nm-1)*(x(nm-1)-(1.0d0-w(nm-2))*x(nm))
//     do i=nm-2,1,-1
//     x(i) = l(i)*(x(i)-x(i+1))-w(i)*x(nm)
//     end do
// }