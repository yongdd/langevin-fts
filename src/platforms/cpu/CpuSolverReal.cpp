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

// This method solves CX=Y, where C is a tridiagonal matrix 
void CpuSolverReal::tridiagonal(
    const double *cu, const double *cd, const double *cl,
    double *x,  const double *d,  const int M)
{
    // cu: a
    // cd: b
    // cl: c

    double c_star[M-1];
    double temp;

    // Forward sweep
    temp = cd[0];
    c_star[0] = cl[0]/cd[0];
    x[0] = d[0]/cd[0];

    for(int i=1; i<M; i++)
    {
        c_star[i-1] = cl[i-1]/temp;
        temp = cd[i]-cu[i]*c_star[i-1];
        x[i] = (d[i]-cu[i]*x[i-1])/temp;
    }

    // Backward substitution
    for(int i=M-2;i>=0; i--)
        x[i] = x[i] - c_star[i]*x[i+1];
}

// This method solves CX=Y, where C is a near-tridiagonal matrix with periodic boundary condition
void CpuSolverReal::tridiagonal_periodic(
    const double *cu, const double *cd, const double *cl,
    double *x,  const double *d,  const int M)
{
    // cu: a
    // cd: b
    // cl: c
    // gamma = 1.0

    double c_star[M-1];
    double q[M];
    double temp, value;

    // Forward sweep
    temp = cd[0] - 1.0 ; 
    c_star[0] = cl[0]/temp;
    x[0] = d[0]/temp;
    q[0] =  1.0/temp;

    for(int i=1; i<M-1; i++)
    {
        c_star[i-1] = cl[i-1]/temp;
        temp = cd[i]-cu[i]*c_star[i-1];
        x[i] = (d[i]-cu[i]*x[i-1])/temp;
        q[i] =     (-cu[i]*q[i-1])/temp;
    }
    c_star[M-2] = cl[M-2]/temp;
    temp = cd[M-1]-cl[M-1]*cu[0] - cu[M-1]*c_star[M-2];
    x[M-1] = ( d[M-1]-cu[M-1]*x[M-2])/temp;
    q[M-1] = (cl[M-1]-cu[M-1]*q[M-2])/temp;

    // Backward substitution
    for(int i=M-2;i>=0; i--)
    {
        x[i] = x[i] - c_star[i]*x[i+1];
        q[i] = q[i] - c_star[i]*q[i+1];
    }

    value = (x[0]+cu[0]*x[M-1])/(1.0+q[0]+cu[0]*q[M-1]);
    for(int i=0; i<M; i++)
        x[i] = x[i] - q[i]*value;
}