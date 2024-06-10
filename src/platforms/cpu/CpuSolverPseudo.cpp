#include <iostream>
#include <cmath>
#include "CpuSolverPseudo.h"

#include "MklFFT3D.h"
#include "MklFFT2D.h"
#include "MklFFT1D.h"

CpuSolverPseudo::CpuSolverPseudo(ComputationBox *cb, Molecules *molecules)
{
    try{
        if (cb->get_dim() == 3)
            this->fft = new MklFFT3D({cb->get_nx(0),cb->get_nx(1),cb->get_nx(2)});
        else if (cb->get_dim() == 2)
            this->fft = new MklFFT2D({cb->get_nx(0),cb->get_nx(1)});
        else if (cb->get_dim() == 1)
            this->fft = new MklFFT1D(cb->get_nx(0));

        this->cb = cb;
        this->molecules = molecules;
        this->chain_model = molecules->get_model_name();

        const int M = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());

        // Create boltz_bond, boltz_bond_half, exp_dw, and exp_dw_half
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            boltz_bond     [monomer_type] = new double[M_COMPLEX];
            boltz_bond_half[monomer_type] = new double[M_COMPLEX]; 
            exp_dw         [monomer_type] = new double[M];
            if(chain_model == "continuous")
                exp_dw_half    [monomer_type] = new double[M]; 
        }

        // Allocate memory for stress calculation: compute_stress()
        fourier_basis_x = new double[M_COMPLEX];
        fourier_basis_y = new double[M_COMPLEX];
        fourier_basis_z = new double[M_COMPLEX];

        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
CpuSolverPseudo::~CpuSolverPseudo()
{
    delete fft;

    delete[] fourier_basis_x;
    delete[] fourier_basis_y;
    delete[] fourier_basis_z;

    for(const auto& item: boltz_bond)
        delete[] item.second;
    for(const auto& item: boltz_bond_half)
        delete[] item.second;
    for(const auto& item: exp_dw)
        delete[] item.second;
    if(chain_model == "continuous")
    {
        for(const auto& item: exp_dw_half)
            delete[] item.second;
    }
}
void CpuSolverPseudo::update_laplacian_operator()
{
    try
    {
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            double bond_length_sq = item.second*item.second;
            Pseudo::get_boltz_bond(cb->get_boundary_conditions(), boltz_bond     [monomer_type], bond_length_sq,   cb->get_nx(), cb->get_dx(), molecules->get_ds() );
            Pseudo::get_boltz_bond(cb->get_boundary_conditions(), boltz_bond_half[monomer_type], bond_length_sq/2, cb->get_nx(), cb->get_dx(), molecules->get_ds() );

            // For stress calculation: compute_stress()
            Pseudo::get_weighted_fourier_basis(cb->get_boundary_conditions(), fourier_basis_x, fourier_basis_y, fourier_basis_z, cb->get_nx(), cb->get_dx());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
void CpuSolverPseudo::update_dw(std::map<std::string, const double*> w_input)
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

        if(chain_model == "continuous")
        {
            for(int i=0; i<M; i++)
            { 
                exp_dw     [monomer_type][i] = exp(-w[i]*ds*0.5);
                exp_dw_half[monomer_type][i] = exp(-w[i]*ds*0.25);
            }
        }
        else if(chain_model == "discrete")
        {
            for(int i=0; i<M; i++)
                exp_dw[monomer_type][i] = exp(-w[i]*ds);
        }
    }
}
void CpuSolverPseudo::advance_propagator_continuous(
    double *q_in, double *q_out, std::string monomer_type, const double *q_mask)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());
        double q_out1[M], q_out2[M];
        std::complex<double> k_q_in1[M_COMPLEX], k_q_in2[M_COMPLEX];

        double *_exp_dw = exp_dw[monomer_type];
        double *_exp_dw_half = exp_dw_half[monomer_type];
        double *_boltz_bond = boltz_bond[monomer_type];
        double *_boltz_bond_half = boltz_bond_half[monomer_type];

        // step 1
        for(int i=0; i<M; i++)
            q_out1[i] = _exp_dw[i]*q_in[i];
        // 3D fourier discrete transform, forward and inplace
        fft->forward(q_out1,k_q_in1);
        // Multiply exp(-k^2 ds/6) in fourier space, in all 3 directions
        for(int i=0; i<M_COMPLEX; i++)
            k_q_in1[i] *= _boltz_bond[i];
        // 3D fourier discrete transform, backward and inplace
        fft->backward(k_q_in1,q_out1);
        // Evaluate exp(-w*ds/2) in real space
        for(int i=0; i<M; i++)
            q_out1[i] *= _exp_dw[i];

        // step 2
        // Evaluate exp(-w*ds/4) in real space
        for(int i=0; i<M; i++)
            q_out2[i] = _exp_dw_half[i]*q_in[i];
        // 3D fourier discrete transform, forward and inplace
        fft->forward(q_out2,k_q_in2);
        // Multiply exp(-k^2 ds/12) in fourier space, in all 3 directions
        for(int i=0; i<M_COMPLEX; i++)
            k_q_in2[i] *= _boltz_bond_half[i];
        // 3D fourier discrete transform, backward and inplace
        fft->backward(k_q_in2,q_out2);
        // Normalization calculation and evaluate exp(-w*ds/2) in real space
        for(int i=0; i<M; i++)
            q_out2[i] *= _exp_dw[i];
        // 3D fourier discrete transform, forward and inplace
        fft->forward(q_out2,k_q_in2);
        // Multiply exp(-k^2 ds/12) in fourier space, in all 3 directions
        for(int i=0; i<M_COMPLEX; i++)
            k_q_in2[i] *= _boltz_bond_half[i];
        // 3D fourier discrete transform, backward and inplace
        fft->backward(k_q_in2,q_out2);
        // Evaluate exp(-w*ds/4) in real space
        for(int i=0; i<M; i++)
            q_out2[i] *= _exp_dw_half[i];

        for(int i=0; i<M; i++)
            q_out[i] = (4.0*q_out2[i] - q_out1[i])/3.0;

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
void CpuSolverPseudo::advance_propagator_discrete(
    double *q_in, double *q_out, std::string monomer_type, const double *q_mask)
{
    try
    {
        const int M = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());
        std::complex<double> k_q_in[M_COMPLEX];

        double *_exp_dw = exp_dw[monomer_type];
        double *_boltz_bond = boltz_bond[monomer_type];

        // 3D fourier discrete transform, forward and inplace
        fft->forward(q_in,k_q_in);
        // Multiply exp(-k^2 ds/6) in fourier space, in all 3 directions
        for(int i=0; i<M_COMPLEX; i++)
            k_q_in[i] *= _boltz_bond[i];
        // 3D fourier discrete transform, backward and inplace
        fft->backward(k_q_in,q_out);
        // Normalization calculation and evaluate exp(-w*ds) in real space
        for(int i=0; i<M; i++)
            q_out[i] *= _exp_dw[i];
        
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

void CpuSolverPseudo::advance_propagator_discrete_half_bond_step(
    double *q_in, double *q_out, std::string monomer_type)
{
    try
    {
        // Const int M = cb->get_n_grid();
        const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());
        std::complex<double> k_q_in[M_COMPLEX];

        double *_boltz_bond_half = boltz_bond_half[monomer_type];

        // 3D fourier discrete transform, forward and inplace
        fft->forward(q_in,k_q_in);
        // Multiply exp(-k^2 ds/12) in fourier space, in all 3 directions
        for(int i=0; i<M_COMPLEX; i++)
            k_q_in[i] *= _boltz_bond_half[i];
        // 3D fourier discrete transform, backward and inplace
        fft->backward(k_q_in,q_out);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

std::vector<double> CpuSolverPseudo::compute_single_segment_stress_continuous(
                double *q_1, double *q_2, std::string monomer_type)
{
    const int DIM  = cb->get_dim();
    // const int M    = cb->get_n_grid();
    const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());
    auto bond_lengths = molecules->get_bond_lengths();
    double bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
    double coeff;
    
    std::vector<double> stress(DIM);
    std::complex<double> qk_1[M_COMPLEX];
    std::complex<double> qk_2[M_COMPLEX];

    fft->forward(q_1, qk_1);
    fft->forward(q_2, qk_2);

    for(int d=0; d<DIM; d++)
        stress[d] = 0.0;

    if ( DIM == 3 )
    {
        for(int i=0; i<M_COMPLEX; i++){
            coeff = bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real();
            stress[0] += coeff*fourier_basis_x[i];
            stress[1] += coeff*fourier_basis_y[i];
            stress[2] += coeff*fourier_basis_z[i];
        }
    }
    if ( DIM == 2 )
    {
        for(int i=0; i<M_COMPLEX; i++){
            coeff = bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real();
            stress[0] += coeff*fourier_basis_y[i];
            stress[1] += coeff*fourier_basis_z[i];
        }
    }
    if ( DIM == 1 )
    {
        for(int i=0; i<M_COMPLEX; i++){
            coeff = bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real();
            stress[0] += coeff*fourier_basis_z[i];
        }
    }

    return stress;
}

std::vector<double> CpuSolverPseudo::compute_single_segment_stress_discrete(
                double *q_1, double *q_2, std::string monomer_type, bool is_half_bond_length)
{
    const int DIM  = cb->get_dim();
    // const int M    = cb->get_n_grid();
    const int M_COMPLEX = Pseudo::get_n_complex_grid(cb->get_nx());
    double coeff;

    std::vector<double> stress(DIM);
    std::complex<double> qk_1[M_COMPLEX];
    std::complex<double> qk_2[M_COMPLEX];

    auto bond_lengths = molecules->get_bond_lengths();
    double bond_length_sq;
    double *_boltz_bond;

    if (is_half_bond_length)
    {
        bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
        _boltz_bond = boltz_bond_half[monomer_type];
    }
    else
    {
        bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
        _boltz_bond = boltz_bond[monomer_type];
    }

    fft->forward(q_1, qk_1);
    fft->forward(q_2, qk_2);

    for(int d=0; d<DIM; d++)
        stress[d] = 0.0;

    if ( DIM == 3 )
    {
        for(int i=0; i<M_COMPLEX; i++){
            coeff = bond_length_sq*_boltz_bond[i]*(qk_1[i]*std::conj(qk_2[i])).real();
            stress[0] += coeff*fourier_basis_x[i];
            stress[1] += coeff*fourier_basis_y[i];
            stress[2] += coeff*fourier_basis_z[i];
        }
    }
    if ( DIM == 2 )
    {
        for(int i=0; i<M_COMPLEX; i++){
            coeff = bond_length_sq*_boltz_bond[i]*(qk_1[i]*std::conj(qk_2[i])).real();
            stress[0] += coeff*fourier_basis_y[i];
            stress[1] += coeff*fourier_basis_z[i];
        }
    }
    if ( DIM == 1 )
    {
        for(int i=0; i<M_COMPLEX; i++){
            coeff = bond_length_sq*_boltz_bond[i]*(qk_1[i]*std::conj(qk_2[i])).real();
            stress[0] += coeff*fourier_basis_z[i];
        }
    }
    return stress;
}