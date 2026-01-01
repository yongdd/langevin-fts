#include <iostream>
#include <cmath>
#include <numbers>

#include "CpuSolverPseudoDiscrete.h"
#include "MklFFT.h"

template <typename T>
CpuSolverPseudoDiscrete<T>::CpuSolverPseudoDiscrete(ComputationBox<T>* cb, Molecules *molecules)
{
    try{
        if (cb->get_dim() == 3)
            this->fft = new MklFFT<T, 3>({cb->get_nx(0),cb->get_nx(1),cb->get_nx(2)});
        else if (cb->get_dim() == 2)
            this->fft = new MklFFT<T, 2>({cb->get_nx(0),cb->get_nx(1)});
        else if (cb->get_dim() == 1)
            this->fft = new MklFFT<T, 1>({cb->get_nx(0)});

        this->cb = cb;
        this->molecules = molecules;
        this->chain_model = molecules->get_model_name();

        pseudo = new Pseudo<T>(
            molecules->get_bond_lengths(),
            cb->get_boundary_conditions(),
            cb->get_nx(), cb->get_dx(), molecules->get_ds());

        // Create exp_dw
        const int M = cb->get_total_grid();
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            this->exp_dw   [monomer_type] = new T[M];
        }
        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CpuSolverPseudoDiscrete<T>::~CpuSolverPseudoDiscrete()
{
    delete fft;
    delete pseudo;

    for(const auto& item: this->exp_dw)
        delete[] item.second;
}
template <typename T>
void CpuSolverPseudoDiscrete<T>::update_laplacian_operator()
{
    try
    {
        pseudo->update(
            this->cb->get_boundary_conditions(),
            this->molecules->get_bond_lengths(),
            this->cb->get_dx(), this->molecules->get_ds());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CpuSolverPseudoDiscrete<T>::update_dw(std::map<std::string, const T*> w_input)
{
    const int M = this->cb->get_total_grid();
    const double ds = this->molecules->get_ds();

    for(const auto& item: w_input)
    {
        if( this->exp_dw.find(item.first) == this->exp_dw.end())
            throw_with_line_number("monomer_type \"" + item.first + "\" is not in exp_dw.");     
    }

    for(const auto& item: w_input)
    {
        std::string monomer_type = item.first;
        const T *w = item.second;

        for(int i=0; i<M; i++)
            this->exp_dw[monomer_type][i] = exp(-w[i]*ds);
    }
}
template <typename T>
void CpuSolverPseudoDiscrete<T>::advance_propagator(
    T *q_in, T *q_out, std::string monomer_type, const double *q_mask)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        std::complex<double> k_q_in[M_COMPLEX];

        T *_exp_dw = this->exp_dw[monomer_type];
        const double* _boltz_bond = pseudo->get_boltz_bond(monomer_type);

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
template <typename T>
void CpuSolverPseudoDiscrete<T>::advance_propagator_half_bond_step(
    T *q_in, T *q_out, std::string monomer_type)
{
    try
    {
        // Const int M = this->cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        std::complex<double> k_q_in[M_COMPLEX];

        const double* _boltz_bond_half = pseudo->get_boltz_bond_half(monomer_type);

        // 3D fourier discrete transform, forward and inplace
        fft->forward(q_in, k_q_in);
        // Multiply exp(-k^2 ds/12) in fourier space, in all 3 directions
        for(int i=0; i<M_COMPLEX; i++)
            k_q_in[i] *= _boltz_bond_half[i];
        // 3D fourier discrete transform, backward and inplace
        fft->backward(k_q_in, q_out);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
std::vector<T> CpuSolverPseudoDiscrete<T>::compute_single_segment_stress(
                T *q_1, T *q_2, std::string monomer_type, bool is_half_bond_length)
{
    try
    {
        const int DIM  = this->cb->get_dim();
        // const int M    = this->cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        T coeff;

        std::vector<T> stress(DIM);
        std::complex<double> qk_1[M_COMPLEX];
        std::complex<double> qk_2[M_COMPLEX];

        auto bond_lengths = this->molecules->get_bond_lengths();
        double bond_length_sq;
        double *_boltz_bond;

        const double* _fourier_basis_x = pseudo->get_fourier_basis_x();
        const double* _fourier_basis_y = pseudo->get_fourier_basis_y();
        const double* _fourier_basis_z = pseudo->get_fourier_basis_z();
        const int* _negative_k_idx = pseudo->get_negative_frequency_mapping();

        if (is_half_bond_length)
        {
            bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
            _boltz_bond = pseudo->get_boltz_bond_half(monomer_type);
        }
        else
        {
            bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
            _boltz_bond = pseudo->get_boltz_bond(monomer_type);
        }

        fft->forward(q_1, qk_1);
        fft->forward(q_2, qk_2);

        for(int d=0; d<DIM; d++)
            stress[d] = 0.0;

        if ( DIM == 3 )
        {
            for(int i=0; i<M_COMPLEX; i++){
                if constexpr (std::is_same<T, double>::value)
                    coeff = bond_length_sq*_boltz_bond[i]*(qk_1[i]*std::conj(qk_2[i])).real();
                else
                    coeff = bond_length_sq*_boltz_bond[i]*qk_1[i]*qk_2[_negative_k_idx[i]];
                stress[0] += coeff*_fourier_basis_x[i];
                stress[1] += coeff*_fourier_basis_y[i];
                stress[2] += coeff*_fourier_basis_z[i];
            }
        }
        if ( DIM == 2 )
        {
            for(int i=0; i<M_COMPLEX; i++){
                if constexpr (std::is_same<T, double>::value)
                    coeff = bond_length_sq*_boltz_bond[i]*(qk_1[i]*std::conj(qk_2[i])).real();
                else
                    coeff = bond_length_sq*_boltz_bond[i]*qk_1[i]*qk_2[_negative_k_idx[i]];
                stress[0] += coeff*_fourier_basis_y[i];
                stress[1] += coeff*_fourier_basis_z[i];
            }
        }
        if ( DIM == 1 )
        {
            for(int i=0; i<M_COMPLEX; i++){
                if constexpr (std::is_same<T, double>::value)
                    coeff = bond_length_sq*_boltz_bond[i]*(qk_1[i]*std::conj(qk_2[i])).real();
                else
                    coeff = bond_length_sq*_boltz_bond[i]*qk_1[i]*qk_2[_negative_k_idx[i]];
                stress[0] += coeff*_fourier_basis_z[i];
            }
        }
        return stress;
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
template class CpuSolverPseudoDiscrete<double>;
template class CpuSolverPseudoDiscrete<std::complex<double>>;