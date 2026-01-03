/**
 * @file CpuSolverPseudoContinuous.cpp
 * @brief CPU pseudo-spectral solver for continuous chain propagators.
 *
 * Implements the operator splitting pseudo-spectral method with
 * 4th-order Richardson extrapolation for continuous Gaussian chains.
 *
 * **Operator Splitting Scheme:**
 *
 * The diffusion equation ∂q/∂s = (b²/6)∇²q - w(r)q is split as:
 * 1. Potential half-step: exp(-w·ds/2)
 * 2. Diffusion step: FFT → multiply by exp(-b²k²ds/6) → IFFT
 * 3. Potential half-step: exp(-w·ds/2)
 *
 * **Richardson Extrapolation:**
 *
 * Combines one full step (ds) and two half-steps (ds/2 each):
 *     q_out = (4·q_half - q_full) / 3
 *
 * This yields 4th-order accuracy in ds.
 *
 * **Template Instantiations:**
 *
 * - CpuSolverPseudoContinuous<double>: Real fields with r2c FFT
 * - CpuSolverPseudoContinuous<std::complex<double>>: Complex fields
 *
 * @see Pseudo for Boltzmann factor computation
 * @see CpuComputationContinuous for usage context
 */

#include <iostream>
#include <cmath>
#include <numbers>

#include "CpuSolverPseudoContinuous.h"
#include "MklFFT.h"

/**
 * @brief Construct pseudo-spectral solver for continuous chains.
 *
 * Initializes FFT plan and Boltzmann factor arrays.
 *
 * @param cb        Computation box for grid info
 * @param molecules Molecular system for segment lengths
 */
template <typename T>
CpuSolverPseudoContinuous<T>::CpuSolverPseudoContinuous(ComputationBox<T>* cb, Molecules *molecules)
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
            cb->get_nx(), cb->get_dx(), molecules->get_ds(),
            cb->get_recip_metric());

        // Create exp_dw, and exp_dw_half
        const int M = cb->get_total_grid();
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            this->exp_dw     [monomer_type] = new T[M];
            this->exp_dw_half[monomer_type] = new T[M]; 
        }
        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CpuSolverPseudoContinuous<T>::~CpuSolverPseudoContinuous()
{
    delete fft;
    delete pseudo;

    for(const auto& item: this->exp_dw)
        delete[] item.second;
    for(const auto& item: this->exp_dw_half)
        delete[] item.second;
}
template <typename T>
void CpuSolverPseudoContinuous<T>::update_laplacian_operator()
{
    try
    {
        pseudo->update(
            this->cb->get_boundary_conditions(),
            this->molecules->get_bond_lengths(),
            this->cb->get_dx(), this->molecules->get_ds(),
            this->cb->get_recip_metric());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CpuSolverPseudoContinuous<T>::update_dw(std::map<std::string, const T*> w_input)
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
        { 
            this->exp_dw     [monomer_type][i] = exp(-w[i]*ds*0.5);
            this->exp_dw_half[monomer_type][i] = exp(-w[i]*ds*0.25);
        }
    }
}
template <typename T>
void CpuSolverPseudoContinuous<T>::advance_propagator(
    T *q_in, T *q_out, std::string monomer_type, const double *q_mask)
{
    try
    {
        const int M = this->cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        T q_out1[M], q_out2[M];
        std::complex<double> k_q_in1[M_COMPLEX], k_q_in2[M_COMPLEX];

        T *_exp_dw      = this->exp_dw     [monomer_type];
        T *_exp_dw_half = this->exp_dw_half[monomer_type];

        const double* _boltz_bond      = pseudo->get_boltz_bond     (monomer_type);
        const double* _boltz_bond_half = pseudo->get_boltz_bond_half(monomer_type);

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
template <typename T>
std::vector<T> CpuSolverPseudoContinuous<T>::compute_single_segment_stress(
                T *q_1, T *q_2, std::string monomer_type, bool is_half_bond_length)
{
    try
    {
        const int DIM  = this->cb->get_dim();
        const int N_STRESS = 6;  // Full stress tensor: xx, yy, zz, xy, xz, yz
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        auto bond_lengths = this->molecules->get_bond_lengths();
        double bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
        T coeff;

        std::vector<T> stress(N_STRESS);
        std::complex<double> qk_1[M_COMPLEX];
        std::complex<double> qk_2[M_COMPLEX];

        // Diagonal Fourier basis
        const double* _fourier_basis_x = pseudo->get_fourier_basis_x();
        const double* _fourier_basis_y = pseudo->get_fourier_basis_y();
        const double* _fourier_basis_z = pseudo->get_fourier_basis_z();
        // Cross-term Fourier basis
        const double* _fourier_basis_xy = pseudo->get_fourier_basis_xy();
        const double* _fourier_basis_xz = pseudo->get_fourier_basis_xz();
        const double* _fourier_basis_yz = pseudo->get_fourier_basis_yz();
        const int* _negative_k_idx = pseudo->get_negative_frequency_mapping();

        fft->forward(q_1, qk_1);
        fft->forward(q_2, qk_2);

        for(int d=0; d<N_STRESS; d++)
            stress[d] = 0.0;

        if ( DIM == 3 )
        {
            for(int i=0; i<M_COMPLEX; i++){
                if constexpr (std::is_same<T, double>::value)
                    coeff = bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real();
                else
                    coeff = bond_length_sq*qk_1[i]*qk_2[_negative_k_idx[i]];

                // Diagonal components
                stress[0] += coeff*_fourier_basis_x[i];   // xx
                stress[1] += coeff*_fourier_basis_y[i];   // yy
                stress[2] += coeff*_fourier_basis_z[i];   // zz
                // Cross-term components (for non-orthogonal systems)
                stress[3] += coeff*_fourier_basis_xy[i];  // xy
                stress[4] += coeff*_fourier_basis_xz[i];  // xz
                stress[5] += coeff*_fourier_basis_yz[i];  // yz
            }
        }
        if ( DIM == 2 )
        {
            // For 2D: stress[0] for lx[0], stress[1] for lx[1], stress[2] for cross term
            // fourier_basis_x/y/xy are remapped to logical dimensions in Pseudo::update_weighted_fourier_basis
            for(int i=0; i<M_COMPLEX; i++){
                if constexpr (std::is_same<T, double>::value)
                    coeff = bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real();
                else
                    coeff = bond_length_sq*qk_1[i]*qk_2[_negative_k_idx[i]];
                stress[0] += coeff*_fourier_basis_x[i];   // lx[0] dimension
                stress[1] += coeff*_fourier_basis_y[i];   // lx[1] dimension
                stress[2] += coeff*_fourier_basis_xy[i];  // cross term
            }
        }
        if ( DIM == 1 )
        {
            // For 1D: stress[0] for lx[0]
            // fourier_basis_x is remapped to logical dimension in Pseudo::update_weighted_fourier_basis
            for(int i=0; i<M_COMPLEX; i++){
                if constexpr (std::is_same<T, double>::value)
                    coeff = bond_length_sq*(qk_1[i]*std::conj(qk_2[i])).real();
                else
                    coeff = bond_length_sq*qk_1[i]*qk_2[_negative_k_idx[i]];
                stress[0] += coeff*_fourier_basis_x[i];   // lx[0] dimension
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
template class CpuSolverPseudoContinuous<double>;
template class CpuSolverPseudoContinuous<std::complex<double>>;
