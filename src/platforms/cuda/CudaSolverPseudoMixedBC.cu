/**
 * @file CudaSolverPseudoMixedBC.cu
 * @brief CUDA pseudo-spectral solver for mixed boundary conditions.
 *
 * Implements 4th-order Richardson extrapolation for advancing chain
 * propagators using DCT/DST transforms for non-periodic boundaries.
 *
 * **Richardson Extrapolation:**
 *
 * The propagator is advanced using two step sizes combined for 4th-order:
 * - Full step ds: O(ds²) accuracy
 * - Two half steps ds/2: O(ds²) accuracy
 * - Combination: (4*q_half - q_full)/3 gives O(ds⁴) accuracy
 *
 * @see CudaSolverPseudoMixedBC.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <thrust/reduce.h>
#include <cub/cub.cuh>

#include "CudaSolverPseudoMixedBC.h"
#include "CudaCommon.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T>
CudaSolverPseudoMixedBC<T>::CudaSolverPseudoMixedBC(
    ComputationBox<T>* cb,
    Molecules* molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    bool reduce_memory_usage)
    : fft_mixed_1d(nullptr), fft_mixed_2d(nullptr), fft_mixed_3d(nullptr)
{
    try
    {
        this->cb = cb;
        this->molecules = molecules;
        this->chain_model = molecules->get_model_name();
        this->n_streams = n_streams;

        // Check if all BCs are periodic
        auto bc_vec = cb->get_boundary_conditions();
        is_periodic_ = true;
        for (const auto& b : bc_vec)
        {
            if (b != BoundaryCondition::PERIODIC)
            {
                is_periodic_ = false;
                break;
            }
        }

        // Copy streams
        for (int i = 0; i < n_streams; i++)
        {
            this->streams[i][0] = streams[i][0];
            this->streams[i][1] = streams[i][1];
        }

        // Extract one BC per dimension
        int dim = cb->get_dim();
        std::vector<BoundaryCondition> bc_per_dim;
        for (int d = 0; d < dim; ++d)
            bc_per_dim.push_back(bc_vec[2 * d]);

        // Create appropriate FFT object
        if (!is_periodic_)
        {
            if (dim == 3)
            {
                std::array<int, 3> nx_arr = {cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)};
                std::array<BoundaryCondition, 3> bc_arr = {bc_per_dim[0], bc_per_dim[1], bc_per_dim[2]};
                fft_mixed_3d = new CudaFFTMixedBC<T, 3>(nx_arr, bc_arr);
            }
            else if (dim == 2)
            {
                std::array<int, 2> nx_arr = {cb->get_nx(0), cb->get_nx(1)};
                std::array<BoundaryCondition, 2> bc_arr = {bc_per_dim[0], bc_per_dim[1]};
                fft_mixed_2d = new CudaFFTMixedBC<T, 2>(nx_arr, bc_arr);
            }
            else if (dim == 1)
            {
                std::array<int, 1> nx_arr = {cb->get_nx(0)};
                std::array<BoundaryCondition, 1> bc_arr = {bc_per_dim[0]};
                fft_mixed_1d = new CudaFFTMixedBC<T, 1>(nx_arr, bc_arr);
            }
        }

        // Create Pseudo object (GPU version)
        pseudo = new CudaPseudoMixedBC<T>(
            molecules->get_bond_lengths(),
            bc_per_dim,
            cb->get_nx(), cb->get_dx(), molecules->get_ds());

        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        // Create exp_dw, and exp_dw_half
        for (const auto& item : molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            this->d_exp_dw[monomer_type] = nullptr;
            this->d_exp_dw_half[monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&this->d_exp_dw[monomer_type], sizeof(T) * M));
            gpu_error_check(cudaMalloc((void**)&this->d_exp_dw_half[monomer_type], sizeof(T) * M));
        }

        // Create cuFFT plans for periodic BC (fallback)
        if (is_periodic_)
        {
            const int NRANK = cb->get_dim();
            int total_grid[NRANK];
            for (int i = 0; i < NRANK; ++i)
                total_grid[i] = cb->get_nx(i);

            cufftType cufft_forward = (std::is_same<T, double>::value) ? CUFFT_D2Z : CUFFT_Z2Z;
            cufftType cufft_backward = (std::is_same<T, double>::value) ? CUFFT_Z2D : CUFFT_Z2Z;

            for (int i = 0; i < n_streams; i++)
            {
                cufftPlanMany(&plan_for_one[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, cufft_forward, 1);
                cufftPlanMany(&plan_bak_one[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, cufft_backward, 1);
                cufftSetStream(plan_for_one[i], streams[i][0]);
                cufftSetStream(plan_bak_one[i], streams[i][0]);
            }
        }

        // Allocate workspace memory
        for (int i = 0; i < n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&d_q_step_1_one[i], sizeof(T) * M));
            gpu_error_check(cudaMalloc((void**)&d_q_step_2_one[i], sizeof(T) * M));
            gpu_error_check(cudaMalloc((void**)&d_q_step_1_two[i], sizeof(T) * 2 * M));

            if (is_periodic_)
            {
                gpu_error_check(cudaMalloc((void**)&d_qk_complex_1[i], sizeof(cuDoubleComplex) * M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&d_qk_complex_2[i], sizeof(cuDoubleComplex) * M_COMPLEX));
                d_qk_in_1[i] = nullptr;
                d_qk_in_2[i] = nullptr;
            }
            else
            {
                gpu_error_check(cudaMalloc((void**)&d_qk_in_1[i], sizeof(double) * M_COMPLEX));
                gpu_error_check(cudaMalloc((void**)&d_qk_in_2[i], sizeof(double) * M_COMPLEX));
                d_qk_complex_1[i] = nullptr;
                d_qk_complex_2[i] = nullptr;
            }
        }

        // Allocate stress computation buffers (always double for DCT/DST)
        for (int i = 0; i < n_streams; i++)
        {
            gpu_error_check(cudaMalloc((void**)&d_stress_sum[i], sizeof(double) * M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_q_multi[i], sizeof(double) * M_COMPLEX));

            d_temp_storage[i] = nullptr;
            temp_storage_bytes[i] = 0;
            // Always use double for DCT/DST stress reduction
            cub::DeviceReduce::Sum(d_temp_storage[i], temp_storage_bytes[i],
                                   d_stress_sum[i], d_stress_sum[i], M_COMPLEX, streams[i][0]);
            gpu_error_check(cudaMalloc(&d_temp_storage[i], temp_storage_bytes[i]));
        }

        update_laplacian_operator();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template <typename T>
CudaSolverPseudoMixedBC<T>::~CudaSolverPseudoMixedBC()
{
    delete fft_mixed_1d;
    delete fft_mixed_2d;
    delete fft_mixed_3d;
    delete pseudo;

    if (is_periodic_)
    {
        for (int i = 0; i < n_streams; i++)
        {
            cufftDestroy(plan_for_one[i]);
            cufftDestroy(plan_bak_one[i]);
        }
    }

    for (const auto& item : this->d_exp_dw)
        cudaFree(item.second);
    for (const auto& item : this->d_exp_dw_half)
        cudaFree(item.second);

    for (int i = 0; i < n_streams; i++)
    {
        cudaFree(d_q_step_1_one[i]);
        cudaFree(d_q_step_2_one[i]);
        cudaFree(d_q_step_1_two[i]);

        if (d_qk_in_1[i]) cudaFree(d_qk_in_1[i]);
        if (d_qk_in_2[i]) cudaFree(d_qk_in_2[i]);
        if (d_qk_complex_1[i]) cudaFree(d_qk_complex_1[i]);
        if (d_qk_complex_2[i]) cudaFree(d_qk_complex_2[i]);

        cudaFree(d_stress_sum[i]);
        cudaFree(d_q_multi[i]);
        cudaFree(d_temp_storage[i]);
    }
}

//------------------------------------------------------------------------------
// Update Laplacian operator
//------------------------------------------------------------------------------
template <typename T>
void CudaSolverPseudoMixedBC<T>::update_laplacian_operator()
{
    try
    {
        auto bc_vec = this->cb->get_boundary_conditions();
        int dim = this->cb->get_dim();
        std::vector<BoundaryCondition> bc_per_dim;
        for (int d = 0; d < dim; ++d)
            bc_per_dim.push_back(bc_vec[2 * d]);

        pseudo->update(
            bc_per_dim,
            this->molecules->get_bond_lengths(),
            this->cb->get_dx(), this->molecules->get_ds());
    }
    catch (std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Update dw (Boltzmann factors from field)
//------------------------------------------------------------------------------
template <typename T>
void CudaSolverPseudoMixedBC<T>::update_dw(std::string device, std::map<std::string, const T*> w_input)
{
    try
    {
        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();
        const double ds = this->molecules->get_ds();

        for (const auto& item : w_input)
        {
            if (this->d_exp_dw.find(item.first) == this->d_exp_dw.end())
                throw_with_line_number("monomer_type \"" + item.first + "\" is not in d_exp_dw.");
        }

        cudaMemcpyKind cudaMemcpyInputToDevice;
        if (device == "gpu")
            cudaMemcpyInputToDevice = cudaMemcpyDeviceToDevice;
        else if (device == "cpu")
            cudaMemcpyInputToDevice = cudaMemcpyHostToDevice;
        else
            throw_with_line_number("Invalid device \"" + device + "\".");

        for (const auto& item : w_input)
        {
            std::string monomer_type = item.first;
            const T* w = item.second;

            gpu_error_check(cudaMemcpyAsync(
                this->d_exp_dw[monomer_type], w,
                sizeof(T) * M, cudaMemcpyInputToDevice));
            gpu_error_check(cudaMemcpyAsync(
                this->d_exp_dw_half[monomer_type], w,
                sizeof(T) * M, cudaMemcpyInputToDevice));

            ker_exp<<<N_BLOCKS, N_THREADS>>>(
                this->d_exp_dw[monomer_type], this->d_exp_dw[monomer_type], 1.0, -0.50 * ds, M);
            ker_exp<<<N_BLOCKS, N_THREADS>>>(
                this->d_exp_dw_half[monomer_type], this->d_exp_dw_half[monomer_type], 1.0, -0.25 * ds, M);
            gpu_error_check(cudaPeekAtLastError());
            gpu_error_check(cudaDeviceSynchronize());
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Transform forward (dispatch to DCT/DST or cuFFT)
//------------------------------------------------------------------------------
template <typename T>
void CudaSolverPseudoMixedBC<T>::transform_forward(const int STREAM, CuDeviceData<T>* d_rdata, double* d_cdata)
{
    // Cast CuDeviceData<T>* to T* (binary compatible: cuDoubleComplex ↔ std::complex<double>)
    T* d_rdata_T = reinterpret_cast<T*>(d_rdata);

    int dim = cb->get_dim();
    if (dim == 3)
        fft_mixed_3d->forward(d_rdata_T, d_cdata, streams[STREAM][0]);
    else if (dim == 2)
        fft_mixed_2d->forward(d_rdata_T, d_cdata, streams[STREAM][0]);
    else if (dim == 1)
        fft_mixed_1d->forward(d_rdata_T, d_cdata, streams[STREAM][0]);
}

//------------------------------------------------------------------------------
// Transform backward (dispatch to DCT/DST or cuFFT)
//------------------------------------------------------------------------------
template <typename T>
void CudaSolverPseudoMixedBC<T>::transform_backward(const int STREAM, double* d_cdata, CuDeviceData<T>* d_rdata)
{
    // Cast CuDeviceData<T>* to T* (binary compatible: cuDoubleComplex ↔ std::complex<double>)
    T* d_rdata_T = reinterpret_cast<T*>(d_rdata);

    int dim = cb->get_dim();
    if (dim == 3)
        fft_mixed_3d->backward(d_cdata, d_rdata_T, streams[STREAM][0]);
    else if (dim == 2)
        fft_mixed_2d->backward(d_cdata, d_rdata_T, streams[STREAM][0]);
    else if (dim == 1)
        fft_mixed_1d->backward(d_cdata, d_rdata_T, streams[STREAM][0]);
}

//------------------------------------------------------------------------------
// Advance propagator using Richardson extrapolation
//------------------------------------------------------------------------------
template <typename T>
void CudaSolverPseudoMixedBC<T>::advance_propagator(
    const int STREAM,
    CuDeviceData<T>* d_q_in, CuDeviceData<T>* d_q_out,
    std::string monomer_type, double* d_q_mask)
{
    try
    {
        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        CuDeviceData<T>* _d_exp_dw = this->d_exp_dw[monomer_type];
        CuDeviceData<T>* _d_exp_dw_half = this->d_exp_dw_half[monomer_type];
        const double* _d_boltz_bond = pseudo->get_boltz_bond(monomer_type);
        const double* _d_boltz_bond_half = pseudo->get_boltz_bond_half(monomer_type);

        // ===== Full step =====
        // Apply exp(-w*ds/2)
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_q_step_1_one[STREAM], d_q_in, _d_exp_dw, 1.0, M);
        gpu_error_check(cudaPeekAtLastError());

        // Forward transform
        transform_forward(STREAM, d_q_step_1_one[STREAM], d_qk_in_1[STREAM]);

        // Multiply by exp(-k^2 ds/6) in Fourier space
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_qk_in_1[STREAM], d_qk_in_1[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);
        gpu_error_check(cudaPeekAtLastError());

        // Backward transform
        transform_backward(STREAM, d_qk_in_1[STREAM], d_q_step_1_one[STREAM]);

        // Apply exp(-w*ds/2)
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_q_step_1_one[STREAM], d_q_step_1_one[STREAM], _d_exp_dw, 1.0, M);
        gpu_error_check(cudaPeekAtLastError());

        // ===== First half step =====
        // Apply exp(-w*ds/4)
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_q_step_2_one[STREAM], d_q_in, _d_exp_dw_half, 1.0, M);
        gpu_error_check(cudaPeekAtLastError());

        // Forward transform
        transform_forward(STREAM, d_q_step_2_one[STREAM], d_qk_in_2[STREAM]);

        // Multiply by exp(-k^2 ds/12)
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_qk_in_2[STREAM], d_qk_in_2[STREAM], _d_boltz_bond_half, 1.0, M_COMPLEX);
        gpu_error_check(cudaPeekAtLastError());

        // Backward transform
        transform_backward(STREAM, d_qk_in_2[STREAM], d_q_step_2_one[STREAM]);

        // Apply exp(-w*ds/2)
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_q_step_2_one[STREAM], d_q_step_2_one[STREAM], _d_exp_dw, 1.0, M);
        gpu_error_check(cudaPeekAtLastError());

        // ===== Second half step =====
        // Forward transform
        transform_forward(STREAM, d_q_step_2_one[STREAM], d_qk_in_2[STREAM]);

        // Multiply by exp(-k^2 ds/12)
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_qk_in_2[STREAM], d_qk_in_2[STREAM], _d_boltz_bond_half, 1.0, M_COMPLEX);
        gpu_error_check(cudaPeekAtLastError());

        // Backward transform
        transform_backward(STREAM, d_qk_in_2[STREAM], d_q_step_2_one[STREAM]);

        // Apply exp(-w*ds/4)
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_q_step_2_one[STREAM], d_q_step_2_one[STREAM], _d_exp_dw_half, 1.0, M);
        gpu_error_check(cudaPeekAtLastError());

        // ===== Richardson extrapolation =====
        ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_q_out, 4.0 / 3.0, d_q_step_2_one[STREAM], -1.0 / 3.0, d_q_step_1_one[STREAM], M);
        gpu_error_check(cudaPeekAtLastError());

        // Apply mask if provided
        if (d_q_mask != nullptr)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_q_out, d_q_out, d_q_mask, 1.0, M);
            gpu_error_check(cudaPeekAtLastError());
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Compute single segment stress
//------------------------------------------------------------------------------
template <typename T>
void CudaSolverPseudoMixedBC<T>::compute_single_segment_stress(
    const int STREAM,
    CuDeviceData<T>* d_q_pair, CuDeviceData<T>* d_segment_stress,
    std::string monomer_type, bool is_half_bond_length)
{
    try
    {
        // Stress computation for complex fields with DCT/DST is not yet supported
        if constexpr (!std::is_same<T, double>::value)
        {
            throw_with_line_number("Stress computation for complex fields with mixed BC is not implemented.");
        }

        const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM = this->cb->get_dim();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        auto bond_lengths = this->molecules->get_bond_lengths();
        double bond_length_sq = bond_lengths[monomer_type] * bond_lengths[monomer_type];

        const double* _d_fourier_basis_x = pseudo->get_fourier_basis_x();
        const double* _d_fourier_basis_y = pseudo->get_fourier_basis_y();
        const double* _d_fourier_basis_z = pseudo->get_fourier_basis_z();

        // For non-periodic BCs, compute stress in Fourier space using DCT/DST
        // Transform forward propagator products (only for real fields)
        transform_forward(STREAM, d_q_pair, d_qk_in_1[STREAM]);
        transform_forward(STREAM, &d_q_pair[pseudo->get_total_complex_grid()], d_qk_in_2[STREAM]);

        // Multiply in Fourier space (all buffers are double* for DCT/DST)
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
            d_q_multi[STREAM], d_qk_in_1[STREAM], d_qk_in_2[STREAM], 1.0, M_COMPLEX);
        gpu_error_check(cudaPeekAtLastError());

        // Cast output pointer to double* for reduction
        double* d_segment_stress_double = reinterpret_cast<double*>(d_segment_stress);

        // Compute stress components
        if (DIM >= 1)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_z, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM],
                                   d_stress_sum[STREAM], &d_segment_stress_double[DIM - 1], M_COMPLEX, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());
        }

        if (DIM >= 2)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_y, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM],
                                   d_stress_sum[STREAM], &d_segment_stress_double[DIM - 2], M_COMPLEX, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());
        }

        if (DIM >= 3)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_x, bond_length_sq, M_COMPLEX);
            gpu_error_check(cudaPeekAtLastError());

            cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM],
                                   d_stress_sum[STREAM], &d_segment_stress_double[DIM - 3], M_COMPLEX, streams[STREAM][0]);
            gpu_error_check(cudaPeekAtLastError());
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiations
template class CudaSolverPseudoMixedBC<double>;
template class CudaSolverPseudoMixedBC<std::complex<double>>;
