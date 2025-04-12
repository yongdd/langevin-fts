#include <iostream>
#include <cmath>
#include <thrust/reduce.h>

#include "CudaPseudo.h"
#include "CudaSolverPseudoDiscrete.h"

template <typename T>
CudaSolverPseudoDiscrete<T>::CudaSolverPseudoDiscrete(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    bool reduce_gpu_memory_usage)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->chain_model = molecules->get_model_name();
        this->n_streams = n_streams;
        
        pseudo = new CudaPseudo<T>(
            molecules->get_bond_lengths(),
            cb->get_boundary_conditions(),
            cb->get_nx(), cb->get_dx(), molecules->get_ds());

        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        // Copy streams
        for(int i=0; i<n_streams; i++)
        {
            this->streams[i][0] = streams[i][0];
            this->streams[i][1] = streams[i][1];
        }

        // Create exp_dw
        for(const auto& item: molecules->get_bond_lengths())
        {
            std::string monomer_type = item.first;
            this->d_exp_dw   [monomer_type] = nullptr;
            gpu_error_check(cudaMalloc((void**)&this->d_exp_dw[monomer_type], sizeof(T)*M));
        }

        // Create FFT plan
        const int NRANK{cb->get_dim()};
        int total_grid[NRANK];

        if(cb->get_dim() == 3)
        {
            total_grid[0] = cb->get_nx(0);
            total_grid[1] = cb->get_nx(1);
            total_grid[2] = cb->get_nx(2);
        }
        else if(cb->get_dim() == 2)
        {
            total_grid[0] = cb->get_nx(0);
            total_grid[1] = cb->get_nx(1);
        }
        else if(cb->get_dim() == 1)
        {
            total_grid[0] = cb->get_nx(0);
        }

        cufftType cufft_forward;
        cufftType cufft_backward;
        if constexpr (std::is_same<T, double>::value)
        {
            cufft_forward = CUFFT_D2Z;
            cufft_backward = CUFFT_Z2D;
        }
        else
        {
            cufft_forward = CUFFT_Z2Z;
            cufft_backward = CUFFT_Z2Z;
        }

        for(int i=0; i<n_streams; i++)
        {
            cufftPlanMany(&plan_for_one[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, cufft_forward, 1);
            cufftPlanMany(&plan_for_two[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, cufft_forward, 2);
            cufftPlanMany(&plan_bak_one[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, cufft_backward, 1);
            cufftPlanMany(&plan_bak_two[i], NRANK, total_grid, NULL, 1, 0, NULL, 1, 0, cufft_backward, 2);
            cufftSetStream(plan_for_one[i], streams[i][0]);
            cufftSetStream(plan_for_two[i], streams[i][0]);
            cufftSetStream(plan_bak_one[i], streams[i][0]);
            cufftSetStream(plan_bak_two[i], streams[i][0]);
        }

        // Allocate memory for pseudo-spectral: advance_propagator()
        for(int i=0; i<n_streams; i++)
        {  
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1_one[i], sizeof(cuDoubleComplex)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_qk_in_1_two[i], sizeof(cuDoubleComplex)*2*M_COMPLEX));
        }

        for(int i=0; i<n_streams; i++)
        {  
            gpu_error_check(cudaMalloc((void**)&d_stress_sum[i],      sizeof(T)*M_COMPLEX));
            gpu_error_check(cudaMalloc((void**)&d_stress_sum_out[i],  sizeof(T)*1));
            gpu_error_check(cudaMalloc((void**)&d_q_multi[i],         sizeof(T)*M_COMPLEX));
        }

        // Allocate memory for cub reduction sum
        for(int i=0; i<n_streams; i++)
        {  
            d_temp_storage[i] = nullptr;
            temp_storage_bytes[i] = 0;
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[i], temp_storage_bytes[i], d_stress_sum[i], d_stress_sum_out[i], M_COMPLEX, streams[i][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[i], temp_storage_bytes[i], d_stress_sum[i], d_stress_sum_out[i], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[i][0]);
            gpu_error_check(cudaMalloc(&d_temp_storage[i], temp_storage_bytes[i]));
        }
        update_laplacian_operator();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
CudaSolverPseudoDiscrete<T>::~CudaSolverPseudoDiscrete()
{
    delete pseudo;

    for(int i=0; i<n_streams; i++)
    {
        cufftDestroy(plan_for_one[i]);
        cufftDestroy(plan_for_two[i]);
        cufftDestroy(plan_bak_one[i]);
        cufftDestroy(plan_bak_two[i]);
    }

    for(const auto& item: this->d_exp_dw)
        cudaFree(item.second);

    // For pseudo-spectral: advance_propagator()
    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_qk_in_1_one[i]);
        cudaFree(d_qk_in_1_two[i]);
    }

    for(int i=0; i<n_streams; i++)
    {
        cudaFree(d_stress_sum[i]);
        cudaFree(d_stress_sum_out[i]);
        cudaFree(d_q_multi[i]);
        cudaFree(d_temp_storage[i]);
    }
}
template <typename T>
void CudaSolverPseudoDiscrete<T>::update_laplacian_operator()
{
    try{
        pseudo->update(
            this->cb->get_boundary_conditions(),
            this->molecules->get_bond_lengths(),
            this->cb->get_dx(), this->molecules->get_ds());
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}
template <typename T>
void CudaSolverPseudoDiscrete<T>::update_dw(std::string device, std::map<std::string, const T*> w_input)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();;
        const double ds = this->molecules->get_ds();

        for(const auto& item: w_input)
        {
            if( this->d_exp_dw.find(item.first) == this->d_exp_dw.end())
                throw_with_line_number("monomer_type \"" + item.first + "\" is not in d_exp_dw.");     
        }

        cudaMemcpyKind cudaMemcpyInputToDevice;
        if (device == "gpu")
            cudaMemcpyInputToDevice = cudaMemcpyDeviceToDevice;
        else if(device == "cpu")
            cudaMemcpyInputToDevice = cudaMemcpyHostToDevice;
        else
        {
            throw_with_line_number("Invalid device \"" + device + "\".");
        }

        // Compute exp_dw and exp_dw_half
        for(const auto& item: w_input)
        {
            std::string monomer_type = item.first;
            const T *w = item.second;

            // Copy field configurations from host to device
            gpu_error_check(cudaMemcpy(
                this->d_exp_dw[monomer_type], w,      
                sizeof(T)*M, cudaMemcpyInputToDevice));

            // Compute exp_dw and exp_dw_half
            ker_exp<<<N_BLOCKS, N_THREADS>>>
                (this->d_exp_dw[monomer_type],
                 this->d_exp_dw[monomer_type], 1.0, -1.0*ds, M);

            gpu_error_check(cudaDeviceSynchronize());
        }
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaSolverPseudoDiscrete<T>::advance_propagator(
    const int STREAM,
    CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
    std::string monomer_type,
    double *d_q_mask)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();;
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        CuDeviceData<T> *_d_exp_dw = this->d_exp_dw[monomer_type];
        const double* _d_boltz_bond = pseudo->get_boltz_bond(monomer_type);

        // Execute a forward FFT
        if constexpr (std::is_same<T, double>::value)
            cufftExecD2Z(plan_for_one[STREAM], d_q_in, d_qk_in_1_one[STREAM]);
        else
            cufftExecZ2Z(plan_for_one[STREAM], d_q_in, d_qk_in_1_one[STREAM], CUFFT_FORWARD);

        // Multiply exp(-k^2 ds/6) in fourier space
        ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_in_1_one[STREAM], _d_boltz_bond, 1.0, M_COMPLEX);

        // Execute a backward FFT
        if constexpr (std::is_same<T, double>::value)
            cufftExecZ2D(plan_bak_one[STREAM], d_qk_in_1_one[STREAM], d_q_out);
        else
            cufftExecZ2Z(plan_bak_one[STREAM], d_qk_in_1_one[STREAM], d_q_out, CUFFT_INVERSE);

        // Evaluate exp(-w*ds) in real space
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, _d_exp_dw, 1.0/static_cast<double>(M), M);

        // Multiply mask
        if (d_q_mask != nullptr)
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_out, d_q_out, d_q_mask, 1.0, M);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaSolverPseudoDiscrete<T>::advance_propagator_half_bond_step(
    const int STREAM,
    CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out, std::string monomer_type)
{
    try
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int M = cb->get_total_grid();;
        const int M_COMPLEX = pseudo->get_total_complex_grid();
        const double* _d_boltz_bond_half = pseudo->get_boltz_bond_half(monomer_type);

        // 3D fourier discrete transform, forward and inplace
        if constexpr (std::is_same<T, double>::value)
            cufftExecD2Z(plan_for_one[STREAM], d_q_in, d_qk_in_1_one[STREAM]);
        else
            cufftExecZ2Z(plan_for_one[STREAM], d_q_in, d_qk_in_1_one[STREAM], CUFFT_FORWARD);
        // Multiply exp(-k^2 ds/12) in fourier space, in all 3 directions
        ker_multi_complex_real<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_in_1_one[STREAM], _d_boltz_bond_half, 1.0/static_cast<double>(M), M_COMPLEX);
        // 3D fourier discrete transform, backward and inplace
        if constexpr (std::is_same<T, double>::value)
            cufftExecZ2D(plan_bak_one[STREAM], d_qk_in_1_one[STREAM], d_q_out);
        else
            cufftExecZ2Z(plan_bak_one[STREAM], d_qk_in_1_one[STREAM], d_q_out, CUFFT_INVERSE);
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}
template <typename T>
void CudaSolverPseudoDiscrete<T>::compute_single_segment_stress(
    const int STREAM,
    CuDeviceData<T> *d_q_pair, CuDeviceData<T> *d_segment_stress,
    std::string monomer_type, bool is_half_bond_length)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM = this->cb->get_dim();
        // const int M   = total_grid;
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        auto bond_lengths = this->molecules->get_bond_lengths();
        double bond_length_sq;
        double *_d_boltz_bond;

        const double* _d_fourier_basis_x = pseudo->get_fourier_basis_x();
        const double* _d_fourier_basis_y = pseudo->get_fourier_basis_y();
        const double* _d_fourier_basis_z = pseudo->get_fourier_basis_z();
        const int* _d_k_idx = pseudo->get_negative_frequency_mapping();

        if (is_half_bond_length)
        {
            bond_length_sq = 0.5*bond_lengths[monomer_type]*bond_lengths[monomer_type];
            _d_boltz_bond = pseudo->get_boltz_bond_half(monomer_type);
        }
        else
        {
            bond_length_sq = bond_lengths[monomer_type]*bond_lengths[monomer_type];
            _d_boltz_bond = pseudo->get_boltz_bond(monomer_type);
        }

        // Execute a forward FFT
        if constexpr (std::is_same<T, double>::value)
            cufftExecD2Z(plan_for_two[STREAM], d_q_pair, d_qk_in_1_two[STREAM]);
        else
            cufftExecZ2Z(plan_for_two[STREAM], d_q_pair, d_qk_in_1_two[STREAM], CUFFT_FORWARD);

        // Multiply two propagators in the fourier spaces
        if constexpr (std::is_same<T, double>::value)
            ker_multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], &d_qk_in_1_two[STREAM][0], &d_qk_in_1_two[STREAM][M_COMPLEX], M_COMPLEX);
        else
        {
            ker_copy_data_with_idx<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_in_1_one[STREAM], &d_qk_in_1_two[STREAM][M_COMPLEX], _d_k_idx, M_COMPLEX);
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], &d_qk_in_1_two[STREAM][0], d_qk_in_1_one[STREAM], 1.0, M_COMPLEX);
        }
        ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], d_q_multi[STREAM], _d_boltz_bond, bond_length_sq, M_COMPLEX);
        
        if ( DIM == 3 )
        {
            // x direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_x, 1.0, M_COMPLEX);
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);

            // y direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_y, 1.0, M_COMPLEX);
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);

            // z direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_z, 1.0, M_COMPLEX);
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[2], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
        }
        if ( DIM == 2 )
        {
            // y direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_y, 1.0, M_COMPLEX);
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);

            // z direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_z, 1.0, M_COMPLEX);
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[1], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);

        }
        if ( DIM == 1 )
        {
            // z direction
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_stress_sum[STREAM], d_q_multi[STREAM], _d_fourier_basis_z, 1.0, M_COMPLEX);
            if constexpr (std::is_same<T, double>::value)
                cub::DeviceReduce::Sum(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, streams[STREAM][0]);
            else
                cub::DeviceReduce::Reduce(d_temp_storage[STREAM], temp_storage_bytes[STREAM], d_stress_sum[STREAM], &d_segment_stress[0], M_COMPLEX, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}, streams[STREAM][0]);
        }

    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation

template class CudaSolverPseudoDiscrete<double>;
template class CudaSolverPseudoDiscrete<std::complex<double>>;