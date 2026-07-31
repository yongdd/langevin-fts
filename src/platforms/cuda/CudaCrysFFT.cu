/**
 * @file CudaCrysFFT.cu
 * @brief CUDA implementation of crystallographic FFT using DCT-II/III.
 *
 * Uses CudaRealTransform for DCT-II (forward) and DCT-III (backward).
 *
 * Reference: Qiang & Li, Macromolecules (2020)
 */

#include "CudaCrysFFT.h"
#include "CudaRealTransform.h"
#include "CudaCommon.h"
#include <cmath>
#include <stdexcept>
#include <atomic>
#include <cstdlib>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//------------------------------------------------------------------------------
// CUDA Kernels
//------------------------------------------------------------------------------

/**
 * @brief Apply Boltzmann factor element-wise.
 *
 * Note: Boltzmann factor includes pre-multiplied normalization (1/M_logical)
 * to eliminate a separate normalize kernel.
 */
__global__ void kernel_apply_boltzmann_crys(
    double* __restrict__ d_data,
    const double* __restrict__ d_boltz,
    int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    d_data[idx] *= d_boltz[idx];
}

/**
 * @brief Apply raw multiplier with explicit normalization factor.
 */
__global__ void kernel_apply_multiplier_crys(
    double* __restrict__ d_data,
    const double* __restrict__ d_multiplier,
    double norm,
    int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    d_data[idx] *= d_multiplier[idx] * norm;
}

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
CudaCrysFFT::CudaCrysFFT(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> /* trans_part - ignored for DCT-II/III */)
    : nx_logical_(nx_logical),
      cell_para_(cell_para),
      d_boltz_current_(nullptr),
      ds_current_(0.0),
      dct_forward_(nullptr),
      dct_backward_(nullptr),
      d_work_(nullptr)
{
    // Validate even grid sizes
    for (int d = 0; d < 3; ++d)
    {
        if (nx_logical_[d] % 2 != 0)
        {
            throw std::invalid_argument("CudaCrysFFT requires even grid dimensions");
        }
    }

    // Physical grid: (N/2) in each dimension
    nx_physical_ = {
        nx_logical_[0] / 2,
        nx_logical_[1] / 2,
        nx_logical_[2] / 2
    };

    M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
    M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];

    // Normalization: 1/M_logical for round-trip
    norm_factor_ = 1.0 / static_cast<double>(M_logical_);

    // Create CudaRealTransform objects for DCT-II and DCT-III
    dct_forward_ = new CudaRealTransform3D(
        nx_physical_[0], nx_physical_[1], nx_physical_[2], CUDA_DCT_2);
    dct_backward_ = new CudaRealTransform3D(
        nx_physical_[0], nx_physical_[1], nx_physical_[2], CUDA_DCT_3);

    // Allocate device work buffer
    gpu_error_check(cudaMalloc(&d_work_, sizeof(double) * M_physical_));
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
CudaCrysFFT::~CudaCrysFFT()
{
    freeBoltzmann();
    freeMultiplierCaches();

    if (dct_forward_) delete dct_forward_;
    if (dct_backward_) delete dct_backward_;
    if (d_work_) cudaFree(d_work_);
}

//------------------------------------------------------------------------------
// Free cached stress multipliers
//------------------------------------------------------------------------------
void CudaCrysFFT::freeMultiplierCaches()
{
    for (int axis = 0; axis < 3; ++axis)
    {
        if (d_axis_k2_[axis])
        {
            cudaFree(d_axis_k2_[axis]);
            d_axis_k2_[axis] = nullptr;
        }
    }
    for (auto& kv : d_axis_boltz_k2_)
    {
        if (kv.second) cudaFree(kv.second);
    }
    d_axis_boltz_k2_.clear();
}

//------------------------------------------------------------------------------
// Free Boltzmann factors
//------------------------------------------------------------------------------
void CudaCrysFFT::freeBoltzmann()
{
    for (auto& kv : d_boltzmann_)
    {
        if (kv.second) cudaFree(kv.second);
    }
    d_boltzmann_.clear();
    d_boltz_current_ = nullptr;
}

//------------------------------------------------------------------------------
// Set cell parameters
//------------------------------------------------------------------------------
void CudaCrysFFT::set_cell_para(const std::array<double, 6>& cell_para)
{
    if (cell_para[0] == cell_para_[0] &&
        cell_para[1] == cell_para_[1] &&
        cell_para[2] == cell_para_[2])
    {
        return;
    }

    cell_para_ = cell_para;
    freeBoltzmann();
    freeMultiplierCaches();
    ds_current_ = 0.0;
}

//------------------------------------------------------------------------------
// Set contour step
//------------------------------------------------------------------------------
void CudaCrysFFT::set_contour_step(double ds)
{
    if (ds == ds_current_ && d_boltzmann_.count(ds) > 0) return;

    ds_current_ = ds;

    if (d_boltzmann_.count(ds) == 0)
    {
        generateBoltzmann(ds);
    }

    d_boltz_current_ = d_boltzmann_[ds];
}

//------------------------------------------------------------------------------
// Generate Boltzmann factors
//------------------------------------------------------------------------------
void CudaCrysFFT::generateBoltzmann(double ds)
{
    // Compute on host, then copy to device
    double* h_boltz = new double[M_physical_];

    double Lx = cell_para_[0];
    double Ly = cell_para_[1];
    double Lz = cell_para_[2];

    int Nx2 = nx_physical_[0];
    int Ny2 = nx_physical_[1];
    int Nz2 = nx_physical_[2];

    int idx = 0;
    for (int ix = 0; ix < Nx2; ++ix)
    {
        double kx = ix * 2.0 * M_PI / Lx;
        double kx2 = kx * kx;

        for (int iy = 0; iy < Ny2; ++iy)
        {
            double ky = iy * 2.0 * M_PI / Ly;
            double ky2 = ky * ky;

            for (int iz = 0; iz < Nz2; ++iz)
            {
                double kz = iz * 2.0 * M_PI / Lz;
                double kz2 = kz * kz;

                // Pre-multiply normalization factor (1/M_logical) into Boltzmann
                // This eliminates a separate normalize kernel in diffusion()
                h_boltz[idx++] = std::exp(-(kx2 + ky2 + kz2) * ds) * norm_factor_;
            }
        }
    }

    double* d_boltz;
    gpu_error_check(cudaMalloc(&d_boltz, sizeof(double) * M_physical_));
    gpu_error_check(cudaMemcpy(d_boltz, h_boltz, sizeof(double) * M_physical_, cudaMemcpyHostToDevice));

    d_boltzmann_[ds] = d_boltz;
    delete[] h_boltz;
}

//------------------------------------------------------------------------------
// Apply diffusion operator
//------------------------------------------------------------------------------
void CudaCrysFFT::diffusion(double* d_q_in, double* d_q_out)
{
    diffusion(d_q_in, d_q_out, stream_);
}

void CudaCrysFFT::diffusion(double* d_q_in, double* d_q_out, cudaStream_t stream)
{
    set_stream(stream);
    int threads = 256;
    int blocks = (M_physical_ + threads - 1) / threads;

    static std::atomic<bool> profile_once{false};
    const bool do_profile = (std::getenv("FTS_PROFILE_CRYSFFT_PMMM") != nullptr);
    const bool profile_this = do_profile && !profile_once.exchange(true);
    cudaEvent_t ev0{}, ev1{}, ev2{}, ev3{}, ev4{};
    if (profile_this)
    {
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventCreate(&ev2);
        cudaEventCreate(&ev3);
        cudaEventCreate(&ev4);
        cudaEventRecord(ev0, stream_);
    }

    // Determine working buffer to minimize memory copies
    // - If in-place (d_q_in == d_q_out): work directly on input
    // - If out-of-place: copy to output once, then work on output
    double* d_work = (d_q_in == d_q_out) ? d_q_in : d_q_out;

    if (d_q_in != d_q_out)
    {
        // Single copy at start (instead of two copies before/after)
        gpu_error_check(cudaMemcpyAsync(d_q_out, d_q_in, sizeof(double) * M_physical_,
                                        cudaMemcpyDeviceToDevice, stream_));
    }

    if (profile_this)
        cudaEventRecord(ev1, stream_);

    // Step 1: Forward DCT-II (in-place)
    dct_forward_->execute(d_work);

    if (profile_this)
        cudaEventRecord(ev2, stream_);

    // Step 2: Apply Boltzmann factor (in-place)
    kernel_apply_boltzmann_crys<<<blocks, threads, 0, stream_>>>(d_work, d_boltz_current_, M_physical_);

    if (profile_this)
        cudaEventRecord(ev3, stream_);

    // Step 3: Backward DCT-III (in-place)
    // Normalization is already included in Boltzmann factor (pre-multiplied in generateBoltzmann)
    dct_backward_->execute(d_work);

    if (profile_this)
    {
        cudaEventRecord(ev4, stream_);
        cudaEventSynchronize(ev4);
        float t_copy = 0.0f, t_fwd = 0.0f, t_boltz = 0.0f, t_bwd = 0.0f, t_total = 0.0f;
        cudaEventElapsedTime(&t_copy, ev0, ev1);
        cudaEventElapsedTime(&t_fwd, ev1, ev2);
        cudaEventElapsedTime(&t_boltz, ev2, ev3);
        cudaEventElapsedTime(&t_bwd, ev3, ev4);
        cudaEventElapsedTime(&t_total, ev0, ev4);

        std::cout << "[CrysFFT-Pmmm profile] M_phys=" << M_physical_
                  << " copy(ms)=" << t_copy
                  << " fwd(ms)=" << t_fwd
                  << " boltz(ms)=" << t_boltz
                  << " bwd(ms)=" << t_bwd
                  << " total(ms)=" << t_total << std::endl;

        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaEventDestroy(ev2);
        cudaEventDestroy(ev3);
        cudaEventDestroy(ev4);
    }
}

void CudaCrysFFT::set_stream(cudaStream_t stream)
{
    if (stream_ == stream)
        return;
    stream_ = stream;
    if (dct_forward_)
        dct_forward_->set_stream(stream_);
    if (dct_backward_)
        dct_backward_->set_stream(stream_);
}

//------------------------------------------------------------------------------
// Apply arbitrary k-space multiplier (for stress computation)
//------------------------------------------------------------------------------
void CudaCrysFFT::apply_multiplier(double* d_q_in, double* d_q_out, const double* d_multiplier, cudaStream_t stream)
{
    set_stream(stream);
    int threads = 256;
    int blocks = (M_physical_ + threads - 1) / threads;

    // Same in-place strategy as diffusion(): work directly on the output buffer
    double* d_work = (d_q_in == d_q_out) ? d_q_in : d_q_out;
    if (d_q_in != d_q_out)
    {
        gpu_error_check(cudaMemcpyAsync(d_q_out, d_q_in, sizeof(double) * M_physical_,
                                        cudaMemcpyDeviceToDevice, stream_));
    }

    // Forward DCT-II (in-place)
    dct_forward_->execute(d_work);

    // Apply raw multiplier with round-trip normalization
    kernel_apply_multiplier_crys<<<blocks, threads, 0, stream_>>>(d_work, d_multiplier, norm_factor_, M_physical_);
    gpu_error_check(cudaPeekAtLastError());

    // Backward DCT-III (in-place)
    dct_backward_->execute(d_work);
}

//------------------------------------------------------------------------------
// Per-axis k² arrays (host computation)
//------------------------------------------------------------------------------
void CudaCrysFFT::computeAxisK2Host(std::vector<double>& kx2, std::vector<double>& ky2, std::vector<double>& kz2) const
{
    kx2.resize(M_physical_);
    ky2.resize(M_physical_);
    kz2.resize(M_physical_);

    const double factor_x = 2.0 * M_PI / cell_para_[0];
    const double factor_y = 2.0 * M_PI / cell_para_[1];
    const double factor_z = 2.0 * M_PI / cell_para_[2];

    const int Nx2 = nx_physical_[0];
    const int Ny2 = nx_physical_[1];
    const int Nz2 = nx_physical_[2];

    int idx = 0;
    for (int ix = 0; ix < Nx2; ++ix)
    {
        double kx = ix * factor_x;
        double vx = kx * kx;
        for (int iy = 0; iy < Ny2; ++iy)
        {
            double ky = iy * factor_y;
            double vy = ky * ky;
            for (int iz = 0; iz < Nz2; ++iz)
            {
                double kz = iz * factor_z;
                kx2[idx] = vx;
                ky2[idx] = vy;
                kz2[idx] = kz * kz;
                ++idx;
            }
        }
    }
}

//------------------------------------------------------------------------------
// Cached device multiplier: k_axis² (continuous chain stress)
//------------------------------------------------------------------------------
const double* CudaCrysFFT::get_axis_k2_multiplier(int axis)
{
    if (axis < 0 || axis > 2)
        throw std::invalid_argument("CudaCrysFFT::get_axis_k2_multiplier: axis must be 0, 1, or 2");

    if (d_axis_k2_[axis] != nullptr)
        return d_axis_k2_[axis];

    std::vector<double> kx2, ky2, kz2;
    computeAxisK2Host(kx2, ky2, kz2);
    const std::vector<double>* selected[3] = {&kx2, &ky2, &kz2};

    // Upload all three axes at once (they share the host computation)
    for (int a = 0; a < 3; ++a)
    {
        if (d_axis_k2_[a] != nullptr)
            continue;
        double* d_ptr;
        gpu_error_check(cudaMalloc(&d_ptr, sizeof(double) * M_physical_));
        gpu_error_check(cudaMemcpy(d_ptr, selected[a]->data(), sizeof(double) * M_physical_, cudaMemcpyHostToDevice));
        d_axis_k2_[a] = d_ptr;
    }

    return d_axis_k2_[axis];
}

//------------------------------------------------------------------------------
// Cached device multiplier: exp(-k²·coeff)·k_axis² (discrete chain stress)
//------------------------------------------------------------------------------
const double* CudaCrysFFT::get_axis_boltz_k2_multiplier(int axis, double coeff)
{
    if (axis < 0 || axis > 2)
        throw std::invalid_argument("CudaCrysFFT::get_axis_boltz_k2_multiplier: axis must be 0, 1, or 2");

    auto key = std::make_pair(axis, coeff);
    auto it = d_axis_boltz_k2_.find(key);
    if (it != d_axis_boltz_k2_.end())
        return it->second;

    std::vector<double> kx2, ky2, kz2;
    computeAxisK2Host(kx2, ky2, kz2);

    std::vector<double> h_mult(M_physical_);
    for (int i = 0; i < M_physical_; ++i)
    {
        double k2 = kx2[i] + ky2[i] + kz2[i];
        double axis_k2 = (axis == 0) ? kx2[i] : (axis == 1) ? ky2[i] : kz2[i];
        h_mult[i] = std::exp(-k2 * coeff) * axis_k2;
    }

    double* d_ptr;
    gpu_error_check(cudaMalloc(&d_ptr, sizeof(double) * M_physical_));
    gpu_error_check(cudaMemcpy(d_ptr, h_mult.data(), sizeof(double) * M_physical_, cudaMemcpyHostToDevice));
    d_axis_boltz_k2_[key] = d_ptr;

    return d_ptr;
}
