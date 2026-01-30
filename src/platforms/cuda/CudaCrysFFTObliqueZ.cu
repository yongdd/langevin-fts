/**
 * @file CudaCrysFFTObliqueZ.cu
 * @brief CUDA implementation of z-mirror CrysFFT (DCT-z + FFT-xy).
 */

#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <mutex>

#include "CudaCrysFFTObliqueZ.h"
#include "CudaRealTransform.h"
#include "CudaCommon.h"
#include "Exception.h"

#include <cufftXt.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {
// transpose kernel removed; DCT3 now reads z-major directly

}  // namespace

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------
std::array<double, 6> CudaCrysFFTObliqueZ::compute_recip_metric(const std::array<double, 6>& cell_para)
{
    const double a = cell_para[0];
    const double b = cell_para[1];
    const double c = cell_para[2];
    const double alpha = cell_para[3];
    const double beta = cell_para[4];
    const double gamma = cell_para[5];

    const double cos_a = std::cos(alpha);
    const double cos_b = std::cos(beta);
    const double cos_g = std::cos(gamma);
    const double sin_g = std::sin(gamma);

    const double vol_factor_sq =
        1.0 - cos_a * cos_a - cos_b * cos_b - cos_g * cos_g + 2.0 * cos_a * cos_b * cos_g;
    const double vol_factor = std::sqrt(vol_factor_sq);
    const double volume = a * b * c * vol_factor;

    const double ax = a, ay = 0.0, az = 0.0;
    const double bx = b * cos_g, by = b * sin_g, bz = 0.0;
    const double cx = c * cos_b;
    const double cy = (sin_g != 0.0) ? c * (cos_a - cos_b * cos_g) / sin_g : 0.0;
    const double cz = volume / (a * b * sin_g);

    const double bc_x = by * cz - bz * cy;
    const double bc_y = bz * cx - bx * cz;
    const double bc_z = bx * cy - by * cx;

    const double ca_x = cy * az - cz * ay;
    const double ca_y = cz * ax - cx * az;
    const double ca_z = cx * ay - cy * ax;

    const double ab_x = ay * bz - az * by;
    const double ab_y = az * bx - ax * bz;
    const double ab_z = ax * by - ay * bx;

    const double a_star_x = bc_x / volume;
    const double a_star_y = bc_y / volume;
    const double a_star_z = bc_z / volume;
    const double b_star_x = ca_x / volume;
    const double b_star_y = ca_y / volume;
    const double b_star_z = ca_z / volume;
    const double c_star_x = ab_x / volume;
    const double c_star_y = ab_y / volume;
    const double c_star_z = ab_z / volume;

    std::array<double, 6> metric{};
    metric[0] = a_star_x * a_star_x + a_star_y * a_star_y + a_star_z * a_star_z;
    metric[1] = a_star_x * b_star_x + a_star_y * b_star_y + a_star_z * b_star_z;
    metric[2] = a_star_x * c_star_x + a_star_y * c_star_y + a_star_z * c_star_z;
    metric[3] = b_star_x * b_star_x + b_star_y * b_star_y + b_star_z * b_star_z;
    metric[4] = b_star_x * c_star_x + b_star_y * c_star_y + b_star_z * c_star_z;
    metric[5] = c_star_x * c_star_x + c_star_y * c_star_y + c_star_z * c_star_z;
    return metric;
}

//------------------------------------------------------------------------------
// Constructor / Destructor
//------------------------------------------------------------------------------
CudaCrysFFTObliqueZ::CudaCrysFFTObliqueZ(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> /*trans_part*/)
    : nx_logical_(nx_logical),
      cell_para_(cell_para),
      recip_metric_(compute_recip_metric(cell_para))
{
    for (int d = 0; d < 3; ++d)
    {
        if (nx_logical_[d] <= 0)
            throw_with_line_number("CudaCrysFFTObliqueZ requires positive grid dimensions.");
    }
    if (nx_logical_[2] % 2 != 0)
        throw_with_line_number("CudaCrysFFTObliqueZ requires even Nz.");

    nx_physical_ = {nx_logical_[0], nx_logical_[1], nx_logical_[2] / 2};
    M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
    M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];
    M_complex_xy_ = nx_logical_[0] * (nx_logical_[1] / 2 + 1) * nx_physical_[2];
    norm_factor_ = 1.0 / static_cast<double>(M_logical_);

    dct_forward_z_ = new CudaRealTransform3D(nx_physical_[0], nx_physical_[1], nx_physical_[2], CUDA_DCT_2);
    dct_backward_z_ = new CudaRealTransform3D(nx_physical_[0], nx_physical_[1], nx_physical_[2], CUDA_DCT_3);

    gpu_error_check(cudaMalloc(reinterpret_cast<void**>(&d_complex_),
                               sizeof(cufftDoubleComplex) * M_complex_xy_));
    gpu_error_check(cudaMalloc(reinterpret_cast<void**>(&d_real_xy_),
                               sizeof(double) * M_physical_));

    init_fft_xy();
}

CudaCrysFFTObliqueZ::~CudaCrysFFTObliqueZ()
{
    freeBoltzmann();
    free_fft_xy();

    if (dct_forward_z_) delete dct_forward_z_;
    if (dct_backward_z_) delete dct_backward_z_;
    dct_forward_z_ = nullptr;
    dct_backward_z_ = nullptr;

    if (d_complex_)
        cudaFree(d_complex_);
    d_complex_ = nullptr;
    if (d_real_xy_)
        cudaFree(d_real_xy_);
    d_real_xy_ = nullptr;
}

//------------------------------------------------------------------------------
// FFT plans
//------------------------------------------------------------------------------
void CudaCrysFFTObliqueZ::init_fft_xy()
{
    long long int n_ll[2] = {nx_logical_[0], nx_logical_[1]};
    long long int inembed_ll[2] = {n_ll[0], n_ll[1]};
    long long int onembed_ll[2] = {n_ll[0], n_ll[1] / 2 + 1};
    long long int istride = 1;
    long long int ostride = 1;
    long long int idist = nx_logical_[0] * nx_logical_[1];
    long long int odist = nx_logical_[0] * (nx_logical_[1] / 2 + 1);
    long long int batch = nx_physical_[2];
    size_t work_size = 0;

    if (cufftCreate(&plan_xy_fwd_) != CUFFT_SUCCESS)
        throw_with_line_number("CudaCrysFFTObliqueZ failed to create cuFFT plan handle for XY.");
    if (cufftCreate(&plan_xy_bwd_) != CUFFT_SUCCESS)
    {
        cufftDestroy(plan_xy_fwd_);
        throw_with_line_number("CudaCrysFFTObliqueZ failed to create cuFFT inverse plan handle for XY.");
    }

    cufftResult rf = cufftXtMakePlanMany(plan_xy_fwd_, 2, n_ll,
                                         inembed_ll, istride, idist, CUDA_R_64F,
                                         onembed_ll, ostride, odist, CUDA_C_64F,
                                         batch, &work_size, CUDA_C_64F);
    if (rf != CUFFT_SUCCESS)
    {
        cufftDestroy(plan_xy_fwd_);
        cufftDestroy(plan_xy_bwd_);
        throw_with_line_number("CudaCrysFFTObliqueZ failed to create cuFFT plan for XY.");
    }

    cufftResult rb = cufftXtMakePlanMany(plan_xy_bwd_, 2, n_ll,
                                         onembed_ll, ostride, odist, CUDA_C_64F,
                                         inembed_ll, istride, idist, CUDA_R_64F,
                                         batch, &work_size, CUDA_C_64F);
    if (rb != CUFFT_SUCCESS)
    {
        cufftDestroy(plan_xy_fwd_);
        cufftDestroy(plan_xy_bwd_);
        throw_with_line_number("CudaCrysFFTObliqueZ failed to create cuFFT inverse plan for XY.");
    }

    plan_xy_initialized_ = true;
    cufftSetStream(plan_xy_fwd_, stream_);
    cufftSetStream(plan_xy_bwd_, stream_);

}

void CudaCrysFFTObliqueZ::free_fft_xy()
{
    if (plan_xy_initialized_)
    {
        cufftDestroy(plan_xy_fwd_);
        cufftDestroy(plan_xy_bwd_);
        plan_xy_initialized_ = false;
    }
}

//------------------------------------------------------------------------------
// Boltzmann cache
//------------------------------------------------------------------------------
void CudaCrysFFTObliqueZ::freeBoltzmann()
{
    for (auto& kv : d_boltzmann_)
        cudaFree(kv.second);
    d_boltzmann_.clear();
    d_boltz_current_ = nullptr;
    ds_current_ = 0.0;
}

void CudaCrysFFTObliqueZ::generateBoltzmann(double ds)
{
    double* h_boltz = new double[M_complex_xy_];

    const double G11 = recip_metric_[0];
    const double G12 = recip_metric_[1];
    const double G13 = recip_metric_[2];
    const double G22 = recip_metric_[3];
    const double G23 = recip_metric_[4];
    const double G33 = recip_metric_[5];
    const double factor = 4.0 * M_PI * M_PI;

    int idx = 0;
    for (int iz = 0; iz < nx_physical_[2]; ++iz)
    {
        int m3 = iz;
        for (int ix = 0; ix < nx_logical_[0]; ++ix)
        {
            int m1 = (ix > nx_logical_[0] / 2) ? (ix - nx_logical_[0]) : ix;
            for (int iy = 0; iy < nx_logical_[1] / 2 + 1; ++iy)
            {
                int m2 = iy;
                double gmm = G11 * m1 * m1 + G22 * m2 * m2 + G33 * m3 * m3
                           + 2.0 * (G12 * m1 * m2 + G13 * m1 * m3 + G23 * m2 * m3);
                double k2 = factor * gmm;
                h_boltz[idx++] = std::exp(-k2 * ds) * norm_factor_;
            }
        }
    }

    double* d_boltz = nullptr;
    gpu_error_check(cudaMalloc(&d_boltz, sizeof(double) * M_complex_xy_));
    gpu_error_check(cudaMemcpy(d_boltz, h_boltz, sizeof(double) * M_complex_xy_, cudaMemcpyHostToDevice));
    d_boltzmann_[ds] = d_boltz;
    delete[] h_boltz;
}

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------
void CudaCrysFFTObliqueZ::set_cell_para(const std::array<double, 6>& cell_para)
{
    if (cell_para == cell_para_)
        return;
    cell_para_ = cell_para;
    recip_metric_ = compute_recip_metric(cell_para_);
    freeBoltzmann();
}

void CudaCrysFFTObliqueZ::set_contour_step(double ds)
{
    if (ds == ds_current_ && d_boltzmann_.count(ds) > 0)
    {
        d_boltz_current_ = d_boltzmann_[ds];
        return;
    }

    if (d_boltzmann_.count(ds) == 0)
        generateBoltzmann(ds);

    d_boltz_current_ = d_boltzmann_[ds];
    ds_current_ = ds;
}

void CudaCrysFFTObliqueZ::set_stream(cudaStream_t stream)
{
    stream_ = stream;
    if (dct_forward_z_) dct_forward_z_->set_stream(stream_);
    if (dct_backward_z_) dct_backward_z_->set_stream(stream_);
    if (plan_xy_initialized_)
    {
        cufftSetStream(plan_xy_fwd_, stream_);
        cufftSetStream(plan_xy_bwd_, stream_);
    }
}

void CudaCrysFFTObliqueZ::diffusion(double* d_q_in, double* d_q_out)
{
    diffusion(d_q_in, d_q_out, stream_);
}

void CudaCrysFFTObliqueZ::diffusion(double* d_q_in, double* d_q_out, cudaStream_t stream)
{
    set_stream(stream);
    if (d_boltz_current_ == nullptr)
        throw_with_line_number("CudaCrysFFTObliqueZ::set_contour_step must be called before diffusion().");

    const int threads = 256;
    const int blocks = (M_complex_xy_ + threads - 1) / threads;

    double* d_work = (d_q_in == d_q_out) ? d_q_in : d_q_out;
    const bool do_profile = (std::getenv("FTS_PROFILE_CRYSFFT_HEX_DETAIL") != nullptr);
    static std::atomic<bool> prof_once{false};
    const bool profile_this = do_profile && !prof_once.exchange(true);

    if (profile_this)
    {
        cudaEvent_t e0{}, e1{}, e2{}, e3{}, e4{}, e5{}, e6{};
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);
        cudaEventCreate(&e3);
        cudaEventCreate(&e4);
        cudaEventCreate(&e5);
        cudaEventCreate(&e6);

        cudaEventRecord(e0, stream_);
        if (d_q_in != d_q_out)
        {
            gpu_error_check(cudaMemcpyAsync(d_q_out, d_q_in, sizeof(double) * M_physical_,
                                            cudaMemcpyDeviceToDevice, stream_));
        }
        cudaEventRecord(e1, stream_);

        // DCT-II along z -> z-major layout (contiguous XY per z)
        dct_forward_z_->execute_z_dct2_to_zmajor(d_work, d_real_xy_, stream_);
        cudaEventRecord(e2, stream_);

        // FFT along x,y (batched over z)
        if (cufftExecD2Z(plan_xy_fwd_, d_real_xy_, d_complex_) != CUFFT_SUCCESS)
            throw_with_line_number("CudaCrysFFTObliqueZ cufftExecD2Z forward failed.");
        cudaEventRecord(e3, stream_);

        // Apply Boltzmann in spectral space (includes normalization)
        ker_multi_complex_real<<<blocks, threads, 0, stream_>>>(d_complex_, d_boltz_current_, 1.0, M_complex_xy_);
        cudaEventRecord(e4, stream_);

        if (cufftExecZ2D(plan_xy_bwd_, d_complex_, d_real_xy_) != CUFFT_SUCCESS)
            throw_with_line_number("CudaCrysFFTObliqueZ cufftExecZ2D inverse failed.");
        cudaEventRecord(e5, stream_);

        // DCT-III along z from z-major (normalization already applied in boltzmann)
        dct_backward_z_->execute_z_dct3_from_zmajor(d_real_xy_, d_work, stream_);
        cudaEventRecord(e6, stream_);

        cudaEventSynchronize(e6);

        float ms_copy = 0.0f, ms_dct2 = 0.0f, ms_fft = 0.0f, ms_boltz = 0.0f, ms_ifft = 0.0f, ms_dct3 = 0.0f, ms_total = 0.0f;
        cudaEventElapsedTime(&ms_copy, e0, e1);
        cudaEventElapsedTime(&ms_dct2, e1, e2);
        cudaEventElapsedTime(&ms_fft, e2, e3);
        cudaEventElapsedTime(&ms_boltz, e3, e4);
        cudaEventElapsedTime(&ms_ifft, e4, e5);
        cudaEventElapsedTime(&ms_dct3, e5, e6);
        cudaEventElapsedTime(&ms_total, e0, e6);

        std::printf("[CrysFFT-ObliqueZ] copy(ms)=%.4f dct2(ms)=%.4f fft(ms)=%.4f boltz(ms)=%.4f ifft(ms)=%.4f dct3(ms)=%.4f total(ms)=%.4f\n",
                    ms_copy, ms_dct2, ms_fft, ms_boltz, ms_ifft, ms_dct3, ms_total);

        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaEventDestroy(e2);
        cudaEventDestroy(e3);
        cudaEventDestroy(e4);
        cudaEventDestroy(e5);
        cudaEventDestroy(e6);
        return;
    }

    if (d_q_in != d_q_out)
    {
        gpu_error_check(cudaMemcpyAsync(d_q_out, d_q_in, sizeof(double) * M_physical_,
                                        cudaMemcpyDeviceToDevice, stream_));
    }

    // DCT-II along z -> z-major layout (contiguous XY per z)
    dct_forward_z_->execute_z_dct2_to_zmajor(d_work, d_real_xy_, stream_);

    // FFT along x,y (batched over z)
    if (cufftExecD2Z(plan_xy_fwd_, d_real_xy_, d_complex_) != CUFFT_SUCCESS)
        throw_with_line_number("CudaCrysFFTObliqueZ cufftExecD2Z forward failed.");

    // Apply Boltzmann in spectral space (includes normalization)
    ker_multi_complex_real<<<blocks, threads, 0, stream_>>>(d_complex_, d_boltz_current_, 1.0, M_complex_xy_);

    if (cufftExecZ2D(plan_xy_bwd_, d_complex_, d_real_xy_) != CUFFT_SUCCESS)
        throw_with_line_number("CudaCrysFFTObliqueZ cufftExecZ2D inverse failed.");

    // DCT-III along z from z-major (normalization already applied in boltzmann)
    dct_backward_z_->execute_z_dct3_from_zmajor(d_real_xy_, d_work, stream_);
}

 
