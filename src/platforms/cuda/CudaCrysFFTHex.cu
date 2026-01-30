/**
 * @file CudaCrysFFTHex.cu
 * @brief CUDA implementation of hexagonal CrysFFT (DCT-z + FFT-xy).
 */

#include <cmath>
#include <limits>
#include <stdexcept>

#include "CudaCrysFFTHex.h"
#include "CudaRealTransform.h"
#include "CudaCommon.h"
#include "Exception.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {
// transpose kernel removed; DCT3 now reads z-major directly
}  // namespace

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------
std::array<double, 6> CudaCrysFFTHex::compute_recip_metric(const std::array<double, 6>& cell_para)
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
CudaCrysFFTHex::CudaCrysFFTHex(
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
            throw_with_line_number("CudaCrysFFTHex requires positive grid dimensions.");
    }
    if (nx_logical_[2] % 2 != 0)
        throw_with_line_number("CudaCrysFFTHex requires even Nz.");

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

CudaCrysFFTHex::~CudaCrysFFTHex()
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
void CudaCrysFFTHex::init_fft_xy()
{
    int n[2] = {nx_logical_[0], nx_logical_[1]};
    int inembed[2] = {n[0], n[1]};
    int onembed[2] = {n[0], n[1] / 2 + 1};
    int istride = 1;
    int ostride = 1;
    int idist = nx_logical_[0] * nx_logical_[1];
    int odist = nx_logical_[0] * (nx_logical_[1] / 2 + 1);
    int batch = nx_physical_[2];

    if (cufftPlanMany(&plan_xy_fwd_, 2, n,
                      inembed, istride, idist,
                      onembed, ostride, odist,
                      CUFFT_D2Z, batch) != CUFFT_SUCCESS)
    {
        throw_with_line_number("CudaCrysFFTHex failed to create cuFFT plan for XY.");
    }
    if (cufftPlanMany(&plan_xy_bwd_, 2, n,
                      onembed, ostride, odist,
                      inembed, istride, idist,
                      CUFFT_Z2D, batch) != CUFFT_SUCCESS)
    {
        cufftDestroy(plan_xy_fwd_);
        throw_with_line_number("CudaCrysFFTHex failed to create cuFFT inverse plan for XY.");
    }
    plan_xy_initialized_ = true;
    cufftSetStream(plan_xy_fwd_, stream_);
    cufftSetStream(plan_xy_bwd_, stream_);
}

void CudaCrysFFTHex::free_fft_xy()
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
void CudaCrysFFTHex::freeBoltzmann()
{
    for (auto& kv : d_boltzmann_)
        cudaFree(kv.second);
    d_boltzmann_.clear();
    d_boltz_current_ = nullptr;
    ds_current_ = 0.0;
}

void CudaCrysFFTHex::generateBoltzmann(double ds)
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
void CudaCrysFFTHex::set_cell_para(const std::array<double, 6>& cell_para)
{
    if (cell_para == cell_para_)
        return;
    cell_para_ = cell_para;
    recip_metric_ = compute_recip_metric(cell_para_);
    freeBoltzmann();
}

void CudaCrysFFTHex::set_contour_step(double ds)
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

void CudaCrysFFTHex::set_stream(cudaStream_t stream)
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

void CudaCrysFFTHex::diffusion(double* d_q_in, double* d_q_out)
{
    diffusion(d_q_in, d_q_out, stream_);
}

void CudaCrysFFTHex::diffusion(double* d_q_in, double* d_q_out, cudaStream_t stream)
{
    set_stream(stream);
    if (d_boltz_current_ == nullptr)
        throw_with_line_number("CudaCrysFFTHex::set_contour_step must be called before diffusion().");

    const int threads = 256;
    const int blocks = (M_complex_xy_ + threads - 1) / threads;

    double* d_work = (d_q_in == d_q_out) ? d_q_in : d_q_out;
    if (d_q_in != d_q_out)
    {
        gpu_error_check(cudaMemcpyAsync(d_q_out, d_q_in, sizeof(double) * M_physical_,
                                        cudaMemcpyDeviceToDevice, stream_));
    }

    // DCT-II along z -> z-major layout (contiguous XY per z)
    dct_forward_z_->execute_z_dct2_to_zmajor(d_work, d_real_xy_, stream_);

    // FFT along x,y (batched over z)
    if (cufftExecD2Z(plan_xy_fwd_, d_real_xy_, d_complex_) != CUFFT_SUCCESS)
        throw_with_line_number("CudaCrysFFTHex cufftExecD2Z forward failed.");

    // Apply Boltzmann in spectral space (includes normalization)
    ker_multi_complex_real<<<blocks, threads, 0, stream_>>>(d_complex_, d_boltz_current_, 1.0, M_complex_xy_);

    if (cufftExecZ2D(plan_xy_bwd_, d_complex_, d_real_xy_) != CUFFT_SUCCESS)
        throw_with_line_number("CudaCrysFFTHex cufftExecZ2D inverse failed.");

    // DCT-III along z from z-major (normalization already applied in boltzmann)
    dct_backward_z_->execute_z_dct3_from_zmajor(d_real_xy_, d_work, stream_);
}
