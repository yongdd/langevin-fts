/**
 * @file CudaCrysFFTRecursive3m.cu
 * @brief CUDA crystallographic FFT using 2x2x2 (3m) algorithm.
 */

#include "CudaCrysFFTRecursive3m.h"

#include "CudaCommon.h"
#include "Exception.h"

#include <cstring>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace {
constexpr double kPi = 3.14159265358979323846;

std::vector<std::array<double, 3>> generate_m3_operations(
    const std::array<double, 9>& g,
    const std::array<int, 3>& s)
{
    return {
        {-2.0 * kPi * (g[0]),                              -2.0 * kPi * (g[1]),                              -2.0 * kPi * (g[2] - 1.0 / s[2])},
        {-2.0 * kPi * (g[3]),                              -2.0 * kPi * (g[4] - 1.0 / s[1]),                 -2.0 * kPi * (g[5])},
        {-2.0 * kPi * (g[0] + g[3]),                       -2.0 * kPi * (-g[1] + g[4] - 1.0 / s[1]),         -2.0 * kPi * (g[2] + g[5] - 1.0 / s[2])},
        {-2.0 * kPi * (g[6] - 1.0 / s[0]),                 -2.0 * kPi * (g[7]),                              -2.0 * kPi * (g[8])},
        {-2.0 * kPi * (g[6] - g[0] - 1.0 / s[0]),          -2.0 * kPi * (g[1] + g[7]),                       -2.0 * kPi * (g[2] + g[8] - 1.0 / s[2])},
        {-2.0 * kPi * (-g[3] + g[6] - 1.0 / s[0]),         -2.0 * kPi * (g[4] + g[7] - 1.0 / s[1]),          -2.0 * kPi * (g[5] + g[8])},
        {-2.0 * kPi * (-g[0] - g[3] + g[6] - 1.0 / s[0]),  -2.0 * kPi * (-g[1] + g[4] + g[7] - 1.0 / s[1]),  -2.0 * kPi * (g[2] + g[5] + g[8] - 1.0 / s[2])}
    };
}

void twiddle_factor(
    const std::vector<std::tuple<double*, double*, std::array<double, 3>>>& data,
    const std::array<int, 3>& range)
{
    const int Nx = range[0];
    const int Ny = range[1];
    const int Nz = range[2];

    for (size_t itr_task = 0; itr_task < data.size(); ++itr_task)
    {
        double* dstcos = std::get<0>(data[itr_task]);
        double* dstsin = std::get<1>(data[itr_task]);
        auto paras = std::get<2>(data[itr_task]);
        for (int ix = 0; ix < Nx; ++ix)
        {
            for (int iy = 0; iy < Ny; ++iy)
            {
                size_t base = static_cast<size_t>(ix * Ny + iy) * Nz;
                for (int iz = 0; iz < Nz; ++iz)
                {
                    double angle = ix * paras[0] + iy * paras[1] + iz * paras[2];
                    dstcos[base + iz] = std::cos(angle);
                    dstsin[base + iz] = std::sin(angle);
                }
            }
        }
    }
}

void mat_split(
    const double* src,
    const std::array<double*, 8>& dst,
    const std::array<int, 3>& size)
{
    const int Nx = size[0];
    const int Ny = size[1];
    const int Nz = size[2];

    const int Nx2 = Nx / 2;
    const int Ny2 = Ny / 2;
    const int Nz2 = Nz / 2;

    for (int part = 0; part < 8; ++part)
    {
        const double* local_src = src
            + ((part & 4) >> 2) * Nx2 * Ny * Nz
            + ((part & 2) >> 1) * Ny2 * Nz
            + (part & 1) * Nz2;
        double* local_dst = dst[part];

        for (int ix = 0; ix < Nx2; ++ix)
        {
            for (int iy = 0; iy < Ny2; ++iy)
            {
                const double* src_row = local_src + (ix * Ny + iy) * Nz;
                double* dst_row = local_dst + (ix * Ny2 + iy) * Nz2;
                std::memcpy(dst_row, src_row, sizeof(double) * Nz2);
            }
        }
    }
}

void add_sub_by_seq(
    const std::array<double*, 8>& dst,
    const std::array<std::pair<int, int>, 4>& seq,
    const std::array<int, 3>& halfsize)
{
    const int Nx = halfsize[0];
    const int Ny = halfsize[1];
    const int Nz = halfsize[2];
    const int NxNy = Nx * Ny;

    for (size_t itr_seq = 0; itr_seq < seq.size(); ++itr_seq)
    {
        double* dst0 = dst[seq[itr_seq].first];
        double* dst1 = dst[seq[itr_seq].second];
        for (int itr_xy = 0; itr_xy < NxNy; ++itr_xy)
        {
            size_t base = static_cast<size_t>(itr_xy) * Nz;
            for (int itr_z = 0; itr_z < Nz; ++itr_z)
            {
                double temp = dst0[base + itr_z] + dst1[base + itr_z];
                dst1[base + itr_z] = dst0[base + itr_z] - dst1[base + itr_z];
                dst0[base + itr_z] = temp;
            }
        }
    }
}

void mul_add_sub_by_seq(
    const std::array<double*, 8>& dst_re,
    const std::array<double*, 8>& dst_im,
    const std::array<double*, 8>& src_re,
    const std::array<double*, 8>& src_im,
    const std::array<double*, 8>& src_mul,
    const std::array<std::pair<int, int>, 4>& seq,
    const std::array<int, 3>& halfsize)
{
    const int Nx = halfsize[0];
    const int Ny = halfsize[1];
    const int Nz = halfsize[2];
    const int NxNy = Nx * Ny;

    for (size_t itr_seq = 0; itr_seq < seq.size(); ++itr_seq)
    {
        int i0 = seq[itr_seq].first;
        int i1 = seq[itr_seq].second;
        double* dst_re0 = dst_re[i0];
        double* dst_re1 = dst_re[i1];
        double* dst_im0 = dst_im[i0];
        double* dst_im1 = dst_im[i1];
        const double* src_re0 = src_re[i0];
        const double* src_re1 = src_re[i1];
        const double* src_im0 = src_im[i0];
        const double* src_im1 = src_im[i1];
        const double* mul0 = src_mul[i0];
        const double* mul1 = src_mul[i1];

        for (int itr_xy = 0; itr_xy < NxNy; ++itr_xy)
        {
            size_t base = static_cast<size_t>(itr_xy) * Nz;
            for (int itr_z = 0; itr_z < Nz; ++itr_z)
            {
                size_t idx = base + itr_z;
                dst_re0[idx] = mul0[idx] * src_re0[idx];
                dst_im0[idx] = mul0[idx] * src_im0[idx];
                dst_re1[idx] = mul1[idx] * src_re1[idx];
                dst_im1[idx] = mul1[idx] * src_im1[idx];

                double tempre = dst_re0[idx] + dst_re1[idx];
                dst_re1[idx] = dst_re0[idx] - dst_re1[idx];
                dst_re0[idx] = tempre;

                double tempim = dst_im0[idx] + dst_im1[idx];
                dst_im1[idx] = dst_im0[idx] - dst_im1[idx];
                dst_im0[idx] = tempim;
            }
        }
    }
}

__global__ void ker_apply_k_3m(
    cufftDoubleComplex* __restrict__ dst,
    const cufftDoubleComplex* __restrict__ src,
    const double* __restrict__ k_re0,
    const double* __restrict__ k_re1,
    const double* __restrict__ k_re2,
    const double* __restrict__ k_re3,
    const double* __restrict__ k_re4,
    const double* __restrict__ k_re5,
    const double* __restrict__ k_re6,
    const double* __restrict__ k_re7,
    const double* __restrict__ k_im0,
    const double* __restrict__ k_im1,
    const double* __restrict__ k_im2,
    const double* __restrict__ k_im3,
    const double* __restrict__ k_im4,
    const double* __restrict__ k_im5,
    const double* __restrict__ k_im6,
    const double* __restrict__ k_im7,
    int Nx2, int Ny2, int Nz2, int Nz2c)
{
    const int total = Nx2 * Ny2 * Nz2c;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total)
    {
        int iz = idx % Nz2c;
        int tmp = idx / Nz2c;
        int iy = tmp % Ny2;
        int ix = tmp / Ny2;

        int ix_m = (Nx2 - ix) % Nx2;
        int iy_m = (Ny2 - iy) % Ny2;

        size_t base_c = static_cast<size_t>(ix * Ny2 + iy) * Nz2c;
        size_t base_k = static_cast<size_t>(ix * Ny2 + iy) * Nz2;
        size_t base_c_110 = static_cast<size_t>(ix_m * Ny2 + iy_m) * Nz2c;
        size_t base_c_010 = static_cast<size_t>(ix * Ny2 + iy_m) * Nz2c;
        size_t base_c_100 = static_cast<size_t>(ix_m * Ny2 + iy) * Nz2c;
        size_t idx_k = base_k + iz;

        cufftDoubleComplex s000 = src[base_c + iz];
        cufftDoubleComplex s110 = src[base_c_110 + iz];
        cufftDoubleComplex s010 = src[base_c_010 + iz];
        cufftDoubleComplex s100 = src[base_c_100 + iz];

        double out_re =
            s000.x * k_re0[idx_k] - s000.y * k_im7[idx_k] +
            s110.x * k_re6[idx_k] - s110.y * k_im1[idx_k] +
            s010.x * k_re2[idx_k] - s010.y * k_im5[idx_k] +
            s100.x * k_re4[idx_k] - s100.y * k_im3[idx_k];

        double out_im =
            s000.x * k_im0[idx_k] + s000.y * k_re7[idx_k] +
            s110.x * k_im6[idx_k] + s110.y * k_re1[idx_k] +
            s010.x * k_im2[idx_k] + s010.y * k_re5[idx_k] +
            s100.x * k_im4[idx_k] + s100.y * k_re3[idx_k];

        dst[base_c + iz].x = out_re;
        dst[base_c + iz].y = out_im;

        idx += blockDim.x * gridDim.x;
    }
}

}  // namespace

CudaCrysFFTRecursive3m::CudaCrysFFTRecursive3m(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> translational_part)
    : nx_logical_(nx_logical),
      cell_para_(cell_para),
      translational_part_(translational_part)
{
    for (int d = 0; d < 3; ++d)
    {
        if (nx_logical_[d] <= 0 || (nx_logical_[d] % 2) != 0)
        {
            throw_with_line_number("CudaCrysFFTRecursive3m requires even, positive grid dimensions.");
        }
    }
    if ((nx_logical_[2] / 2) % 8 != 0)
    {
        throw_with_line_number("CudaCrysFFTRecursive3m requires Nz/2 aligned to 8 for 3m algorithm.");
    }

    nx_physical_ = { nx_logical_[0] / 2, nx_logical_[1] / 2, nx_logical_[2] / 2 };
    M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
    M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];
    M_complex_ = nx_physical_[0] * nx_physical_[1] * (nx_physical_[2] / 2 + 1);

    gpu_error_check(cudaMalloc(&d_work_, sizeof(double) * M_physical_));
    gpu_error_check(cudaMalloc(&d_step1_, sizeof(cufftDoubleComplex) * M_complex_));
    gpu_error_check(cudaMalloc(&d_step2_, sizeof(cufftDoubleComplex) * M_complex_));

    init_plans();
    generate_twiddle_factors();
}

CudaCrysFFTRecursive3m::~CudaCrysFFTRecursive3m()
{
    for (auto& kv : k_cache_)
    {
        for (int i = 0; i < 8; ++i)
        {
            if (kv.second.re[i])
                cudaFree(kv.second.re[i]);
            if (kv.second.im[i])
                cudaFree(kv.second.im[i]);
        }
    }
    k_cache_.clear();

    if (d_step1_)
        cudaFree(d_step1_);
    if (d_step2_)
        cudaFree(d_step2_);
    if (d_work_)
        cudaFree(d_work_);

    free_plans();
}

void CudaCrysFFTRecursive3m::init_plans()
{
    if (plans_initialized_)
        return;
    if (cufftPlan3d(&plan_r2c_, nx_physical_[0], nx_physical_[1], nx_physical_[2], CUFFT_D2Z) != CUFFT_SUCCESS)
        throw_with_line_number("Failed to create cuFFT D2Z plan for CudaCrysFFTRecursive3m.");
    if (cufftPlan3d(&plan_c2r_, nx_physical_[0], nx_physical_[1], nx_physical_[2], CUFFT_Z2D) != CUFFT_SUCCESS)
        throw_with_line_number("Failed to create cuFFT Z2D plan for CudaCrysFFTRecursive3m.");
    plans_initialized_ = true;
}

void CudaCrysFFTRecursive3m::free_plans()
{
    if (!plans_initialized_)
        return;
    cufftDestroy(plan_r2c_);
    cufftDestroy(plan_c2r_);
    plans_initialized_ = false;
}

void CudaCrysFFTRecursive3m::generate_twiddle_factors()
{
    for (int i = 0; i < 8; ++i)
    {
        r_re_[i].assign(M_physical_, 0.0);
        r_im_[i].assign(M_physical_, 0.0);
    }

    auto operations = generate_m3_operations(translational_part_, nx_logical_);
    std::vector<std::tuple<double*, double*, std::array<double, 3>>> twiddle;
    for (size_t i = 0; i < operations.size(); ++i)
    {
        twiddle.emplace_back(r_re_[i + 1].data(), r_im_[i + 1].data(), operations[i]);
    }

    std::fill(r_re_[0].begin(), r_re_[0].end(), 1.0);
    std::fill(r_im_[0].begin(), r_im_[0].end(), 0.0);

    twiddle_factor(twiddle, nx_physical_);
}

CudaCrysFFTRecursive3m::KCacheDevice CudaCrysFFTRecursive3m::generate_k_cache(double coeff)
{
    std::vector<double> kx(nx_logical_[0]);
    std::vector<double> ky(nx_logical_[1]);
    std::vector<double> kz(nx_logical_[2]);

    double factor_Lx = 2.0 * kPi / cell_para_[0];
    factor_Lx *= factor_Lx;
    double factor_Ly = 2.0 * kPi / cell_para_[1];
    factor_Ly *= factor_Ly;
    double factor_Lz = 2.0 * kPi / cell_para_[2];
    factor_Lz *= factor_Lz;

    for (int ix = 0; ix < nx_logical_[0]; ++ix)
    {
        int temp = (ix > nx_logical_[0] / 2) ? (nx_logical_[0] - ix) : ix;
        kx[ix] = temp * temp * factor_Lx;
    }
    for (int iy = 0; iy < nx_logical_[1]; ++iy)
    {
        int temp = (iy > nx_logical_[1] / 2) ? (nx_logical_[1] - iy) : iy;
        ky[iy] = temp * temp * factor_Ly;
    }
    for (int iz = 0; iz < nx_logical_[2]; ++iz)
    {
        int temp = (iz > nx_logical_[2] / 2) ? (nx_logical_[2] - iz) : iz;
        kz[iz] = temp * temp * factor_Lz;
    }

    std::vector<double> tempmat(static_cast<size_t>(M_logical_));
    double factor = 1.0 / static_cast<double>(M_logical_);

    for (int ix = 0; ix < nx_logical_[0]; ++ix)
    {
        for (int iy = 0; iy < nx_logical_[1]; ++iy)
        {
            size_t base = static_cast<size_t>(ix * nx_logical_[1] + iy) * nx_logical_[2];
            for (int iz = 0; iz < nx_logical_[2]; ++iz)
            {
                tempmat[base + iz] = std::exp(-(kx[ix] + ky[iy] + kz[iz]) * coeff) * factor;
            }
        }
    }

    std::array<std::vector<double>, 8> k_split;
    std::array<double*, 8> k_split_ptr{};
    for (int i = 0; i < 8; ++i)
    {
        k_split[i].assign(M_physical_, 0.0);
        k_split_ptr[i] = k_split[i].data();
    }

    mat_split(tempmat.data(), k_split_ptr, nx_logical_);

    std::array<int, 3> halfsize = nx_physical_;
    constexpr std::array<std::pair<int, int>, 4> seq0{{{0,4},{1,5},{2,6},{3,7}}};
    constexpr std::array<std::pair<int, int>, 4> seq1{{{0,2},{1,3},{4,6},{5,7}}};
    constexpr std::array<std::pair<int, int>, 4> seq2{{{0,1},{2,3},{4,5},{6,7}}};
    add_sub_by_seq(k_split_ptr, seq0, halfsize);
    add_sub_by_seq(k_split_ptr, seq1, halfsize);
    add_sub_by_seq(k_split_ptr, seq2, halfsize);

    std::array<std::vector<double>, 8> k_re_host;
    std::array<std::vector<double>, 8> k_im_host;
    std::array<double*, 8> k_re_ptr{};
    std::array<double*, 8> k_im_ptr{};
    std::array<double*, 8> r_re_ptr{};
    std::array<double*, 8> r_im_ptr{};

    for (int i = 0; i < 8; ++i)
    {
        k_re_host[i].assign(M_physical_, 0.0);
        k_im_host[i].assign(M_physical_, 0.0);
        k_re_ptr[i] = k_re_host[i].data();
        k_im_ptr[i] = k_im_host[i].data();
        r_re_ptr[i] = r_re_[i].data();
        r_im_ptr[i] = r_im_[i].data();
    }

    constexpr std::array<std::pair<int, int>, 4> seq3{{{0,7},{6,1},{2,5},{4,3}}};
    mul_add_sub_by_seq(k_re_ptr, k_im_ptr, r_re_ptr, r_im_ptr, k_split_ptr, seq3, halfsize);

    KCacheDevice cache;
    for (int i = 0; i < 8; ++i)
    {
        gpu_error_check(cudaMalloc(&cache.re[i], sizeof(double) * M_physical_));
        gpu_error_check(cudaMalloc(&cache.im[i], sizeof(double) * M_physical_));
        gpu_error_check(cudaMemcpy(cache.re[i], k_re_host[i].data(), sizeof(double) * M_physical_, cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(cache.im[i], k_im_host[i].data(), sizeof(double) * M_physical_, cudaMemcpyHostToDevice));
    }

    return cache;
}

void CudaCrysFFTRecursive3m::set_cell_para(const std::array<double, 6>& cell_para)
{
    if (cell_para[0] == cell_para_[0] &&
        cell_para[1] == cell_para_[1] &&
        cell_para[2] == cell_para_[2])
    {
        return;
    }
    cell_para_ = cell_para;

    for (auto& kv : k_cache_)
    {
        for (int i = 0; i < 8; ++i)
        {
            if (kv.second.re[i])
                cudaFree(kv.second.re[i]);
            if (kv.second.im[i])
                cudaFree(kv.second.im[i]);
        }
    }
    k_cache_.clear();
    k_current_ = nullptr;
}

void CudaCrysFFTRecursive3m::set_contour_step(double coeff)
{
    if (k_current_ && coeff == coeff_current_)
        return;

    auto it = k_cache_.find(coeff);
    if (it == k_cache_.end())
    {
        auto inserted = k_cache_.emplace(coeff, generate_k_cache(coeff));
        it = inserted.first;
    }
    coeff_current_ = coeff;
    k_current_ = &it->second;
}

void CudaCrysFFTRecursive3m::diffusion(double* d_q_in, double* d_q_out)
{
    diffusion(d_q_in, d_q_out, stream_);
}

void CudaCrysFFTRecursive3m::diffusion(double* d_q_in, double* d_q_out, cudaStream_t stream)
{
    if (!k_current_)
        throw_with_line_number("CudaCrysFFTRecursive3m::set_contour_step must be called before diffusion().");

    set_stream(stream);

    double* d_work = (d_q_in == d_q_out) ? d_q_in : d_q_out;
    if (d_q_in != d_q_out)
    {
        gpu_error_check(cudaMemcpyAsync(d_q_out, d_q_in, sizeof(double) * M_physical_, cudaMemcpyDeviceToDevice, stream_));
    }

    if (cufftExecD2Z(plan_r2c_, d_work, d_step1_) != CUFFT_SUCCESS)
        throw_with_line_number("CudaCrysFFTRecursive3m cufftExecD2Z failed.");

    const int threads = 256;
    const int blocks = (M_complex_ + threads - 1) / threads;

    ker_apply_k_3m<<<blocks, threads, 0, stream_>>>(
        d_step2_, d_step1_,
        k_current_->re[0], k_current_->re[1], k_current_->re[2], k_current_->re[3],
        k_current_->re[4], k_current_->re[5], k_current_->re[6], k_current_->re[7],
        k_current_->im[0], k_current_->im[1], k_current_->im[2], k_current_->im[3],
        k_current_->im[4], k_current_->im[5], k_current_->im[6], k_current_->im[7],
        nx_physical_[0], nx_physical_[1], nx_physical_[2], nx_physical_[2] / 2 + 1);
    gpu_error_check(cudaPeekAtLastError());

    if (cufftExecZ2D(plan_c2r_, d_step2_, d_work) != CUFFT_SUCCESS)
        throw_with_line_number("CudaCrysFFTRecursive3m cufftExecZ2D failed.");
}

void CudaCrysFFTRecursive3m::set_stream(cudaStream_t stream)
{
    if (stream_ == stream)
        return;
    stream_ = stream;
    cufftSetStream(plan_r2c_, stream_);
    cufftSetStream(plan_c2r_, stream_);
}
