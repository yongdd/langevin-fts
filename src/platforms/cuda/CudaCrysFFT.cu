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
    while (idx < M) {
        d_data[idx] *= d_boltz[idx];
        idx += blockDim.x * gridDim.x;
    }
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
    while (idx < M) {
        d_data[idx] *= d_multiplier[idx] * norm;
        idx += blockDim.x * gridDim.x;
    }
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
      tables_(std::make_shared<Tables>()),
      d_boltz_current_(nullptr),
      ds_current_(0.0)
{
    // The owning members release what was acquired if a step below throws.
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
    tables_->cell_para = cell_para_;

    // Create CudaRealTransform objects for DCT-II and DCT-III
    dct_forward_ = std::make_unique<CudaRealTransform3D>(nx_physical_[0], nx_physical_[1], nx_physical_[2], CUDA_DCT_2);
    dct_backward_ = std::make_unique<CudaRealTransform3D>(nx_physical_[0], nx_physical_[1], nx_physical_[2], CUDA_DCT_3);
}

CudaCrysFFT::CudaCrysFFT(const CudaCrysFFT& other, std::shared_ptr<Tables> tables)
    : nx_logical_(other.nx_logical_),
      nx_physical_(other.nx_physical_),
      M_logical_(other.M_logical_),
      M_physical_(other.M_physical_),
      cell_para_(other.cell_para_),
      tables_(std::move(tables)),
      d_boltz_current_(nullptr),
      ds_current_(0.0),
      norm_factor_(other.norm_factor_)
{
    dct_forward_ = std::make_unique<CudaRealTransform3D>(nx_physical_[0], nx_physical_[1], nx_physical_[2], CUDA_DCT_2);
    dct_backward_ = std::make_unique<CudaRealTransform3D>(nx_physical_[0], nx_physical_[1], nx_physical_[2], CUDA_DCT_3);
}

std::unique_ptr<CudaCrysFFTBase> CudaCrysFFT::clone_sharing_tables() const
{
    return std::unique_ptr<CudaCrysFFTBase>(new CudaCrysFFT(*this, tables_));
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
CudaCrysFFT::~CudaCrysFFT() = default;   // the owning members release the tables, transforms and buffers

//------------------------------------------------------------------------------
// Set cell parameters
//------------------------------------------------------------------------------
void CudaCrysFFT::set_cell_para(const std::array<double, 6>& cell_para)
{
    auto same_lengths = [](const std::array<double, 6>& a, const std::array<double, 6>& b) {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    };
    if (same_lengths(cell_para, cell_para_))
        return;

    cell_para_ = cell_para;
    boltz_seen_.clear();
    d_boltz_current_ = nullptr;
    ds_current_ = 0.0;

    // The first object to see the new cell drops the shared tables, keeping their
    // device buffers for the tables the next steps rebuild. Reusing the buffers is
    // safe because the computations synchronize every stream before
    // update_laplacian_operator() can change the cell; set_cell_para() must not
    // run while a kernel may still read the old tables.
    std::lock_guard<std::mutex> lock(tables_->mutex);
    if (same_lengths(cell_para, tables_->cell_para))
        return;
    tables_->cell_para = cell_para;
    for (auto& kv : tables_->boltzmann)
        if (!kv.second.empty()) tables_->spare.push_back(std::move(kv.second));
    tables_->boltzmann.clear();
    for (CudaDeviceBuffer<double>& d_k2 : tables_->axis_k2)
        if (!d_k2.empty()) tables_->spare.push_back(std::move(d_k2));
    for (auto& kv : tables_->axis_boltz_k2)
        if (!kv.second.empty()) tables_->spare.push_back(std::move(kv.second));
    tables_->axis_boltz_k2.clear();
}

CudaDeviceBuffer<double> CudaCrysFFT::uploadTable(const std::vector<double>& h)
{
    CudaDeviceBuffer<double> d_table;
    if (!tables_->spare.empty())
    {
        d_table = std::move(tables_->spare.back());
        tables_->spare.pop_back();
    }
    else
    {
        d_table.allocate(M_physical_);
    }
    // on failure the local releases the table
    CUDA_CHECK(cudaMemcpyAsync(d_table, h.data(), sizeof(double) * M_physical_, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return d_table;
}

//------------------------------------------------------------------------------
// Set contour step
//------------------------------------------------------------------------------
void CudaCrysFFT::set_contour_step(double ds)
{
    if (ds == ds_current_ && d_boltz_current_ != nullptr) return;

    auto seen = boltz_seen_.find(ds);
    if (seen == boltz_seen_.end())
    {
        std::lock_guard<std::mutex> lock(tables_->mutex);
        auto it = tables_->boltzmann.find(ds);
        if (it == tables_->boltzmann.end())
            it = tables_->boltzmann.emplace(ds, generateBoltzmann(ds)).first;
        seen = boltz_seen_.emplace(ds, it->second.get()).first;
    }

    ds_current_ = ds;
    d_boltz_current_ = seen->second;
}

//------------------------------------------------------------------------------
// Generate Boltzmann factors
//------------------------------------------------------------------------------
CudaDeviceBuffer<double> CudaCrysFFT::generateBoltzmann(double ds)
{
    // Compute on host, then copy to device
    std::vector<double> h_boltz(M_physical_);

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

    return uploadTable(h_boltz);
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
    if (d_boltz_current_ == nullptr)
        throw_with_line_number("CudaCrysFFT::set_contour_step must be called before diffusion().");
    int threads = 256;
    int blocks = grid_blocks(M_physical_, threads);

    static std::atomic<bool> profile_once{false};
    static const bool do_profile = (std::getenv("FTS_PROFILE_CRYSFFT_PMMM") != nullptr);
    const bool profile_this = do_profile && !profile_once.exchange(true);
    // Created only when profiling; CudaEvent destroys them on every exit
    std::vector<CudaEvent> ev(profile_this ? 5 : 0);
    if (profile_this)
    {
        CUDA_CHECK(cudaEventRecord(ev[0], stream_));
    }

    // Determine working buffer to minimize memory copies
    // - If in-place (d_q_in == d_q_out): work directly on input
    // - If out-of-place: copy to output once, then work on output
    double* d_work = (d_q_in == d_q_out) ? d_q_in : d_q_out;

    if (d_q_in != d_q_out)
    {
        // Single copy at start (instead of two copies before/after)
        CUDA_CHECK(cudaMemcpyAsync(d_q_out, d_q_in, sizeof(double) * M_physical_,
                                        cudaMemcpyDeviceToDevice, stream_));
    }

    if (profile_this)
        CUDA_CHECK(cudaEventRecord(ev[1], stream_));

    // Step 1: Forward DCT-II (in-place)
    dct_forward_->execute(d_work);

    if (profile_this)
        CUDA_CHECK(cudaEventRecord(ev[2], stream_));

    // Step 2: Apply Boltzmann factor (in-place)
    kernel_apply_boltzmann_crys<<<blocks, threads, 0, stream_>>>(d_work, d_boltz_current_, M_physical_);
    CUDA_CHECK_LAUNCH();

    if (profile_this)
        CUDA_CHECK(cudaEventRecord(ev[3], stream_));

    // Step 3: Backward DCT-III (in-place)
    // Normalization is already included in Boltzmann factor (pre-multiplied in generateBoltzmann)
    dct_backward_->execute(d_work);

    if (profile_this)
    {
        CUDA_CHECK(cudaEventRecord(ev[4], stream_));
        CUDA_CHECK(cudaEventSynchronize(ev[4]));
        float t_copy = 0.0f, t_fwd = 0.0f, t_boltz = 0.0f, t_bwd = 0.0f, t_total = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&t_copy, ev[0], ev[1]));
        CUDA_CHECK(cudaEventElapsedTime(&t_fwd, ev[1], ev[2]));
        CUDA_CHECK(cudaEventElapsedTime(&t_boltz, ev[2], ev[3]));
        CUDA_CHECK(cudaEventElapsedTime(&t_bwd, ev[3], ev[4]));
        CUDA_CHECK(cudaEventElapsedTime(&t_total, ev[0], ev[4]));

        std::cout << "[CrysFFT-Pmmm profile] M_phys=" << M_physical_
                  << " copy(ms)=" << t_copy
                  << " fwd(ms)=" << t_fwd
                  << " boltz(ms)=" << t_boltz
                  << " bwd(ms)=" << t_bwd
                  << " total(ms)=" << t_total << std::endl;

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
    int blocks = grid_blocks(M_physical_, threads);

    // Same in-place strategy as diffusion(): work directly on the output buffer
    double* d_work = (d_q_in == d_q_out) ? d_q_in : d_q_out;
    if (d_q_in != d_q_out)
    {
        CUDA_CHECK(cudaMemcpyAsync(d_q_out, d_q_in, sizeof(double) * M_physical_,
                                        cudaMemcpyDeviceToDevice, stream_));
    }

    // Forward DCT-II (in-place)
    dct_forward_->execute(d_work);

    // Apply raw multiplier with round-trip normalization
    kernel_apply_multiplier_crys<<<blocks, threads, 0, stream_>>>(d_work, d_multiplier, norm_factor_, M_physical_);
    CUDA_CHECK_LAUNCH();

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
// Cached device multiplier: Σ k_d² over an axis mask (continuous chain stress)
//------------------------------------------------------------------------------
const double* CudaCrysFFT::get_axis_k2_multiplier(int axes)
{
    if (axes < 1 || axes > 7)
        throw std::invalid_argument("CudaCrysFFT::get_axis_k2_multiplier: axes must be a mask of 1 (x), 2 (y), 4 (z)");

    std::lock_guard<std::mutex> lock(tables_->mutex);
    CudaDeviceBuffer<double>& d_table = tables_->axis_k2[axes];
    if (!d_table.empty())
        return d_table.get();

    std::vector<double> kx2, ky2, kz2;
    computeAxisK2Host(kx2, ky2, kz2);
    std::vector<double> h_mult(M_physical_);
    for (int i = 0; i < M_physical_; ++i)
        h_mult[i] = ((axes & 1) ? kx2[i] : 0.0) + ((axes & 2) ? ky2[i] : 0.0) + ((axes & 4) ? kz2[i] : 0.0);

    d_table = uploadTable(h_mult);
    return d_table.get();
}

//------------------------------------------------------------------------------
// Cached device multiplier: exp(-k²·coeff)·Σ k_d² (discrete chain stress)
//------------------------------------------------------------------------------
const double* CudaCrysFFT::get_axis_boltz_k2_multiplier(int axes, double coeff)
{
    if (axes < 1 || axes > 7)
        throw std::invalid_argument("CudaCrysFFT::get_axis_boltz_k2_multiplier: axes must be a mask of 1 (x), 2 (y), 4 (z)");

    auto key = std::make_pair(axes, coeff);
    std::lock_guard<std::mutex> lock(tables_->mutex);
    auto it = tables_->axis_boltz_k2.find(key);
    if (it != tables_->axis_boltz_k2.end())
        return it->second.get();

    std::vector<double> kx2, ky2, kz2;
    computeAxisK2Host(kx2, ky2, kz2);

    std::vector<double> h_mult(M_physical_);
    for (int i = 0; i < M_physical_; ++i)
    {
        double k2 = kx2[i] + ky2[i] + kz2[i];
        double axis_k2 = ((axes & 1) ? kx2[i] : 0.0) + ((axes & 2) ? ky2[i] : 0.0) + ((axes & 4) ? kz2[i] : 0.0);
        h_mult[i] = std::exp(-k2 * coeff) * axis_k2;
    }

    CudaDeviceBuffer<double> d_table = uploadTable(h_mult);
    const double* d_ptr = d_table.get();     // stays valid: the map takes the same allocation
    tables_->axis_boltz_k2[key] = std::move(d_table);
    return d_ptr;
}
