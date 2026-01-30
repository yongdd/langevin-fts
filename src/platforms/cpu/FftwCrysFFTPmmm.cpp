/**
 * @file FftwCrysFFTPmmm.cpp
 * @brief CPU implementation of crystallographic FFT using DCT-II/III.
 *
 * DCT-II (forward) and DCT-III (backward) for Pmmm symmetry.
 *
 * Reference: Qiang & Li, Macromolecules (2020)
 */

#include <cmath>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <unordered_map>

#include "FftwCrysFFTPmmm.h"
#include "Exception.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
FftwCrysFFTPmmm::FftwCrysFFTPmmm(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> /* trans_part - ignored for DCT-II/III */)
    : nx_logical_(nx_logical),
      cell_para_(cell_para),
      instance_id_(next_instance_id_.fetch_add(1, std::memory_order_relaxed)),
      plan_forward_(nullptr),
      plan_backward_(nullptr),
      io_buffer_(nullptr),
      temp_buffer_(nullptr)
{
    // Validate even grid sizes
    for (int d = 0; d < 3; ++d)
    {
        if (nx_logical_[d] % 2 != 0)
        {
            throw_with_line_number("FftwCrysFFTPmmm requires even grid dimensions. "
                "Dimension " + std::to_string(d) + " has size " + std::to_string(nx_logical_[d]));
        }
        if (nx_logical_[d] <= 0)
        {
            throw_with_line_number("FftwCrysFFTPmmm requires positive grid dimensions.");
        }
    }

    // Physical grid: (N/2) in each dimension (excluding boundary)
    nx_physical_ = {
        nx_logical_[0] / 2,
        nx_logical_[1] / 2,
        nx_logical_[2] / 2
    };

    M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
    M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];

    // DCT-II/III normalization: 1 / M_logical for round-trip
    // DCT-II output is scaled by 2^3 = 8 relative to "orthogonal" normalization
    // DCT-III reverses this, so round-trip scales by 8 * M_physical = M_logical
    norm_factor_ = 1.0 / static_cast<double>(M_logical_);

    // Initialize FFTW plans
    initFFTPlans();

}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
FftwCrysFFTPmmm::~FftwCrysFFTPmmm()
{
    // Free Boltzmann factors
    freeBoltzmann();

    // Destroy FFTW plans
    if (plan_forward_) fftw_destroy_plan(plan_forward_);
    if (plan_backward_) fftw_destroy_plan(plan_backward_);

    // Free work buffers
    if (io_buffer_) fftw_free(io_buffer_);
    if (temp_buffer_) fftw_free(temp_buffer_);
}

//------------------------------------------------------------------------------
// Initialize FFTW DCT-II/III plans
//------------------------------------------------------------------------------
void FftwCrysFFTPmmm::initFFTPlans()
{
    // Allocate work buffers
    io_buffer_ = (double*)fftw_malloc(sizeof(double) * M_physical_);
    temp_buffer_ = (double*)fftw_malloc(sizeof(double) * M_physical_);

    if (!io_buffer_ || !temp_buffer_)
    {
        throw_with_line_number("Failed to allocate work buffer memory");
    }

    // DCT-II (REDFT10) forward, DCT-III (REDFT01) backward
    plan_forward_ = fftw_plan_r2r_3d(
        nx_physical_[0], nx_physical_[1], nx_physical_[2],
        io_buffer_, temp_buffer_,
        FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10,
        FFTW_MEASURE);

    plan_backward_ = fftw_plan_r2r_3d(
        nx_physical_[0], nx_physical_[1], nx_physical_[2],
        temp_buffer_, io_buffer_,
        FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01,
        FFTW_MEASURE);

    if (!plan_forward_ || !plan_backward_)
    {
        throw_with_line_number("Failed to create FFTW DCT-II/III plans");
    }
}

//------------------------------------------------------------------------------
// Free Boltzmann factors
//------------------------------------------------------------------------------
void FftwCrysFFTPmmm::freeBoltzmann()
{
    cache_epoch_.fetch_add(1, std::memory_order_acq_rel);
}

//------------------------------------------------------------------------------
// Set cell parameters
//------------------------------------------------------------------------------
void FftwCrysFFTPmmm::set_cell_para(const std::array<double, 6>& cell_para)
{
    // Only check Lx, Ly, Lz (first 3 parameters)
    if (cell_para[0] == cell_para_[0] &&
        cell_para[1] == cell_para_[1] &&
        cell_para[2] == cell_para_[2])
    {
        return;
    }

    cell_para_ = cell_para;

    // Invalidate all Boltzmann factors
    freeBoltzmann();
}

//------------------------------------------------------------------------------
// Set contour step and prepare Boltzmann factors
//------------------------------------------------------------------------------
void FftwCrysFFTPmmm::set_contour_step(double ds)
{
    ThreadState& state = get_thread_state();
    if (state.boltz_current != nullptr && state.ds_current == ds)
        return;

    auto it = state.boltzmann.find(ds);
    if (it == state.boltzmann.end())
    {
        std::unique_ptr<double, BoltzDeleter> boltz(generateBoltzmann(ds));
        it = state.boltzmann.emplace(ds, std::move(boltz)).first;
    }

    state.ds_current = ds;
    state.boltz_current = it->second.get();
}

//------------------------------------------------------------------------------
// Generate Boltzmann factors for a specific ds value
//------------------------------------------------------------------------------
double* FftwCrysFFTPmmm::generateBoltzmann(double ds) const
{
    double* boltz = new double[M_physical_];

    double Lx = cell_para_[0];
    double Ly = cell_para_[1];
    double Lz = cell_para_[2];

    int Nx2 = nx_physical_[0];
    int Ny2 = nx_physical_[1];
    int Nz2 = nx_physical_[2];

    // For DCT-II/III, frequencies are k_n = n * 2π / L for n = 0, 1, ..., N/2-1
    // This corresponds to the physical grid indices
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

                boltz[idx++] = std::exp(-(kx2 + ky2 + kz2) * ds);
            }
        }
    }

    return boltz;
}

//------------------------------------------------------------------------------
// Apply diffusion operator
//------------------------------------------------------------------------------
void FftwCrysFFTPmmm::diffusion(double* q_in, double* q_out)
{
    ThreadState& state = get_thread_state();
    if (!state.boltz_current)
    {
        throw_with_line_number("FftwCrysFFTPmmm::set_contour_step must be called before diffusion().");
    }

    struct AlignedDeleter {
        void operator()(double* ptr) const { if (ptr) fftw_free(ptr); }
    };
    struct ThreadBuffers {
        std::unique_ptr<double, AlignedDeleter> io;
        std::unique_ptr<double, AlignedDeleter> temp;
        int size = 0;
    };

    thread_local ThreadBuffers buffers;
    if (buffers.size != M_physical_)
    {
        buffers.io.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
        buffers.temp.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
        if (!buffers.io || !buffers.temp)
        {
            throw_with_line_number("Failed to allocate thread-local FFTW buffers for FftwCrysFFTPmmm.");
        }
        buffers.size = M_physical_;
    }

    // Step 1: Copy input
    std::memcpy(buffers.io.get(), q_in, sizeof(double) * M_physical_);

    // Step 2: Forward DCT-II (new-array execute)
    fftw_execute_r2r(plan_forward_, buffers.io.get(), buffers.temp.get());

    // Step 3: Apply Boltzmann factor (no internal threading)
    for (int i = 0; i < M_physical_; ++i)
        buffers.temp.get()[i] *= state.boltz_current[i];

    // Step 4: Backward DCT-III (new-array execute)
    fftw_execute_r2r(plan_backward_, buffers.temp.get(), buffers.io.get());

    // Step 5: Apply normalization and copy result to output (no internal threading)
    for (int i = 0; i < M_physical_; ++i)
        q_out[i] = buffers.io.get()[i] * norm_factor_;
}

FftwCrysFFTPmmm::ThreadState& FftwCrysFFTPmmm::get_thread_state() const
{
    struct ThreadLocalStates
    {
        std::unordered_map<const FftwCrysFFTPmmm*, ThreadState> states;
    };
    thread_local ThreadLocalStates tls;

    ThreadState& state = tls.states[this];
    const uint64_t epoch = cache_epoch_.load(std::memory_order_acquire);
    if (state.instance_id != instance_id_ || state.epoch != epoch)
    {
        state.boltzmann.clear();
        state.boltz_current = nullptr;
        state.ds_current = std::numeric_limits<double>::quiet_NaN();
        state.epoch = epoch;
        state.instance_id = instance_id_;
    }
    return state;
}

//------------------------------------------------------------------------------
// Apply custom multiplier in Fourier space
//------------------------------------------------------------------------------
void FftwCrysFFTPmmm::apply_multiplier(const double* q_in, double* q_out, const double* multiplier)
{
    struct AlignedDeleter {
        void operator()(double* ptr) const { if (ptr) fftw_free(ptr); }
    };
    struct ThreadBuffers {
        std::unique_ptr<double, AlignedDeleter> io;
        std::unique_ptr<double, AlignedDeleter> temp;
        int size = 0;
    };

    thread_local ThreadBuffers buffers;
    if (buffers.size != M_physical_)
    {
        buffers.io.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
        buffers.temp.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
        if (!buffers.io || !buffers.temp)
        {
            throw_with_line_number("Failed to allocate thread-local FFTW buffers for FftwCrysFFTPmmm.");
        }
        buffers.size = M_physical_;
    }

    // Step 1: Copy input
    std::memcpy(buffers.io.get(), q_in, sizeof(double) * M_physical_);

    // Step 2: Forward DCT-II
    fftw_execute_r2r(plan_forward_, buffers.io.get(), buffers.temp.get());

    // Step 3: Apply multiplier (no internal threading)
    for (int i = 0; i < M_physical_; ++i)
    {
        buffers.temp.get()[i] *= multiplier[i];
    }

    // Step 4: Backward DCT-III
    fftw_execute_r2r(plan_backward_, buffers.temp.get(), buffers.io.get());

    // Step 5: Apply normalization and copy result to output (no internal threading)
    for (int i = 0; i < M_physical_; ++i)
    {
        q_out[i] = buffers.io.get()[i] * norm_factor_;
    }
}
