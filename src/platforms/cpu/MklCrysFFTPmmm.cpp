/**
 * @file MklCrysFFTPmmm.cpp
 * @brief MKL implementation of crystallographic FFT using DCT-II/III.
 *
 * DCT-II (forward) and DCT-III (backward) for Pmmm symmetry,
 * using O(N log N) FFT algorithm via Intel MKL.
 *
 * Reference: Qiang & Li, Macromolecules (2020)
 */

#include <cmath>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <numbers>

#include "MklCrysFFTPmmm.h"
#include "Exception.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
MklCrysFFTPmmm::MklCrysFFTPmmm(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> /* trans_part - ignored for DCT-II/III */)
    : nx_logical_(nx_logical),
      cell_para_(cell_para),
      instance_id_(next_instance_id_.fetch_add(1, std::memory_order_relaxed)),
      io_buffer_(nullptr),
      temp_buffer_(nullptr)
{
    // Initialize DFTI handles to nullptr
    for (int d = 0; d < 3; ++d)
    {
        dct_fft_forward_[d] = nullptr;
        dct_fft_backward_[d] = nullptr;
    }

    // Validate even grid sizes
    for (int d = 0; d < 3; ++d)
    {
        if (nx_logical_[d] % 2 != 0)
        {
            throw_with_line_number("MklCrysFFTPmmm requires even grid dimensions. "
                "Dimension " + std::to_string(d) + " has size " + std::to_string(nx_logical_[d]));
        }
        if (nx_logical_[d] <= 0)
        {
            throw_with_line_number("MklCrysFFTPmmm requires positive grid dimensions.");
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
    norm_factor_ = 1.0 / static_cast<double>(M_logical_);

    // Initialize MKL FFT plans
    initFFTPlans();

    // Precompute twiddle factors
    precomputeTwiddleFactors();
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
MklCrysFFTPmmm::~MklCrysFFTPmmm()
{
    // Free Boltzmann factors
    freeBoltzmann();

    // Destroy MKL DFTI descriptors
    for (int d = 0; d < 3; ++d)
    {
        if (dct_fft_forward_[d]) DftiFreeDescriptor(&dct_fft_forward_[d]);
        if (dct_fft_backward_[d]) DftiFreeDescriptor(&dct_fft_backward_[d]);
    }

    // Free work buffers
    delete[] io_buffer_;
    delete[] temp_buffer_;
}

//------------------------------------------------------------------------------
// Initialize MKL DFTI descriptors for DCT via FFT
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::initFFTPlans()
{
    // Allocate work buffers
    io_buffer_ = new double[M_physical_];
    temp_buffer_ = new double[M_physical_];

    if (!io_buffer_ || !temp_buffer_)
    {
        throw_with_line_number("Failed to allocate work buffer memory");
    }

    // Create 1D FFT plans for each dimension of physical grid
    for (int d = 0; d < 3; ++d)
    {
        int n = nx_physical_[d];
        MKL_LONG status;

        // Forward FFT for DCT-II (D2Z equivalent via IFFT)
        status = DftiCreateDescriptor(&dct_fft_forward_[d], DFTI_DOUBLE, DFTI_REAL, 1, n);
        status = DftiSetValue(dct_fft_forward_[d], DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        status = DftiSetValue(dct_fft_forward_[d], DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        status = DftiCommitDescriptor(dct_fft_forward_[d]);

        // Backward FFT for DCT-III (Z2D equivalent via IFFT)
        status = DftiCreateDescriptor(&dct_fft_backward_[d], DFTI_DOUBLE, DFTI_REAL, 1, n);
        status = DftiSetValue(dct_fft_backward_[d], DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        status = DftiSetValue(dct_fft_backward_[d], DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        status = DftiCommitDescriptor(dct_fft_backward_[d]);

        if (status != 0)
            throw_with_line_number("MKL FFT plan creation failed for dimension " + std::to_string(d));
    }
}

//------------------------------------------------------------------------------
// Precompute twiddle factors for DCT via FFT
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::precomputeTwiddleFactors()
{
    const double PI = std::numbers::pi;

    for (int d = 0; d < 3; ++d)
    {
        int n = nx_physical_[d];
        int num_twiddles = n / 2 + 1;
        cos_tables_[d].resize(num_twiddles);
        sin_tables_[d].resize(num_twiddles);

        for (int k = 0; k <= n / 2; ++k)
        {
            cos_tables_[d][k] = std::cos(k * PI / (2.0 * n));
            sin_tables_[d][k] = std::sin(k * PI / (2.0 * n));
        }
    }
}

//------------------------------------------------------------------------------
// Free Boltzmann factors
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::freeBoltzmann()
{
    cache_epoch_.fetch_add(1, std::memory_order_acq_rel);
}

//------------------------------------------------------------------------------
// Set cell parameters
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::set_cell_para(const std::array<double, 6>& cell_para)
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
void MklCrysFFTPmmm::set_contour_step(double ds)
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
double* MklCrysFFTPmmm::generateBoltzmann(double ds) const
{
    double* boltz = new double[M_physical_];

    double Lx = cell_para_[0];
    double Ly = cell_para_[1];
    double Lz = cell_para_[2];

    int Nx2 = nx_physical_[0];
    int Ny2 = nx_physical_[1];
    int Nz2 = nx_physical_[2];

    const double PI = std::numbers::pi;

    // For DCT-II/III, frequencies are k_n = n * 2π / L for n = 0, 1, ..., N/2-1
    int idx = 0;
    for (int ix = 0; ix < Nx2; ++ix)
    {
        double kx = ix * 2.0 * PI / Lx;
        double kx2 = kx * kx;

        for (int iy = 0; iy < Ny2; ++iy)
        {
            double ky = iy * 2.0 * PI / Ly;
            double ky2 = ky * ky;

            for (int iz = 0; iz < Nz2; ++iz)
            {
                double kz = iz * 2.0 * PI / Lz;
                double kz2 = kz * kz;

                boltz[idx++] = std::exp(-(kx2 + ky2 + kz2) * ds);
            }
        }
    }

    return boltz;
}

//------------------------------------------------------------------------------
// Get thread-local state
//------------------------------------------------------------------------------
MklCrysFFTPmmm::ThreadState& MklCrysFFTPmmm::get_thread_state() const
{
    struct ThreadLocalStates
    {
        std::unordered_map<const MklCrysFFTPmmm*, ThreadState> states;
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
// Apply DCT-II forward along one dimension
//------------------------------------------------------------------------------
static void applyDCT2Forward1D(
    double* data, double* temp,
    int n, int stride, int num_transforms,
    DFTI_DESCRIPTOR_HANDLE fft_backward,
    const double* cos_tbl, const double* sin_tbl)
{
    int complex_size = n / 2 + 1;

    // Thread-local buffers
    thread_local std::vector<double> slice;
    thread_local std::vector<std::complex<double>> fft_in;
    thread_local std::vector<double> fft_out;

    if (static_cast<int>(slice.size()) < n)
    {
        slice.resize(n);
        fft_in.resize(complex_size);
        fft_out.resize(n);
    }

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            // Extract 1D slice (strided)
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            // DCT-II preprocessing: rearrange into complex format
            fft_in[0] = std::complex<double>(slice[0], 0.0);
            for (int k = 1; k < complex_size - 1; ++k)
            {
                double x_2k = slice[2 * k];
                double x_2k_1 = slice[2 * k - 1];
                fft_in[k] = std::complex<double>((x_2k + x_2k_1) / 2.0, -(x_2k_1 - x_2k) / 2.0);
            }
            if (n % 2 == 0)
            {
                fft_in[n / 2] = std::complex<double>(slice[n - 1], 0.0);
            }
            else
            {
                int k = n / 2;
                double x_2k = slice[2 * k];
                double x_2k_1 = slice[2 * k - 1];
                fft_in[k] = std::complex<double>((x_2k + x_2k_1) / 2.0, -(x_2k_1 - x_2k) / 2.0);
            }

            // Inverse FFT (c2r)
            DftiComputeBackward(fft_backward,
                                reinterpret_cast<double*>(fft_in.data()),
                                fft_out.data());

            // DCT-II postprocessing: apply twiddle factors
            temp[offset] = fft_out[0];  // DC component
            for (int k = 1; k <= n / 2; ++k)
            {
                double Ta = fft_out[k] + fft_out[n - k];
                double Tb = fft_out[k] - fft_out[n - k];

                double result_k = (Ta * cos_tbl[k] + Tb * sin_tbl[k]) * 0.5;
                double result_nk = (Ta * sin_tbl[k] - Tb * cos_tbl[k]) * 0.5;

                temp[offset + k * stride] = result_k;
                if (k < n - k)
                    temp[offset + (n - k) * stride] = result_nk;
            }
        }
    }

    // Copy temp back to data
    int total = num_transforms * n * stride;
    std::memcpy(data, temp, total * sizeof(double));
}

//------------------------------------------------------------------------------
// Apply DCT-III backward along one dimension
//------------------------------------------------------------------------------
static void applyDCT3Backward1D(
    double* data, double* temp,
    int n, int stride, int num_transforms,
    DFTI_DESCRIPTOR_HANDLE fft_forward,
    const double* cos_tbl, const double* sin_tbl)
{
    int complex_size = n / 2 + 1;

    // Thread-local buffers
    thread_local std::vector<double> slice;
    thread_local std::vector<double> fft_in;
    thread_local std::vector<std::complex<double>> fft_out;

    if (static_cast<int>(slice.size()) < n)
    {
        slice.resize(n);
        fft_in.resize(n);
        fft_out.resize(complex_size);
    }

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            // Extract 1D slice (strided)
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            // DCT-III preprocessing: apply twiddle factors
            fft_in[0] = slice[0];
            for (int k = 0; k < n / 2; ++k)
            {
                double val_k = slice[k + 1];
                double val_nk = slice[n - k - 1];

                double Ta = val_k + val_nk;
                double Tb = val_k - val_nk;

                fft_in[k + 1] = Ta * sin_tbl[k + 1] + Tb * cos_tbl[k + 1];
                fft_in[n - k - 1] = Ta * cos_tbl[k + 1] - Tb * sin_tbl[k + 1];
            }

            // Forward FFT (r2c)
            DftiComputeForward(fft_forward, fft_in.data(), fft_out.data());

            // DCT-III postprocessing: rearrange complex to real
            double scale = 1.0 / n;
            temp[offset] = fft_out[0].real() * scale;
            for (int k = 1; k <= n / 2; ++k)
            {
                double re = fft_out[k].real();
                double im = fft_out[k].imag();

                temp[offset + (2 * k - 1) * stride] = (re - im) * scale;
                if (2 * k < n)
                    temp[offset + (2 * k) * stride] = (re + im) * scale;
            }
        }
    }

    // Copy temp back to data
    int total = num_transforms * n * stride;
    std::memcpy(data, temp, total * sizeof(double));
}

//------------------------------------------------------------------------------
// Apply DCT-II forward along all 3 dimensions
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::applyDCT2Forward3D(double* data, double* temp)
{
    int nx = nx_physical_[0];
    int ny = nx_physical_[1];
    int nz = nx_physical_[2];

    // Dimension 0 (x): stride = ny*nz, num_transforms = 1
    applyDCT2Forward1D(data, temp, nx, ny * nz, 1,
                       dct_fft_backward_[0],
                       cos_tables_[0].data(), sin_tables_[0].data());

    // Dimension 1 (y): stride = nz, num_transforms = nx
    applyDCT2Forward1D(data, temp, ny, nz, nx,
                       dct_fft_backward_[1],
                       cos_tables_[1].data(), sin_tables_[1].data());

    // Dimension 2 (z): stride = 1, num_transforms = nx*ny
    applyDCT2Forward1D(data, temp, nz, 1, nx * ny,
                       dct_fft_backward_[2],
                       cos_tables_[2].data(), sin_tables_[2].data());
}

//------------------------------------------------------------------------------
// Apply DCT-III backward along all 3 dimensions
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::applyDCT3Backward3D(double* data, double* temp)
{
    int nx = nx_physical_[0];
    int ny = nx_physical_[1];
    int nz = nx_physical_[2];

    // Dimension 2 (z): stride = 1, num_transforms = nx*ny
    applyDCT3Backward1D(data, temp, nz, 1, nx * ny,
                        dct_fft_forward_[2],
                        cos_tables_[2].data(), sin_tables_[2].data());

    // Dimension 1 (y): stride = nz, num_transforms = nx
    applyDCT3Backward1D(data, temp, ny, nz, nx,
                        dct_fft_forward_[1],
                        cos_tables_[1].data(), sin_tables_[1].data());

    // Dimension 0 (x): stride = ny*nz, num_transforms = 1
    applyDCT3Backward1D(data, temp, nx, ny * nz, 1,
                        dct_fft_forward_[0],
                        cos_tables_[0].data(), sin_tables_[0].data());
}

//------------------------------------------------------------------------------
// Apply diffusion operator
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::diffusion(double* q_in, double* q_out)
{
    ThreadState& state = get_thread_state();
    if (!state.boltz_current)
    {
        throw_with_line_number("MklCrysFFTPmmm::set_contour_step must be called before diffusion().");
    }

    // Thread-local buffers
    thread_local std::vector<double> io_local;
    thread_local std::vector<double> temp_local;

    if (static_cast<int>(io_local.size()) < M_physical_)
    {
        io_local.resize(M_physical_);
        temp_local.resize(M_physical_);
    }

    // Step 1: Copy input
    std::memcpy(io_local.data(), q_in, sizeof(double) * M_physical_);

    // Step 2: Forward DCT-II (3D)
    applyDCT2Forward3D(io_local.data(), temp_local.data());

    // Step 3: Apply Boltzmann factor
    for (int i = 0; i < M_physical_; ++i)
        io_local[i] *= state.boltz_current[i];

    // Step 4: Backward DCT-III (3D)
    applyDCT3Backward3D(io_local.data(), temp_local.data());

    // Step 5: Apply normalization and copy result to output
    for (int i = 0; i < M_physical_; ++i)
        q_out[i] = io_local[i] * norm_factor_;
}

//------------------------------------------------------------------------------
// Apply custom multiplier in Fourier space
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::apply_multiplier(const double* q_in, double* q_out, const double* multiplier)
{
    // Thread-local buffers
    thread_local std::vector<double> io_local;
    thread_local std::vector<double> temp_local;

    if (static_cast<int>(io_local.size()) < M_physical_)
    {
        io_local.resize(M_physical_);
        temp_local.resize(M_physical_);
    }

    // Step 1: Copy input
    std::memcpy(io_local.data(), q_in, sizeof(double) * M_physical_);

    // Step 2: Forward DCT-II
    applyDCT2Forward3D(io_local.data(), temp_local.data());

    // Step 3: Apply multiplier
    for (int i = 0; i < M_physical_; ++i)
        io_local[i] *= multiplier[i];

    // Step 4: Backward DCT-III
    applyDCT3Backward3D(io_local.data(), temp_local.data());

    // Step 5: Apply normalization and copy result to output
    for (int i = 0; i < M_physical_; ++i)
        q_out[i] = io_local[i] * norm_factor_;
}
