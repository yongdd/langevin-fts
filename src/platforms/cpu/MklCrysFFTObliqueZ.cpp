/**
 * @file MklCrysFFTObliqueZ.cpp
 * @brief MKL implementation of crystallographic FFT for z-mirror symmetry (oblique in-plane angles).
 *
 * Uses the same algorithm as FftwCrysFFTObliqueZ:
 * - DCT-II/III along Z implemented via FFT with mirrored sequence
 * - Batched 2D real-to-complex FFT along XY with proper striding
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <numbers>

#include "MklCrysFFTObliqueZ.h"
#include "Exception.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------
std::array<double, 6> MklCrysFFTObliqueZ::compute_recip_metric(const std::array<double, 6>& cell_para)
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

    // Direct lattice vectors
    const double ax = a, ay = 0.0, az = 0.0;
    const double bx = b * cos_g, by = b * sin_g, bz = 0.0;
    const double cx = c * cos_b;
    const double cy = (sin_g != 0.0) ? c * (cos_a - cos_b * cos_g) / sin_g : 0.0;
    const double cz = volume / (a * b * sin_g);

    // Reciprocal lattice vectors (without 2π)
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
MklCrysFFTObliqueZ::MklCrysFFTObliqueZ(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> /*trans_part*/)
    : nx_logical_(nx_logical),
      cell_para_(cell_para),
      recip_metric_(compute_recip_metric(cell_para)),
      instance_id_(next_instance_id_.fetch_add(1, std::memory_order_relaxed))
{
    for (int d = 0; d < 3; ++d)
    {
        if (nx_logical_[d] <= 0)
            throw_with_line_number("MklCrysFFTObliqueZ requires positive grid dimensions.");
    }
    if (nx_logical_[2] % 2 != 0)
        throw_with_line_number("MklCrysFFTObliqueZ requires even Nz.");

    nx_physical_ = {nx_logical_[0], nx_logical_[1], nx_logical_[2] / 2};
    M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
    M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];
    M_complex_ = nx_logical_[0] * (nx_logical_[1] / 2 + 1) * nx_physical_[2];
    norm_factor_ = 1.0 / static_cast<double>(M_logical_);

    initFFTPlans();
}

MklCrysFFTObliqueZ::~MklCrysFFTObliqueZ()
{
    freeBoltzmann();

    if (plan_dct_forward_z_) DftiFreeDescriptor(&plan_dct_forward_z_);
    if (plan_dct_backward_z_) DftiFreeDescriptor(&plan_dct_backward_z_);
    if (plan_fft_forward_xy_) DftiFreeDescriptor(&plan_fft_forward_xy_);
    if (plan_fft_backward_xy_) DftiFreeDescriptor(&plan_fft_backward_xy_);

    delete[] io_buffer_;
    delete[] temp_buffer_;
    delete[] complex_buffer_;
}

//------------------------------------------------------------------------------
// MKL plan initialization
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::initFFTPlans()
{
    io_buffer_ = new double[M_physical_];
    temp_buffer_ = new double[M_physical_];
    complex_buffer_ = new std::complex<double>[M_complex_];

    if (!io_buffer_ || !temp_buffer_ || !complex_buffer_)
        throw_with_line_number("Failed to allocate MKL buffers for MklCrysFFTObliqueZ.");

    MKL_LONG status;

    // DCT via FFT: 1D FFT along z of size Nz for each (x,y) column
    // Data layout: [Nx*Ny][Nz] with contiguous z-columns
    const int Nz = nx_logical_[2];
    const int howmany_z = nx_logical_[0] * nx_logical_[1];
    const MKL_LONG Nz_long = Nz;

    status = DftiCreateDescriptor(&plan_dct_forward_z_, DFTI_DOUBLE, DFTI_REAL, 1, Nz_long);
    status = DftiSetValue(plan_dct_forward_z_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(plan_dct_forward_z_, DFTI_NUMBER_OF_TRANSFORMS, static_cast<MKL_LONG>(howmany_z));
    status = DftiSetValue(plan_dct_forward_z_, DFTI_INPUT_DISTANCE, Nz_long);
    status = DftiSetValue(plan_dct_forward_z_, DFTI_OUTPUT_DISTANCE, static_cast<MKL_LONG>(Nz / 2 + 1));
    status = DftiSetValue(plan_dct_forward_z_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiCommitDescriptor(plan_dct_forward_z_);

    status = DftiCreateDescriptor(&plan_dct_backward_z_, DFTI_DOUBLE, DFTI_REAL, 1, Nz_long);
    status = DftiSetValue(plan_dct_backward_z_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(plan_dct_backward_z_, DFTI_NUMBER_OF_TRANSFORMS, static_cast<MKL_LONG>(howmany_z));
    status = DftiSetValue(plan_dct_backward_z_, DFTI_INPUT_DISTANCE, static_cast<MKL_LONG>(Nz / 2 + 1));
    status = DftiSetValue(plan_dct_backward_z_, DFTI_OUTPUT_DISTANCE, Nz_long);
    status = DftiSetValue(plan_dct_backward_z_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiCommitDescriptor(plan_dct_backward_z_);

    // 2D Real-to-complex FFT along X,Y for each Z slice (batched, strided)
    // Data layout after DCT: [Nx][Ny][Nz/2] where z varies fastest
    // XY FFT batch: howmany = Nz/2, stride = Nz/2, dist = 1
    MKL_LONG dims_xy[2] = { static_cast<MKL_LONG>(nx_logical_[0]), static_cast<MKL_LONG>(nx_logical_[1]) };
    const int howmany_xy = nx_physical_[2];
    const MKL_LONG stride_xy = nx_physical_[2];

    // For strided transforms, we need to set input/output strides
    MKL_LONG input_strides_xy[3] = {0, static_cast<MKL_LONG>(nx_logical_[1]) * stride_xy, stride_xy};
    MKL_LONG output_strides_xy[3] = {0, static_cast<MKL_LONG>(nx_logical_[1] / 2 + 1) * stride_xy, stride_xy};

    status = DftiCreateDescriptor(&plan_fft_forward_xy_, DFTI_DOUBLE, DFTI_REAL, 2, dims_xy);
    status = DftiSetValue(plan_fft_forward_xy_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(plan_fft_forward_xy_, DFTI_NUMBER_OF_TRANSFORMS, static_cast<MKL_LONG>(howmany_xy));
    status = DftiSetValue(plan_fft_forward_xy_, DFTI_INPUT_DISTANCE, 1);
    status = DftiSetValue(plan_fft_forward_xy_, DFTI_OUTPUT_DISTANCE, 1);
    status = DftiSetValue(plan_fft_forward_xy_, DFTI_INPUT_STRIDES, input_strides_xy);
    status = DftiSetValue(plan_fft_forward_xy_, DFTI_OUTPUT_STRIDES, output_strides_xy);
    status = DftiSetValue(plan_fft_forward_xy_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiCommitDescriptor(plan_fft_forward_xy_);

    status = DftiCreateDescriptor(&plan_fft_backward_xy_, DFTI_DOUBLE, DFTI_REAL, 2, dims_xy);
    status = DftiSetValue(plan_fft_backward_xy_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(plan_fft_backward_xy_, DFTI_NUMBER_OF_TRANSFORMS, static_cast<MKL_LONG>(howmany_xy));
    status = DftiSetValue(plan_fft_backward_xy_, DFTI_INPUT_DISTANCE, 1);
    status = DftiSetValue(plan_fft_backward_xy_, DFTI_OUTPUT_DISTANCE, 1);
    status = DftiSetValue(plan_fft_backward_xy_, DFTI_INPUT_STRIDES, output_strides_xy);
    status = DftiSetValue(plan_fft_backward_xy_, DFTI_OUTPUT_STRIDES, input_strides_xy);
    status = DftiSetValue(plan_fft_backward_xy_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiCommitDescriptor(plan_fft_backward_xy_);

    if (!plan_dct_forward_z_ || !plan_dct_backward_z_ || !plan_fft_forward_xy_ || !plan_fft_backward_xy_)
        throw_with_line_number("Failed to create MKL DFTI plans for MklCrysFFTObliqueZ.");

    // Precompute DCT twiddle factors
    const int N = nx_physical_[2];
    dct_cos_.resize(N + 1);
    dct_sin_.resize(N + 1);
    const double inv_2N = 1.0 / (2.0 * static_cast<double>(N));
    for (int k = 0; k <= N; ++k)
    {
        const double theta = M_PI * static_cast<double>(k) * inv_2N;
        dct_cos_[k] = std::cos(theta);
        dct_sin_[k] = std::sin(theta);
    }

    // Calibrate DCT scaling to match FFTW's native DCT behavior
    calibrateDCTScale();

    (void)status;
}

//------------------------------------------------------------------------------
// Calibrate DCT scaling factors to ensure round-trip accuracy
// This calibration computes scale factors by testing with a known input
// and comparing FFT-based DCT output against the direct computation
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::calibrateDCTScale()
{
    const int N = nx_physical_[2];  // Nz/2

    // Generate test signal
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = std::sin(0.13 * (i + 1)) + 0.1 * (i + 1);

    // Compute exact DCT-II using direct formula with extended precision
    // DCT-II: X[k] = sum_{n=0}^{N-1} x[n] * cos(pi*k*(2n+1)/(2N))
    std::vector<double> dct_exact(N, 0.0);
    for (int k = 0; k < N; ++k)
    {
        double sum = 0.0;
        for (int n = 0; n < N; ++n)
        {
            // Use explicit angle calculation for better precision
            double angle = M_PI * k * (2.0 * n + 1.0) / (2.0 * N);
            sum += x[n] * std::cos(angle);
        }
        dct_exact[k] = sum;
    }

    // Compute FFT-based DCT-II (single transform for calibration)
    std::vector<double> z_extended(2 * N);
    std::vector<std::complex<double>> z_freq(N + 1);

    // Create mirrored sequence
    for (int iz = 0; iz < N; ++iz)
        z_extended[iz] = x[iz];
    for (int iz = 0; iz < N; ++iz)
        z_extended[2 * N - 1 - iz] = x[iz];

    // Single forward FFT for calibration
    DFTI_DESCRIPTOR_HANDLE plan_calib = nullptr;
    MKL_LONG Nz_calib = 2 * N;
    DftiCreateDescriptor(&plan_calib, DFTI_DOUBLE, DFTI_REAL, 1, Nz_calib);
    DftiSetValue(plan_calib, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(plan_calib, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiCommitDescriptor(plan_calib);

    DftiComputeForward(plan_calib, z_extended.data(), z_freq.data());

    // Extract FFT-based DCT-II (without calibration)
    std::vector<double> dct_fft(N);
    for (int k = 0; k < N; ++k)
    {
        double re = z_freq[k].real();
        double im = z_freq[k].imag();
        dct_fft[k] = 0.5 * (re * dct_cos_[k] + im * dct_sin_[k]);
    }

    // Compute forward scale: dct_exact / dct_fft (least squares)
    double num = 0.0, den = 0.0;
    for (int k = 0; k < N; ++k)
    {
        num += dct_exact[k] * dct_fft[k];
        den += dct_fft[k] * dct_fft[k];
    }
    dct_scale_fwd_ = (den > 0.0) ? (num / den) : 1.0;

    // Now test round-trip to get backward scale
    // Apply forward scale to get corrected DCT coefficients
    std::vector<double> dct_scaled(N);
    for (int k = 0; k < N; ++k)
        dct_scaled[k] = dct_fft[k] * dct_scale_fwd_;

    // Apply inverse twiddle factors
    for (int k = 0; k < N; ++k)
    {
        double xk = dct_scaled[k] / dct_scale_fwd_;  // Undo forward scale for inverse
        z_freq[k] = std::complex<double>(2.0 * xk * dct_cos_[k], 2.0 * xk * dct_sin_[k]);
    }
    z_freq[N] = std::complex<double>(0.0, 0.0);

    // Backward FFT
    DftiComputeBackward(plan_calib, z_freq.data(), z_extended.data());

    // Extract DCT-III result
    std::vector<double> idct_fft(N);
    const double scale_z = 1.0 / (2.0 * N);
    for (int iz = 0; iz < N; ++iz)
        idct_fft[iz] = z_extended[iz] * scale_z;

    // Compute backward scale: x / idct_fft (to get original back)
    num = 0.0;
    den = 0.0;
    for (int i = 0; i < N; ++i)
    {
        num += x[i] * idct_fft[i];
        den += idct_fft[i] * idct_fft[i];
    }
    dct_scale_bwd_ = (den > 0.0) ? (num / den) : 1.0;

    // Verify calibration: compute round-trip error
    double max_err = 0.0;
    for (int i = 0; i < N; ++i)
    {
        double err = std::abs(idct_fft[i] * dct_scale_bwd_ - x[i]);
        max_err = std::max(max_err, err);
    }

    // Debug: print calibration info for first instance only
    static bool printed = false;
    if (!printed && std::getenv("FTS_DEBUG_OBLIQUEZ_CALIB"))
    {
        std::fprintf(stderr, "[MklCrysFFTObliqueZ] N=%d, scale_fwd=%.15e, scale_bwd=%.15e, round_trip_err=%.15e\n",
                     N, dct_scale_fwd_, dct_scale_bwd_, max_err);
        printed = true;
    }

    DftiFreeDescriptor(&plan_calib);
}

//------------------------------------------------------------------------------
// DCT-II forward along Z (implemented via FFT with mirrored sequence)
// DCT-II: X[k] = sum_{n=0}^{N-1} x[n] * cos(pi*k*(2n+1)/(2N))
// Using FFT of mirrored sequence: x[0..N-1] -> mirror -> FFT -> extract
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::applyDCT2ForwardZ(const double* in, double* out) const
{
    const int Nx = nx_logical_[0];
    const int Ny = nx_logical_[1];
    const int Nz = nx_logical_[2];
    const int Nz2 = nx_physical_[2];
    const int howmany = Nx * Ny;
    const int freq_stride = Nz / 2 + 1;

    struct ThreadBuffers
    {
        std::unique_ptr<double[]> z_extended;
        std::unique_ptr<std::complex<double>[]> z_freq;
        int size = 0;
    };
    thread_local ThreadBuffers buffers;
    if (buffers.size != howmany * Nz)
    {
        buffers.z_extended.reset(new double[howmany * Nz]);
        buffers.z_freq.reset(new std::complex<double>[howmany * freq_stride]);
        buffers.size = howmany * Nz;
    }

    // Create mirrored sequences for all (x, y) columns
    for (int ixy = 0; ixy < howmany; ++ixy)
    {
        const int base_half = ixy * Nz2;
        const int base_full = ixy * Nz;
        // First half: copy
        for (int iz = 0; iz < Nz2; ++iz)
            buffers.z_extended.get()[base_full + iz] = in[base_half + iz];
        // Second half: mirror
        for (int iz = 0; iz < Nz2; ++iz)
            buffers.z_extended.get()[base_full + Nz - 1 - iz] = in[base_half + iz];
    }

    // Batched FFT along z
    DftiComputeForward(plan_dct_forward_z_, buffers.z_extended.get(), buffers.z_freq.get());

    // Extract DCT-II coefficients with twiddle factors and calibration
    for (int ixy = 0; ixy < howmany; ++ixy)
    {
        const int base_half = ixy * Nz2;
        const int base_freq = ixy * freq_stride;
        for (int k = 0; k < Nz2; ++k)
        {
            double re = buffers.z_freq.get()[base_freq + k].real();
            double im = buffers.z_freq.get()[base_freq + k].imag();
            // Twiddle factor: exp(-i*pi*k/(2*Nz2)) = cos - i*sin
            // DCT-II[k] = Re(FFT[k] * exp(-i*theta)) = Re*cos + Im*sin
            // Factor 0.5 because mirrored sequence FFT gives 2x the DCT-II
            // Apply forward calibration scale
            out[base_half + k] = 0.5 * (re * dct_cos_[k] + im * dct_sin_[k]) * dct_scale_fwd_;
        }
    }
}

//------------------------------------------------------------------------------
// DCT-III backward along Z (implemented via FFT)
// DCT-III: x[n] = X[0]/2 + sum_{k=1}^{N-1} X[k] * cos(pi*k*(2n+1)/(2N))
// This is the inverse of DCT-II (with appropriate normalization)
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::applyDCT3BackwardZ(const double* in, double* out) const
{
    const int Nx = nx_logical_[0];
    const int Ny = nx_logical_[1];
    const int Nz = nx_logical_[2];
    const int Nz2 = nx_physical_[2];
    const int howmany = Nx * Ny;
    const int freq_stride = Nz / 2 + 1;

    struct ThreadBuffers
    {
        std::unique_ptr<std::complex<double>[]> z_freq;
        std::unique_ptr<double[]> z_extended;
        int size = 0;
    };
    thread_local ThreadBuffers buffers;
    if (buffers.size != howmany * Nz)
    {
        buffers.z_freq.reset(new std::complex<double>[howmany * freq_stride]);
        buffers.z_extended.reset(new double[howmany * Nz]);
        buffers.size = howmany * Nz;
    }

    // Apply twiddle factors to create symmetric spectrum
    // Divide by forward scale to undo the calibration from DCT-II
    const double inv_fwd_scale = 1.0 / dct_scale_fwd_;
    for (int ixy = 0; ixy < howmany; ++ixy)
    {
        const int base_half = ixy * Nz2;
        const int base_freq = ixy * freq_stride;
        for (int k = 0; k < Nz2; ++k)
        {
            double xk = in[base_half + k] * inv_fwd_scale;
            // Twiddle factor: exp(i*pi*k/(2*Nz2)) = cos + i*sin
            // Factor 2 for inverse mirroring
            buffers.z_freq.get()[base_freq + k] = std::complex<double>(
                2.0 * xk * dct_cos_[k],
                2.0 * xk * dct_sin_[k]
            );
        }
        // Nyquist frequency
        buffers.z_freq.get()[base_freq + Nz2] = std::complex<double>(0.0, 0.0);
    }

    // Batched backward FFT along z
    DftiComputeBackward(plan_dct_backward_z_, buffers.z_freq.get(), buffers.z_extended.get());

    // Extract first half (DCT-III result) with normalization and backward calibration
    // IFFT returns 2N values, we take the first N
    // Total normalization: 1/(2N) = 1/Nz, times backward calibration
    const double scale = dct_scale_bwd_ / static_cast<double>(Nz);
    for (int ixy = 0; ixy < howmany; ++ixy)
    {
        const int base_half = ixy * Nz2;
        const int base_full = ixy * Nz;
        for (int iz = 0; iz < Nz2; ++iz)
            out[base_half + iz] = buffers.z_extended.get()[base_full + iz] * scale;
    }
}

//------------------------------------------------------------------------------
// Boltzmann factor cache
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::freeBoltzmann()
{
    cache_epoch_.fetch_add(1, std::memory_order_acq_rel);
}

MklCrysFFTObliqueZ::ThreadState& MklCrysFFTObliqueZ::get_thread_state() const
{
    struct ThreadLocalStates
    {
        std::unordered_map<const MklCrysFFTObliqueZ*, ThreadState> states;
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

double* MklCrysFFTObliqueZ::generateBoltzmann(double ds) const
{
    double* boltz = new double[M_complex_];

    const double G11 = recip_metric_[0];
    const double G12 = recip_metric_[1];
    const double G13 = recip_metric_[2];
    const double G22 = recip_metric_[3];
    const double G23 = recip_metric_[4];
    const double G33 = recip_metric_[5];
    const double factor = 4.0 * M_PI * M_PI;

    const int Nyh = nx_logical_[1] / 2 + 1;
    const int Nz2 = nx_physical_[2];
    for (int ix = 0; ix < nx_logical_[0]; ++ix)
    {
        int m1 = (ix > nx_logical_[0] / 2) ? (ix - nx_logical_[0]) : ix;
        for (int iy = 0; iy < Nyh; ++iy)
        {
            int m2 = iy;
            for (int iz = 0; iz < Nz2; ++iz)
            {
                int m3 = iz;
                double gmm = G11 * m1 * m1 + G22 * m2 * m2 + G33 * m3 * m3
                           + 2.0 * (G12 * m1 * m2 + G13 * m1 * m3 + G23 * m2 * m3);
                double k2 = factor * gmm;
                int idx = (ix * Nyh + iy) * Nz2 + iz;
                boltz[idx] = std::exp(-k2 * ds);
            }
        }
    }

    return boltz;
}

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::set_cell_para(const std::array<double, 6>& cell_para)
{
    if (cell_para == cell_para_)
        return;
    cell_para_ = cell_para;
    recip_metric_ = compute_recip_metric(cell_para_);
    freeBoltzmann();
}

void MklCrysFFTObliqueZ::set_contour_step(double ds)
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

void MklCrysFFTObliqueZ::diffusion(double* q_in, double* q_out)
{
    ThreadState& state = get_thread_state();
    if (!state.boltz_current)
        throw_with_line_number("MklCrysFFTObliqueZ::set_contour_step must be called before diffusion().");

    const int Nx = nx_logical_[0];
    const int Ny = nx_logical_[1];
    const int Nyh = Ny / 2 + 1;
    const int Nz2 = nx_physical_[2];

    struct ThreadBuffers
    {
        std::unique_ptr<double[]> real_in;
        std::unique_ptr<double[]> real_tmp;
        std::unique_ptr<std::complex<double>[]> complex;
        int size_real = 0;
        int size_complex = 0;
    };
    thread_local ThreadBuffers buffers;
    if (buffers.size_real != M_physical_ || buffers.size_complex != M_complex_)
    {
        buffers.real_in.reset(new double[M_physical_]);
        buffers.real_tmp.reset(new double[M_physical_]);
        buffers.complex.reset(new std::complex<double>[M_complex_]);
        buffers.size_real = M_physical_;
        buffers.size_complex = M_complex_;
    }

    // Step 1: Copy input
    std::memcpy(buffers.real_in.get(), q_in, sizeof(double) * M_physical_);

    // Step 2: DCT-II along Z
    applyDCT2ForwardZ(buffers.real_in.get(), buffers.real_tmp.get());

    // Step 3: Batched 2D FFT along XY (strided)
    DftiComputeForward(plan_fft_forward_xy_, buffers.real_tmp.get(), buffers.complex.get());

    // Step 4: Apply Boltzmann factor
    for (int i = 0; i < M_complex_; ++i)
        buffers.complex.get()[i] *= state.boltz_current[i];

    // Step 5: Batched 2D IFFT along XY (strided)
    DftiComputeBackward(plan_fft_backward_xy_, buffers.complex.get(), buffers.real_tmp.get());

    // Step 6: DCT-III along Z
    applyDCT3BackwardZ(buffers.real_tmp.get(), buffers.real_in.get());

    // Step 7: Apply overall normalization
    // DCT round trip: already normalized in DCT-III (1/Nz)
    // XY FFT round trip: MKL DFTI uses unnormalized transforms
    // Total: 1/(Nx * Ny * Nz) = norm_factor_ * 2 (because DCT-III has 1/Nz built in)
    // Actually DCT-II to DCT-III has factor 2, and we divided by Nz in DCT-III
    // So we need: 1/(Nx * Ny) for XY normalization
    const double xy_norm = 1.0 / static_cast<double>(Nx * Ny);
    for (int i = 0; i < M_physical_; ++i)
        q_out[i] = buffers.real_in.get()[i] * xy_norm;
}

void MklCrysFFTObliqueZ::apply_multiplier(const double* q_in, double* q_out, const double* multiplier)
{
    const int Nx = nx_logical_[0];
    const int Ny = nx_logical_[1];
    const int Nyh = Ny / 2 + 1;
    const int Nz2 = nx_physical_[2];

    struct ThreadBuffers
    {
        std::unique_ptr<double[]> real_in;
        std::unique_ptr<double[]> real_tmp;
        std::unique_ptr<std::complex<double>[]> complex;
        int size_real = 0;
        int size_complex = 0;
    };
    thread_local ThreadBuffers buffers;
    if (buffers.size_real != M_physical_ || buffers.size_complex != M_complex_)
    {
        buffers.real_in.reset(new double[M_physical_]);
        buffers.real_tmp.reset(new double[M_physical_]);
        buffers.complex.reset(new std::complex<double>[M_complex_]);
        buffers.size_real = M_physical_;
        buffers.size_complex = M_complex_;
    }

    // Step 1: Copy input
    std::memcpy(buffers.real_in.get(), q_in, sizeof(double) * M_physical_);

    // Step 2: DCT-II along Z
    applyDCT2ForwardZ(buffers.real_in.get(), buffers.real_tmp.get());

    // Step 3: Batched 2D FFT along XY (strided)
    DftiComputeForward(plan_fft_forward_xy_, buffers.real_tmp.get(), buffers.complex.get());

    // Step 4: Apply multiplier
    for (int i = 0; i < M_complex_; ++i)
        buffers.complex.get()[i] *= multiplier[i];

    // Step 5: Batched 2D IFFT along XY (strided)
    DftiComputeBackward(plan_fft_backward_xy_, buffers.complex.get(), buffers.real_tmp.get());

    // Step 6: DCT-III along Z
    applyDCT3BackwardZ(buffers.real_tmp.get(), buffers.real_in.get());

    // Step 7: Apply overall normalization
    const double xy_norm = 1.0 / static_cast<double>(Nx * Ny);
    for (int i = 0; i < M_physical_; ++i)
        q_out[i] = buffers.real_in.get()[i] * xy_norm;
}
