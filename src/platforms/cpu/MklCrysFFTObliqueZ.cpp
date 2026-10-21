/**
 * @file MklCrysFFTObliqueZ.cpp
 * @brief MKL implementation of crystallographic FFT for z-mirror symmetry.
 *
 * Uses FFT-based DCT-II/III with mirrored sequences since MKL lacks native DCT.
 *
 * @see CrysFFTObliqueZBase for common functionality
 * @see FftwCrysFFTObliqueZ for FFTW implementation
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "MklCrysFFTObliqueZ.h"
#include "MklStatusCheck.h"
#include "Exception.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace
{
/// Work arrays of the z DCT steps, one set per thread and shared by every
/// instance on that thread. Each array is sized from its own requirement: two
/// grids with the same Nx*Ny*Nz but a different Nz need the same real size and
/// different spectrum sizes.
struct ZWork
{
    std::vector<double> z_extended;               ///< mirrored sequences, howmany * Nz
    std::vector<std::complex<double>> z_freq;     ///< their spectra, howmany * (Nz/2 + 1)
};
thread_local ZWork z_work;

ZWork& z_work_for(size_t n_extended, size_t n_freq)
{
    if (z_work.z_extended.size() != n_extended)
        z_work.z_extended.resize(n_extended);
    if (z_work.z_freq.size() != n_freq)
        z_work.z_freq.resize(n_freq);
    return z_work;
}
}  // namespace

std::array<size_t, 2> MklCrysFFTObliqueZ::thread_z_work_sizes()
{
    return {z_work.z_extended.size(), z_work.z_freq.size()};
}

//------------------------------------------------------------------------------
// Constructor / Destructor
//------------------------------------------------------------------------------
MklCrysFFTObliqueZ::MklCrysFFTObliqueZ(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> trans_part)
    : Base(nx_logical, cell_para, trans_part)
{
    try
    {
        initFFTPlans();
    }
    catch (...)
    {
        // The destructor does not run for a partly constructed object
        release();
        throw;
    }
}

MklCrysFFTObliqueZ::~MklCrysFFTObliqueZ()
{
    release();
}

void MklCrysFFTObliqueZ::release()
{
    if (plan_dct_forward_z_) DftiFreeDescriptor(&plan_dct_forward_z_);
    if (plan_dct_backward_z_) DftiFreeDescriptor(&plan_dct_backward_z_);
    if (plan_fft_forward_xy_) DftiFreeDescriptor(&plan_fft_forward_xy_);
    if (plan_fft_backward_xy_) DftiFreeDescriptor(&plan_fft_backward_xy_);
    plan_dct_forward_z_ = plan_dct_backward_z_ = nullptr;
    plan_fft_forward_xy_ = plan_fft_backward_xy_ = nullptr;
}

//------------------------------------------------------------------------------
// MKL plan initialization
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::initFFTPlans()
{
    // DCT via FFT: 1D FFT along z of size Nz for each (x,y) column
    // Data layout: [Nx*Ny][Nz] with contiguous z-columns
    const int Nz = nx_logical_[2];
    const int howmany_z = nx_logical_[0] * nx_logical_[1];
    const MKL_LONG Nz_long = Nz;

    mkl_dfti_check(DftiCreateDescriptor(&plan_dct_forward_z_, DFTI_DOUBLE, DFTI_REAL, 1, Nz_long), "DftiCreateDescriptor (z forward)");
    mkl_dfti_check(DftiSetValue(plan_dct_forward_z_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, z forward)");
    mkl_dfti_check(DftiSetValue(plan_dct_forward_z_, DFTI_NUMBER_OF_TRANSFORMS, static_cast<MKL_LONG>(howmany_z)), "DftiSetValue(DFTI_NUMBER_OF_TRANSFORMS, z forward)");
    mkl_dfti_check(DftiSetValue(plan_dct_forward_z_, DFTI_INPUT_DISTANCE, Nz_long), "DftiSetValue(DFTI_INPUT_DISTANCE, z forward)");
    mkl_dfti_check(DftiSetValue(plan_dct_forward_z_, DFTI_OUTPUT_DISTANCE, static_cast<MKL_LONG>(Nz / 2 + 1)), "DftiSetValue(DFTI_OUTPUT_DISTANCE, z forward)");
    mkl_dfti_check(DftiSetValue(plan_dct_forward_z_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, z forward)");
    mkl_dfti_check(DftiCommitDescriptor(plan_dct_forward_z_), "DftiCommitDescriptor (z forward)");

    mkl_dfti_check(DftiCreateDescriptor(&plan_dct_backward_z_, DFTI_DOUBLE, DFTI_REAL, 1, Nz_long), "DftiCreateDescriptor (z backward)");
    mkl_dfti_check(DftiSetValue(plan_dct_backward_z_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, z backward)");
    mkl_dfti_check(DftiSetValue(plan_dct_backward_z_, DFTI_NUMBER_OF_TRANSFORMS, static_cast<MKL_LONG>(howmany_z)), "DftiSetValue(DFTI_NUMBER_OF_TRANSFORMS, z backward)");
    mkl_dfti_check(DftiSetValue(plan_dct_backward_z_, DFTI_INPUT_DISTANCE, static_cast<MKL_LONG>(Nz / 2 + 1)), "DftiSetValue(DFTI_INPUT_DISTANCE, z backward)");
    mkl_dfti_check(DftiSetValue(plan_dct_backward_z_, DFTI_OUTPUT_DISTANCE, Nz_long), "DftiSetValue(DFTI_OUTPUT_DISTANCE, z backward)");
    mkl_dfti_check(DftiSetValue(plan_dct_backward_z_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, z backward)");
    mkl_dfti_check(DftiCommitDescriptor(plan_dct_backward_z_), "DftiCommitDescriptor (z backward)");

    // 2D Real-to-complex FFT along X,Y for each Z slice (batched, strided)
    // Data layout after DCT: [Nx][Ny][Nz/2] where z varies fastest
    // XY FFT batch: howmany = Nz/2, stride = Nz/2, dist = 1
    MKL_LONG dims_xy[2] = { static_cast<MKL_LONG>(nx_logical_[0]), static_cast<MKL_LONG>(nx_logical_[1]) };
    const int howmany_xy = nx_physical_[2];
    const MKL_LONG stride_xy = nx_physical_[2];

    // For strided transforms, we need to set input/output strides
    MKL_LONG input_strides_xy[3] = {0, static_cast<MKL_LONG>(nx_logical_[1]) * stride_xy, stride_xy};
    MKL_LONG output_strides_xy[3] = {0, static_cast<MKL_LONG>(nx_logical_[1] / 2 + 1) * stride_xy, stride_xy};

    mkl_dfti_check(DftiCreateDescriptor(&plan_fft_forward_xy_, DFTI_DOUBLE, DFTI_REAL, 2, dims_xy), "DftiCreateDescriptor (xy forward)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, xy forward)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, DFTI_NUMBER_OF_TRANSFORMS, static_cast<MKL_LONG>(howmany_xy)), "DftiSetValue(DFTI_NUMBER_OF_TRANSFORMS, xy forward)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, DFTI_INPUT_DISTANCE, 1), "DftiSetValue(DFTI_INPUT_DISTANCE, xy forward)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, DFTI_OUTPUT_DISTANCE, 1), "DftiSetValue(DFTI_OUTPUT_DISTANCE, xy forward)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, DFTI_INPUT_STRIDES, input_strides_xy), "DftiSetValue(DFTI_INPUT_STRIDES, xy forward)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, DFTI_OUTPUT_STRIDES, output_strides_xy), "DftiSetValue(DFTI_OUTPUT_STRIDES, xy forward)");
    mkl_dfti_check(DftiSetValue(plan_fft_forward_xy_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, xy forward)");
    mkl_dfti_check(DftiCommitDescriptor(plan_fft_forward_xy_), "DftiCommitDescriptor (xy forward)");

    mkl_dfti_check(DftiCreateDescriptor(&plan_fft_backward_xy_, DFTI_DOUBLE, DFTI_REAL, 2, dims_xy), "DftiCreateDescriptor (xy backward)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, xy backward)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, DFTI_NUMBER_OF_TRANSFORMS, static_cast<MKL_LONG>(howmany_xy)), "DftiSetValue(DFTI_NUMBER_OF_TRANSFORMS, xy backward)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, DFTI_INPUT_DISTANCE, 1), "DftiSetValue(DFTI_INPUT_DISTANCE, xy backward)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, DFTI_OUTPUT_DISTANCE, 1), "DftiSetValue(DFTI_OUTPUT_DISTANCE, xy backward)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, DFTI_INPUT_STRIDES, output_strides_xy), "DftiSetValue(DFTI_INPUT_STRIDES, xy backward)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, DFTI_OUTPUT_STRIDES, input_strides_xy), "DftiSetValue(DFTI_OUTPUT_STRIDES, xy backward)");
    mkl_dfti_check(DftiSetValue(plan_fft_backward_xy_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, xy backward)");
    mkl_dfti_check(DftiCommitDescriptor(plan_fft_backward_xy_), "DftiCommitDescriptor (xy backward)");

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

    // No scale factors: 0.5*(Re cos + Im sin) of the mirrored-sequence FFT is
    // exactly the textbook DCT-II, and the unnormalized 2N-point inverse of the
    // twiddled spectrum divided by 2N returns the signal.
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

    ZWork& buffers = z_work_for(static_cast<size_t>(howmany) * Nz, static_cast<size_t>(howmany) * freq_stride);

    // Create mirrored sequences for all (x, y) columns
    for (int ixy = 0; ixy < howmany; ++ixy)
    {
        const int base_half = ixy * Nz2;
        const int base_full = ixy * Nz;
        // First half: copy
        for (int iz = 0; iz < Nz2; ++iz)
            buffers.z_extended[base_full + iz] = in[base_half + iz];
        // Second half: mirror
        for (int iz = 0; iz < Nz2; ++iz)
            buffers.z_extended[base_full + Nz - 1 - iz] = in[base_half + iz];
    }

    // Batched FFT along z
    mkl_dfti_check(DftiComputeForward(plan_dct_forward_z_, buffers.z_extended.data(), buffers.z_freq.data()), "DftiComputeForward (z DCT-II)");

    // Extract DCT-II coefficients with twiddle factors
    for (int ixy = 0; ixy < howmany; ++ixy)
    {
        const int base_half = ixy * Nz2;
        const int base_freq = ixy * freq_stride;
        for (int k = 0; k < Nz2; ++k)
        {
            double re = buffers.z_freq[base_freq + k].real();
            double im = buffers.z_freq[base_freq + k].imag();
            // Twiddle factor: exp(-i*pi*k/(2*Nz2)) = cos - i*sin
            // DCT-II[k] = Re(FFT[k] * exp(-i*theta)) = Re*cos + Im*sin
            // Factor 0.5 because mirrored sequence FFT gives 2x the DCT-II
            out[base_half + k] = 0.5 * (re * dct_cos_[k] + im * dct_sin_[k]);
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

    ZWork& buffers = z_work_for(static_cast<size_t>(howmany) * Nz, static_cast<size_t>(howmany) * freq_stride);

    // Apply twiddle factors to create symmetric spectrum
    for (int ixy = 0; ixy < howmany; ++ixy)
    {
        const int base_half = ixy * Nz2;
        const int base_freq = ixy * freq_stride;
        for (int k = 0; k < Nz2; ++k)
        {
            const double xk = in[base_half + k];
            // Twiddle factor: exp(i*pi*k/(2*Nz2)) = cos + i*sin
            // Factor 2 for inverse mirroring
            buffers.z_freq[base_freq + k] = std::complex<double>(
                2.0 * xk * dct_cos_[k],
                2.0 * xk * dct_sin_[k]
            );
        }
        // Nyquist frequency
        buffers.z_freq[base_freq + Nz2] = std::complex<double>(0.0, 0.0);
    }

    // Batched backward FFT along z
    mkl_dfti_check(DftiComputeBackward(plan_dct_backward_z_, buffers.z_freq.data(), buffers.z_extended.data()), "DftiComputeBackward (z DCT-III)");

    // Extract first half (DCT-III result) with normalization
    // IFFT returns 2N values, we take the first N
    // Total normalization: 1/(2N) = 1/Nz
    const double scale = 1.0 / static_cast<double>(Nz);
    for (int ixy = 0; ixy < howmany; ++ixy)
    {
        const int base_half = ixy * Nz2;
        const int base_full = ixy * Nz;
        for (int iz = 0; iz < Nz2; ++iz)
            out[base_half + iz] = buffers.z_extended[base_full + iz] * scale;
    }
}

//------------------------------------------------------------------------------
// Shared pipeline of diffusion() and apply_multiplier():
// DCT-II(z) -> FFT(xy) -> multiply -> IFFT(xy) -> DCT-III(z) -> normalize
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::apply_spectral_multiplier(const double* q_in, double* q_out, const double* multiplier) const
{
    const int Nx = nx_logical_[0];
    const int Ny = nx_logical_[1];

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
        buffers.real_in = std::make_unique<double[]>(M_physical_);
        buffers.real_tmp = std::make_unique<double[]>(M_physical_);
        buffers.complex = std::make_unique<std::complex<double>[]>(M_complex_);
        buffers.size_real = M_physical_;
        buffers.size_complex = M_complex_;
    }

    // Step 1: Copy input
    std::memcpy(buffers.real_in.get(), q_in, sizeof(double) * M_physical_);

    // Step 2: DCT-II along Z
    applyDCT2ForwardZ(buffers.real_in.get(), buffers.real_tmp.get());

    // Step 3: Batched 2D FFT along XY (strided)
    mkl_dfti_check(DftiComputeForward(plan_fft_forward_xy_, buffers.real_tmp.get(), buffers.complex.get()), "DftiComputeForward (xy)");

    // Step 4: Apply the spectral multiplier (Boltzmann factor or stress multiplier)
    for (int i = 0; i < M_complex_; ++i)
        buffers.complex.get()[i] *= multiplier[i];

    // Step 5: Batched 2D IFFT along XY (strided)
    mkl_dfti_check(DftiComputeBackward(plan_fft_backward_xy_, buffers.complex.get(), buffers.real_tmp.get()), "DftiComputeBackward (xy)");

    // Step 6: DCT-III along Z
    applyDCT3BackwardZ(buffers.real_tmp.get(), buffers.real_in.get());

    // Step 7: Apply overall normalization
    // DCT round trip: already normalized in DCT-III (1/Nz)
    // XY FFT round trip: MKL DFTI uses unnormalized transforms,
    // so only 1/(Nx * Ny) remains
    const double xy_norm = 1.0 / static_cast<double>(Nx * Ny);
    for (int i = 0; i < M_physical_; ++i)
        q_out[i] = buffers.real_in.get()[i] * xy_norm;
}

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------
void MklCrysFFTObliqueZ::diffusion(double* q_in, double* q_out)
{
    ThreadState& state = get_thread_state();
    if (!state.boltz_current)
        throw_with_line_number("MklCrysFFTObliqueZ::set_contour_step must be called before diffusion().");

    apply_spectral_multiplier(q_in, q_out, state.boltz_current);
}

void MklCrysFFTObliqueZ::apply_multiplier(const double* q_in, double* q_out, const double* multiplier)
{
    apply_spectral_multiplier(q_in, q_out, multiplier);
}
