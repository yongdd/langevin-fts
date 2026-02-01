/**
 * @file MklCrysFFTRecursive3m.cpp
 * @brief MKL implementation of recursive crystallographic FFT (2x2y2z).
 */

#include "MklCrysFFTRecursive3m.h"
#include <cstring>
#include <vector>

namespace {

constexpr size_t kAlignment = 64;

inline size_t align_up(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
MklCrysFFTRecursive3m::MklCrysFFTRecursive3m(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> translational_part)
    : Base(nx_logical, cell_para, translational_part)
{
    init_fft_plans();
    generate_twiddle_factors();
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
MklCrysFFTRecursive3m::~MklCrysFFTRecursive3m()
{
    if (plan_forward_) DftiFreeDescriptor(&plan_forward_);
    if (plan_backward_) DftiFreeDescriptor(&plan_backward_);
}

//------------------------------------------------------------------------------
// Initialize MKL DFTI descriptors
//------------------------------------------------------------------------------
void MklCrysFFTRecursive3m::init_fft_plans()
{
    MKL_LONG dims[3] = { nx_physical_[0], nx_physical_[1], nx_physical_[2] };
    MKL_LONG strides_real[4] = { 0, nx_physical_[1] * nx_physical_[2], nx_physical_[2], 1 };
    MKL_LONG strides_complex[4] = { 0, nx_physical_[1] * nx_physical_[2], nx_physical_[2], 1 };

    MKL_LONG status;

    // Forward r2c plan
    status = DftiCreateDescriptor(&plan_forward_, DFTI_DOUBLE, DFTI_REAL, 3, dims);
    if (status != 0)
        throw_with_line_number("Failed to create MKL DFTI descriptor for forward FFT.");
    status = DftiSetValue(plan_forward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(plan_forward_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(plan_forward_, DFTI_INPUT_STRIDES, strides_real);
    status = DftiSetValue(plan_forward_, DFTI_OUTPUT_STRIDES, strides_complex);
    status = DftiCommitDescriptor(plan_forward_);
    if (status != 0)
        throw_with_line_number("Failed to commit MKL DFTI descriptor for forward FFT.");

    // Backward c2r plan
    status = DftiCreateDescriptor(&plan_backward_, DFTI_DOUBLE, DFTI_REAL, 3, dims);
    if (status != 0)
        throw_with_line_number("Failed to create MKL DFTI descriptor for backward FFT.");
    status = DftiSetValue(plan_backward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(plan_backward_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(plan_backward_, DFTI_INPUT_STRIDES, strides_complex);
    status = DftiSetValue(plan_backward_, DFTI_OUTPUT_STRIDES, strides_real);
    status = DftiCommitDescriptor(plan_backward_);
    if (status != 0)
        throw_with_line_number("Failed to commit MKL DFTI descriptor for backward FFT.");
}

//------------------------------------------------------------------------------
// Apply FFT with cached Boltzmann factors
//------------------------------------------------------------------------------
void MklCrysFFTRecursive3m::apply_with_cache(
    const KCache& cache, const double* q_in, double* q_out) const
{
    struct ThreadBuffers {
        std::vector<double> iomat;
        std::vector<std::complex<double>> step1;
        std::vector<std::complex<double>> step2;
        int size = 0;
    };

    thread_local ThreadBuffers buffers;
    if (buffers.size != M_physical_)
    {
        buffers.iomat.resize(M_physical_);
        buffers.step1.resize(M_physical_);
        buffers.step2.resize(M_physical_);
        buffers.size = M_physical_;
    }

    std::memcpy(buffers.iomat.data(), q_in, sizeof(double) * M_physical_);

    // Forward r2c FFT
    DftiComputeForward(plan_forward_, buffers.iomat.data(), buffers.step1.data());

    const int Nx2 = nx_physical_[0];
    const int Ny2 = nx_physical_[1];
    const int Nz2 = nx_physical_[2];
    const size_t Nz_qua = align_up(static_cast<size_t>(Nz2 / 2 + 1), kAlignment / 8);

    const auto& k_re = cache.re;
    const auto& k_im = cache.im;

    for (int ix = 0; ix < Nx2; ++ix)
    {
        for (int iy = 0; iy < Ny2; ++iy)
        {
            const size_t base_shift = static_cast<size_t>(ix * Ny2 + iy) * Nz2;
            const std::complex<double>* src000 = buffers.step1.data() + base_shift;
            const std::complex<double>* src110 = buffers.step1.data() + (static_cast<size_t>((Nx2 - ix) % Nx2) * Ny2 + (Ny2 - iy) % Ny2) * Nz2;
            const std::complex<double>* src010 = buffers.step1.data() + (static_cast<size_t>(ix) * Ny2 + (Ny2 - iy) % Ny2) * Nz2;
            const std::complex<double>* src100 = buffers.step1.data() + (static_cast<size_t>((Nx2 - ix) % Nx2) * Ny2 + iy) * Nz2;
            std::complex<double>* dst = buffers.step2.data() + base_shift;

            for (size_t iz = 0; iz < Nz_qua; ++iz)
            {
                size_t idx = base_shift + iz;
                double src000_re = src000[iz].real();
                double src000_im = src000[iz].imag();
                double src110_re = src110[iz].real();
                double src110_im = src110[iz].imag();
                double src010_re = src010[iz].real();
                double src010_im = src010[iz].imag();
                double src100_re = src100[iz].real();
                double src100_im = src100[iz].imag();

                double dst_re
                    = src000_re * k_re[0].get()[idx] - src000_im * k_im[7].get()[idx]
                    + src110_re * k_re[6].get()[idx] - src110_im * k_im[1].get()[idx]
                    + src010_re * k_re[2].get()[idx] - src010_im * k_im[5].get()[idx]
                    + src100_re * k_re[4].get()[idx] - src100_im * k_im[3].get()[idx];

                double dst_im
                    = src000_re * k_im[0].get()[idx] + src000_im * k_re[7].get()[idx]
                    + src110_re * k_im[6].get()[idx] + src110_im * k_re[1].get()[idx]
                    + src010_re * k_im[2].get()[idx] + src010_im * k_re[5].get()[idx]
                    + src100_re * k_im[4].get()[idx] + src100_im * k_re[3].get()[idx];

                dst[iz] = std::complex<double>(dst_re, dst_im);
            }
        }
    }

    // Backward c2r FFT
    DftiComputeBackward(plan_backward_, buffers.step2.data(), buffers.iomat.data());

    std::memcpy(q_out, buffers.iomat.data(), sizeof(double) * M_physical_);
}
