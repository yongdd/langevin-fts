/**
 * @file FftwCrysFFTObliqueZ.cpp
 * @brief FFTW implementation of crystallographic FFT for z-mirror symmetry.
 */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "FftwCrysFFTObliqueZ.h"

//------------------------------------------------------------------------------
// Constructor / Destructor
//------------------------------------------------------------------------------
FftwCrysFFTObliqueZ::FftwCrysFFTObliqueZ(
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

FftwCrysFFTObliqueZ::~FftwCrysFFTObliqueZ()
{
    release();
}

void FftwCrysFFTObliqueZ::release()
{
    if (plan_dct_forward_z_) fftw_destroy_plan(plan_dct_forward_z_);
    if (plan_dct_backward_z_) fftw_destroy_plan(plan_dct_backward_z_);
    if (plan_fft_forward_xy_) fftw_destroy_plan(plan_fft_forward_xy_);
    if (plan_fft_backward_xy_) fftw_destroy_plan(plan_fft_backward_xy_);
    plan_dct_forward_z_ = plan_dct_backward_z_ = nullptr;
    plan_fft_forward_xy_ = plan_fft_backward_xy_ = nullptr;
}

//------------------------------------------------------------------------------
// FFTW plan initialization
//------------------------------------------------------------------------------
void FftwCrysFFTObliqueZ::initFFTPlans()
{
    const unsigned plan_flags = FFTW_MEASURE;

    // Planning arrays only, freed on return: the transforms run on the
    // thread-local buffers of get_thread_buffers() (new-array execute)
    std::unique_ptr<double, RealDeleter> io_buffer(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
    std::unique_ptr<double, RealDeleter> temp_buffer(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
    std::unique_ptr<fftw_complex, ComplexDeleter> complex_buffer(
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M_complex_)));

    if (!io_buffer || !temp_buffer || !complex_buffer)
        throw_with_line_number("Failed to allocate FFTW buffers for FftwCrysFFTObliqueZ.");

    // DCT-II/III along Z for each (x,y) column
    int n_z[1] = {nx_physical_[2]};
    int howmany_z = nx_physical_[0] * nx_physical_[1];
    int inembed_z[1] = {n_z[0]};
    int onembed_z[1] = {n_z[0]};
    int istride_z = 1;
    int ostride_z = 1;
    int idist_z = n_z[0];
    int odist_z = n_z[0];

    fftw_r2r_kind kind_forward[1] = {FFTW_REDFT10};
    fftw_r2r_kind kind_backward[1] = {FFTW_REDFT01};

    plan_dct_forward_z_ = fftw_plan_many_r2r(
        1, n_z, howmany_z,
        io_buffer.get(), inembed_z, istride_z, idist_z,
        temp_buffer.get(), onembed_z, ostride_z, odist_z,
        kind_forward, plan_flags);

    plan_dct_backward_z_ = fftw_plan_many_r2r(
        1, n_z, howmany_z,
        temp_buffer.get(), inembed_z, istride_z, idist_z,
        io_buffer.get(), onembed_z, ostride_z, odist_z,
        kind_backward, plan_flags);

    // Real-to-complex FFT along X,Y for each Z slice (batched, strided)
    int n_xy[2] = {nx_logical_[0], nx_logical_[1]};
    int howmany_xy = nx_physical_[2];
    int inembed_xy[2] = {n_xy[0], n_xy[1]};
    int onembed_xy[2] = {n_xy[0], n_xy[1] / 2 + 1};
    int istride_xy = nx_physical_[2];
    int ostride_xy = nx_physical_[2];
    int idist_xy = 1;
    int odist_xy = 1;

    plan_fft_forward_xy_ = fftw_plan_many_dft_r2c(
        2, n_xy, howmany_xy,
        temp_buffer.get(), inembed_xy, istride_xy, idist_xy,
        complex_buffer.get(), onembed_xy, ostride_xy, odist_xy,
        plan_flags);

    plan_fft_backward_xy_ = fftw_plan_many_dft_c2r(
        2, n_xy, howmany_xy,
        complex_buffer.get(), onembed_xy, ostride_xy, odist_xy,
        temp_buffer.get(), inembed_xy, istride_xy, idist_xy,
        plan_flags);

    if (!plan_dct_forward_z_ || !plan_dct_backward_z_ || !plan_fft_forward_xy_ || !plan_fft_backward_xy_)
        throw_with_line_number("Failed to create FFTW plans for FftwCrysFFTObliqueZ.");
}

//------------------------------------------------------------------------------
// Thread-local work buffers and z-direction DCT helpers shared by
// diffusion() and apply_multiplier()
//------------------------------------------------------------------------------
FftwCrysFFTObliqueZ::ThreadBuffers& FftwCrysFFTObliqueZ::get_thread_buffers() const
{
    thread_local ThreadBuffers buffers;
    if (buffers.size_real != M_physical_ || buffers.size_complex != M_complex_)
    {
        buffers.real_in.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
        buffers.real_tmp.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
        buffers.complex.reset(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M_complex_)));
        if (!buffers.real_in || !buffers.real_tmp || !buffers.complex)
            throw_with_line_number("Failed to allocate thread-local FFTW buffers for FftwCrysFFTObliqueZ.");
        buffers.size_real = M_physical_;
        buffers.size_complex = M_complex_;
    }
    return buffers;
}

void FftwCrysFFTObliqueZ::dct2_execute(const double* in, double* out) const
{
    fftw_execute_r2r(plan_dct_forward_z_, const_cast<double*>(in), out);
}

void FftwCrysFFTObliqueZ::dct3_execute(const double* in, double* out) const
{
    fftw_execute_r2r(plan_dct_backward_z_, const_cast<double*>(in), out);
}

void FftwCrysFFTObliqueZ::apply_spectral_multiplier(const double* q_in, double* q_out, const double* multiplier) const
{
    ThreadBuffers& buffers = get_thread_buffers();

    std::memcpy(buffers.real_in.get(), q_in, sizeof(double) * M_physical_);

    dct2_execute(buffers.real_in.get(), buffers.real_tmp.get());
    fftw_execute_dft_r2c(plan_fft_forward_xy_, buffers.real_tmp.get(), buffers.complex.get());

    for (int i = 0; i < M_complex_; ++i)
    {
        buffers.complex.get()[i][0] *= multiplier[i];
        buffers.complex.get()[i][1] *= multiplier[i];
    }

    fftw_execute_dft_c2r(plan_fft_backward_xy_, buffers.complex.get(), buffers.real_tmp.get());
    dct3_execute(buffers.real_tmp.get(), buffers.real_in.get());

    for (int i = 0; i < M_physical_; ++i)
        q_out[i] = buffers.real_in.get()[i] * norm_factor_;
}

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------
void FftwCrysFFTObliqueZ::diffusion(double* q_in, double* q_out)
{
    const double* boltz = current_boltzmann();
    if (!boltz)
        throw_with_line_number("FftwCrysFFTObliqueZ::set_contour_step must be called before diffusion().");

    static const bool do_profile = (std::getenv("FTS_PROFILE_CRYSFFT_HEX_CPU_DETAIL") != nullptr);   // read once
    static std::atomic<bool> prof_once{false};
    const bool profile_this = do_profile && !prof_once.exchange(true);

    if (profile_this)
    {
        ThreadBuffers& buffers = get_thread_buffers();

        auto t0 = std::chrono::steady_clock::now();
        std::memcpy(buffers.real_in.get(), q_in, sizeof(double) * M_physical_);
        auto t1 = std::chrono::steady_clock::now();

        dct2_execute(buffers.real_in.get(), buffers.real_tmp.get());
        auto t2 = std::chrono::steady_clock::now();

        fftw_execute_dft_r2c(plan_fft_forward_xy_, buffers.real_tmp.get(), buffers.complex.get());
        auto t3 = std::chrono::steady_clock::now();

        for (int i = 0; i < M_complex_; ++i)
        {
            buffers.complex.get()[i][0] *= boltz[i];
            buffers.complex.get()[i][1] *= boltz[i];
        }
        auto t5 = std::chrono::steady_clock::now();

        fftw_execute_dft_c2r(plan_fft_backward_xy_, buffers.complex.get(), buffers.real_tmp.get());
        auto t6 = std::chrono::steady_clock::now();

        dct3_execute(buffers.real_tmp.get(), buffers.real_in.get());
        auto t7 = std::chrono::steady_clock::now();

        for (int i = 0; i < M_physical_; ++i)
            q_out[i] = buffers.real_in.get()[i] * norm_factor_;
        auto t8 = std::chrono::steady_clock::now();

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::printf("[CrysFFT-ObliqueZ CPU] copy(ms)=%.4f dct2(ms)=%.4f r2c(ms)=%.4f "
                    "boltz(ms)=%.4f c2r(ms)=%.4f dct3(ms)=%.4f store(ms)=%.4f total(ms)=%.4f\n",
                    ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t5),
                    ms(t5, t6), ms(t6, t7), ms(t7, t8), ms(t0, t8));
        return;
    }

    apply_spectral_multiplier(q_in, q_out, boltz);
}

void FftwCrysFFTObliqueZ::apply_multiplier(const double* q_in, double* q_out, const double* multiplier)
{
    apply_spectral_multiplier(q_in, q_out, multiplier);
}
