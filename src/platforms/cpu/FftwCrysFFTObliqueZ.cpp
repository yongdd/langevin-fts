/**
 * @file FftwCrysFFTObliqueZ.cpp
 * @brief CPU implementation of crystallographic FFT for z-mirror symmetry (oblique in-plane angles).
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>

#include "FftwCrysFFTObliqueZ.h"
#include "Exception.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------
std::array<double, 6> FftwCrysFFTObliqueZ::compute_recip_metric(const std::array<double, 6>& cell_para)
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
FftwCrysFFTObliqueZ::FftwCrysFFTObliqueZ(
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
            throw_with_line_number("FftwCrysFFTObliqueZ requires positive grid dimensions.");
    }
    if (nx_logical_[2] % 2 != 0)
        throw_with_line_number("FftwCrysFFTObliqueZ requires even Nz.");

    nx_physical_ = {nx_logical_[0], nx_logical_[1], nx_logical_[2] / 2};
    M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
    M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];
    M_complex_ = nx_logical_[0] * (nx_logical_[1] / 2 + 1) * nx_physical_[2];
    norm_factor_ = 1.0 / static_cast<double>(M_logical_);

    initFFTPlans();
}

FftwCrysFFTObliqueZ::~FftwCrysFFTObliqueZ()
{
    freeBoltzmann();

    if (plan_dct_forward_z_) fftw_destroy_plan(plan_dct_forward_z_);
    if (plan_dct_backward_z_) fftw_destroy_plan(plan_dct_backward_z_);
    if (plan_fft_forward_xy_) fftw_destroy_plan(plan_fft_forward_xy_);
    if (plan_fft_backward_xy_) fftw_destroy_plan(plan_fft_backward_xy_);
    if (plan_fft_z_forward_) fftw_destroy_plan(plan_fft_z_forward_);
    if (plan_fft_z_backward_) fftw_destroy_plan(plan_fft_z_backward_);

    if (io_buffer_) fftw_free(io_buffer_);
    if (temp_buffer_) fftw_free(temp_buffer_);
    if (complex_buffer_) fftw_free(complex_buffer_);
    if (fft_z_real_) fftw_free(fft_z_real_);
    if (fft_z_complex_) fftw_free(fft_z_complex_);
}

//------------------------------------------------------------------------------
// FFTW plan initialization
//------------------------------------------------------------------------------
void FftwCrysFFTObliqueZ::initFFTPlans()
{
    unsigned plan_flags = FFTW_MEASURE;
    if (std::getenv("FTS_FFTW_EXHAUSTIVE"))
        plan_flags = FFTW_EXHAUSTIVE;
    else if (std::getenv("FTS_FFTW_PATIENT"))
        plan_flags = FFTW_PATIENT;
    else if (std::getenv("FTS_FFTW_ESTIMATE"))
        plan_flags = FFTW_ESTIMATE;
    if (std::getenv("FTS_FFTW_WISDOM_ONLY"))
        plan_flags |= FFTW_WISDOM_ONLY;

    const char* wisdom_file = std::getenv("FTS_FFTW_WISDOM_FILE");
    if (wisdom_file && wisdom_file[0] != '\0')
        fftw_import_wisdom_from_filename(wisdom_file);

    io_buffer_ = static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_));
    temp_buffer_ = static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_));
    complex_buffer_ = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M_complex_));

    if (!io_buffer_ || !temp_buffer_ || !complex_buffer_)
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
        io_buffer_, inembed_z, istride_z, idist_z,
        temp_buffer_, onembed_z, ostride_z, odist_z,
        kind_forward, plan_flags);

    plan_dct_backward_z_ = fftw_plan_many_r2r(
        1, n_z, howmany_z,
        temp_buffer_, inembed_z, istride_z, idist_z,
        io_buffer_, onembed_z, ostride_z, odist_z,
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
        temp_buffer_, inembed_xy, istride_xy, idist_xy,
        complex_buffer_, onembed_xy, ostride_xy, odist_xy,
        plan_flags);

    plan_fft_backward_xy_ = fftw_plan_many_dft_c2r(
        2, n_xy, howmany_xy,
        complex_buffer_, onembed_xy, ostride_xy, odist_xy,
        temp_buffer_, inembed_xy, istride_xy, idist_xy,
        plan_flags);

    use_fft_dct_ = (std::getenv("FTS_CRYSFFT_HEX_DCT_FFT") != nullptr);
    if (use_fft_dct_)
        initFFTPlansZ(plan_flags);

    if (!plan_dct_forward_z_ || !plan_dct_backward_z_ || !plan_fft_forward_xy_ || !plan_fft_backward_xy_)
        throw_with_line_number("Failed to create FFTW plans for FftwCrysFFTObliqueZ.");

    if (wisdom_file && wisdom_file[0] != '\0')
        fftw_export_wisdom_to_filename(wisdom_file);
}

void FftwCrysFFTObliqueZ::initFFTPlansZ(unsigned plan_flags)
{
    const int Nz = nx_logical_[2];
    const int howmany = nx_logical_[0] * nx_logical_[1];
    M_complex_z_ = nx_logical_[0] * nx_logical_[1] * (Nz / 2 + 1);

    fft_z_real_ = static_cast<double*>(fftw_malloc(sizeof(double) * M_logical_));
    fft_z_complex_ = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M_complex_z_));
    if (!fft_z_real_ || !fft_z_complex_)
        throw_with_line_number("Failed to allocate FFTW buffers for ObliqueZ FFT-DCT.");

    int n_z[1] = {Nz};
    int inembed_z[1] = {n_z[0]};
    int onembed_z[1] = {n_z[0] / 2 + 1};
    int istride_z = 1;
    int ostride_z = 1;
    int idist_z = n_z[0];
    int odist_z = n_z[0] / 2 + 1;

    plan_fft_z_forward_ = fftw_plan_many_dft_r2c(
        1, n_z, howmany,
        fft_z_real_, inembed_z, istride_z, idist_z,
        fft_z_complex_, onembed_z, ostride_z, odist_z,
        plan_flags);

    plan_fft_z_backward_ = fftw_plan_many_dft_c2r(
        1, n_z, howmany,
        fft_z_complex_, onembed_z, ostride_z, odist_z,
        fft_z_real_, inembed_z, istride_z, idist_z,
        plan_flags);

    const int N = nx_physical_[2];
    dct_fft_cos_.resize(N);
    dct_fft_sin_.resize(N);
    const double inv_2N = 1.0 / (2.0 * static_cast<double>(N));
    for (int k = 0; k < N; ++k)
    {
        const double theta = M_PI * static_cast<double>(k) * inv_2N;
        dct_fft_cos_[k] = std::cos(theta);
        dct_fft_sin_[k] = std::sin(theta);
    }

    calibrate_fft_dct_scale();
    if (!plan_fft_z_forward_ || !plan_fft_z_backward_)
        throw_with_line_number("Failed to create FFTW plans for ObliqueZ FFT-DCT.");
}

void FftwCrysFFTObliqueZ::calibrate_fft_dct_scale()
{
    const int N = nx_physical_[2];
    std::vector<double> x(N), tmp(N), fftw_dct(N);
    for (int i = 0; i < N; ++i)
        x[i] = std::sin(0.13 * (i + 1)) + 0.1 * (i + 1);

    fftw_plan p_fwd = fftw_plan_r2r_1d(N, x.data(), fftw_dct.data(), FFTW_REDFT10, FFTW_ESTIMATE);
    if (!p_fwd)
        return;
    fftw_execute(p_fwd);
    fftw_destroy_plan(p_fwd);

    std::vector<double> real_z(2 * N, 0.0);
    std::vector<fftw_complex> freq(N + 1);
    for (int i = 0; i < N; ++i)
    {
        real_z[i] = x[i];
        real_z[2 * N - 1 - i] = x[i];
    }

    fftw_plan p_fft = fftw_plan_dft_r2c_1d(2 * N, real_z.data(), freq.data(), FFTW_ESTIMATE);
    if (!p_fft)
        return;
    fftw_execute(p_fft);
    fftw_destroy_plan(p_fft);

    std::vector<double> fft_dct(N, 0.0);
    for (int k = 0; k < N; ++k)
    {
        const double a = freq[k][0];
        const double b = freq[k][1];
        fft_dct[k] = 0.5 * (a * dct_fft_cos_[k] + b * dct_fft_sin_[k]);
    }

    double num = 0.0, den = 0.0;
    for (int k = 0; k < N; ++k)
    {
        num += fftw_dct[k] * fft_dct[k];
        den += fft_dct[k] * fft_dct[k];
    }
    fft_dct_scale_fwd_ = (den > 0.0) ? (num / den) : 1.0;

    // DCT-III scaling
    std::vector<double> fftw_idct(N, 0.0);
    fftw_plan p_bwd = fftw_plan_r2r_1d(N, fftw_dct.data(), fftw_idct.data(), FFTW_REDFT01, FFTW_ESTIMATE);
    if (p_bwd)
    {
        fftw_execute(p_bwd);
        fftw_destroy_plan(p_bwd);
    }

    std::vector<fftw_complex> freq_inv(N + 1);
    for (int k = 0; k < N; ++k)
    {
        const double xk = fftw_dct[k] / fft_dct_scale_fwd_;
        const double c = dct_fft_cos_[k];
        const double s = dct_fft_sin_[k];
        freq_inv[k][0] = 2.0 * xk * c;
        freq_inv[k][1] = 2.0 * xk * s;
    }
    freq_inv[N][0] = 0.0;
    freq_inv[N][1] = 0.0;

    fftw_plan p_ifft = fftw_plan_dft_c2r_1d(2 * N, freq_inv.data(), real_z.data(), FFTW_ESTIMATE);
    if (p_ifft)
    {
        fftw_execute(p_ifft);
        fftw_destroy_plan(p_ifft);
    }

    const double inv_2N = 1.0 / (2.0 * static_cast<double>(N));
    num = 0.0;
    den = 0.0;
    for (int i = 0; i < N; ++i)
    {
        const double x_std = real_z[i] * inv_2N;
        num += fftw_idct[i] * x_std;
        den += x_std * x_std;
    }
    fft_dct_scale_bwd_ = (den > 0.0) ? (num / den) : 1.0;
}

//------------------------------------------------------------------------------
// Boltzmann factor cache
//------------------------------------------------------------------------------
void FftwCrysFFTObliqueZ::freeBoltzmann()
{
    cache_epoch_.fetch_add(1, std::memory_order_acq_rel);
}

FftwCrysFFTObliqueZ::ThreadState& FftwCrysFFTObliqueZ::get_thread_state() const
{
    struct ThreadLocalStates
    {
        std::unordered_map<const FftwCrysFFTObliqueZ*, ThreadState> states;
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

double* FftwCrysFFTObliqueZ::generateBoltzmann(double ds) const
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
void FftwCrysFFTObliqueZ::set_cell_para(const std::array<double, 6>& cell_para)
{
    if (cell_para == cell_para_)
        return;
    cell_para_ = cell_para;
    recip_metric_ = compute_recip_metric(cell_para_);
    freeBoltzmann();
}

void FftwCrysFFTObliqueZ::set_contour_step(double ds)
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

void FftwCrysFFTObliqueZ::diffusion(double* q_in, double* q_out)
{
    ThreadState& state = get_thread_state();
    if (!state.boltz_current)
        throw_with_line_number("FftwCrysFFTObliqueZ::set_contour_step must be called before diffusion().");

    const bool do_profile = (std::getenv("FTS_PROFILE_CRYSFFT_HEX_CPU_DETAIL") != nullptr);
    static std::atomic<bool> prof_once{false};
    const bool profile_this = do_profile && !prof_once.exchange(true);

    struct RealDeleter { void operator()(double* ptr) const { if (ptr) fftw_free(ptr); } };
    struct ComplexDeleter { void operator()(fftw_complex* ptr) const { if (ptr) fftw_free(ptr); } };
    struct ThreadBuffers
    {
        std::unique_ptr<double, RealDeleter> real_in;
        std::unique_ptr<double, RealDeleter> real_tmp;
        std::unique_ptr<fftw_complex, ComplexDeleter> complex;
        std::unique_ptr<double, RealDeleter> z_real;
        std::unique_ptr<fftw_complex, ComplexDeleter> z_complex;
        int size_real = 0;
        int size_complex = 0;
        int size_z_real = 0;
        int size_z_complex = 0;
    };
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
    if (use_fft_dct_ && (buffers.size_z_real != M_logical_ || buffers.size_z_complex != M_complex_z_))
    {
        buffers.z_real.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_logical_)));
        buffers.z_complex.reset(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M_complex_z_)));
        if (!buffers.z_real || !buffers.z_complex)
            throw_with_line_number("Failed to allocate thread-local FFTW Z-FFT buffers for FftwCrysFFTObliqueZ.");
        buffers.size_z_real = M_logical_;
        buffers.size_z_complex = M_complex_z_;
    }

    auto dct2_execute = [&](const double* in, double* out) {
        if (!use_fft_dct_)
        {
            fftw_execute_r2r(plan_dct_forward_z_, const_cast<double*>(in), out);
            return;
        }
        const int Nz = nx_logical_[2];
        const int Nz2 = nx_physical_[2];
        for (int ix = 0; ix < nx_logical_[0]; ++ix)
        {
            for (int iy = 0; iy < nx_logical_[1]; ++iy)
            {
                const int base_half = (ix * nx_logical_[1] + iy) * Nz2;
                const int base_full = (ix * nx_logical_[1] + iy) * Nz;
                std::memcpy(buffers.z_real.get() + base_full, in + base_half, sizeof(double) * Nz2);
                for (int iz = 0; iz < Nz2; ++iz)
                    buffers.z_real.get()[base_full + Nz - 1 - iz] = in[base_half + iz];
            }
        }

        fftw_execute_dft_r2c(plan_fft_z_forward_, buffers.z_real.get(), buffers.z_complex.get());

        const int stride = Nz / 2 + 1;
        for (int ix = 0; ix < nx_logical_[0]; ++ix)
        {
            for (int iy = 0; iy < nx_logical_[1]; ++iy)
            {
                const int base_half = (ix * nx_logical_[1] + iy) * Nz2;
                const int base_freq = (ix * nx_logical_[1] + iy) * stride;
                for (int k = 0; k < Nz2; ++k)
                {
                    const int idx = base_freq + k;
                    const double a = buffers.z_complex.get()[idx][0];
                    const double b = buffers.z_complex.get()[idx][1];
                    out[base_half + k] = 0.5 * (a * dct_fft_cos_[k] + b * dct_fft_sin_[k]) * fft_dct_scale_fwd_;
                }
            }
        }
    };

    auto dct3_execute = [&](const double* in, double* out) {
        if (!use_fft_dct_)
        {
            fftw_execute_r2r(plan_dct_backward_z_, const_cast<double*>(in), out);
            return;
        }
        const int Nz = nx_logical_[2];
        const int Nz2 = nx_physical_[2];
        const int stride = Nz / 2 + 1;
        for (int ix = 0; ix < nx_logical_[0]; ++ix)
        {
            for (int iy = 0; iy < nx_logical_[1]; ++iy)
            {
                const int base_half = (ix * nx_logical_[1] + iy) * Nz2;
                const int base_freq = (ix * nx_logical_[1] + iy) * stride;
                for (int k = 0; k < Nz2; ++k)
                {
                    const double xk = in[base_half + k] / fft_dct_scale_fwd_;
                    buffers.z_complex.get()[base_freq + k][0] = 2.0 * xk * dct_fft_cos_[k];
                    buffers.z_complex.get()[base_freq + k][1] = 2.0 * xk * dct_fft_sin_[k];
                }
                buffers.z_complex.get()[base_freq + Nz2][0] = 0.0;
                buffers.z_complex.get()[base_freq + Nz2][1] = 0.0;
            }
        }

        fftw_execute_dft_c2r(plan_fft_z_backward_, buffers.z_complex.get(), buffers.z_real.get());

        const double scale = fft_dct_scale_bwd_ / (2.0 * static_cast<double>(Nz2));
        for (int ix = 0; ix < nx_logical_[0]; ++ix)
        {
            for (int iy = 0; iy < nx_logical_[1]; ++iy)
            {
                const int base_half = (ix * nx_logical_[1] + iy) * Nz2;
                const int base_full = (ix * nx_logical_[1] + iy) * Nz;
                for (int iz = 0; iz < Nz2; ++iz)
                    out[base_half + iz] = buffers.z_real.get()[base_full + iz] * scale;
            }
        }
    };

    if (profile_this)
    {
        auto t0 = std::chrono::steady_clock::now();
        std::memcpy(buffers.real_in.get(), q_in, sizeof(double) * M_physical_);
        auto t1 = std::chrono::steady_clock::now();

        dct2_execute(buffers.real_in.get(), buffers.real_tmp.get());
        auto t2 = std::chrono::steady_clock::now();

        fftw_execute_dft_r2c(plan_fft_forward_xy_, buffers.real_tmp.get(), buffers.complex.get());
        auto t3 = std::chrono::steady_clock::now();

        for (int i = 0; i < M_complex_; ++i)
        {
            buffers.complex.get()[i][0] *= state.boltz_current[i];
            buffers.complex.get()[i][1] *= state.boltz_current[i];
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

    std::memcpy(buffers.real_in.get(), q_in, sizeof(double) * M_physical_);

    dct2_execute(buffers.real_in.get(), buffers.real_tmp.get());
    fftw_execute_dft_r2c(plan_fft_forward_xy_, buffers.real_tmp.get(), buffers.complex.get());

    for (int i = 0; i < M_complex_; ++i)
    {
        buffers.complex.get()[i][0] *= state.boltz_current[i];
        buffers.complex.get()[i][1] *= state.boltz_current[i];
    }

    fftw_execute_dft_c2r(plan_fft_backward_xy_, buffers.complex.get(), buffers.real_tmp.get());
    dct3_execute(buffers.real_tmp.get(), buffers.real_in.get());

    for (int i = 0; i < M_physical_; ++i)
        q_out[i] = buffers.real_in.get()[i] * norm_factor_;
}

void FftwCrysFFTObliqueZ::apply_multiplier(const double* q_in, double* q_out, const double* multiplier)
{
    struct RealDeleter { void operator()(double* ptr) const { if (ptr) fftw_free(ptr); } };
    struct ComplexDeleter { void operator()(fftw_complex* ptr) const { if (ptr) fftw_free(ptr); } };
    struct ThreadBuffers
    {
        std::unique_ptr<double, RealDeleter> real_in;
        std::unique_ptr<double, RealDeleter> real_tmp;
        std::unique_ptr<fftw_complex, ComplexDeleter> complex;
        std::unique_ptr<double, RealDeleter> z_real;
        std::unique_ptr<fftw_complex, ComplexDeleter> z_complex;
        int size_real = 0;
        int size_complex = 0;
        int size_z_real = 0;
        int size_z_complex = 0;
    };
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
    if (use_fft_dct_ && (buffers.size_z_real != M_logical_ || buffers.size_z_complex != M_complex_z_))
    {
        buffers.z_real.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_logical_)));
        buffers.z_complex.reset(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M_complex_z_)));
        if (!buffers.z_real || !buffers.z_complex)
            throw_with_line_number("Failed to allocate thread-local FFTW Z-FFT buffers for FftwCrysFFTObliqueZ.");
        buffers.size_z_real = M_logical_;
        buffers.size_z_complex = M_complex_z_;
    }

    auto dct2_execute = [&](const double* in, double* out) {
        if (!use_fft_dct_)
        {
            fftw_execute_r2r(plan_dct_forward_z_, const_cast<double*>(in), out);
            return;
        }
        const int Nz = nx_logical_[2];
        const int Nz2 = nx_physical_[2];
        for (int ix = 0; ix < nx_logical_[0]; ++ix)
        {
            for (int iy = 0; iy < nx_logical_[1]; ++iy)
            {
                const int base_half = (ix * nx_logical_[1] + iy) * Nz2;
                const int base_full = (ix * nx_logical_[1] + iy) * Nz;
                std::memcpy(buffers.z_real.get() + base_full, in + base_half, sizeof(double) * Nz2);
                for (int iz = 0; iz < Nz2; ++iz)
                    buffers.z_real.get()[base_full + Nz - 1 - iz] = in[base_half + iz];
            }
        }

        fftw_execute_dft_r2c(plan_fft_z_forward_, buffers.z_real.get(), buffers.z_complex.get());

        const int stride = Nz / 2 + 1;
        for (int ix = 0; ix < nx_logical_[0]; ++ix)
        {
            for (int iy = 0; iy < nx_logical_[1]; ++iy)
            {
                const int base_half = (ix * nx_logical_[1] + iy) * Nz2;
                const int base_freq = (ix * nx_logical_[1] + iy) * stride;
                for (int k = 0; k < Nz2; ++k)
                {
                    const int idx = base_freq + k;
                    const double a = buffers.z_complex.get()[idx][0];
                    const double b = buffers.z_complex.get()[idx][1];
                    out[base_half + k] = 0.5 * (a * dct_fft_cos_[k] + b * dct_fft_sin_[k]) * fft_dct_scale_fwd_;
                }
            }
        }
    };

    auto dct3_execute = [&](const double* in, double* out) {
        if (!use_fft_dct_)
        {
            fftw_execute_r2r(plan_dct_backward_z_, const_cast<double*>(in), out);
            return;
        }
        const int Nz = nx_logical_[2];
        const int Nz2 = nx_physical_[2];
        const int stride = Nz / 2 + 1;
        for (int ix = 0; ix < nx_logical_[0]; ++ix)
        {
            for (int iy = 0; iy < nx_logical_[1]; ++iy)
            {
                const int base_half = (ix * nx_logical_[1] + iy) * Nz2;
                const int base_freq = (ix * nx_logical_[1] + iy) * stride;
                for (int k = 0; k < Nz2; ++k)
                {
                    const double xk = in[base_half + k] / fft_dct_scale_fwd_;
                    buffers.z_complex.get()[base_freq + k][0] = 2.0 * xk * dct_fft_cos_[k];
                    buffers.z_complex.get()[base_freq + k][1] = 2.0 * xk * dct_fft_sin_[k];
                }
                buffers.z_complex.get()[base_freq + Nz2][0] = 0.0;
                buffers.z_complex.get()[base_freq + Nz2][1] = 0.0;
            }
        }

        fftw_execute_dft_c2r(plan_fft_z_backward_, buffers.z_complex.get(), buffers.z_real.get());

        const double scale = fft_dct_scale_bwd_ / (2.0 * static_cast<double>(Nz2));
        for (int ix = 0; ix < nx_logical_[0]; ++ix)
        {
            for (int iy = 0; iy < nx_logical_[1]; ++iy)
            {
                const int base_half = (ix * nx_logical_[1] + iy) * Nz2;
                const int base_full = (ix * nx_logical_[1] + iy) * Nz;
                for (int iz = 0; iz < Nz2; ++iz)
                    out[base_half + iz] = buffers.z_real.get()[base_full + iz] * scale;
            }
        }
    };

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
