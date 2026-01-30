/**
 * @file FftwCrysFFTHex.cpp
 * @brief CPU implementation of crystallographic FFT for hexagonal z-mirror symmetry.
 */

#include <cmath>
#include <cstring>
#include <memory>
#include <unordered_map>

#include "FftwCrysFFTHex.h"
#include "Exception.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------
std::array<double, 6> FftwCrysFFTHex::compute_recip_metric(const std::array<double, 6>& cell_para)
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
FftwCrysFFTHex::FftwCrysFFTHex(
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
            throw_with_line_number("FftwCrysFFTHex requires positive grid dimensions.");
    }
    if (nx_logical_[2] % 2 != 0)
        throw_with_line_number("FftwCrysFFTHex requires even Nz.");

    nx_physical_ = {nx_logical_[0], nx_logical_[1], nx_logical_[2] / 2};
    M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
    M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];
    norm_factor_ = 1.0 / static_cast<double>(M_logical_);

    initFFTPlans();
}

FftwCrysFFTHex::~FftwCrysFFTHex()
{
    freeBoltzmann();

    if (plan_dct_forward_z_) fftw_destroy_plan(plan_dct_forward_z_);
    if (plan_dct_backward_z_) fftw_destroy_plan(plan_dct_backward_z_);
    if (plan_fft_forward_xy_) fftw_destroy_plan(plan_fft_forward_xy_);
    if (plan_fft_backward_xy_) fftw_destroy_plan(plan_fft_backward_xy_);

    if (io_buffer_) fftw_free(io_buffer_);
    if (temp_buffer_) fftw_free(temp_buffer_);
    if (complex_buffer_) fftw_free(complex_buffer_);
}

//------------------------------------------------------------------------------
// FFTW plan initialization
//------------------------------------------------------------------------------
void FftwCrysFFTHex::initFFTPlans()
{
    io_buffer_ = static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_));
    temp_buffer_ = static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_));
    complex_buffer_ = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M_physical_));

    if (!io_buffer_ || !temp_buffer_ || !complex_buffer_)
        throw_with_line_number("Failed to allocate FFTW buffers for FftwCrysFFTHex.");

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
        kind_forward, FFTW_MEASURE);

    plan_dct_backward_z_ = fftw_plan_many_r2r(
        1, n_z, howmany_z,
        temp_buffer_, inembed_z, istride_z, idist_z,
        io_buffer_, onembed_z, ostride_z, odist_z,
        kind_backward, FFTW_MEASURE);

    // Complex FFT along X,Y for each Z slice (batched, strided)
    int n_xy[2] = {nx_logical_[0], nx_logical_[1]};
    int howmany_xy = nx_physical_[2];
    int inembed_xy[2] = {n_xy[0], n_xy[1]};
    int onembed_xy[2] = {n_xy[0], n_xy[1]};
    int istride_xy = nx_physical_[2];
    int ostride_xy = nx_physical_[2];
    int idist_xy = 1;
    int odist_xy = 1;

    plan_fft_forward_xy_ = fftw_plan_many_dft(
        2, n_xy, howmany_xy,
        complex_buffer_, inembed_xy, istride_xy, idist_xy,
        complex_buffer_, onembed_xy, ostride_xy, odist_xy,
        FFTW_FORWARD, FFTW_MEASURE);

    plan_fft_backward_xy_ = fftw_plan_many_dft(
        2, n_xy, howmany_xy,
        complex_buffer_, inembed_xy, istride_xy, idist_xy,
        complex_buffer_, onembed_xy, ostride_xy, odist_xy,
        FFTW_BACKWARD, FFTW_MEASURE);

    if (!plan_dct_forward_z_ || !plan_dct_backward_z_ || !plan_fft_forward_xy_ || !plan_fft_backward_xy_)
        throw_with_line_number("Failed to create FFTW plans for FftwCrysFFTHex.");
}

//------------------------------------------------------------------------------
// Boltzmann factor cache
//------------------------------------------------------------------------------
void FftwCrysFFTHex::freeBoltzmann()
{
    cache_epoch_.fetch_add(1, std::memory_order_acq_rel);
}

FftwCrysFFTHex::ThreadState& FftwCrysFFTHex::get_thread_state() const
{
    struct ThreadLocalStates
    {
        std::unordered_map<const FftwCrysFFTHex*, ThreadState> states;
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

double* FftwCrysFFTHex::generateBoltzmann(double ds) const
{
    double* boltz = new double[M_physical_];

    const double G11 = recip_metric_[0];
    const double G12 = recip_metric_[1];
    const double G13 = recip_metric_[2];
    const double G22 = recip_metric_[3];
    const double G23 = recip_metric_[4];
    const double G33 = recip_metric_[5];
    const double factor = 4.0 * M_PI * M_PI;

    int idx = 0;
    for (int ix = 0; ix < nx_logical_[0]; ++ix)
    {
        int m1 = (ix > nx_logical_[0] / 2) ? (ix - nx_logical_[0]) : ix;
        for (int iy = 0; iy < nx_logical_[1]; ++iy)
        {
            int m2 = (iy > nx_logical_[1] / 2) ? (iy - nx_logical_[1]) : iy;
            for (int iz = 0; iz < nx_physical_[2]; ++iz)
            {
                int m3 = iz;
                double gmm = G11 * m1 * m1 + G22 * m2 * m2 + G33 * m3 * m3
                           + 2.0 * (G12 * m1 * m2 + G13 * m1 * m3 + G23 * m2 * m3);
                double k2 = factor * gmm;
                boltz[idx++] = std::exp(-k2 * ds);
            }
        }
    }

    return boltz;
}

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------
void FftwCrysFFTHex::set_cell_para(const std::array<double, 6>& cell_para)
{
    if (cell_para == cell_para_)
        return;
    cell_para_ = cell_para;
    recip_metric_ = compute_recip_metric(cell_para_);
    freeBoltzmann();
}

void FftwCrysFFTHex::set_contour_step(double ds)
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

void FftwCrysFFTHex::diffusion(double* q_in, double* q_out)
{
    ThreadState& state = get_thread_state();
    if (!state.boltz_current)
        throw_with_line_number("FftwCrysFFTHex::set_contour_step must be called before diffusion().");

    struct RealDeleter { void operator()(double* ptr) const { if (ptr) fftw_free(ptr); } };
    struct ComplexDeleter { void operator()(fftw_complex* ptr) const { if (ptr) fftw_free(ptr); } };
    struct ThreadBuffers
    {
        std::unique_ptr<double, RealDeleter> real_in;
        std::unique_ptr<double, RealDeleter> real_tmp;
        std::unique_ptr<fftw_complex, ComplexDeleter> complex;
        int size = 0;
    };
    thread_local ThreadBuffers buffers;
    if (buffers.size != M_physical_)
    {
        buffers.real_in.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
        buffers.real_tmp.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
        buffers.complex.reset(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M_physical_)));
        if (!buffers.real_in || !buffers.real_tmp || !buffers.complex)
            throw_with_line_number("Failed to allocate thread-local FFTW buffers for FftwCrysFFTHex.");
        buffers.size = M_physical_;
    }

    std::memcpy(buffers.real_in.get(), q_in, sizeof(double) * M_physical_);

    fftw_execute_r2r(plan_dct_forward_z_, buffers.real_in.get(), buffers.real_tmp.get());

    for (int i = 0; i < M_physical_; ++i)
    {
        buffers.complex.get()[i][0] = buffers.real_tmp.get()[i];
        buffers.complex.get()[i][1] = 0.0;
    }

    fftw_execute_dft(plan_fft_forward_xy_, buffers.complex.get(), buffers.complex.get());

    for (int i = 0; i < M_physical_; ++i)
    {
        buffers.complex.get()[i][0] *= state.boltz_current[i];
        buffers.complex.get()[i][1] *= state.boltz_current[i];
    }

    fftw_execute_dft(plan_fft_backward_xy_, buffers.complex.get(), buffers.complex.get());

    for (int i = 0; i < M_physical_; ++i)
        buffers.real_tmp.get()[i] = buffers.complex.get()[i][0];

    fftw_execute_r2r(plan_dct_backward_z_, buffers.real_tmp.get(), buffers.real_in.get());

    for (int i = 0; i < M_physical_; ++i)
        q_out[i] = buffers.real_in.get()[i] * norm_factor_;
}

void FftwCrysFFTHex::apply_multiplier(const double* q_in, double* q_out, const double* multiplier)
{
    struct RealDeleter { void operator()(double* ptr) const { if (ptr) fftw_free(ptr); } };
    struct ComplexDeleter { void operator()(fftw_complex* ptr) const { if (ptr) fftw_free(ptr); } };
    struct ThreadBuffers
    {
        std::unique_ptr<double, RealDeleter> real_in;
        std::unique_ptr<double, RealDeleter> real_tmp;
        std::unique_ptr<fftw_complex, ComplexDeleter> complex;
        int size = 0;
    };
    thread_local ThreadBuffers buffers;
    if (buffers.size != M_physical_)
    {
        buffers.real_in.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
        buffers.real_tmp.reset(static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_)));
        buffers.complex.reset(static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * M_physical_)));
        if (!buffers.real_in || !buffers.real_tmp || !buffers.complex)
            throw_with_line_number("Failed to allocate thread-local FFTW buffers for FftwCrysFFTHex.");
        buffers.size = M_physical_;
    }

    std::memcpy(buffers.real_in.get(), q_in, sizeof(double) * M_physical_);

    fftw_execute_r2r(plan_dct_forward_z_, buffers.real_in.get(), buffers.real_tmp.get());

    for (int i = 0; i < M_physical_; ++i)
    {
        buffers.complex.get()[i][0] = buffers.real_tmp.get()[i];
        buffers.complex.get()[i][1] = 0.0;
    }

    fftw_execute_dft(plan_fft_forward_xy_, buffers.complex.get(), buffers.complex.get());

    for (int i = 0; i < M_physical_; ++i)
    {
        buffers.complex.get()[i][0] *= multiplier[i];
        buffers.complex.get()[i][1] *= multiplier[i];
    }

    fftw_execute_dft(plan_fft_backward_xy_, buffers.complex.get(), buffers.complex.get());

    for (int i = 0; i < M_physical_; ++i)
        buffers.real_tmp.get()[i] = buffers.complex.get()[i][0];

    fftw_execute_r2r(plan_dct_backward_z_, buffers.real_tmp.get(), buffers.real_in.get());

    for (int i = 0; i < M_physical_; ++i)
        q_out[i] = buffers.real_in.get()[i] * norm_factor_;
}
