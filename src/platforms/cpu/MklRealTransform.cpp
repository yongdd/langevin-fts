/**
 * @file MklRealTransform.cpp
 * @brief MKL real-to-real DCT/DST implementation using TT interface and FFT.
 *
 * Uses Intel MKL Trigonometric Transform (TT) routines where available,
 * and FFT-based O(N log N) algorithms for other transform types.
 *
 * MKL TT to FFTW mapping (empirically verified):
 *   - STAGGERED_COSINE backward × 2 = DCT-II (FFTW REDFT10)
 *   - STAGGERED_COSINE forward × N  = DCT-III (FFTW REDFT01)
 *   - STAGGERED2_COSINE forward × N = DCT-IV (FFTW REDFT11)
 *   - STAGGERED2_SINE forward × N   = DST-IV (FFTW RODFT11)
 *
 * FFT-based O(N log N) implementations:
 *   - DCT-I: via 2(N-1) point FFT symmetric extension
 *   - DST-I: via 2(N+1) point FFT antisymmetric extension
 *   - DST-II, DST-III: cuHelmholtz algorithm (Makhoul 1980)
 */

#include "MklRealTransform.h"

#include <cmath>
#include <complex>
#include <cstring>
#include <numbers>
#include <string>
#include <vector>

#include "mkl_dfti.h"
#include "mkl_trig_transforms.h"

/**
 * @brief Get normalization factor for round-trip transform (FFTW convention).
 */
static double get_norm_factor(MklTransformType type, int N)
{
    switch (type)
    {
        case MKL_DCT_1: return 2.0 * (N - 1);
        case MKL_DST_1: return 2.0 * (N + 1);
        case MKL_DCT_2:
        case MKL_DCT_3:
        case MKL_DCT_4:
        case MKL_DST_2:
        case MKL_DST_3:
        case MKL_DST_4:
            return 2.0 * N;
        default:
            return 1.0;
    }
}

//==============================================================================
// FFT-based O(N log N) implementations
//==============================================================================

/**
 * @brief FFT handle for DCT-I/DST-I/DST-II/DST-III transforms
 */
class MklFFTHandle
{
public:
    MklFFTHandle(int fft_size)
        : fft_size_(fft_size)
    {
        MKL_LONG status;
        status = DftiCreateDescriptor(&hand_forward_, DFTI_DOUBLE, DFTI_REAL, 1, fft_size);
        status = DftiSetValue(hand_forward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        status = DftiSetValue(hand_forward_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        status = DftiCommitDescriptor(hand_forward_);

        status = DftiCreateDescriptor(&hand_backward_, DFTI_DOUBLE, DFTI_REAL, 1, fft_size);
        status = DftiSetValue(hand_backward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        status = DftiSetValue(hand_backward_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        status = DftiCommitDescriptor(hand_backward_);

        if (status != 0)
            throw_with_line_number("MKL FFT handle creation failed");
    }

    ~MklFFTHandle()
    {
        if (hand_forward_ != nullptr)
            DftiFreeDescriptor(&hand_forward_);
        if (hand_backward_ != nullptr)
            DftiFreeDescriptor(&hand_backward_);
    }

    void forward(double* in, std::complex<double>* out)
    {
        DftiComputeForward(hand_forward_, in, out);
    }

    void backward(std::complex<double>* in, double* out)
    {
        DftiComputeBackward(hand_backward_, in, out);
    }

    int size() const { return fft_size_; }

private:
    int fft_size_;
    DFTI_DESCRIPTOR_HANDLE hand_forward_{nullptr};
    DFTI_DESCRIPTOR_HANDLE hand_backward_{nullptr};
};

/**
 * @brief DCT-I via FFT: O(N log N) using 2(N-1) point symmetric extension
 */
static void fft_dct1(double* data, int N, MklFFTHandle& fft)
{
    int M = 2 * (N - 1);

    thread_local std::vector<double> extended;
    thread_local std::vector<std::complex<double>> fft_out;

    if (static_cast<int>(extended.size()) < M)
    {
        extended.resize(M);
        fft_out.resize(M / 2 + 1);
    }

    // Create symmetric extension: [x0, x1, ..., x_{N-1}, x_{N-2}, ..., x_1]
    for (int i = 0; i < N; ++i)
        extended[i] = data[i];
    for (int i = 1; i < N - 1; ++i)
        extended[N - 1 + i] = data[N - 1 - i];

    fft.forward(extended.data(), fft_out.data());

    // Extract DCT-I coefficients (real parts of first N FFT coefficients)
    for (int k = 0; k < N; ++k)
        data[k] = fft_out[k].real();
}

/**
 * @brief DST-I via FFT: O(N log N) using 2(N+1) point antisymmetric extension
 */
static void fft_dst1(double* data, int N, MklFFTHandle& fft)
{
    int M = 2 * (N + 1);

    thread_local std::vector<double> extended;
    thread_local std::vector<std::complex<double>> fft_out;

    if (static_cast<int>(extended.size()) < M)
    {
        extended.resize(M);
        fft_out.resize(M / 2 + 1);
    }

    // Create antisymmetric extension: [0, x0, x1, ..., x_{N-1}, 0, -x_{N-1}, ..., -x_0]
    extended[0] = 0.0;
    for (int i = 0; i < N; ++i)
        extended[i + 1] = data[i];
    extended[N + 1] = 0.0;
    for (int i = 0; i < N; ++i)
        extended[N + 2 + i] = -data[N - 1 - i];

    fft.forward(extended.data(), fft_out.data());

    // Extract DST-I coefficients: Y[k] = -Im(X[k+1])
    for (int k = 0; k < N; ++k)
        data[k] = -fft_out[k + 1].imag();
}

/**
 * @brief DST-II via FFT: O(N log N) using cuHelmholtz algorithm
 *
 * DST-II formula (FFTW RODFT10):
 * Y[k] = 2 * sum_{n=0}^{N-1} x[n] sin(π*(2n+1)*(k+1)/(2N))
 */
static void fft_dst2(double* data, int N, MklFFTHandle& fft,
                     const std::vector<double>& cos_tbl,
                     const std::vector<double>& sin_tbl)
{
    int complex_size = N / 2 + 1;

    thread_local std::vector<std::complex<double>> fft_in;
    thread_local std::vector<double> fft_out;
    thread_local std::vector<double> result;

    if (static_cast<int>(fft_in.size()) < complex_size)
    {
        fft_in.resize(complex_size);
        fft_out.resize(N);
        result.resize(N);
    }

    // DST-II preprocessing: create complex for IFFT
    fft_in[0] = std::complex<double>(data[0], 0.0);
    for (int k = 1; k < complex_size - 1; ++k)
    {
        double x_2k = data[2 * k];
        double x_2k_1 = data[2 * k - 1];
        fft_in[k] = std::complex<double>((x_2k - x_2k_1) / 2.0, -((x_2k + x_2k_1) / 2.0));
    }
    if (N % 2 == 0)
    {
        fft_in[N / 2] = std::complex<double>(-data[N - 1], 0.0);
    }
    else
    {
        int k = N / 2;
        double x_2k = data[2 * k];
        double x_2k_1 = data[2 * k - 1];
        fft_in[k] = std::complex<double>((x_2k - x_2k_1) / 2.0, -((x_2k + x_2k_1) / 2.0));
    }

    // IFFT (c2r)
    fft.backward(fft_in.data(), fft_out.data());

    // DST-II postprocessing: apply twiddle factors
    // FFTW has factor of 2 in formula, so no 0.5 scaling here
    for (int k = 1; k <= N / 2; ++k)
    {
        double Ta = fft_out[k] + fft_out[N - k];
        double Tb = fft_out[k] - fft_out[N - k];

        double result_k = Ta * sin_tbl[k] + Tb * cos_tbl[k];
        double result_nk = Ta * cos_tbl[k] - Tb * sin_tbl[k];

        result[k - 1] = result_k;
        if (k < N - k)
            result[N - k - 1] = result_nk;
    }
    result[N - 1] = fft_out[0] * 2.0;  // DC also needs 2x for FFTW convention

    std::memcpy(data, result.data(), N * sizeof(double));
}

/**
 * @brief DST-III via FFT: O(N log N) using cuHelmholtz algorithm
 *
 * DST-III formula (FFTW RODFT01):
 * Y[k] = (-1)^k * x[N-1] + 2 * sum_{n=0}^{N-2} x[n] sin(π*(n+1)*(2k+1)/(2N))
 */
static void fft_dst3(double* data, int N, MklFFTHandle& fft,
                     const std::vector<double>& cos_tbl,
                     const std::vector<double>& sin_tbl)
{
    int complex_size = N / 2 + 1;

    thread_local std::vector<double> fft_in;
    thread_local std::vector<std::complex<double>> fft_out;
    thread_local std::vector<double> result;

    if (static_cast<int>(fft_in.size()) < N)
    {
        fft_in.resize(N);
        fft_out.resize(complex_size);
        result.resize(N);
    }

    // Save last element for the (-1)^k term
    double x_last = data[N - 1];

    // DST-III preprocessing: apply twiddle factors (only first N-1 elements)
    fft_in[0] = 0.0;
    for (int k = 1; k <= N / 2; ++k)
    {
        double val_k = data[k - 1];
        // For val_nk, ensure we don't use data[N-1] in the FFT part
        double val_nk = (k == N - k) ? val_k : ((N - k - 1 >= 0 && N - k - 1 < N - 1) ? data[N - k - 1] : 0.0);

        double Ta = val_k + val_nk;
        double Tb = val_k - val_nk;

        fft_in[k] = Ta * cos_tbl[k] + Tb * sin_tbl[k];
        if (k < N - k)
            fft_in[N - k] = Ta * sin_tbl[k] - Tb * cos_tbl[k];
    }

    // FFT (r2c)
    fft.forward(fft_in.data(), fft_out.data());

    // DST-III postprocessing (no scaling - FFTW is unnormalized)
    result[0] = fft_out[0].real();
    for (int k = 1; k <= N / 2; ++k)
    {
        double re = fft_out[k].real();
        double im = -fft_out[k].imag();

        if (2 * k - 1 < N)
            result[2 * k - 1] = im - re;
        if (2 * k < N)
            result[2 * k] = re + im;
    }

    // Add the (-1)^k * X[N-1] term for FFTW RODFT01 formula
    for (int k = 0; k < N; ++k)
    {
        result[k] += ((k % 2 == 0) ? 1.0 : -1.0) * x_last;
    }

    std::memcpy(data, result.data(), N * sizeof(double));
}

//==============================================================================
// MKL TT Wrapper Classes
//==============================================================================

/**
 * @brief Wrapper for MKL TT transforms.
 *
 * Supported transforms:
 *   - DCT-II: STAGGERED_COSINE backward × 2
 *   - DCT-III: STAGGERED_COSINE forward × N
 *   - DCT-IV: STAGGERED2_COSINE forward × N
 *   - DST-IV: STAGGERED2_SINE forward × N
 */
class MklTTHandle
{
public:
    MklTTHandle(int N, MklTransformType type)
        : N_(N), type_(type)
    {
        // Determine MKL TT type and direction
        switch (type)
        {
            case MKL_DCT_2:
                tt_type_ = MKL_STAGGERED_COSINE_TRANSFORM;
                use_backward_ = true;
                scale_ = 2.0;
                break;
            case MKL_DCT_3:
                tt_type_ = MKL_STAGGERED_COSINE_TRANSFORM;
                use_backward_ = false;
                scale_ = static_cast<double>(N);
                break;
            case MKL_DCT_4:
                tt_type_ = MKL_STAGGERED2_COSINE_TRANSFORM;
                use_backward_ = false;
                scale_ = static_cast<double>(N);
                break;
            case MKL_DST_4:
                tt_type_ = MKL_STAGGERED2_SINE_TRANSFORM;
                use_backward_ = false;
                scale_ = static_cast<double>(N);
                break;
            default:
                throw_with_line_number("Transform type not supported by MKL TT: " +
                                       std::string(getTransformName(type)));
        }

        // Allocate work buffer
        work_.resize(N + 16);

        // dpar size requirement from MKL docs
        dpar_.resize(10 * N + 100);

        MKL_INT n = N_;
        MKL_INT stat = 0;

        // Initialize TT parameters
        d_init_trig_transform(&n, &tt_type_, ipar_, dpar_.data(), &stat);
        if (stat != 0)
            throw_with_line_number("d_init_trig_transform failed, stat: " + std::to_string(stat));

        // Commit the transform
        d_commit_trig_transform(work_.data(), &handle_, ipar_, dpar_.data(), &stat);
        if (stat != 0)
            throw_with_line_number("d_commit_trig_transform failed, stat: " + std::to_string(stat));
    }

    ~MklTTHandle()
    {
        if (handle_ != nullptr)
        {
            MKL_INT stat = 0;
            free_trig_transform(&handle_, ipar_, &stat);
        }
    }

    void execute(double* data)
    {
        // Copy data to work buffer (MKL TT works in-place)
        std::memcpy(work_.data(), data, N_ * sizeof(double));

        MKL_INT stat = 0;

        if (use_backward_)
            d_backward_trig_transform(work_.data(), &handle_, ipar_, dpar_.data(), &stat);
        else
            d_forward_trig_transform(work_.data(), &handle_, ipar_, dpar_.data(), &stat);

        if (stat != 0)
            throw_with_line_number("MKL TT transform failed, stat: " + std::to_string(stat));

        // Apply scaling to match FFTW output
        for (int i = 0; i < N_; ++i)
            data[i] = work_[i] * scale_;
    }

private:
    int N_;
    MklTransformType type_;
    MKL_INT tt_type_;
    bool use_backward_;
    double scale_;
    DFTI_DESCRIPTOR_HANDLE handle_{nullptr};
    MKL_INT ipar_[128]{};
    std::vector<double> dpar_;
    std::vector<double> work_;
};

//==============================================================================
// MklRealTransform1D
//==============================================================================

MklRealTransform1D::MklRealTransform1D(int N, MklTransformType type)
    : N_(N), type_(type)
{
    if (N < 2)
        throw_with_line_number("Transform size must be >= 2");
    init();
}

MklRealTransform1D::~MklRealTransform1D()
{
    // Handles are freed by unique_ptr automatically
}

void MklRealTransform1D::init()
{
    const double PI = std::numbers::pi;

    // DCT-2, DCT-3, DCT-4, DST-4 use MKL TT
    if (type_ == MKL_DCT_2 || type_ == MKL_DCT_3 ||
        type_ == MKL_DCT_4 || type_ == MKL_DST_4)
    {
        tt_handle_ = std::make_unique<MklTTHandle>(N_, type_);
    }
    // DCT-1 uses 2(N-1) point FFT
    else if (type_ == MKL_DCT_1)
    {
        fft_handle_ = std::make_unique<MklFFTHandle>(2 * (N_ - 1));
    }
    // DST-1 uses 2(N+1) point FFT
    else if (type_ == MKL_DST_1)
    {
        fft_handle_ = std::make_unique<MklFFTHandle>(2 * (N_ + 1));
    }
    // DST-2, DST-3 use N point FFT with twiddle factors (cuHelmholtz algorithm)
    else if (type_ == MKL_DST_2 || type_ == MKL_DST_3)
    {
        fft_handle_ = std::make_unique<MklFFTHandle>(N_);

        // Precompute twiddle factors
        int num_twiddles = N_ / 2 + 1;
        cos_tbl_.resize(num_twiddles);
        sin_tbl_.resize(num_twiddles);
        for (int k = 0; k <= N_ / 2; ++k)
        {
            cos_tbl_[k] = std::cos(k * PI / (2.0 * N_));
            sin_tbl_[k] = std::sin(k * PI / (2.0 * N_));
        }
    }

    initialized_ = true;
}

void MklRealTransform1D::execute(double* data)
{
    if (!initialized_)
        throw_with_line_number("MklRealTransform1D not initialized");

    if (tt_handle_)
    {
        tt_handle_->execute(data);
    }
    else if (fft_handle_)
    {
        switch (type_)
        {
            case MKL_DCT_1:
                fft_dct1(data, N_, *fft_handle_);
                break;
            case MKL_DST_1:
                fft_dst1(data, N_, *fft_handle_);
                break;
            case MKL_DST_2:
                fft_dst2(data, N_, *fft_handle_, cos_tbl_, sin_tbl_);
                break;
            case MKL_DST_3:
                fft_dst3(data, N_, *fft_handle_, cos_tbl_, sin_tbl_);
                break;
            default:
                throw_with_line_number("Unsupported transform type for FFT handle.");
        }
    }
    else
    {
        throw_with_line_number("No transform handle available for type: " +
                               std::string(getTransformName(type_)));
    }
}

double MklRealTransform1D::get_normalization() const
{
    return get_norm_factor(type_, N_);
}

//==============================================================================
// MklRealTransform2D
//==============================================================================

MklRealTransform2D::MklRealTransform2D(int Nx, int Ny, MklTransformType type)
    : MklRealTransform2D(Nx, Ny, type, type)
{
}

MklRealTransform2D::MklRealTransform2D(int Nx, int Ny, MklTransformType type_x, MklTransformType type_y)
    : nx_{Nx, Ny}, type_x_(type_x), type_y_(type_y)
{
    if (Nx < 2 || Ny < 2)
        throw_with_line_number("Transform sizes must be >= 2");

    init();
}

MklRealTransform2D::~MklRealTransform2D()
{
}

void MklRealTransform2D::init()
{
    transform_x_ = std::make_unique<MklRealTransform1D>(nx_[0], type_x_);
    transform_y_ = std::make_unique<MklRealTransform1D>(nx_[1], type_y_);
    initialized_ = true;
}

void MklRealTransform2D::getStrides(int dim, int& stride, int& num_transforms) const
{
    stride = 1;
    for (int d = dim + 1; d < 2; ++d)
        stride *= nx_[d];

    num_transforms = 1;
    for (int d = 0; d < dim; ++d)
        num_transforms *= nx_[d];
}

void MklRealTransform2D::apply1D(double* data, int dim, MklRealTransform1D* transform) const
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    thread_local std::vector<double> slice;
    if (static_cast<int>(slice.size()) < n)
        slice.resize(n);

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            // Extract slice
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            // Transform
            transform->execute(slice.data());

            // Write back
            for (int j = 0; j < n; ++j)
                data[offset + j * stride] = slice[j];
        }
    }
}

void MklRealTransform2D::execute(double* data)
{
    if (!initialized_)
        throw_with_line_number("MklRealTransform2D not initialized");

    // Transform along y (fastest varying) first, then x
    apply1D(data, 1, transform_y_.get());
    apply1D(data, 0, transform_x_.get());
}

double MklRealTransform2D::get_normalization() const
{
    return get_norm_factor(type_x_, nx_[0]) * get_norm_factor(type_y_, nx_[1]);
}

//==============================================================================
// MklRealTransform3D
//==============================================================================

MklRealTransform3D::MklRealTransform3D(int Nx, int Ny, int Nz, MklTransformType type)
    : MklRealTransform3D(Nx, Ny, Nz, type, type, type)
{
}

MklRealTransform3D::MklRealTransform3D(int Nx, int Ny, int Nz,
                                       MklTransformType type_x,
                                       MklTransformType type_y,
                                       MklTransformType type_z)
    : nx_{Nx, Ny, Nz}, type_x_(type_x), type_y_(type_y), type_z_(type_z)
{
    if (Nx < 2 || Ny < 2 || Nz < 2)
        throw_with_line_number("Transform sizes must be >= 2");

    init();
}

MklRealTransform3D::~MklRealTransform3D()
{
}

void MklRealTransform3D::init()
{
    transform_x_ = std::make_unique<MklRealTransform1D>(nx_[0], type_x_);
    transform_y_ = std::make_unique<MklRealTransform1D>(nx_[1], type_y_);
    transform_z_ = std::make_unique<MklRealTransform1D>(nx_[2], type_z_);
    initialized_ = true;
}

void MklRealTransform3D::getStrides(int dim, int& stride, int& num_transforms) const
{
    stride = 1;
    for (int d = dim + 1; d < 3; ++d)
        stride *= nx_[d];

    num_transforms = 1;
    for (int d = 0; d < dim; ++d)
        num_transforms *= nx_[d];
}

void MklRealTransform3D::apply1D(double* data, int dim, MklRealTransform1D* transform) const
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    thread_local std::vector<double> slice;
    if (static_cast<int>(slice.size()) < n)
        slice.resize(n);

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            // Extract slice
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            // Transform
            transform->execute(slice.data());

            // Write back
            for (int j = 0; j < n; ++j)
                data[offset + j * stride] = slice[j];
        }
    }
}

void MklRealTransform3D::execute(double* data)
{
    if (!initialized_)
        throw_with_line_number("MklRealTransform3D not initialized");

    // Transform along z (fastest varying) first, then y, then x
    apply1D(data, 2, transform_z_.get());
    apply1D(data, 1, transform_y_.get());
    apply1D(data, 0, transform_x_.get());
}

double MklRealTransform3D::get_normalization() const
{
    return get_norm_factor(type_x_, nx_[0]) *
           get_norm_factor(type_y_, nx_[1]) *
           get_norm_factor(type_z_, nx_[2]);
}
