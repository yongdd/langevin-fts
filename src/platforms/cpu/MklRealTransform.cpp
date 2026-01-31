/**
 * @file MklRealTransform.cpp
 * @brief MKL real-to-real DCT/DST implementation using TT interface.
 *
 * Uses Intel MKL Trigonometric Transform (TT) routines where available.
 * MKL TT is designed for PDE solving (Poisson/Laplace equations), not as
 * general-purpose DCT/DST. Only some transforms have direct MKL TT equivalents.
 *
 * MKL TT to FFTW mapping (empirically verified):
 *   - STAGGERED_COSINE backward × 2 = DCT-II (FFTW REDFT10)
 *   - STAGGERED_COSINE forward × N  = DCT-III (FFTW REDFT01)
 *   - STAGGERED2_COSINE forward × N = DCT-IV (FFTW REDFT11)
 *   - STAGGERED2_SINE forward × N   = DST-IV (FFTW RODFT11)
 *
 * No MKL TT equivalent (use direct O(N²) computation):
 *   - DCT-I (MKL COSINE_TRANSFORM uses different formula)
 *   - DST-I, DST-II, DST-III (MKL SINE transforms are PDE-specific)
 */

#include "MklRealTransform.h"

#include <cmath>
#include <cstring>
#include <numbers>
#include <string>
#include <vector>

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
// Direct O(N²) computation for transforms without MKL TT equivalent
//==============================================================================

/**
 * @brief Direct DCT-I computation: Y[k] = x[0] + (-1)^k x[N-1] + 2 * sum_{n=1}^{N-2} x[n] cos(π*n*k/(N-1))
 */
static void direct_dct1(double* data, int N)
{
    const double PI = std::numbers::pi;
    std::vector<double> result(N);

    for (int k = 0; k < N; ++k)
    {
        double sum = data[0] + (k % 2 == 0 ? data[N-1] : -data[N-1]);
        for (int n = 1; n < N - 1; ++n)
            sum += 2.0 * data[n] * std::cos(PI * n * k / (N - 1));
        result[k] = sum;
    }
    std::memcpy(data, result.data(), N * sizeof(double));
}

/**
 * @brief Direct DST-I computation: Y[k] = 2 * sum_{n=0}^{N-1} x[n] sin(π*(n+1)*(k+1)/(N+1))
 */
static void direct_dst1(double* data, int N)
{
    const double PI = std::numbers::pi;
    std::vector<double> result(N);

    for (int k = 0; k < N; ++k)
    {
        double sum = 0.0;
        for (int n = 0; n < N; ++n)
            sum += data[n] * std::sin(PI * (n + 1) * (k + 1) / (N + 1));
        result[k] = 2.0 * sum;
    }
    std::memcpy(data, result.data(), N * sizeof(double));
}

/**
 * @brief Direct DST-II computation: Y[k] = 2 * sum_{n=0}^{N-1} x[n] sin(π*(n+0.5)*(k+1)/N)
 */
static void direct_dst2(double* data, int N)
{
    const double PI = std::numbers::pi;
    std::vector<double> result(N);

    for (int k = 0; k < N; ++k)
    {
        double sum = 0.0;
        for (int n = 0; n < N; ++n)
            sum += data[n] * std::sin(PI * (n + 0.5) * (k + 1) / N);
        result[k] = 2.0 * sum;
    }
    std::memcpy(data, result.data(), N * sizeof(double));
}

/**
 * @brief Direct DST-III computation: Y[k] = (-1)^k x[N-1] + 2 * sum_{n=0}^{N-2} x[n] sin(π*(n+1)*(k+0.5)/N)
 */
static void direct_dst3(double* data, int N)
{
    const double PI = std::numbers::pi;
    std::vector<double> result(N);

    for (int k = 0; k < N; ++k)
    {
        double sum = (k % 2 == 0 ? 1.0 : -1.0) * data[N - 1];
        for (int n = 0; n < N - 1; ++n)
            sum += 2.0 * data[n] * std::sin(PI * (n + 1) * (k + 0.5) / N);
        result[k] = sum;
    }
    std::memcpy(data, result.data(), N * sizeof(double));
}

//==============================================================================
// MKL TT Wrapper Classes
//==============================================================================

/**
 * @brief Wrapper for MKL TT transforms that have direct FFTW equivalents.
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
    // Only DCT-2, DCT-3, DCT-4, DST-4 use MKL TT
    // Others use direct computation
    if (type_ == MKL_DCT_2 || type_ == MKL_DCT_3 ||
        type_ == MKL_DCT_4 || type_ == MKL_DST_4)
    {
        tt_handle_ = std::make_unique<MklTTHandle>(N_, type_);
    }
    // DCT-1, DST-1, DST-2, DST-3 use direct computation (no MKL TT equivalent)

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
    else
    {
        // Direct computation for transforms without MKL TT equivalent
        switch (type_)
        {
            case MKL_DCT_1:
                direct_dct1(data, N_);
                break;
            case MKL_DST_1:
                direct_dst1(data, N_);
                break;
            case MKL_DST_2:
                direct_dst2(data, N_);
                break;
            case MKL_DST_3:
                direct_dst3(data, N_);
                break;
            default:
                throw_with_line_number("Unsupported transform type.");
        }
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
