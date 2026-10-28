/**
 * @file MklFFT.cpp
 * @brief Intel MKL FFT implementation with O(N log N) DCT/DST via FFT.
 *
 * Provides Fast Fourier Transform functionality using Intel MKL DFTI for
 * all boundary conditions with O(N log N) complexity.
 *
 * **Transform Types by Boundary Condition:**
 *
 * - PERIODIC: MKL DFTI r2c/c2r transforms
 * - REFLECTING: DCT-II/III via FFT (O(N log N))
 * - ABSORBING: DST-II/III via FFT (O(N log N))
 *
 * **DCT/DST via FFT Algorithm:**
 *
 * Based on cuHelmholtz library:
 * M. Ren et al., "Discrete Sine and Cosine Transform and Helmholtz Equation
 * Solver on GPU," IEEE ISPA 2020.
 * DOI: 10.1109/ISPA-BDCloud-SocialCom-SustainCom51426.2020.00034
 *
 * DCT-II: preprocess → IFFT → postprocess (twiddle multiply)
 * DCT-III: preprocess (twiddle multiply) → FFT → postprocess
 * DST-II/III: Input/output transformations + DCT-II/III
 *
 * **Normalization:**
 *
 * Backward transforms include 1/N scaling so forward(backward(x)) = x.
 *
 * @see MklFFT.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <numbers>
#include <stdexcept>
#include "MklFFT.h"
#include "MklDctLines.h"
#include "MklDftiLayout.h"
#include "MklStatusCheck.h"

//------------------------------------------------------------------------------
// Constructor (periodic BC, backward compatible)
//------------------------------------------------------------------------------
template <typename T, int DIM>
MklFFT<T, DIM>::MklFFT(std::array<int, DIM> nx)
    : nx_(nx), is_all_periodic_(true)
{
    // Initialize all BCs to periodic
    for (int d = 0; d < DIM; ++d)
        bc_[d] = BoundaryCondition::PERIODIC;

    // Compute grid sizes
    total_grid_ = 1;
    for (int d = 0; d < DIM; ++d)
        total_grid_ *= nx_[d];

    // Compute complex grid size for r2c FFT
    if constexpr (std::is_same<T, double>::value)
    {
        if (DIM == 3)
            total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
        else if (DIM == 2)
            total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
        else
            total_complex_grid_ = nx_[0] / 2 + 1;
    }
    else
    {
        total_complex_grid_ = total_grid_;
    }

    try
    {
        initPeriodicFFT();
    }
    catch (...)
    {
        // The destructor does not run for a partly constructed object
        release();
        throw;
    }
}

//------------------------------------------------------------------------------
// Constructor (with BC specification)
//------------------------------------------------------------------------------
template <typename T, int DIM>
MklFFT<T, DIM>::MklFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc)
    : nx_(nx), bc_(bc)
{
    try
    {
        // Compute total grid size
        total_grid_ = 1;
        for (int d = 0; d < DIM; ++d)
            total_grid_ *= nx_[d];

        // Check if all periodic
        is_all_periodic_ = true;
        for (int d = 0; d < DIM; ++d)
        {
            if (bc_[d] != BoundaryCondition::PERIODIC)
            {
                is_all_periodic_ = false;
                break;
            }
        }

        // For simplicity, require either all periodic or all non-periodic
        if (!is_all_periodic_)
        {
            for (int d = 0; d < DIM; ++d)
            {
                if (bc_[d] == BoundaryCondition::PERIODIC)
                {
                    throw_with_line_number("Mixed periodic and non-periodic BCs not yet supported. "
                                          "Use all periodic or all non-periodic (reflecting/absorbing).");
                }
            }
        }

        // Compute complex grid size
        if (is_all_periodic_)
        {
            if constexpr (std::is_same<T, double>::value)
            {
                if (DIM == 3)
                    total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
                else if (DIM == 2)
                    total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
                else
                    total_complex_grid_ = nx_[0] / 2 + 1;
            }
            else
            {
                total_complex_grid_ = total_grid_;
            }
            initPeriodicFFT();
        }
        else
        {
            // DCT/DST: real-to-real, same size
            total_complex_grid_ = total_grid_;
            initNonPeriodicFFT();
        }
    }
    catch (std::exception& exc)
    {
        // The destructor does not run for a partly constructed object
        release();
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template <typename T, int DIM>
MklFFT<T, DIM>::~MklFFT()
{
    release();
}

template <typename T, int DIM>
void MklFFT<T, DIM>::release()
{
    if (hand_forward_ != nullptr)
        DftiFreeDescriptor(&hand_forward_);
    if (hand_backward_ != nullptr)
        DftiFreeDescriptor(&hand_backward_);
    hand_forward_ = nullptr;
    hand_backward_ = nullptr;

    // Free 1D FFT plans for DCT/DST
    for (auto& handle : dct_fft_forward_)
        if (handle != nullptr)
            DftiFreeDescriptor(&handle);
    for (auto& handle : dct_fft_backward_)
        if (handle != nullptr)
            DftiFreeDescriptor(&handle);
    dct_fft_forward_.clear();
    dct_fft_backward_.clear();
}

//------------------------------------------------------------------------------
// Initialize periodic FFT (MKL DFTI)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::initPeriodicFFT()
{
    try
    {
        MKL_LONG NX[DIM];
        for (int i = 0; i < DIM; i++)
            NX[i] = nx_[i];

        // Strides describe data layout in real and conjugate-even domain
        MKL_LONG rs[DIM + 1] = {0};
        MKL_LONG cs[DIM + 1] = {0};
        rs[DIM] = 1;
        cs[DIM] = 1;

        // Determine if we're using real or complex data type
        constexpr bool is_real = std::is_same<T, double>::value;
        constexpr DFTI_CONFIG_VALUE dtype = is_real ? DFTI_REAL : DFTI_COMPLEX;

        // Calculate strides for real domain
        for (int i = DIM - 1; i > 0; i--)
            rs[i] = rs[i + 1] * nx_[i];

        // Calculate strides for complex domain
        if constexpr (std::is_same<T, double>::value)
        {
            for (int i = DIM - 1; i > 0; i--)
                cs[i] = cs[i + 1] * (i == DIM - 1 ? nx_[i] / 2 + 1 : nx_[i]);
        }
        else
        {
            for (int i = DIM - 1; i > 0; i--)
                cs[i] = cs[i + 1] * nx_[i];
        }

        if (DIM == 1)
        {
            mkl_dfti_check(DftiCreateDescriptor(&hand_forward_, DFTI_DOUBLE, dtype, 1, NX[0]), "DftiCreateDescriptor (forward)");
            mkl_dfti_check(DftiCreateDescriptor(&hand_backward_, DFTI_DOUBLE, dtype, 1, NX[0]), "DftiCreateDescriptor (backward)");
        }
        else
        {
            mkl_dfti_check(DftiCreateDescriptor(&hand_forward_, DFTI_DOUBLE, dtype, DIM, NX), "DftiCreateDescriptor (forward)");
            mkl_dfti_check(DftiSetValue(hand_forward_, mkl_dfti::forward_input_strides, rs), "DftiSetValue (forward input strides)");
            mkl_dfti_check(DftiSetValue(hand_forward_, mkl_dfti::forward_output_strides, cs), "DftiSetValue (forward output strides)");

            mkl_dfti_check(DftiCreateDescriptor(&hand_backward_, DFTI_DOUBLE, dtype, DIM, NX), "DftiCreateDescriptor (backward)");
            mkl_dfti_check(DftiSetValue(hand_backward_, mkl_dfti::backward_input_strides, cs), "DftiSetValue (backward input strides)");
            mkl_dfti_check(DftiSetValue(hand_backward_, mkl_dfti::backward_output_strides, rs), "DftiSetValue (backward output strides)");
        }

        mkl_dfti_check(DftiSetValue(hand_forward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, forward)");
        mkl_dfti_check(DftiSetValue(hand_backward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, backward)");

        if (is_real)
        {
            mkl_dfti_check(DftiSetValue(hand_forward_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, forward)");
            mkl_dfti_check(DftiSetValue(hand_backward_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, backward)");
        }

        mkl_dfti_check(DftiCommitDescriptor(hand_forward_), "DftiCommitDescriptor (forward)");
        mkl_dfti_check(DftiSetValue(hand_backward_, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(total_grid_)), "DftiSetValue(DFTI_BACKWARD_SCALE, backward)");
        mkl_dfti_check(DftiCommitDescriptor(hand_backward_), "DftiCommitDescriptor (backward)");
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Initialize non-periodic FFT (DCT/DST via FFT)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::initNonPeriodicFFT()
{
    try
    {
        dct_fft_forward_.resize(DIM, nullptr);
        dct_fft_backward_.resize(DIM, nullptr);

        // Create 1D FFT plans for each dimension
        // DCT-II uses IFFT (backward), DCT-III uses FFT (forward)
        for (int dim = 0; dim < DIM; ++dim)
        {
            int n = nx_[dim];

            // Forward FFT for DCT-III (D2Z equivalent)
            mkl_dfti_check(DftiCreateDescriptor(&dct_fft_forward_[dim], DFTI_DOUBLE, DFTI_REAL, 1, n), "DftiCreateDescriptor (DCT/DST forward)");
            mkl_dfti_check(DftiSetValue(dct_fft_forward_[dim], DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, DCT/DST forward)");
            mkl_dfti_check(DftiSetValue(dct_fft_forward_[dim], DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, DCT/DST forward)");
            mkl_dfti_check(DftiCommitDescriptor(dct_fft_forward_[dim]), "DftiCommitDescriptor (DCT/DST forward)");

            // Backward FFT for DCT-II (Z2D equivalent)
            // NOTE: No scaling - cuHelmholtz algorithm expects unscaled IFFT
            mkl_dfti_check(DftiCreateDescriptor(&dct_fft_backward_[dim], DFTI_DOUBLE, DFTI_REAL, 1, n), "DftiCreateDescriptor (DCT/DST backward)");
            mkl_dfti_check(DftiSetValue(dct_fft_backward_[dim], DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, DCT/DST backward)");
            mkl_dfti_check(DftiSetValue(dct_fft_backward_[dim], DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, DCT/DST backward)");
            mkl_dfti_check(DftiCommitDescriptor(dct_fft_backward_[dim]), "DftiCommitDescriptor (DCT/DST backward)");
        }

        // Precompute twiddle factors for each dimension
        precomputeTwiddleFactors();
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Precompute twiddle factors for O(N log N) DCT/DST
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::precomputeTwiddleFactors()
{
    const double PI = std::numbers::pi;
    sin_tables_.resize(DIM);
    cos_tables_.resize(DIM);

    for (int dim = 0; dim < DIM; ++dim)
    {
        int n = nx_[dim];

        // Twiddle factors for DCT-II/III: cos(k*pi/(2*n)), sin(k*pi/(2*n))
        // We need k = 0, 1, ..., n/2
        int num_twiddles = n / 2 + 1;
        cos_tables_[dim].resize(num_twiddles);
        sin_tables_[dim].resize(num_twiddles);

        for (int k = 0; k <= n / 2; ++k)
        {
            cos_tables_[dim][k] = std::cos(k * PI / (2.0 * n));
            sin_tables_[dim][k] = std::sin(k * PI / (2.0 * n));
        }
    }
}

//------------------------------------------------------------------------------
// Get strides for dimension-by-dimension transform
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::getStrides(int dim, int& stride, int& num_transforms) const
{
    stride = 1;
    for (int d = dim + 1; d < DIM; ++d)
        stride *= nx_[d];

    num_transforms = 1;
    for (int d = 0; d < dim; ++d)
        num_transforms *= nx_[d];
}

//------------------------------------------------------------------------------
// DCT-II Forward via FFT (O(N log N), cuHelmholtz algorithm; MklDctLines.h)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::applyDCT2Forward(double* data, double* temp, int dim)
{
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);
    // 0.5 x FFTW's REDFT10, the standard DCT-II (as FftwFFT)
    mkl_dct2_lines(data, temp, nx_[dim], stride, num_transforms, dct_fft_backward_[dim],
                   cos_tables_[dim].data(), sin_tables_[dim].data(), 0.5);
}

//------------------------------------------------------------------------------
// DCT-III Backward via FFT (O(N log N), cuHelmholtz algorithm; MklDctLines.h)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::applyDCT3Backward(double* data, double* temp, int dim)
{
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);
    // FFTW's REDFT01 / n, the inverse of applyDCT2Forward
    mkl_dct3_lines(data, temp, nx_[dim], stride, num_transforms, dct_fft_forward_[dim],
                   cos_tables_[dim].data(), sin_tables_[dim].data(), 1.0 / nx_[dim]);
}

//------------------------------------------------------------------------------
// DST-II Forward via FFT (O(N log N))
// Based on cuHelmholtz FST algorithm (Makhoul 1980)
// DST-II: preprocess → IFFT → postprocess with twiddles
//
// FFTW DST-II (RODFT10) formula:
// Y_k = 2 * sum_{j=0}^{N-1} X_j * sin(π*(2j+1)*(k+1)/(2N))
// Our wrapper scales output by 0.5 to get standard DST-II.
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::applyDST2Forward(double* data, double* temp, int dim)
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    const double* cos_tbl = cos_tables_[dim].data();
    const double* sin_tbl = sin_tables_[dim].data();

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

            // DST-II preprocessing: create complex for Z2D FFT
            // Based on cuHelmholtz kernel_dst2_preOp
            fft_in[0] = std::complex<double>(slice[0], 0.0);
            for (int k = 1; k < complex_size - 1; ++k)
            {
                double x_2k = slice[2 * k];
                double x_2k_1 = slice[2 * k - 1];
                // DST preOp: different signs from DCT
                fft_in[k] = std::complex<double>((x_2k - x_2k_1) / 2.0, -((x_2k + x_2k_1) / 2.0));
            }
            if (n % 2 == 0)
            {
                // Even n: last complex element
                fft_in[n / 2] = std::complex<double>(-slice[n - 1], 0.0);
            }
            else if (n > 1)  // odd n; a single point has only fft_in[0]
            {
                // Odd n: last complex element
                int k = n / 2;
                double x_2k = slice[2 * k];
                double x_2k_1 = slice[2 * k - 1];
                fft_in[k] = std::complex<double>((x_2k - x_2k_1) / 2.0, -((x_2k + x_2k_1) / 2.0));
            }

            // Inverse FFT (c2r)
            mkl_dfti_check(DftiComputeBackward(dct_fft_backward_[dim],
                                               reinterpret_cast<double*>(fft_in.data()),
                                               fft_out.data()),
                           "DftiComputeBackward (DST-II)");

            // DST-II postprocessing: apply twiddle factors
            // Based on cuHelmholtz kernel_dst2_postOp
            // Output mapping: CUDA writes to buffer[0..N], but our output is 0..N-1
            //
            // CUDA has: pin[0] = 0, pin[k] for k=1..N-1, pin[N] = DC*2
            // FFTW expects: y[0..N-1] where y[k] = DST-II coefficient k
            //
            // We shift CUDA's output: our y[k] = CUDA's pin[k+1]
            // So we compute pin[1..N] and store as y[0..N-1]
            //
            // CUDA postOp for k != 0:
            //   Ta = fft_out[k] + fft_out[N-k]
            //   Tb = fft_out[k] - fft_out[N-k]
            //   pin[k] = Ta * sin[k] + Tb * cos[k]
            //   pin[N-k] = Ta * cos[k] - Tb * sin[k]
            // For k = 0:
            //   pin[0] = 0 (not used in our output)
            //   pin[N] = fft_out[0] * 2

            // We need output y[0..N-1] matching FFTW
            // FFTW indexing: y[k] is coefficient for frequency k+1 (sine starts at 1)

            for (int k = 1; k <= n / 2; ++k)
            {
                double Ta = fft_out[k] + fft_out[n - k];
                double Tb = fft_out[k] - fft_out[n - k];

                // DST-II: use sin/cos (swapped from DCT postOp)
                // CUDA: pin[k] = Ta * sin + Tb * cos
                // CUDA: pin[N-k] = Ta * cos - Tb * sin
                // Scale by 0.5 to match FFTW convention
                double result_k = (Ta * sin_tbl[k] + Tb * cos_tbl[k]) * 0.5;
                double result_nk = (Ta * cos_tbl[k] - Tb * sin_tbl[k]) * 0.5;

                // Map: y[k-1] = pin[k], y[N-k-1] = pin[N-k]
                // But wait, this would give y[-1] for k=0 which is wrong
                // Actually we want: y[k] = pin[k+1] (0-indexed to 1-indexed shift)
                // So we compute pin[k] for k=1..N, and assign y[k-1] = pin[k]

                temp[offset + (k - 1) * stride] = result_k;
                if (k < n - k)
                    temp[offset + (n - k - 1) * stride] = result_nk;
            }
            // Middle element for even n (k = n/2): k = n - k, so pin[n/2] and
            // pin[N - n/2] are the same element (result_k == result_nk) and the
            // loop above has already written it once.
            // Handle pin[N] = fft_out[0] * 2 -> y[N-1]
            // This is the last coefficient
            temp[offset + (n - 1) * stride] = fft_out[0];  // CUDA has *2, scale by 0.5 = *1
        }
    }

    // Copy temp to data
    std::memcpy(data, temp, total_grid_ * sizeof(double));
}

//------------------------------------------------------------------------------
// DST-III Backward via FFT (O(N log N))
// Based on cuHelmholtz FST algorithm (Makhoul 1980)
// DST-III: preprocess with twiddles → FFT → postprocess
//
// FFTW DST-III (RODFT01) formula:
// Y_k = (-1)^k * X_{N-1} + 2 * sum_{j=0}^{N-2} X_j * sin(π*(2k+1)*(j+1)/(2N))
// We apply 1/N scaling for round-trip identity.
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::applyDST3Backward(double* data, double* temp, int dim)
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    const double* cos_tbl = cos_tables_[dim].data();
    const double* sin_tbl = sin_tables_[dim].data();

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
            // Input is FFTW-format DST coefficients: y[0..N-1]
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            // Map FFTW indexing to CUDA buffer indexing
            // FFTW y[k] is coefficient for frequency k+1
            // CUDA expects pin[k] for k=1..N (1-indexed)
            // So: pin[k] = y[k-1] for k=1..N
            //
            // For DST-III preOp, CUDA reads from buffer[1..N] and writes to buffer[0..N-1]
            // We need to adapt this for our 0-indexed arrays

            // DST-III preprocessing: apply twiddle factors
            // Based on cuHelmholtz kernel_dst3_preOp
            // For k = 0..N/2:
            //   Ta = pin[k] + pin[N-k]
            //   Tb = pin[k] - pin[N-k]
            //   pin[k] = Ta * cos[k] + Tb * sin[k]
            //   pin[N-k] = Ta * sin[k] - Tb * cos[k]
            //
            // But CUDA works on buffer[0..N] where buffer[0]=0 for boundary
            // We adapt: work on fft_in[0..N-1], treating it as if there's a 0 at position N

            // First, create buffer with boundary condition:
            // pin_cuda[0] = 0 (implicit)
            // pin_cuda[k] = slice[k-1] for k=1..N-1
            // pin_cuda[N] = slice[N-1] (the last coefficient maps to position N)
            //
            // Actually looking at kernel_dst3_preOp more carefully:
            // It reads pin[k] and pin[N-k] where pin is size N+2 (for complex array)
            // For k=0..N/2, it processes pairs (k, N-k)

            // For our implementation, let's use slice as the "pin" array shifted by 1
            // slice_extended[k] = slice[k-1] for k=1..N, slice_extended[0] = 0
            // But this is just a conceptual mapping

            // Apply twiddles to prepare for FFT
            fft_in[0] = 0.0;  // DST boundary condition: coefficient 0 is always 0
            for (int k = 1; k <= n / 2; ++k)
            {
                // CUDA reads pin[k] and pin[N-k]
                // Our mapping: pin_cuda[k] = slice[k-1] for k >= 1
                // pin_cuda[N] is slice[N-1] (the last element)
                double val_k = slice[k - 1];
                double val_nk = (k == n - k) ? val_k : ((n - k - 1 >= 0) ? slice[n - k - 1] : 0.0);

                if (k == n)  // This shouldn't happen since k <= n/2
                    val_nk = slice[n - 1];

                double Ta = val_k + val_nk;
                double Tb = val_k - val_nk;

                // DST-III preOp: pin[k] = Ta*cos + Tb*sin, pin[N-k] = Ta*sin - Tb*cos
                fft_in[k] = Ta * cos_tbl[k] + Tb * sin_tbl[k];
                if (k < n - k)
                    fft_in[n - k] = Ta * sin_tbl[k] - Tb * cos_tbl[k];
            }
            // Middle element for even n: when k = n/2, k = n-k, so fft_in[n/2]
            // is set exactly once by the loop above; nothing else to do.

            // Forward FFT (r2c)
            mkl_dfti_check(DftiComputeForward(dct_fft_forward_[dim], fft_in.data(), fft_out.data()),
                           "DftiComputeForward (DST-III)");

            // DST-III postprocessing: extract real output from complex FFT
            // Based on cuHelmholtz kernel_dst3_postOp
            // CUDA post-processes and writes to buffer[0..N]
            // Output: buffer[0] = 0, buffer[1..N] = DST-III result
            //
            // kernel_dst3_postOp:
            //   Negate imaginary parts
            //   pin[0] = 0
            //   pin[1] = sh_in[0]  (real part of DC)
            //   pin[2*k] = sh_in[k*2+1] - sh_in[k*2]  (imag - real)
            //   pin[2*k+1] = sh_in[k*2] + sh_in[k*2+1]  (real + imag)
            //
            // sh_in is the FFT output as interleaved real/imag
            // sh_in[k*2] = real part of fft_out[k]
            // sh_in[k*2+1] = imag part of fft_out[k]
            //
            // After negating imag: sh_in[k*2+1] = -imag

            double scale = 1.0 / n;

            // Extract result with scaling
            // Our output temp[0..N-1] = CUDA's buffer[1..N] with 1/n scaling
            temp[offset] = fft_out[0].real() * scale;  // CUDA: pin[1] = sh_in[0]
            for (int k = 1; k <= n / 2; ++k)
            {
                double re = fft_out[k].real();
                double im = -fft_out[k].imag();  // Negate as per kernel_dst3_postOp

                // CUDA: pin[2*k] = im - re, pin[2*k+1] = re + im
                // These go to our output indices 2*k-1 and 2*k (shifted by 1)
                if (2 * k - 1 < n)
                    temp[offset + (2 * k - 1) * stride] = (im - re) * scale;
                if (2 * k < n)
                    temp[offset + (2 * k) * stride] = (re + im) * scale;
            }

            // The FFT-based preprocessing loop above (k = 1..n/2) never reads the
            // highest DST-III coefficient slice[n-1], so the Nyquist mode was being
            // dropped (backward of a delta at k=n-1 returned all zeros instead of the
            // correct (1/n)(-1)^i, which made absorbing-BC results differ from FFTW/CUDA
            // for fields with (-1)^i content). Add that mode's contribution explicitly.
            double c_high = slice[n - 1] * scale;
            for (int i = 0; i < n; ++i)
                temp[offset + i * stride] += (i % 2 == 0 ? c_high : -c_high);
        }
    }

    // Copy temp to data
    std::memcpy(data, temp, total_grid_ * sizeof(double));
}

//------------------------------------------------------------------------------
// Forward (complex output interface - FFT<T> compatibility)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::forward(T *rdata, std::complex<double> *cdata)
{
    if (!is_all_periodic_)
    {
        throw_with_line_number("Complex output interface requires all periodic BC. "
                              "Use forward(T*, double*) for non-periodic BC.");
    }

    mkl_dfti_check(DftiComputeForward(hand_forward_, rdata, cdata), "DftiComputeForward (periodic)");
}

//------------------------------------------------------------------------------
// Backward (complex input interface - FFT<T> compatibility)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::backward(std::complex<double> *cdata, T *rdata)
{
    if (!is_all_periodic_)
    {
        throw_with_line_number("Complex input interface requires all periodic BC. "
                              "Use backward(double*, T*) for non-periodic BC.");
    }

    mkl_dfti_check(DftiComputeBackward(hand_backward_, cdata, rdata), "DftiComputeBackward (periodic)");
}

//------------------------------------------------------------------------------
// Forward (real coefficient interface - all BCs)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::forward(T *rdata, double *cdata)
{
    try
    {
        if (is_all_periodic_)
        {
            // Use MKL DFTI (treat double* as complex<double>*)
            std::complex<double>* cdata_complex = reinterpret_cast<std::complex<double>*>(cdata);
            mkl_dfti_check(DftiComputeForward(hand_forward_, rdata, cdata_complex), "DftiComputeForward (periodic)");
        }
        else
        {
            // DCT/DST: dimension-by-dimension transform using O(N log N) FFT
            thread_local std::vector<double> work;
            thread_local std::vector<double> temp;

            if (static_cast<int>(work.size()) < total_grid_)
            {
                work.resize(total_grid_);
                temp.resize(total_grid_);
            }

            // The DCT/DST is a real-linear operator: for complex fields
            // (CL-FTS) transform the real and imaginary parts independently
            // and store the coefficients interleaved as complex pairs
            // (cdata must then hold 2*total_complex_grid_ doubles, matching
            // the periodic complex layout used by the solvers).
            const int n_parts = std::is_same<T, double>::value ? 1 : 2;
            for (int part = 0; part < n_parts; ++part)
            {
                // Copy input component to work buffer
                for (int i = 0; i < total_grid_; ++i)
                {
                    if constexpr (std::is_same<T, double>::value)
                        work[i] = rdata[i];
                    else
                        work[i] = (part == 0) ? std::real(rdata[i])
                                              : std::imag(rdata[i]);
                }

                // Apply transforms dimension by dimension
                for (int dim = 0; dim < DIM; ++dim)
                {
                    if (bc_[dim] == BoundaryCondition::REFLECTING)
                        applyDCT2Forward(work.data(), temp.data(), dim);
                    else if (bc_[dim] == BoundaryCondition::ABSORBING)
                        applyDST2Forward(work.data(), temp.data(), dim);
                }

                // Copy to output (interleaved for complex fields)
                if constexpr (std::is_same<T, double>::value)
                {
                    for (int i = 0; i < total_complex_grid_; ++i)
                        cdata[i] = work[i];
                }
                else
                {
                    for (int i = 0; i < total_complex_grid_; ++i)
                        cdata[2 * i + part] = work[i];
                }
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Backward (real coefficient interface - all BCs)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::backward(double *cdata, T *rdata)
{
    try
    {
        if (is_all_periodic_)
        {
            // Use MKL DFTI (treat double* as complex<double>*)
            std::complex<double>* cdata_complex = reinterpret_cast<std::complex<double>*>(cdata);
            mkl_dfti_check(DftiComputeBackward(hand_backward_, cdata_complex, rdata), "DftiComputeBackward (periodic)");
        }
        else
        {
            // DCT/DST: dimension-by-dimension inverse transform using O(N log N) FFT
            thread_local std::vector<double> work;
            thread_local std::vector<double> temp;

            if (static_cast<int>(work.size()) < total_grid_)
            {
                work.resize(total_grid_);
                temp.resize(total_grid_);
            }

            // Complex fields: coefficients arrive interleaved (see forward);
            // inverse-transform the real and imaginary parts independently.
            const int n_parts = std::is_same<T, double>::value ? 1 : 2;
            for (int part = 0; part < n_parts; ++part)
            {
                // Copy input component to work buffer
                if constexpr (std::is_same<T, double>::value)
                {
                    for (int i = 0; i < total_complex_grid_; ++i)
                        work[i] = cdata[i];
                }
                else
                {
                    for (int i = 0; i < total_complex_grid_; ++i)
                        work[i] = cdata[2 * i + part];
                }

                // Apply inverse transforms dimension by dimension (reverse order)
                for (int dim = DIM - 1; dim >= 0; --dim)
                {
                    if (bc_[dim] == BoundaryCondition::REFLECTING)
                        applyDCT3Backward(work.data(), temp.data(), dim);
                    else if (bc_[dim] == BoundaryCondition::ABSORBING)
                        applyDST3Backward(work.data(), temp.data(), dim);
                }

                // Copy to output component
                for (int i = 0; i < total_grid_; ++i)
                {
                    if constexpr (std::is_same<T, double>::value)
                        rdata[i] = work[i];
                    else if constexpr (!std::is_same<T, double>::value)
                    {
                        double* out = reinterpret_cast<double*>(rdata);
                        out[2 * i + part] = work[i];
                    }
                }
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiations
#include "TemplateInstantiations.h"
INSTANTIATE_FFT_CLASS(MklFFT);
